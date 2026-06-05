"""
GPU master scheduler — GPU-native rewriting.

Pattern matching uses the Turbo CSP solver (GPU path when CUDA.functional()).
Rewriting is applied in-place on the GPUACSet via deletion and addition
kernels, with no world round-trip through the CPU.

For PLAYER_RULE boxes the world is downloaded once per box execution so
the agent can see it; no re-upload is needed after rewriting.
"""

"""
Pre-allocated GPU scratch buffers for a single scheduler state.
Re-used across solve-and-rewrite cycles to eliminate per-step GPU allocations.
Fields that depend on world size (buf_hf_flat, buf_to_del) are grown on demand.
"""
mutable struct GPUScratchBuffers
    buf_domains       :: CuVector{UInt64}       # n_vars_max * nc
    buf_hf_flat       :: CuVector{UInt64}       # total hom-fwd words (grown as world grows)
    buf_hf_offs       :: CuVector{Int32}        # n_homs + 1
    buf_bytecodes     :: CuVector{TCNBytecode}  # max bytecodes
    buf_solutions     :: CuMatrix{Int32}        # n_vars_max × max_solutions
    buf_sol_count     :: CuVector{Int32}        # [1]
    buf_workspace     :: CuMatrix{UInt64}       # n_vars_max * MAX_CHUNKS × 16
    buf_type_mask     :: CuVector{UInt64}       # nc
    buf_to_del        :: CuVector{Bool}         # sum(n_alloc) (grown as world grows)
    buf_violation     :: CuVector{Int32}        # [1] dangling-check flag (0=ok, 1=violated)
    buf_attr_mask     :: CuVector{UInt64}       # nc
    buf_pushout_slots :: CuVector{Int32}        # staging for slot indices (B5/B6, grown on demand)
    buf_pushout_vals  :: CuVector{Int32}        # staging for fk/attr values (B5/B6, grown on demand)
    cached_hf_offs    :: Vector{Int32}          # B15: cached CPU copy of hom_fwd_offs; skip GPU upload when unchanged
    buf_match         :: CuVector{Int32}        # GPU-resident chosen match (n_vars_max)
    buf_ub_info       :: CuVector{Int32}        # [ub_var, n_subs, ok] for GPU branching-point detection
    buf_ub_elems      :: CuVector{Int32}        # element indices of first unbound variable (nc_max*64 capacity)
    buf_fired         :: CuVector{Int32}        # [1] — fired flag for single-sync native pipeline
    buf_g_type_offs   :: CuVector{Int32}        # type_idx → 0-based offset in buf_to_del (length = n_obj_types)
    buf_turbo_nextsub :: CuVector{Int32}        # [1] global atomic subproblem counter for turbo_block_kernel!
    buf_sample_ws     :: CuMatrix{UInt64}       # per-thread descent workspace for "take N" sampling (grown on demand)
end

mutable struct GPUSchedulerState
    sched         :: CompiledGPUSched
    g             :: GPUACSet
    schema        :: SchemaInfo
    enc           :: AttributeEncoder
    world_type    :: Any
    agents        :: Dict
    trajectory    :: Union{GPUTrajectoryLog, Nothing}
    compact_every :: Int
    rng           :: Xoshiro
    turn          :: Ref{Int}
    step_log      :: Vector{NamedTuple}
    scratch       :: Union{GPUScratchBuffers, Nothing}   # nothing on CPU-only path
    graph_data      :: Union{GPUGraphData, Nothing}        # built lazily for GNN players
    graph_dirty     :: Bool                               # set true after each rewrite
    zone_partition  :: Any                                 # Union{ZonePartition,Nothing} — defined in ZonePartition.jl
    take            :: Any                                 # nothing | Int | (turn,box_idx)->Int : "take N" sampling cap
    sample_seed     :: Int                                 # base PRNG seed for "take N" sampling
end

# All CSPs reachable from a compiled schedule, including those nested inside
# agent-loop body sub-schedules.  The shared scratch buffers must be sized for
# the largest CSP that any box — top-level or agent-loop body — will solve.
function _all_csps(sched::CompiledGPUSched)
    csps = collect(sched.csps)
    for sub in sched.sub_schedules
        append!(csps, _all_csps(sub))
    end
    csps
end

function GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                            log_trajectory=false, compact_every=100,
                            take=nothing, sample_seed::Int=0)
    traj = log_trajectory ? GPUTrajectoryLog(schema) : nothing

    scratch = if CUDA.functional()
        # Include agent-loop body CSPs: a top-level agent sched keeps only the
        # interface CSP in sched.csps, while the body rule (often more variables)
        # lives in a sub-schedule and is solved via the same scratch buffers.
        all_csps = _all_csps(sched)
        max_nc = isempty(all_csps) ? 1 :
                 maximum(Int(csp.n_chunks) for csp in all_csps; init=1)
        nc_max = _select_nc_max(max_nc)
        nc     = max_nc   # for hf_flat sizing
        max_n_vars = isempty(all_csps) ? 1 :
                     maximum(Int(csp.n_vars) for csp in all_csps; init=1)
        max_n_bc   = isempty(all_csps) ? 1 :
                     maximum(length(csp.bytecodes) for csp in all_csps; init=1)
        n_homs     = length(schema.homs)
        total_alloc = sum(values(g.n_alloc); init=0)
        # initial hf_flat capacity: sum of per-hom source sizes × nc, × 4 headroom
        hf_flat_init = max(sum(max(get(g.n_alloc, schema.hom_dom[h], 0), 1) * nc
                               for h in schema.homs; init=1) * 4, 1)

        GPUScratchBuffers(
            CUDA.zeros(UInt64, max(max_n_vars * nc, 1)),
            CUDA.zeros(UInt64, hf_flat_init),
            CUDA.zeros(Int32,  n_homs + 1),
            CuArray{TCNBytecode}(undef, max(max_n_bc, 1)),
            CUDA.zeros(Int32,  max(max_n_vars, 1), 10_001),
            CUDA.zeros(Int32,  1),
            CUDA.zeros(UInt64, max(max_n_vars * nc_max, 1), 16),
            CUDA.zeros(UInt64, max(nc, 1)),
            CUDA.zeros(Bool,   max(total_alloc * 4, 1)),
            CUDA.zeros(Int32,  1),            # buf_violation
            CUDA.zeros(UInt64, max(nc, 1)),
            CUDA.zeros(Int32,  256),          # buf_pushout_slots initial capacity
            CUDA.zeros(Int32,  256),          # buf_pushout_vals initial capacity
            Int32[],                          # cached_hf_offs: empty = always upload on first call
            CUDA.zeros(Int32,  max(max_n_vars, 1)),  # buf_match: GPU-resident chosen match
            CUDA.zeros(Int32,  3),            # buf_ub_info: [ub_var, n_subs, ok]
            CUDA.zeros(Int32,  nc_max * 64),  # buf_ub_elems: max domain elements per variable
            CUDA.zeros(Int32,  1),            # buf_fired: single-sync fired flag
            CUDA.zeros(Int32,  max(length(schema.obj_types), 1)),  # buf_g_type_offs
            CUDA.zeros(Int32,  1),                                  # buf_turbo_nextsub
            CUDA.zeros(UInt64, 1, 1),                               # buf_sample_ws (grown on demand)
        )
    else
        nothing
    end

    GPUSchedulerState(sched, g, schema, enc, world_type, agents,
                      traj, compact_every, Xoshiro(42), Ref(1), NamedTuple[], scratch,
                      nothing, false, nothing, take, sample_seed)
end

# ── GPU kernels for domain and hom-forward building ───────────────────────────

# Build per-type active-element bitmask via atomic OR.
# One thread per source element; writes one bit per active element.
@kernel function _build_type_mask_kernel!(
    mask   :: AbstractVector{UInt64},
    active :: AbstractVector{Bool},
    nc     :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(active) && active[i]
        ci, bi = elem_to_chunk(i)
        if ci <= Int(nc)
            Atomix.@atomic mask[ci] |= UInt64(1) << bi
        end
    end
end

# Build hom-forward flat array from FK + active arrays.
# One thread per source element; each writes to a unique position — no atomics needed.
@kernel function _build_hom_fwd_kernel!(
    hom_fwd :: AbstractVector{UInt64},
    homs_h  :: AbstractVector{Int32},
    active  :: AbstractVector{Bool},
    off_h   :: Int32,   # 0-based word offset for this morphism in hom_fwd
    nc      :: Int32,
)
    w = @index(Global, Linear)
    if w <= length(active) && active[w]
        tgt = Int(homs_h[w])
        if tgt > 0
            ci, bi = elem_to_chunk(tgt)
            if ci <= Int(nc)
                hom_fwd[Int(off_h) + (w - 1) * Int(nc) + ci] = UInt64(1) << bi
            end
        end
    end
end

"""
Build flat hom_fwd and offset arrays entirely on GPU, writing into `scratch`
buffers.  Grows `scratch.buf_hf_flat` if the current world exceeds capacity.
Returns `(hf_flat_gpu, hf_offs_gpu)` views into the scratch buffers.
"""
function _build_hom_fwd_gpu!(backend, g::GPUACSet, schema::SchemaInfo, nc::Int,
                              scratch::GPUScratchBuffers)
    hom_fwd_offs = Int32[Int32(0)]
    total_words  = 0
    for h in schema.homs
        n = max(g.n_alloc[schema.hom_dom[h]], 1)
        total_words += n * nc
        push!(hom_fwd_offs, Int32(total_words))
    end
    total_words = max(total_words, 1)

    if length(scratch.buf_hf_flat) < total_words
        scratch.buf_hf_flat = CUDA.zeros(UInt64, total_words * 2)
    end
    hf_flat = @view scratch.buf_hf_flat[1:total_words]
    KernelAbstractions.fill!(hf_flat, UInt64(0))

    for (h_idx, h) in enumerate(schema.homs)
        n = g.n_alloc[schema.hom_dom[h]]
        n == 0 && continue
        off = hom_fwd_offs[h_idx]
        _build_hom_fwd_kernel!(backend, 256)(
            hf_flat, g.homs[h], g.active[schema.hom_dom[h]],
            off, Int32(nc); ndrange = n)
    end

    if scratch.cached_hf_offs != hom_fwd_offs
        KernelAbstractions.copyto!(backend, scratch.buf_hf_offs, hom_fwd_offs)
        scratch.cached_hf_offs = copy(hom_fwd_offs)
    end

    hf_flat, scratch.buf_hf_offs
end

"""
Fallback (no scratch): allocates fresh GPU buffers for hom_fwd.
Used by `turbo_homomorphisms` and the CPU test path.
"""
function _build_hom_fwd_gpu(backend, g::GPUACSet, schema::SchemaInfo, nc::Int)
    hom_fwd_offs = Int32[Int32(0)]
    total_words  = 0
    for h in schema.homs
        n = max(g.n_alloc[schema.hom_dom[h]], 1)
        total_words += n * nc
        push!(hom_fwd_offs, Int32(total_words))
    end
    total_words = max(total_words, 1)

    hf_flat = KernelAbstractions.allocate(backend, UInt64, total_words)
    KernelAbstractions.fill!(hf_flat, UInt64(0))

    for (h_idx, h) in enumerate(schema.homs)
        n = g.n_alloc[schema.hom_dom[h]]
        n == 0 && continue
        off = hom_fwd_offs[h_idx]
        _build_hom_fwd_kernel!(backend, 256)(
            hf_flat, g.homs[h], g.active[schema.hom_dom[h]],
            off, Int32(nc); ndrange = n)
    end

    hf_offs = KernelAbstractions.allocate(backend, Int32, length(hom_fwd_offs))
    KernelAbstractions.copyto!(backend, hf_offs, hom_fwd_offs)

    hf_flat, hf_offs
end

"""
Build CSP domain array into `scratch.buf_domains` entirely on GPU.
Reuses `scratch.buf_type_mask` for the per-type bitmask (no per-call allocations).
No explicit synchronize after `_build_type_mask_kernel!` — the subsequent
GPU-to-GPU copyto! is ordered by CUDA's implicit stream dependencies.
"""
function _build_domains_gpu!(backend, csp::CSPProblem, g::GPUACSet, schema::SchemaInfo,
                              scratch::GPUScratchBuffers)
    nc = csp.n_chunks
    nv = Int(csp.n_vars)
    d  = @view scratch.buf_domains[1:max(nv * nc, 1)]
    KernelAbstractions.fill!(d, UInt64(0))
    isempty(csp.var_offset) && return d

    type_mask = scratch.buf_type_mask
    type_bases = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        n         = g.n_alloc[o]
        n == 0 && continue

        KernelAbstractions.fill!(type_mask, UInt64(0))
        _build_type_mask_kernel!(backend, 256)(
            type_mask, g.active[o], Int32(nc); ndrange = n)

        for v in base:(next_base - 1)
            off = (v - 1) * nc
            copyto!(d, off + 1, type_mask, 1, nc)
        end
    end
    d
end

"""
Build CSP domain array entirely on GPU.
Uses the chunked flat layout: `domains[(v-1)*nc + c]` = chunk c of variable v.
Solutions will be GPU-local slot indices.
"""
function _build_domains_gpu(backend, csp::CSPProblem, g::GPUACSet, schema::SchemaInfo)
    nc = csp.n_chunks
    nv = Int(csp.n_vars)
    d  = KernelAbstractions.allocate(backend, UInt64, max(nv * nc, 1))
    KernelAbstractions.fill!(d, UInt64(0))
    isempty(csp.var_offset) && return d

    type_bases = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        n_vars_o  = next_base - base
        n         = g.n_alloc[o]
        n == 0 && continue

        # Build per-type bitmask on GPU (nc words), then copy to each variable slot.
        # No explicit synchronize needed here: the subsequent GPU-to-GPU copyto! is
        # ordered by CUDA's implicit stream dependencies.
        type_mask = KernelAbstractions.allocate(backend, UInt64, nc)
        KernelAbstractions.fill!(type_mask, UInt64(0))
        _build_type_mask_kernel!(backend, 256)(
            type_mask, g.active[o], Int32(nc); ndrange = n)

        for v in base:(next_base - 1)
            off = (v - 1) * nc
            copyto!(d, off + 1, type_mask, 1, nc)
        end
    end
    d
end

# ── GPU attribute-mask kernels (B2) ──────────────────────────────────────────

# Pass 1: fill staging mask with bits for elements whose attribute == req.
@kernel function _attr_mask_fill_kernel!(
    mask   :: AbstractVector{UInt64},
    attrs  :: AbstractVector{Int32},
    active :: AbstractVector{Bool},
    req    :: Int32,
    nc     :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(active) && active[i] && attrs[i] == req
        ci, bi = elem_to_chunk(i)
        if ci <= Int(nc)
            Atomix.@atomic mask[ci] |= UInt64(1) << bi
        end
    end
end

# Pass 2: AND the staging mask into the variable's domain slice.
@kernel function _attr_mask_and_kernel!(
    domains :: AbstractVector{UInt64},
    mask    :: AbstractVector{UInt64},
    var_off :: Int32,
    nc      :: Int32,
)
    c = @index(Global, Linear)
    if c <= Int(nc)
        domains[Int(var_off) + c] &= mask[c]
    end
end

"""
Apply PROP_ATTR_EQ masks to a GPU-resident domain array.
When `scratch` is provided (GPU path), builds the mask entirely on-device
via two-pass kernel (no `Array(g.attrs[a])` download).  Falls back to the
CPU-download path when scratch is nothing.
"""
function _apply_attr_masks_gpu_device!(d_gpu, csp::CSPProblem,
                                        g::GPUACSet, schema::SchemaInfo,
                                        enc::AttributeEncoder,
                                        backend = nothing,
                                        scratch::Union{GPUScratchBuffers, Nothing} = nothing)
    nc = csp.n_chunks
    for bc in csp.bytecodes
        bc.op != PROP_ATTR_EQ && continue
        v     = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        n_elems = g.n_alloc[owner]

        off = Int32((v - 1) * nc)

        if scratch !== nothing && backend !== nothing && n_elems > 0
            # GPU-native two-pass mask build (no CPU round-trip)
            KernelAbstractions.fill!(scratch.buf_attr_mask, UInt64(0))
            _attr_mask_fill_kernel!(backend, 256)(
                scratch.buf_attr_mask, g.attrs[a], g.active[owner],
                req, Int32(nc); ndrange = n_elems)
            _attr_mask_and_kernel!(backend, 256)(
                d_gpu, scratch.buf_attr_mask, off, Int32(nc); ndrange = nc)
        else
            # CPU fallback: download attrs, build mask on host, re-upload
            h_attrs  = Array(g.attrs[a])
            n_capped = min(n_elems, nc * 64)
            mask     = zeros(UInt64, nc)
            for i in 1:n_capped
                h_attrs[i] == req || continue
                ci, bi = elem_to_chunk(i)
                ci <= nc && (mask[ci] |= UInt64(1) << bi)
            end
            dom_slice = Array(d_gpu[off+1:off+nc])
            for c in 1:nc; dom_slice[c] &= mask[c]; end
            copyto!(d_gpu, off+1, CuArray(dom_slice), 1, nc)
        end
    end
end

# ── CPU-path helpers (still used for non-CUDA backend and turbo_homomorphisms) ─

"""
Build CSP domains from the active-flag arrays of a GPUACSet (CPU path).
Uses the chunked flat layout: `domains[(v-1)*nc + c]` = chunk c of variable v.
"""
function _init_gpu_domains(csp::CSPProblem, g::GPUACSet, schema::SchemaInfo)
    nc  = csp.n_chunks
    nv  = Int(csp.n_vars)
    domains = zeros(UInt64, nv * nc)
    isempty(csp.var_offset) && return domains
    type_bases = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        host_active = Array(g.active[o])
        n_elems = min(length(host_active), nc * 64)
        mask = zeros(UInt64, nc)
        for i in 1:n_elems
            host_active[i] || continue
            ci, bi = elem_to_chunk(i)
            ci <= nc && (mask[ci] |= UInt64(1) << bi)
        end
        for v in base:(next_base - 1)
            off = (v - 1) * nc
            for c in 1:nc; domains[off + c] = mask[c]; end
        end
    end
    domains
end

"""Apply PROP_ATTR_EQ domain masks (CPU path, used for non-CUDA backend)."""
function _apply_attr_masks_gpu!(domains::Vector{UInt64}, csp::CSPProblem,
                                 g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder)
    nc = csp.n_chunks
    for bc in csp.bytecodes
        bc.op != PROP_ATTR_EQ && continue
        v     = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        h_attrs = Array(g.attrs[a])
        n_elems = min(g.n_alloc[owner], nc * 64)
        mask    = zeros(UInt64, nc)
        for i in 1:n_elems
            h_attrs[i] == req || continue
            ci, bi = elem_to_chunk(i)
            ci <= nc && (mask[ci] |= UInt64(1) << bi)
        end
        off = (v - 1) * nc
        for c in 1:nc; domains[off + c] &= mask[c]; end
    end
end

"""
Build hom-forward bitmask tables from the GPU-resident FK arrays (CPU path).
Used as fallback when CUDA is not available.
"""
function _recompute_hom_forward_gpu(g::GPUACSet, schema::SchemaInfo, nc::Int)
    hom_forward = Vector{UInt64}[]
    for h in schema.homs
        owner    = schema.hom_dom[h]
        n        = g.n_alloc[owner]
        host_fk  = Array(g.homs[h])
        host_act = Array(g.active[owner])
        fwd      = zeros(UInt64, max(n, 1) * nc)
        for w in 1:n
            host_act[w] || continue
            tgt = Int(host_fk[w])
            tgt > 0 || continue
            ci, bi = elem_to_chunk(tgt)
            ci <= nc && (fwd[(w-1)*nc + ci] |= UInt64(1) << bi)
        end
        push!(hom_forward, fwd)
    end
    hom_forward
end

# ── Index mapping between GPU-slot space and compact world ────────────────────

"""
Build bidirectional index maps between GPU-slot indices (with tombstones)
and compact-world indices (1-based among live elements).
Returns `(gpu_to_compact, compact_to_gpu)` per object type.
"""
function _gpu_to_compact_mapping(g::GPUACSet, schema::SchemaInfo)
    gpu_to_compact = Dict{Symbol, Vector{Int}}()
    compact_to_gpu = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        h_active    = Array(g.active[o])
        n           = length(h_active)
        g_to_c      = zeros(Int, n)
        c_to_g      = Int[]
        cursor      = 0
        for (i, alive) in enumerate(h_active)
            alive || continue
            cursor += 1
            g_to_c[i] = cursor
            push!(c_to_g, i)
        end
        gpu_to_compact[o] = g_to_c
        compact_to_gpu[o] = c_to_g
    end
    gpu_to_compact, compact_to_gpu
end

"""
Translate a CSP solution from GPU-slot indices to compact-world indices,
so it can be passed to `_assignment_to_hom` with a compact world ACSet.
"""
function _sol_gpu_to_compact(sol::Vector{Int32}, csp::CSPProblem,
                              schema::SchemaInfo,
                              gpu_to_compact::Dict{Symbol, Vector{Int}})
    compact = copy(sol)
    type_bases = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : Int(csp.n_vars) + 1
        g_to_c    = gpu_to_compact[o]
        for v in base:(next_base - 1)
            v > Int(csp.n_vars) && break
            gpu_idx = Int(sol[v])
            gpu_idx == 0 && continue
            compact[v] = Int32(gpu_idx <= length(g_to_c) ? g_to_c[gpu_idx] : 0)
        end
    end
    compact
end

# ── In-place GPU rewrite ───────────────────────────────────────────────────────

"""
Apply a DPO rewrite in-place on the GPUACSet.
`sol` contains GPU-local slot indices (as returned by the Turbo solver
when domains are built with `_init_gpu_domains`).
`scratch` is the pre-allocated buffer set from the parent `GPUSchedulerState`.
"""
function _gpu_apply_inplace!(g::GPUACSet, sol::Vector{Int32},
                              cube::AdhesiveCube, rule,
                              schema::SchemaInfo, enc::AttributeEncoder,
                              scratch::Union{GPUScratchBuffers, Nothing} = nothing;
                              gpu_cube::Union{GPUAdhesiveCube, Nothing} = nothing,
                              d_match::Union{AbstractVector{Int32}, Nothing} = nothing)
    backend = CUDA.functional() ? CUDA.CUDABackend() : CPU()

    # 1. Build deletion mask (CPU-computed, upload to GPU using pre-allocated buf_to_del)
    to_del = build_to_del_mask(sol, cube, schema, g; buf_to_del = scratch !== nothing ? scratch.buf_to_del : nothing)

    # 2. Dangling check entirely on GPU (single kernel + single sync)
    buf_viol = scratch !== nothing ? scratch.buf_violation : nothing
    gpu_dangling_ok(to_del, g, schema, backend; buf_violation = buf_viol) || return false

    # 3. Delete in-place via GPU kernel
    g_off = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        g_off[o] = cursor
        cursor += g.n_alloc[o]
    end

    for o in schema.obj_types
        n = g.n_alloc[o]
        n == 0 && continue
        off      = g_off[o]
        to_del_o = @view to_del[off+1:off+n]
        n_del    = Int(sum(to_del_o))
        n_del == 0 && continue
        dpo_deletion_kernel!(backend, 256)(g.active[o], to_del_o; ndrange=n)
        g.n_live[o][] -= n_del
    end

    # Snapshot pre-alloc before pushout modifies g.n_alloc (needed for FK target resolution)
    pre_alloc = (gpu_cube !== nothing && d_match !== nothing && CUDA.functional()) ?
                Dict{Symbol, Int32}(o => Int32(g.n_alloc[o]) for o in schema.obj_types) :
                nothing

    # 4. Add R\K elements via GPU scatter kernels
    r_to_local = apply_pushout!(g, sol, cube, rule, schema, enc;
                                scratch   = scratch,
                                gpu_cube  = gpu_cube,
                                d_match   = d_match)

    # 5. Patch preserved K elements that differ in R via GPU scatter kernels
    _update_preserved!(g, sol, cube, rule, schema, enc, r_to_local;
                       scratch    = scratch,
                       gpu_cube   = gpu_cube,
                       d_match    = d_match,
                       pre_alloc  = pre_alloc)

    KernelAbstractions.synchronize(backend)
    true
end

# ── Agent dispatch for PLAYER_RULE ────────────────────────────────────────────

"""
Build an `ACSetTransformation` directly from a CSP solution vector without
calling Catlab's `homomorphisms` search (B8A).  The solver already guarantees
that `compact_sol` satisfies all FK and attribute constraints, so the
construction is O(n_vars) rather than O(backtracking-search).

AttrVar bindings are left at their default value; downstream code that needs
concrete attribute values reads from `g.attrs[a]` via the encoder.
"""
function _sol_to_hom(compact_sol::Vector{Int32},
                     L, world_host,
                     csp::CSPProblem,
                     schema::SchemaInfo)
    comps = Dict{Symbol, Vector{Int}}()
    S = acset_schema(L)
    for o in ob(S)
        base = get(csp.var_offset, o, 0)
        base == 0 && continue
        n = nparts(L, o)
        comps[o] = [Int(compact_sol[base + i - 1]) for i in 1:n]
    end
    try
        ACSetTransformation(comps, L, world_host)
    catch
        nothing
    end
end

function _choose_gpu_match(solutions::Vector{Vector{Int32}},
                            agent, rule, csp::CSPProblem,
                            schema::SchemaInfo,
                            g::GPUACSet, enc::AttributeEncoder,
                            world_type, turn::Int;
                            can_pass::Bool = false,
                            state = nothing)
    agent === nothing && return solutions[1]

    # GPU player path: pass the (already NAC-filtered) candidates matrix directly.
    if agent isa AbstractGPUPlayer
        n_sols    = length(solutions)
        cands_raw = CUDA.functional() ? CuArray(reduce(hcat, solutions)) :
                                        reduce(hcat, solutions)
        if can_pass
            # Append a zero column representing the pass option.  FalconGPUPlayer
            # scores all n_presented columns; a zero-feature column lets the MLP
            # learn a pass score relative to real moves.
            pass_col  = CUDA.functional() ? CUDA.zeros(Int32, size(cands_raw, 1), 1) :
                                            zeros(Int32, size(cands_raw, 1), 1)
            cands_ext = hcat(cands_raw, pass_col)
        else
            cands_ext = cands_raw
        end
        n_presented = can_pass ? n_sols + 1 : n_sols
        # GNN players need the world graph; build/refresh it like the fast path
        # so conditional (NAC) rules — which always route here — work for every
        # AbstractGPUPlayer, not just those that ignore graph_data.
        gd = nothing
        if agent isa AbstractGNNPlayer && state !== nothing && CUDA.functional()
            backend = CUDA.CUDABackend()
            if state.graph_data === nothing
                state.graph_data = build_gpu_graph(g, schema, enc; backend)
                state.graph_dirty = false
            elseif state.graph_dirty
                rebuild_gpu_graph!(state.graph_data, g, schema, enc; backend)
                state.graph_dirty = false
            end
            gd = state.graph_data
        end
        idx = select_action_gpu(agent, g, enc, schema, cands_ext, n_presented, turn;
                                graph_data = gd)
        can_pass && Int(idx) > n_sols && return nothing   # player passed
        return solutions[clamp(Int(idx), 1, n_sols)]
    end

    inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
    if hasproperty(inner_rule, :rule) && hasmethod(left, Tuple{typeof(inner_rule.rule)})
        inner_rule = inner_rule.rule
    end
    hasmethod(left, Tuple{typeof(inner_rule)}) || return solutions[1]
    L = codom(left(inner_rule))

    # Download compact world once for agent's view (B8C: already done once here)
    gpu_to_compact, _ = _gpu_to_compact_mapping(g, schema)
    world_host = download_acset(g, enc, world_type)

    action_pairs = Pair{Action, Vector{Int32}}[]
    for sol in solutions
        compact_sol = _sol_gpu_to_compact(sol, csp, schema, gpu_to_compact)
        # B8A: build ACSetTransformation directly — no Catlab backtracking search
        hom = _sol_to_hom(compact_sol, L, world_host, csp, schema)
        hom === nothing && continue
        push!(action_pairs, Action(rule, hom) => sol)
    end
    if can_pass && !isempty(action_pairs)
        push!(action_pairs, Action(rule, nothing) => Int32[])
    end
    isempty(action_pairs) && return solutions[1]

    actions = [p.first for p in action_pairs]
    chosen  = select_action(agent, GameState(world_host, turn), actions)
    chosen  === nothing && return solutions[1]
    chosen.match === nothing && return nothing   # player passed

    for (act, sol) in action_pairs
        act === chosen && return sol
    end
    solutions[1]
end

# ── NAC/PAC post-filter (CPU-side) ───────────────────────────────────────────
#
# After the GPU/CPU solver finds candidate L-matches, we filter out any that
# trigger a Negative Application Condition (NAC) or fail to satisfy a Positive
# Application Condition (PAC).
#
# The GPU propagation kernel does not yet process NAC_REIF/PAC_REIF bytecodes
# (they require extra variables for elements introduced only in the extended
# pattern).  This CPU post-filter is the correct enforcement path until a
# full GPU reification implementation is added.

"""
    _is_nac_condition(cond) -> Bool

Returns `true` if `cond` is a Negative Application Condition.

`AppCond(f, false)` wraps its quantifier expression in `BoolNot`, which has an
`:expr` field but no `:kind` field.  `AppCond(f, true)` produces a `Quantifier`
directly, which has a `:kind` field.
"""
_is_nac_condition(cond) = !hasproperty(cond.d, :kind)

"""
    _solution_passes_conditions(compact_sol, conditions, L, world_host, csp, schema) -> Bool

Returns `false` if any NAC fires or any PAC is unsatisfied for `compact_sol`.

For each condition, the extended pattern `ac_L = codom(f)` is extracted from
`cond.g[1, :vlabel]` (vertex 1 of the AppCond CGraph).  The morphism
`f : L → ac_L` is at `cond.g[1, :elabel]` (edge 1).  L-element assignments
from `compact_sol` are pinned as the `initial` map for a hom-search from
`ac_L` into `world_host`; new NAC-only elements are searched freely.
"""
function _solution_passes_conditions(compact_sol::Vector{Int32},
                                     conditions,
                                     L,
                                     world_host,
                                     csp::CSPProblem,
                                     schema::SchemaInfo)
    for cond in conditions
        hasproperty(cond, :g)          || continue
        nparts(cond.g, :V) >= 1       || continue
        nparts(cond.g, :E) >= 1       || continue
        ac_L   = subpart(cond.g, 1, :vlabel)
        ac_L isa ACSet                 || continue
        f_edge = subpart(cond.g, 1, :elabel)
        f_edge isa ACSetTransformation || continue

        # Build the initial assignment: pin each L-element to its world value.
        S_L       = acset_schema(L)
        init_comps = Dict{Symbol, Dict{Int,Int}}()
        for o in ob(S_L)
            base = get(csp.var_offset, o, 0)
            base == 0 && continue
            n = nparts(L, o)
            n == 0 && continue
            d = Dict{Int,Int}()
            for i in 1:n
                j = f_edge[o](i)    # image of L-part i in ac_L
                w = Int(compact_sol[base + i - 1])
                w > 0 && (d[j] = w)
            end
            isempty(d) || (init_comps[o] = d)
        end

        init_nt  = NamedTuple(Symbol(k) => v for (k, v) in init_comps)
        # no_bind=true: allow free-floating attr vars (pre-allocated but unreferenced
        # in some NAC patterns); they are quantified freely over attr-var codomains.
        ext_homs = homomorphisms(ac_L, world_host; initial = init_nt, no_bind = true)

        if _is_nac_condition(cond) && !isempty(ext_homs)
            return false   # NAC fires → reject
        elseif !_is_nac_condition(cond) && isempty(ext_homs)
            return false   # PAC not satisfied → reject
        end
    end
    return true
end

"""
    _filter_nac_solutions(solutions, rule, csp, g, enc, schema, world_type)
        -> Vector{Vector{Int32}}

Downloads the world once and post-filters GPU solutions against all NAC/PAC
application conditions attached to the rewrite rule.  Returns only solutions
that satisfy all conditions.  Returns the input unchanged if the rule has no
conditions or if `L` cannot be recovered.
"""
function _filter_nac_solutions(solutions::Vector{Vector{Int32}},
                                rule,
                                csp::CSPProblem,
                                g::GPUACSet,
                                enc::AttributeEncoder,
                                schema::SchemaInfo,
                                world_type)
    inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
    hasproperty(inner_rule, :conditions)        || return solutions
    isempty(inner_rule.conditions)              && return solutions
    hasmethod(left, Tuple{typeof(inner_rule)})  || return solutions

    L          = codom(left(inner_rule))
    world_host = download_acset(g, enc, world_type)
    gpu_to_compact, _ = _gpu_to_compact_mapping(g, schema)

    filter(solutions) do sol
        compact = _sol_gpu_to_compact(sol, csp, schema, gpu_to_compact)
        _solution_passes_conditions(compact, inner_rule.conditions, L,
                                    world_host, csp, schema)
    end
end

# ── GPU-native NAC/PAC checking ───────────────────────────────────────────────
#
# The CPU `_filter_nac_solutions` downloads the whole world and runs a host-side
# homomorphism search per candidate.  For the common application-condition shape
# — a single new element whose foreign keys all point at *pinned* L-elements
# (e.g. rtb_bingo's "a FuelToken on this Platform", move's "a DestroyedPlatform
# on this Platform", isr's "a TargetFix for this target by this platform") — the
# check is just an existence query that can run on-device against `g.homs` /
# `g.active`, with no GPU→CPU world round-trip.
#
# `NacSpec` captures one such condition.  `_extract_nac_specs` returns `nothing`
# when *any* condition is not of this simple shape, so the caller falls back to
# the (fully general) CPU filter rather than checking a condition incorrectly.

struct NacSpec
    is_nac      :: Bool
    new_type    :: Symbol                      # type of the single new element
    constraints :: Vector{Tuple{Symbol, Int}}  # (fk hom, target CSP-variable index)
end

function _extract_nac_specs(rule, csp::CSPProblem, schema::SchemaInfo)
    inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
    hasproperty(inner_rule, :conditions)       || return NacSpec[]
    isempty(inner_rule.conditions)             && return NacSpec[]
    hasmethod(left, Tuple{typeof(inner_rule)}) || return nothing
    L = codom(left(inner_rule))

    specs = NacSpec[]
    for cond in inner_rule.conditions
        hasproperty(cond, :g)          || return nothing
        nparts(cond.g, :V) >= 1        || return nothing
        nparts(cond.g, :E) >= 1        || return nothing
        ac_L = subpart(cond.g, 1, :vlabel)
        ac_L isa ACSet                 || return nothing
        f_edge = subpart(cond.g, 1, :elabel)
        f_edge isa ACSetTransformation || return nothing
        S_ac = acset_schema(ac_L)

        # Pinned ac_L parts (the image of f_edge) and their L preimage indices.
        pinned = Dict{Symbol, Dict{Int,Int}}()   # type → (ac_L part → L part)
        for o in ob(acset_schema(L))
            d = Dict{Int,Int}()
            for i in parts(L, o)
                d[f_edge[o](i)] = i
            end
            pinned[o] = d
        end

        # New elements = ac_L parts not in the f_edge image.
        new_elems = Tuple{Symbol,Int}[]
        for o in ob(S_ac)
            img = get(pinned, o, Dict{Int,Int}())
            for p in parts(ac_L, o)
                haskey(img, p) || push!(new_elems, (o, p))
            end
        end
        length(new_elems) == 1 || return nothing   # only single-new-element NACs
        (T, p) = new_elems[1]

        constraints = Tuple{Symbol,Int}[]
        for h in schema.homs
            schema.hom_dom[h] == T || continue
            tgt = subpart(ac_L, p, h)
            tgt == 0 && continue
            cod   = schema.hom_cod[h]
            base  = get(csp.var_offset, cod, 0)
            base == 0 && return nothing
            Lpart = get(get(pinned, cod, Dict{Int,Int}()), tgt, 0)
            Lpart == 0 && return nothing            # FK points at another new elem
            push!(constraints, (h, base + Lpart - 1))
        end
        # Concrete (non-AttrVar) attributes on the new element aren't handled here.
        for a in schema.attrs
            schema.attr_dom[a] == T || continue
            subpart(ac_L, p, a) isa AttrVar || return nothing
        end

        push!(specs, NacSpec(_is_nac_condition(cond), T, constraints))
    end
    specs
end

# Does the world contain an instance of `spec.new_type` whose FKs match the
# pinned slots in `sol`?  Evaluated entirely on-device.
function _gpu_nac_exists(g::GPUACSet, spec::NacSpec, sol::Vector{Int32})::Bool
    T = spec.new_type
    n = get(g.n_alloc, T, 0)
    n == 0 && return false
    mask = copy(@view g.active[T][1:n])          # live instances of T
    for (h, var) in spec.constraints
        (var < 1 || var > length(sol)) && return false
        s    = sol[var]
        mask = mask .& (@view(g.homs[h][1:n]) .== s)
    end
    any(mask)
end

"""
    _gpu_filter_nac_solutions(solutions, rule, csp, g, schema)

GPU-resident NAC/PAC post-filter: returns the kept solutions, or `nothing` if
the rule has a condition this fast path does not support (caller then uses the
CPU `_filter_nac_solutions`).  Unlike the CPU path it never downloads the world.
"""
function _gpu_filter_nac_solutions(solutions::Vector{Vector{Int32}},
                                    rule, csp::CSPProblem,
                                    g::GPUACSet, schema::SchemaInfo)
    specs = _extract_nac_specs(rule, csp, schema)
    specs === nothing && return nothing
    isempty(specs)    && return solutions
    filter(solutions) do sol
        for spec in specs
            exists = _gpu_nac_exists(g, spec, sol)
            (spec.is_nac && exists)  && return false   # NAC fired
            (!spec.is_nac && !exists) && return false  # PAC unsatisfied
        end
        true
    end
end

# ── General GPU-native NAC/PAC checking via the rule solver ───────────────────
#
# The `NacSpec` fast path above only covers a single new element pinned to L.
# For a general application condition (e.g. sam_aim's NAC3: a fresh ThreatSystem
# AND a ShotAt referencing it), the principled check is exactly a homomorphism
# search of the condition's extended pattern `ac_L` into the world, with the
# shared L-elements pinned to the candidate match — and a NAC fires iff that
# search finds ANY extension (a PAC is satisfied iff it finds one).  We lower
# each condition's `ac_L` to its own `CSPProblem` (`lower_pattern_to_csp`) and
# run the SAME `gpu_dive_solve` a rule uses, so this needs no world download and
# no host-side Catlab homsearch.  This mirrors the CPU filter's
# `homomorphisms(ac_L, world; no_bind=true)` (non-monic, free AttrVars), so an
# existence test is exactly equivalent.

struct ConditionCSP
    csp    :: CSPProblem
    shared :: Vector{Tuple{Int,Int}}   # (ac_L variable, rule-L variable) for shared elements
    is_nac :: Bool
end

# Cache lowered condition CSPs per (rule identity, n_chunks).  The pattern
# structure is fixed; domains/hom-forward are rebuilt from live g at solve time.
const _COND_CSP_CACHE = IdDict{Any, Dict{Int, Union{Nothing, Vector{ConditionCSP}}}}()

function _build_condition_csps(rule, csp::CSPProblem, g::GPUACSet,
                               schema::SchemaInfo, enc::AttributeEncoder)
    inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
    hasproperty(inner_rule, :conditions)       || return ConditionCSP[]
    isempty(inner_rule.conditions)             && return ConditionCSP[]
    hasmethod(left, Tuple{typeof(inner_rule)}) || return nothing
    L  = codom(left(inner_rule))
    nc = csp.n_chunks
    n_alloc = Dict{Symbol,Int}(o => Int(get(g.n_alloc, o, 0)) for o in schema.obj_types)

    out = ConditionCSP[]
    for cond in inner_rule.conditions
        hasproperty(cond, :g) || return nothing
        ac_L = try
            raw = subpart(cond.g, 1, :vlabel); raw isa ACSet ? raw : (return nothing)
        catch; return nothing; end
        f = try
            subpart(cond.g, 1, :elabel)
        catch; return nothing; end
        f isa Catlab.CategoricalAlgebra.ACSetTransformation || return nothing

        ccsp = lower_pattern_to_csp(ac_L, schema, enc; n_chunks=nc, n_alloc=n_alloc)

        shared = Tuple{Int,Int}[]
        for o in ob(acset_schema(L))
            (haskey(csp.var_offset, o) && haskey(ccsp.var_offset, o)) || continue
            for i in parts(L, o)
                rv  = csp.var_offset[o]  + (i - 1)
                acp = f[o](i)
                cv  = ccsp.var_offset[o] + (acp - 1)
                push!(shared, (cv, rv))
            end
        end
        push!(out, ConditionCSP(ccsp, shared, _is_nac_condition(cond)))
    end
    out
end

function _condition_csps_cached(rule, csp::CSPProblem, g::GPUACSet,
                                schema::SchemaInfo, enc::AttributeEncoder)
    bync = get!(() -> Dict{Int, Union{Nothing, Vector{ConditionCSP}}}(),
                _COND_CSP_CACHE, rule)
    get!(() -> _build_condition_csps(rule, csp, g, schema, enc), bync, Int(csp.n_chunks))
end

# Pin a CSP variable to a single world slot, entirely on-device (no host scalar
# access).  For the variable's `nc` domain chunks based at `base`: AND chunk `ci`
# with the slot's bit (preserving any attribute mask there) and zero every other
# chunk.  The previous `@allowscalar` read+write cost two host↔device syncs per
# pin — a large fraction of the agent loop (315 pins/turn) and the NAC filter.
@kernel function _pin_var_kernel!(d, base::Int, nc::Int, ci::Int, bit::Int)
    c = @index(Global, Linear)
    @inbounds if c <= nc
        d[base + c] = (c == ci) ? (d[base + c] & (UInt64(1) << bit)) : UInt64(0)
    end
end

# Launch the pin for variable `v` (domain based at `(v-1)*nc`).  Async: the
# launch is stream-ordered before any subsequent solve kernel on `d_gpu`, so no
# explicit synchronize is needed here.
function _pin_var!(d_gpu, nc::Int, v::Int, slot::Int)
    ci, bi = elem_to_chunk(slot)
    (ci < 1 || ci > nc) && return
    base    = (v - 1) * nc
    backend = KernelAbstractions.get_backend(d_gpu)
    _pin_var_kernel!(backend, min(nc, 256))(d_gpu, base, nc, ci, bi; ndrange = nc)
    return
end

# Pin CSP variable `v` to world slot `slot` by ANDing its domain to {slot}.
function _pin_csp_var!(d_gpu, csp::CSPProblem, v::Int, slot::Int)
    _pin_var!(d_gpu, csp.n_chunks, v, slot)
end

# Per-candidate workspace is `rows*16` UInt64 words; cap the tile so the batched
# filter's extra GPU memory stays bounded (~64 MB) regardless of candidate count.
# Typical N fits in a single tile (one kernel launch per condition).
function _nac_batch_tile(rows::Int, N::Int)
    budget = 8_000_000                       # UInt64 words ≈ 64 MB
    per    = max(rows * 16, 1)
    cap    = max(div(budget, per), 1)
    return min(N, cap)
end

"""
    _gpu_filter_conditions(solutions, rule, csp, g, schema, enc) -> kept | nothing

GPU-native general NAC/PAC post-filter.  Returns the kept solutions, or `nothing`
if a condition could not be lowered (caller then uses the CPU `homomorphisms`
fallback).  Each condition is checked by pinning its shared-L variables to the
candidate match and running an existence dive on the condition pattern: NAC fires
on ≥1 solution, PAC requires ≥1.

Two implementations with identical kept-sets:
  * `_gpu_filter_conditions_batched` — one batched kernel launch per *tile* of
    candidates per condition, pinning + diving every candidate on-device.
    O(tiles) launches per condition instead of the serial path's
    O(N × shared_vars + N) per-candidate `_pin_var!`/dive launches.  Wins only
    once the candidate count `N` is large enough to amortise its per-solve
    upload/alloc overhead.
  * `_gpu_filter_conditions_serial` — the original per-candidate host loop.

Dispatch (default): batched when `N ≥ RG_NAC_BATCH_MIN` (the launch storm only
exists at large N; small solves keep the lighter serial path so small games do
not regress), serial otherwise.  Overrides: `RG_NO_BATCH_NAC` forces serial
everywhere (kill-switch / A-B reference); `RG_FORCE_BATCH_NAC` forces batched
at every N (so the batched kernel can be exercised on small worlds in tests).
"""
# Cached so the dispatcher does no per-solve ENV string-parse on the hot path.
# Read once; override at process start via env (tests set it before any solve).
const _NAC_BATCH_MIN = Ref(-1)
function _nac_batch_min()
    m = _NAC_BATCH_MIN[]
    m >= 0 && return m
    m = parse(Int, get(ENV, "RG_NAC_BATCH_MIN", "32"))
    _NAC_BATCH_MIN[] = m
    return m
end

function _gpu_filter_conditions(solutions::Vector{Vector{Int32}}, rule,
                                csp::CSPProblem, g::GPUACSet,
                                schema::SchemaInfo, enc::AttributeEncoder)
    haskey(ENV, "RG_NO_BATCH_NAC") &&
        return _gpu_filter_conditions_serial(solutions, rule, csp, g, schema, enc)
    if !haskey(ENV, "RG_FORCE_BATCH_NAC") && length(solutions) < _nac_batch_min()
        return _gpu_filter_conditions_serial(solutions, rule, csp, g, schema, enc)
    end
    return _gpu_filter_conditions_batched(solutions, rule, csp, g, schema, enc)
end

# Batched candidate filter: see `_gpu_filter_conditions` docstring.  For each
# condition, all N candidates are checked by `nac_exist_batch_kernel!` — one
# work-item per candidate that copies the shared base domain `d0` into its own
# workspace slab, pins its candidate's shared-L variables on-device, and runs an
# existence dive.  Candidates are tiled to bound workspace memory; tiles on one
# stream are serialised so they safely reuse the `work` buffer.
function _gpu_filter_conditions_batched(solutions::Vector{Vector{Int32}}, rule,
                                csp::CSPProblem, g::GPUACSet,
                                schema::SchemaInfo, enc::AttributeEncoder)
    haskey(ENV, "RG_FORCE_CPU_NAC") && return nothing
    conds = _condition_csps_cached(rule, csp, g, schema, enc)
    conds === nothing && return nothing
    isempty(conds)    && return solutions
    N = length(solutions)
    N == 0 && return solutions

    # A uniform solution length lets us pin from a dense [R × N] device matrix and
    # filter shared pairs once per condition — equivalent to the serial path's
    # per-candidate `rv > length(sol)` guard.  Differing lengths shouldn't occur
    # (all solutions span the rule's vars); fall back to serial if they do.
    R = length(solutions[1])
    all(s -> length(s) == R, solutions) ||
        return _gpu_filter_conditions_serial(solutions, rule, csp, g, schema, enc)

    backend = CUDA.CUDABackend()
    nc      = csp.n_chunks
    nc_max  = _select_nc_max(nc)
    hf_flat, hf_offs = _build_hom_fwd_gpu(backend, g, schema, nc)   # world-only; shared by all conds

    # Upload all candidate solutions ONCE (not per candidate, as the serial path
    # implicitly did by reading `solutions[i]` each iteration).
    sol_host = Matrix{Int32}(undef, max(R, 1), N)
    @inbounds for i in 1:N
        s = solutions[i]
        for r in 1:R; sol_host[r, i] = s[r]; end
    end
    sol_gpu = KernelAbstractions.allocate(backend, Int32, max(R, 1), N)
    KernelAbstractions.copyto!(backend, sol_gpu, sol_host)

    keep     = trues(N)
    keep_i32 = Vector{Int32}(undef, N)
    keep_gpu = KernelAbstractions.allocate(backend, Int32, N)
    cnt_gpu  = KernelAbstractions.allocate(backend, Int32, N)
    kern     = nac_exist_batch_kernel!(backend)

    for cc in conds
        ccsp    = cc.csp
        cn_vars = Int(ccsp.n_vars)
        n_bc    = length(ccsp.bytecodes)

        d0 = _build_domains_gpu(backend, ccsp, g, schema)              # world domains
        _apply_attr_masks_gpu_device!(d0, ccsp, g, schema, enc)        # concrete-attr masks
        b_gpu = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, ccsp.bytecodes)

        # Shared-L pins for this condition (drop invalid rule vars once, up front).
        cv_host = Int32[]; rv_host = Int32[]
        for (cv, rv) in cc.shared
            (rv < 1 || rv > R) && continue
            push!(cv_host, Int32(cv)); push!(rv_host, Int32(rv))
        end
        n_pin  = length(cv_host)
        cv_gpu = KernelAbstractions.allocate(backend, Int32, max(n_pin, 1))
        rv_gpu = KernelAbstractions.allocate(backend, Int32, max(n_pin, 1))
        n_pin > 0 && KernelAbstractions.copyto!(backend, cv_gpu, cv_host)
        n_pin > 0 && KernelAbstractions.copyto!(backend, rv_gpu, rv_host)

        # Upload the current keep mask (lets the kernel skip already-rejected
        # candidates, matching the serial `keep[i] || continue`); reset counts.
        @inbounds for i in 1:N; keep_i32[i] = keep[i] ? Int32(1) : Int32(0); end
        KernelAbstractions.copyto!(backend, keep_gpu, keep_i32)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))

        rows = max(cn_vars * nc_max, 1)
        tile = _nac_batch_tile(rows, N)
        work = KernelAbstractions.allocate(backend, UInt64, rows, 16, tile)

        base = 0
        while base < N
            this = min(tile, N - base)
            kern(d0, b_gpu, n_bc, cn_vars, nc,
                 cv_gpu, rv_gpu, n_pin, sol_gpu, R, N, base,
                 keep_gpu, cnt_gpu, work, hf_flat, hf_offs, Val(nc_max);
                 ndrange = this)
            base += this
        end
        KernelAbstractions.synchronize(backend)        # ONE sync per condition

        e = Array(cnt_gpu)
        @inbounds for i in 1:N
            keep[i] || continue
            exists = e[i] > 0
            ((cc.is_nac && exists) || (!cc.is_nac && !exists)) && (keep[i] = false)
        end
    end
    solutions[keep]
end

function _gpu_filter_conditions_serial(solutions::Vector{Vector{Int32}}, rule,
                                csp::CSPProblem, g::GPUACSet,
                                schema::SchemaInfo, enc::AttributeEncoder)
    haskey(ENV, "RG_FORCE_CPU_NAC") && return nothing
    conds = _condition_csps_cached(rule, csp, g, schema, enc)
    conds === nothing && return nothing
    isempty(conds)    && return solutions
    N = length(solutions)
    N == 0 && return solutions

    backend = CUDA.CUDABackend()
    nc = csp.n_chunks
    hf_flat, hf_offs = _build_hom_fwd_gpu(backend, g, schema, nc)   # world-only; shared by all conds

    # For each condition, check all N candidates with a SINGLE host sync: queue
    # one (stream-ordered) existence sub-solve per candidate, each writing its own
    # count slot, then read all N counts at once.  The previous code synced once
    # per (candidate × condition) — N×C host round-trips, the dominant full-game
    # cost.  The per-candidate dives reuse one domain/workspace buffer; stream
    # ordering serialises them (dive i completes before candidate i+1 overwrites
    # the domain), so this is the same per-candidate existence test, batched.
    keep = trues(N)
    for cc in conds
        ccsp    = cc.csp
        cn_vars = Int(ccsp.n_vars)
        n_bc    = length(ccsp.bytecodes)
        nc_max  = _select_nc_max(nc)

        d0 = _build_domains_gpu(backend, ccsp, g, schema)              # world domains
        _apply_attr_masks_gpu_device!(d0, ccsp, g, schema, enc)        # concrete-attr masks
        d        = KernelAbstractions.allocate(backend, UInt64, max(cn_vars * nc, 1))
        b_gpu    = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, ccsp.bytecodes)
        sol_gpu  = KernelAbstractions.allocate(backend, Int32, max(cn_vars, 1), 1)  # existence only
        work_gpu = KernelAbstractions.allocate(backend, UInt64, max(cn_vars, 1) * MAX_CHUNKS, 16)
        cnt_all  = KernelAbstractions.allocate(backend, Int32, N)
        KernelAbstractions.fill!(cnt_all, Int32(0))
        kern = dive_solve_kernel!(backend)

        @inbounds for i in 1:N
            keep[i] || continue                       # already rejected by an earlier condition
            sol = solutions[i]
            KernelAbstractions.copyto!(backend, d, d0)
            for (cv, rv) in cc.shared
                (rv < 1 || rv > length(sol)) && continue
                _pin_var!(d, nc, cv, Int(sol[rv]))    # on-device pin (async)
            end
            # max_solutions=1: we only need existence (count>0); one solution slot.
            kern(d, b_gpu, n_bc, cn_vars, nc, sol_gpu, view(cnt_all, i:i), 1,
                 work_gpu, hf_flat, hf_offs, Val(nc_max); ndrange = 1)
        end
        KernelAbstractions.synchronize(backend)        # ONE sync for all N candidates
        e = Array(cnt_all)
        @inbounds for i in 1:N
            keep[i] || continue
            exists = e[i] > 0
            ((cc.is_nac && exists) || (!cc.is_nac && !exists)) && (keep[i] = false)
        end
    end
    solutions[keep]
end

# Equivalence harness (env RG_NAC_DIAG): compare the ACTUAL kept solution sets
# from the GPU-native filter vs the CPU homsearch filter, for the same
# solve/world.  This is the true correctness check — immune to RNG/ordering: if
# the sets ever differ, the GPU NAC is wrong; if they never differ, any
# trajectory divergence between the two paths is only RNG consumption (the CPU
# Catlab homsearch touches the global RNG; the GPU kernels do not), not a NAC
# logic difference.  Counters let a test assert zero mismatches over many solves.
const _NAC_DIAG_CHECKS = Ref(0)
const _NAC_DIAG_MISM   = Ref(0)
function _nac_diag(solutions::Vector{Vector{Int32}}, rule, csp::CSPProblem,
                   g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder, world_type)
    saved = get(ENV, "RG_FORCE_CPU_NAC", nothing)
    saved !== nothing && delete!(ENV, "RG_FORCE_CPU_NAC")     # force the GPU path on
    kept_gpu = _gpu_filter_conditions(copy(solutions), rule, csp, g, schema, enc)
    saved !== nothing && (ENV["RG_FORCE_CPU_NAC"] = saved)
    kept_gpu === nothing && return
    kept_cpu = _filter_nac_solutions(copy(solutions), rule, csp, g, enc, schema, world_type)
    _NAC_DIAG_CHECKS[] += 1
    if Set(kept_gpu) != Set(kept_cpu)
        _NAC_DIAG_MISM[] += 1
        @warn "NAC SET mismatch" ngpu=length(kept_gpu) ncpu=length(kept_cpu) nsol=length(solutions) maxlog=40
    end
end

# ── Agent-loop per-instance pin ───────────────────────────────────────────────
#
# Inside a BOX_AGENT_LOOP the body must be solved once per agent instance, with
# the interface element pinned to that instance — mirroring the CPU runner's
# `homomorphisms(L, world; initial=initial_map)` (sched_runner.jl).  The GPU port
# previously dropped this pin and re-solved the body globally every iteration
# (O(instances²)).  `_pin_agent_var!` constrains the domain of the agent
# object's CSP variable to the single live slot `slot`, so the body solve only
# considers that instance's matches.  No-op when this csp has no variable of the
# agent object's type.  `agent_pin = (agent_obj::Symbol, slot::Int)`.
function _pin_agent_var!(d_gpu, csp::CSPProblem, agent_pin::Tuple{Symbol,Int})
    agent_obj, slot = agent_pin
    va = get(csp.var_offset, agent_obj, 0)
    va == 0 && return
    _pin_var!(d_gpu, csp.n_chunks, va, slot)   # on-device pin (no host scalar sync)
end

# ── Per-box solve-and-apply ───────────────────────────────────────────────────

# Resolve the "take N" sampling cap for a given box / turn.  `state.take` may be
# `nothing` (no sampling), an `Int`, or a function `(turn, box_idx) -> Int|nothing`
# (for annealing schedules).  Returns `nothing` to mean "present all matches".
function _effective_take(state, b_idx::Int, turn::Int)
    t = state.take
    t === nothing && return nothing
    val = t isa Function ? t(turn, b_idx) : t
    (val === nothing || val <= 0) && return nothing
    return Int(val)
end

function _gpu_solve_inplace!(g::GPUACSet, csp::CSPProblem, rule,
                              cube::AdhesiveCube,
                              gpu_cube::Union{GPUAdhesiveCube, Nothing},
                              schema::SchemaInfo, enc::AttributeEncoder,
                              box, b_idx::Int, sched, state, turn::Int;
                              agent_pin::Union{Nothing,Tuple{Symbol,Int}}=nothing)::Bool
    scratch = state.scratch

    # Detect agent early: GPU players get a fast path that skips bulk download
    player_sym = box.box_type == BOX_PLAYER_RULE ? sched.box_players[b_idx] : nothing
    agent      = player_sym !== nothing ? get(state.agents, player_sym, nothing) : nothing

    can_pass = box.box_type == BOX_PLAYER_RULE && box.params[1] > 0f0

    chosen_sol = if agent isa AbstractGPUPlayer && CUDA.functional() && scratch !== nothing &&
                    !_rule_has_conditions(rule)
        # ── GPU player fast path: fill scratch.buf_solutions on device,
        #    download only the 4-byte count + one chosen column ───────────────
        # NOTE: this fast path presents raw CSP solutions to the player WITHOUT
        # running _filter_nac_solutions, so it is only safe for rules with no
        # NAC/PAC conditions.  Conditional rules fall through to the standard
        # path below, which filters NAC-violating candidates before the player
        # ever sees them.  (NAC enforcement requires a host-side hom search over
        # the extended pattern; there is no GPU NAC reification yet.)
        backend          = CUDA.CUDABackend()
        nc               = csp.n_chunks
        hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, nc, scratch)
        d_gpu            = _build_domains_gpu!(backend, csp, g, schema, scratch)
        _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc, backend, scratch)
        agent_pin !== nothing && _pin_agent_var!(d_gpu, csp, agent_pin)
        KernelAbstractions.synchronize(backend)

        # "take N": when a sampling cap is active for this box/turn, present a
        # count-weighted random subset of matches instead of the full set.
        take_eff = _effective_take(state, b_idx, turn)
        n_sols = take_eff === nothing ?
            _gpu_turbo_fill_scratch!(backend, csp, d_gpu, hf_flat, hf_offs;
                                     scratch = scratch) :
            _gpu_turbo_sample_scratch!(backend, csp, d_gpu, hf_flat, hf_offs;
                                       take = take_eff,
                                       seed = state.sample_seed + 1009 * turn + 9176 * b_idx,
                                       scratch = scratch)
        n_sols == 0 && return false

        n_vars      = Int(csp.n_vars)
        n_presented = can_pass ? n_sols + 1 : n_sols
        if can_pass
            # Zero column n_sols+1 (may be dirty from a previous turn) so
            # FalconGPUPlayer sees a clean zero-feature vector for the pass option.
            scratch.buf_solutions[1:n_vars, n_sols + 1] .= Int32(0)
            KernelAbstractions.synchronize(backend)
        end
        candidates = @view scratch.buf_solutions[1:n_vars, 1:n_presented]

        # Lazy graph build/refresh for GNN players
        gd = nothing
        if agent isa AbstractGNNPlayer
            if state.graph_data === nothing
                state.graph_data = build_gpu_graph(g, schema, enc; backend)
                state.graph_dirty = false
            elseif state.graph_dirty
                rebuild_gpu_graph!(state.graph_data, g, schema, enc; backend)
                state.graph_dirty = false
            end
            gd = state.graph_data
        end

        idx = select_action_gpu(agent, g, enc, schema, candidates, n_presented, turn;
                                 graph_data = gd)
        can_pass && Int(idx) > n_sols && return false   # player passed
        chosen_col = clamp(Int(idx), 1, n_sols)
        Array(@view scratch.buf_solutions[1:n_vars, chosen_col])
    else
        # ── Standard path: solve on GPU/CPU, download all solutions, choose ──
        solutions = if CUDA.functional() && scratch !== nothing
            backend          = CUDA.CUDABackend()
            nc               = csp.n_chunks
            hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, nc, scratch)
            d_gpu            = _build_domains_gpu!(backend, csp, g, schema, scratch)
            _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc, backend, scratch)
            agent_pin !== nothing && _pin_agent_var!(d_gpu, csp, agent_pin)
            KernelAbstractions.synchronize(backend)
            gpu_turbo_solve(backend, csp, d_gpu, hf_flat, hf_offs; scratch = scratch)
        else
            if CUDA.functional()
                backend          = CUDA.CUDABackend()
                nc               = csp.n_chunks
                hf_flat, hf_offs = _build_hom_fwd_gpu(backend, g, schema, nc)
                d_gpu            = _build_domains_gpu(backend, csp, g, schema)
                _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc)
                agent_pin !== nothing && _pin_agent_var!(d_gpu, csp, agent_pin)
                KernelAbstractions.synchronize(backend)
                gpu_dive_solve(backend, csp, d_gpu, hf_flat, hf_offs)
            else
                hf        = _recompute_hom_forward_gpu(g, schema, csp.n_chunks)
                fresh_csp = CSPProblem(csp.n_vars, csp.var_offset, csp.domain_sizes,
                                       csp.bytecodes, csp.nac_groups, csp.pac_groups,
                                       csp.agent_var_map, hf, csp.n_chunks,
                                       csp.sorted_type_bases)
                domains   = _init_gpu_domains(fresh_csp, g, schema)
                _apply_attr_masks_gpu!(domains, fresh_csp, g, schema, enc)
                agent_pin !== nothing && _pin_agent_var!(domains, fresh_csp, agent_pin)
                cpu_dive_solve(fresh_csp, domains)
            end
        end
        isempty(solutions) && return false

        # ── NAC/PAC post-filter ──
        # Tier 1: fast single-new-element GPU existence check (NacSpec).
        # Tier 2: general GPU-native check — lower each condition pattern and
        #         run the rule solver pinned to the match (no world download).
        # Tier 3: CPU host-side Catlab homsearch, only if a pattern won't lower.
        gpu_filtered = _gpu_filter_nac_solutions(solutions, rule, csp, g, schema)
        if gpu_filtered === nothing
            haskey(ENV, "RG_NAC_DIAG") && _nac_diag(solutions, rule, csp, g, schema, enc, state.world_type)
            gpu_general = _gpu_filter_conditions(solutions, rule, csp, g, schema, enc)
            solutions = gpu_general === nothing ?
                _filter_nac_solutions(solutions, rule, csp, g, enc, schema,
                                      state.world_type) :
                gpu_general
        else
            solutions = gpu_filtered
        end
        isempty(solutions) && return false

        if box.box_type == BOX_PLAYER_RULE
            _choose_gpu_match(solutions, agent, rule, csp, schema,
                              g, enc, state.world_type, turn; can_pass=can_pass,
                              state=state)
        else
            solutions[1]
        end
    end
    chosen_sol === nothing && return false

    if rule !== nothing
        d_match = nothing
        if CUDA.functional() && scratch !== nothing && gpu_cube !== nothing
            n_v = length(chosen_sol)
            if length(scratch.buf_match) < n_v
                scratch.buf_match = CUDA.zeros(Int32, n_v * 2)
            end
            d_match = @view scratch.buf_match[1:n_v]
            copyto!(d_match, chosen_sol)
        end
        _gpu_apply_inplace!(g, chosen_sol, cube, rule, schema, enc, scratch;
                            gpu_cube = gpu_cube, d_match = d_match)
        state.graph_dirty = true
    end
    true
end

# ── Single-sync native-rule pipeline ─────────────────────────────────────────

"""
    _gpu_native_pipeline!(g, csp, cube, gpu_cube, rule, schema, enc, state) -> Bool

Execute a NATIVE_RULE box with a single host-device synchronization:

1. Speculatively grow arrays for new elements (no n_alloc/n_live update yet).
2. Build hom_fwd + CSP domains (async).
3. `dive_solve_kernel!` — single-thread DFS, writes first solution (async).
4. `write_match_from_sols_kernel!` — writes buf_match + buf_fired (async).
5. `build_to_del_kernel!` — GPU-resident to_del mask (async, guarded).
6. `dangling_check_fired_kernel!` — ANDs violation into buf_fired (async).
7. `dpo_deletion_kernel_g!` — clears active flags (async, guarded).
8. Addition kernels — activate (guarded) + FK/attr writes (unguarded; safe slots).
9. `_update_preserved!` — attr/FK updates (async; slot=0 guard is automatic).
10. ONE `synchronize` + ONE `Array(buf_fired)` download (4 bytes).
11. Post-sync: update n_alloc and n_live from static cube counts (CPU only).
"""
function _gpu_native_pipeline!(g::GPUACSet, csp::CSPProblem,
                                cube::AdhesiveCube,
                                gpu_cube::GPUAdhesiveCube,
                                rule, schema::SchemaInfo, enc::AttributeEncoder,
                                state::GPUSchedulerState)::Bool
    scratch  = state.scratch
    backend  = CUDA.CUDABackend()
    nc       = csp.n_chunks
    nc_max   = _select_nc_max(nc)
    n_vars   = Int(csp.n_vars)
    n_bc     = length(csp.bytecodes)

    # ── Phase 0: Speculative slot allocation (arrays grown, n_alloc NOT yet updated) ──
    pre_alloc = Dict{Symbol, Int32}(o => Int32(g.n_alloc[o]) for o in schema.obj_types)
    d_globals_per_type = Dict{Symbol, Any}()
    for o in schema.obj_types
        n_add = get(gpu_cube.add_per_type, o, 0)
        n_add == 0 && continue
        n_cur  = g.n_alloc[o]
        n_next = n_cur + n_add
        cap    = length(g.active[o])
        if n_next > cap
            new_cap  = max(2 * cap, n_next)
            new_active = CUDA.zeros(Bool, new_cap)
            n_cur > 0 && copyto!(new_active, 1, g.active[o], 1, n_cur)
            g.active[o] = new_active
            for h in schema.homs
                schema.hom_dom[h] == o || continue
                new_fk = CUDA.zeros(Int32, new_cap)
                n_cur > 0 && copyto!(new_fk, 1, g.homs[h], 1, n_cur)
                g.homs[h] = new_fk
            end
            for a in schema.attrs
                schema.attr_dom[a] == o || continue
                new_av = CUDA.zeros(Int32, new_cap)
                n_cur > 0 && copyto!(new_av, 1, g.attrs[a], 1, n_cur)
                g.attrs[a] = new_av
            end
        end
        globals = Int32[Int32(n_cur + j) for j in 1:n_add]
        d_gl = CuArray(globals)
        d_globals_per_type[o] = d_gl
    end

    # ── Phase 1: Build hf + domains (async) ──────────────────────────────────
    hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, nc, scratch)
    d_gpu = _build_domains_gpu!(backend, csp, g, schema, scratch)
    _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc, backend, scratch)

    # ── Phase 2: Single-thread DFS (async) ───────────────────────────────────
    b_gpu    = scratch.buf_bytecodes
    sol_gpu  = scratch.buf_solutions
    cnt_gpu  = scratch.buf_sol_count
    work_gpu = scratch.buf_workspace
    if length(scratch.buf_match) < n_vars
        scratch.buf_match = CUDA.zeros(Int32, n_vars * 2)
    end
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))
    dive_solve_kernel!(backend)(
        d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, 1,
        work_gpu, hf_flat, hf_offs, Val(nc_max); ndrange=1)

    # ── Phase 3: Write match + fired flag (async) ─────────────────────────────
    write_match_from_sols_kernel!(backend, 1)(
        scratch.buf_match, scratch.buf_fired, sol_gpu, cnt_gpu, n_vars; ndrange=1)

    # ── Phase 4: Build to_del mask (async, guarded by buf_fired) ─────────────
    n_del_l = length(gpu_cube.del_l_flats_gpu)
    total_alloc = sum(g.n_alloc[o] for o in schema.obj_types; init=0)
    if length(scratch.buf_to_del) < total_alloc
        scratch.buf_to_del = CUDA.zeros(Bool, total_alloc * 2)
    end
    buf_del = @view scratch.buf_to_del[1:total_alloc]
    KernelAbstractions.fill!(buf_del, false)

    g_off = Dict{Symbol, Int}()
    cursor_off = 0
    for o in schema.obj_types
        g_off[o] = cursor_off
        cursor_off += g.n_alloc[o]
    end

    if n_del_l > 0
        # Build g_type_offs: 1-based type index → 0-based flat offset in buf_to_del
        g_type_offs_h = zeros(Int32, length(schema.obj_types))
        for (t, o) in enumerate(schema.obj_types)
            g_type_offs_h[t] = Int32(g_off[o])
        end
        if length(scratch.buf_g_type_offs) < length(g_type_offs_h)
            scratch.buf_g_type_offs = CuArray(g_type_offs_h)
        else
            copyto!(scratch.buf_g_type_offs, g_type_offs_h)
        end

        build_to_del_kernel!(backend, 256)(
            buf_del, scratch.buf_fired, scratch.buf_match,
            gpu_cube.del_l_flats_gpu, gpu_cube.del_l_types_gpu,
            scratch.buf_g_type_offs, Int32(n_del_l); ndrange=n_del_l)

        # ── Phase 5: Dangling check → AND into buf_fired (async) ─────────────
        for h in schema.homs
            src_type = schema.hom_dom[h]
            tgt_type = schema.hom_cod[h]
            n_src = g.n_alloc[src_type]
            n_tgt = g.n_alloc[tgt_type]
            (n_src == 0 || n_tgt == 0) && continue
            o_src = g_off[src_type]
            o_tgt = g_off[tgt_type]
            dangling_check_fired_kernel!(backend, 256)(
                scratch.buf_fired,
                @view(g.active[src_type][1:n_src]),
                @view(g.homs[h][1:n_src]),
                @view(buf_del[o_src+1:o_src+n_src]),
                @view(buf_del[o_tgt+1:o_tgt+n_tgt]),
                Int32(n_src), Int32(n_tgt); ndrange=n_src)
        end

        # ── Phase 6: Deletion (async, guarded) ───────────────────────────────
        for o in schema.obj_types
            n = g.n_alloc[o]
            n == 0 && continue
            off = g_off[o]
            dpo_deletion_kernel_g!(backend, 256)(
                g.active[o],
                @view(buf_del[off+1:off+n]),
                scratch.buf_fired; ndrange=n)
        end
    end

    # ── Phase 7: Addition (async; activate guarded, FK/attr writes unguarded) ─
    d_match = @view scratch.buf_match[1:n_vars]
    for o in schema.obj_types
        n_add = get(gpu_cube.add_per_type, o, 0)
        n_add == 0 && continue
        d_globals = d_globals_per_type[o]

        activate_slots_kernel_g!(backend, 256)(
            g.active[o], d_globals, scratch.buf_fired; ndrange=n_add)

        fk_o_gpu   = get(gpu_cube.new_r_fk_gpu,  o, Dict{Symbol,Any}())
        attr_o_gpu = get(gpu_cube.new_r_attr_gpu, o, Dict{Symbol,Any}())

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            tgt_type = schema.hom_cod[h]
            haskey(fk_o_gpu, h) || continue
            fk_pre_gpu = fk_o_gpu[h]
            n_pre = length(fk_pre_gpu)
            n_pre == 0 && continue
            if length(scratch.buf_pushout_vals) < n_pre
                scratch.buf_pushout_vals = CUDA.zeros(Int32, n_pre * 2)
            end
            d_vals = @view scratch.buf_pushout_vals[1:n_pre]
            compute_fk_vals_kernel!(backend, 256)(
                d_vals, fk_pre_gpu, d_match,
                gpu_cube.k_to_l_gpu, pre_alloc[tgt_type]; ndrange=n_pre)
            write_fk_kernel!(backend, 256)(g.homs[h], d_globals, d_vals; ndrange=n_add)
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            haskey(attr_o_gpu, a) || continue
            attr_gpu = attr_o_gpu[a]
            n_pre = length(attr_gpu)
            n_pre == 0 && continue
            write_attr_kernel!(backend, 256)(g.attrs[a], d_globals, attr_gpu; ndrange=n_add)
        end
    end

    # ── Phase 8: Preserved element updates (async; safe kernels auto-guard via slot=0) ──
    # buf_match is zeroed when no solution → gathered slots = 0 → safe write kernels skip.
    dummy_match = zeros(Int32, cube.n_l_elems)
    r_to_local  = Dict{Symbol, Vector{Int32}}(o => Int32[] for o in schema.obj_types)
    _update_preserved!(g, dummy_match, cube, rule, schema, enc, r_to_local;
                       scratch   = scratch,
                       gpu_cube  = gpu_cube,
                       d_match   = d_match,
                       pre_alloc = pre_alloc)

    # ── Phase 9: ONE sync + ONE 4-byte download ───────────────────────────────
    KernelAbstractions.synchronize(backend)
    fired = CUDA.@allowscalar(scratch.buf_fired[1]) != Int32(0)

    if fired
        for (o, cnt) in gpu_cube.del_per_type
            g.n_live[o][] -= cnt
        end
        for (o, cnt) in gpu_cube.add_per_type
            g.n_alloc[o] += cnt
            g.n_live[o][] += cnt
        end
    end

    fired
end

# ── Main scheduler ────────────────────────────────────────────────────────────

const _DEFAULT_TERMINAL = (W) -> (false, nothing)

"""
    _rule_has_conditions(rule) -> Bool

True if the rule carries NAC/PAC application conditions.  Such rules must NOT use
the fast single-sync native pipeline, which skips NAC enforcement: new-element
NACs (e.g. rtb_bingo's "platform still has a FuelToken") are not lowered into the
CSP (see CSPLowering.jl) and are only enforced by `_filter_nac_solutions` on the
standard solve path.  Bypassing it lets a conditional rule fire when its NAC
should block it (rtb_bingo deleting a fuelled platform's PlatformZone).
"""
function _rule_has_conditions(rule)::Bool
    rule === nothing && return false
    inner = hasproperty(rule, :rule) ? rule.rule : rule
    hasproperty(inner, :conditions) && !isempty(inner.conditions)
end

"""
    _dispatch_gpu_box!(box, b_idx, sched, g, schema, enc, state, turn,
                       wire_active, events, rewrite_count, backend) -> Bool

Execute a single compiled box against the live GPU world `g`, updating
`wire_active`.  Shared by the main per-turn loop and the `BOX_AGENT_LOOP`
sub-schedule runner so both honour identical box semantics.

`sched` is the schedule that *owns* `box` (the parent schedule for top-level
boxes, or a sub-schedule for an agent-loop body); every csp/rule/cube/player
index on `box` is resolved against it.  Returns `true` if the box changed any
wire state (the old inline `any_changed`).
"""
function _dispatch_gpu_box!(box::CompiledBox, b_idx::Int, sched::CompiledGPUSched,
                            g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder,
                            state::GPUSchedulerState, turn::Int,
                            wire_active::Vector{Bool},
                            events::Vector,   # Vector{GpuRewriteEvent}; type defined later
                            rewrite_count::Base.RefValue{Int}, backend;
                            event_box_idx::Int = b_idx,
                            agent_pin::Union{Nothing,Tuple{Symbol,Int}}=nothing)::Bool
    # `event_box_idx` is the index recorded in GpuRewriteEvents.  It equals
    # `b_idx` for top-level boxes; for boxes inside an agent-loop body it is the
    # parent agent-loop box's top-level index, so the event maps back into
    # `sched.boxes` (and to the body player) during experience reconstruction.
    in_w = Int(box.in_wire)
    (in_w == 0 || !wire_active[in_w]) && return false

    if box.box_type == BOX_WEAKEN
        wire_active[in_w] = false
        for ow in box.out_wires
            Int(ow) == 0 && break
            wire_active[Int(ow)] = true
        end
        return true

    elseif box.box_type == BOX_COIN
        wire_active[in_w] = false
        p      = Float64(box.params[1])
        branch = rand(state.rng) < p ? 1 : 2
        ow     = Int(box.out_wires[branch])
        ow != 0 && (wire_active[ow] = true)
        return true

    elseif box.box_type == BOX_AGENT_LOOP
        # Run the body sub-schedule once per live agent instance (dynamic, no
        # cap), mirroring the CPU runner's `for am in homomorphisms(iface,world)`.
        # Fast path: a batched-PINNED solve (build the whole-world hom_forward
        # once per box, pin each agent like the sequential loop).  Falls back to
        # the per-instance `_exec_agent_loop!` for bodies it can't handle.
        wire_active[in_w] = false
        handled = _exec_agent_loop_batched!(box, b_idx, sched, g, schema, enc, state,
                                            turn, events, rewrite_count, backend;
                                            event_box_idx=event_box_idx)
        handled || _exec_agent_loop!(box, b_idx, sched, g, schema, enc, state, turn,
                                     events, rewrite_count, backend; event_box_idx=event_box_idx)
        ow = Int(box.out_wires[1])
        ow != 0 && (wire_active[ow] = true)
        return true

    elseif box.box_type == BOX_NATIVE_RULE || box.box_type == BOX_PLAYER_RULE
        wire_active[in_w] = false
        ridx     = Int(box.csp_idx)
        csp      = sched.csps[ridx]
        rule     = sched.rules[ridx]
        adh_idx  = Int(box.adh_idx)
        cube     = sched.adhesive_cubes[adh_idx]
        gpu_cube = adh_idx <= length(sched.gpu_cubes) ?
                   sched.gpu_cubes[adh_idx] : nothing

        # NATIVE_RULE on CUDA uses the single-sync pipeline; PLAYER_RULE keeps
        # the multi-sync path so the agent can inspect solutions.  Rules with
        # NAC/PAC conditions must take the standard path so _filter_nac_solutions
        # runs — the fast pipeline does not enforce application conditions.
        fired = if box.box_type == BOX_NATIVE_RULE &&
                   CUDA.functional() &&
                   state.scratch !== nothing &&
                   gpu_cube !== nothing &&
                   !_rule_has_conditions(rule)
            _gpu_native_pipeline!(g, csp, cube, gpu_cube, rule,
                                  schema, enc, state)
        else
            _gpu_solve_inplace!(g, csp, rule, cube, gpu_cube, schema, enc,
                                box, b_idx, sched, state, turn; agent_pin=agent_pin)
        end
        ow = Int(box.out_wires[fired ? 1 : 2])
        ow != 0 && (wire_active[ow] = true)
        if fired
            push!(events, GpuRewriteEvent(Int32(turn), Int32(event_box_idx), true))
            rewrite_count[] += 1
            if state.compact_every > 0 && rewrite_count[] % state.compact_every == 0
                compact_gpu_acset!(g, schema, backend)
            end
        end
        return true
    end

    return false
end

"""
    _exec_agent_loop!(box, b_idx, sched, g, schema, enc, state, turn,
                      events, rewrite_count, backend)

Execute a `BOX_AGENT_LOOP`: dispatch the body sub-schedule once per live
instance of the agent object, mirroring the CPU runner's

    for am in homomorphisms(agent_interface, world); run_body(am); end

The body mutates the GPU world `g` in place, so the world threads through
iterations automatically.  The instance count `k` is the live part count of the
agent object read directly from `g.n_live` — a single host-side scalar, no GPU
data round-trip and no fixed cap, so larger worlds iterate over *more* matches
rather than silently dropping the overflow.
"""
function _exec_agent_loop!(box::CompiledBox, b_idx::Int, sched::CompiledGPUSched,
                           g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder,
                           state::GPUSchedulerState, turn::Int,
                           events::Vector,   # Vector{GpuRewriteEvent}; type defined later
                           rewrite_count::Base.RefValue{Int}, backend;
                           event_box_idx::Int = b_idx)
    sub_idx = Int(box.sub_sched_idx)
    (sub_idx < 1 || sub_idx > length(sched.sub_schedules)) && return
    sub = sched.sub_schedules[sub_idx]
    isempty(sub.boxes) && return

    # Agent object symbol (e.g. :Platform) is recorded in box_players at compile.
    agent_obj = b_idx <= length(sched.box_players) ? sched.box_players[b_idx] : :_none
    k = haskey(g.n_live, agent_obj) ? g.n_live[agent_obj][] : 0
    k <= 0 && return

    # Enumerate the live slot-ids of the agent object ONCE (the agent population is
    # fixed for the loop's duration, mirroring the CPU runner's single
    # `homomorphisms(interface, world)` call).  Each iteration pins the body solve
    # to its instance via `_pin_agent_var!`, so the body only searches that
    # instance's matches.  Without the pin the body re-solves globally k times
    # (O(instances²)) AND loses the per-instance agent-box semantics.
    n_alloc_obj = get(g.n_alloc, agent_obj, 0)
    live_ids    = n_alloc_obj > 0 ?
        findall(Array(@view g.active[agent_obj][1:n_alloc_obj])) : Int[]
    isempty(live_ids) && return

    for inst_id in live_ids
        pin = (agent_obj, Int(inst_id))
        # One pass of the (acyclic, tiny) body sub-schedule: seed its init wires,
        # then sweep its boxes until an exit wire fires or nothing changes.
        sub_wires = zeros(Bool, sub.n_wires)
        for w in sub.init_wires
            sub_wires[w] = true
        end
        for _pass in 1:(length(sub.boxes) + 1)
            changed = false
            for (sb_idx, sbox) in enumerate(sub.boxes)
                changed |= _dispatch_gpu_box!(sbox, sb_idx, sub, g, schema, enc,
                                              state, turn, sub_wires, events,
                                              rewrite_count, backend;
                                              event_box_idx=event_box_idx,
                                              agent_pin=pin)
            end
            (any(sub_wires[w] for w in sub.exit_wires) || !changed) && break
        end
    end
end

# ── Batched-PINNED agent loop ─────────────────────────────────────────────────
#
# Equivalent to `_exec_agent_loop!` for the common "each agent picks one move"
# body (a single PLAYER_RULE box wrapped in WEAKEN plumbing), but builds the
# whole-world hom_forward + base domains ONCE per box instead of once per agent.
# Each agent's matches come from a PINNED solve (pin the agent var to its slot,
# exactly as the sequential loop does) → IDENTICAL witnesses.  This fixes the
# reverted `perf/batch-agent-loop` failure mode, where a single unpinned solve
# grouped by agent var picked different interchangeable-resource witnesses
# (a Platform with several FuelTokens) than the per-agent pinned solves.
#
# Matches are enumerated against the BOX-ENTRY world (a snapshot, so all agents
# are solved before any apply — required to collapse the per-agent host syncs).
# The per-agent NAC re-filter then runs against the LIVE (mutating) world, so a
# move invalidated by an earlier apply is still dropped.  This equals the
# sequential loop exactly when one agent's applied move can't change another
# agent's match set (within-turn independence; holds for Falcon `move`: each
# platform consumes its own fuel and moves itself).  Slot indices are stable
# across applies (deletes tombstone, adds extend the high-water mark; no
# compaction inside the loop), so box-entry matches remain valid at apply time.
#
# Returns true if it handled the loop; false → caller falls back to
# `_exec_agent_loop!`.  Disable via env RG_NO_BATCH_AGENT.  RG_AGENT_DIAG
# cross-checks each agent's turbo solve against the reference `gpu_dive_solve`
# (counters `_AGENT_DIAG_{CHECKS,MISM}`; a test asserts 0 mismatches).
const _AGENT_DIAG_CHECKS = Ref(0)
const _AGENT_DIAG_MISM   = Ref(0)
# RG_DECOMP_DIAG cross-checks each agent's compact codomain-decomposed solve against the
# full-world turbo solve (back-translated to world indices); a test asserts 0 mismatches.
const _DECOMP_DIAG_CHECKS = Ref(0)
const _DECOMP_DIAG_MISM   = Ref(0)
function _exec_agent_loop_batched!(box::CompiledBox, b_idx::Int, sched::CompiledGPUSched,
                                    g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder,
                                    state::GPUSchedulerState, turn::Int, events::Vector,
                                    rewrite_count::Base.RefValue{Int}, backend;
                                    event_box_idx::Int = b_idx)::Bool
    haskey(ENV, "RG_NO_BATCH_AGENT")                 && return false
    (CUDA.functional() && state.scratch !== nothing) || return false
    sub_idx = Int(box.sub_sched_idx)
    (sub_idx < 1 || sub_idx > length(sched.sub_schedules)) && return false
    sub = sched.sub_schedules[sub_idx]
    isempty(sub.boxes) && return false

    # Body must be exactly one PLAYER_RULE box surrounded by WEAKEN plumbing.
    eff_idxs = findall(b -> b.box_type == BOX_PLAYER_RULE || b.box_type == BOX_NATIVE_RULE,
                       sub.boxes)
    (length(eff_idxs) == 1 &&
     all(b -> b.box_type == BOX_PLAYER_RULE || b.box_type == BOX_NATIVE_RULE ||
              b.box_type == BOX_WEAKEN, sub.boxes)) || return false
    sbox = sub.boxes[eff_idxs[1]]
    sbox.box_type == BOX_PLAYER_RULE                 || return false

    agent_obj = b_idx <= length(sched.box_players) ? sched.box_players[b_idx] : :_none
    ridx      = Int(sbox.csp_idx)
    csp       = sub.csps[ridx]
    rule      = sub.rules[ridx]
    av        = get(csp.var_offset, agent_obj, 0)    # agent object's CSP variable
    av == 0                                          && return false
    adh_idx   = Int(sbox.adh_idx)
    gpu_cube  = adh_idx <= length(sub.gpu_cubes) ? sub.gpu_cubes[adh_idx] : nothing
    gpu_cube === nothing                             && return false
    cube      = sub.adhesive_cubes[adh_idx]
    player_sym = eff_idxs[1] <= length(sub.box_players) ? sub.box_players[eff_idxs[1]] : :_none
    agent      = get(state.agents, player_sym, nothing)
    agent isa AbstractGPUPlayer                      || return false
    can_pass   = sbox.params[1] > 0f0
    scratch    = state.scratch

    n_alloc_obj = get(g.n_alloc, agent_obj, 0)
    live_ids    = n_alloc_obj > 0 ?
        findall(Array(@view g.active[agent_obj][1:n_alloc_obj])) : Int[]
    isempty(live_ids) && return true

    # ── Build whole-world hom_forward + base domains ONCE (box-entry snapshot) ──
    nc               = csp.n_chunks
    hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, nc, scratch)
    d0               = _build_domains_gpu!(backend, csp, g, schema, scratch)
    _apply_attr_masks_gpu_device!(d0, csp, g, schema, enc, backend, scratch)
    KernelAbstractions.synchronize(backend)
    base_d = copy(d0)               # standalone: per-agent pinning won't corrupt it
    d_work = similar(base_d)

    # ── Per-agent PINNED solve ────────────────────────────────────────────────
    # Default: compact codomain-decomposed solve (restrict to the pinned agent's
    # morphism-neighborhood, remapped to a small local nc → fast shared-memory solve).
    # RG_NO_DECOMP forces the full-world solve.  RG_AGENT_DIAG / RG_DECOMP_DIAG also run
    # the full-world (and dive) solve to cross-check per-solve solution SETS.
    diag        = haskey(ENV, "RG_AGENT_DIAG")
    decomp_diag = haskey(ENV, "RG_DECOMP_DIAG")
    use_decomp  = !haskey(ENV, "RG_NO_DECOMP")
    need_world  = (!use_decomp) || diag || decomp_diag
    base_d_host = UInt64[]
    fk_cols_dd  = Dict{Symbol,Vector{Int32}}()
    if use_decomp || decomp_diag                      # per-box: download base domain + FK cols once
        base_d_host = Array(base_d)
        for h in _decomp_relevant_homs(schema, csp)
            nh = g.n_alloc[schema.hom_dom[h]]
            fk_cols_dd[h] = nh > 0 ? Array(@view g.homs[h][1:nh]) : Int32[]
        end
    end
    groups = Dict{Int, Vector{Vector{Int32}}}()
    for inst_id in live_ids
        sols_world = Vector{Vector{Int32}}()
        if need_world
            copyto!(d_work, base_d)
            _pin_agent_var!(d_work, csp, (agent_obj, Int(inst_id)))
            sols_world = gpu_turbo_solve(backend, csp, d_work, hf_flat, hf_offs; scratch = scratch)
        end
        sols_decomp = Vector{Vector{Int32}}()
        if use_decomp || decomp_diag
            sols_decomp = decomposed_pinned_solve(backend, csp, schema, agent_obj, Int(inst_id),
                                                  base_d_host, fk_cols_dd, g.n_alloc)
        end
        sols = use_decomp ? sols_decomp : sols_world
        groups[Int(inst_id)] = sols
        if decomp_diag
            nv2 = Int(csp.n_vars)
            _DECOMP_DIAG_CHECKS[] += 1
            dset = Set(Vector{Int32}(s[1:min(nv2, length(s))]) for s in sols_decomp)
            wset = Set(Vector{Int32}(s[1:min(nv2, length(s))]) for s in sols_world)
            dset != wset && (_DECOMP_DIAG_MISM[] += 1)
        end
        if diag
            # Cross-check the full-world shared-memory turbo solve against the reference
            # dive solver (n_vars rows only; the scratch turbo path pads with garbage).
            hf2, ho2 = _build_hom_fwd_gpu(backend, g, schema, nc)
            d2       = _build_domains_gpu(backend, csp, g, schema)
            _apply_attr_masks_gpu_device!(d2, csp, g, schema, enc, backend, scratch)
            _pin_agent_var!(d2, csp, (agent_obj, Int(inst_id)))
            KernelAbstractions.synchronize(backend)
            ref = gpu_dive_solve(backend, csp, d2, hf2, ho2)
            nv  = Int(csp.n_vars)
            _AGENT_DIAG_CHECKS[] += 1
            sset = Set(Vector{Int32}(s[1:min(nv, length(s))]) for s in sols_world)
            rset = Set(Vector{Int32}(r[1:min(nv, length(r))]) for r in ref)
            sset != rset && (_AGENT_DIAG_MISM[] += 1)
        end
    end

    # ── Phase 1: NAC-filter every agent (box-entry world) + build its candidate
    #    matrix (mirrors _choose_gpu_match's GPU-player branch).  No applies yet,
    #    so all agents are filtered against the same box-entry world — consistent
    #    with the simultaneous-move semantics and a prerequisite for batched
    #    scoring (one forward over all agents). ──
    sel_ids      = Int[]
    sel_filtered = Vector{Vector{Vector{Int32}}}()
    sel_cands    = Any[]
    sel_nsols    = Int[]
    for inst_id in live_ids
        cands = get(groups, Int(inst_id), nothing)
        (cands === nothing || isempty(cands)) && continue
        gpu_filtered = _gpu_filter_nac_solutions(cands, rule, csp, g, schema)
        filtered = if gpu_filtered === nothing
            gpu_general = _gpu_filter_conditions(cands, rule, csp, g, schema, enc)
            gpu_general === nothing ?
                _filter_nac_solutions(cands, rule, csp, g, enc, schema, state.world_type) :
                gpu_general
        else
            gpu_filtered
        end
        isempty(filtered) && continue
        cands_raw = CUDA.functional() ? CuArray(reduce(hcat, filtered)) : reduce(hcat, filtered)
        cands_ext = can_pass ?
            hcat(cands_raw, CUDA.functional() ? CUDA.zeros(Int32, size(cands_raw, 1), 1) :
                                                zeros(Int32, size(cands_raw, 1), 1)) :
            cands_raw
        push!(sel_ids, Int(inst_id))
        push!(sel_filtered, filtered)
        push!(sel_cands, cands_ext)
        push!(sel_nsols, length(filtered))
    end
    isempty(sel_ids) && return true

    # ── Phase 2: batched player scoring — ONE forward over all agents. ──
    gd = nothing
    if agent isa AbstractGNNPlayer
        if state.graph_data === nothing
            state.graph_data = build_gpu_graph(g, schema, enc; backend); state.graph_dirty = false
        elseif state.graph_dirty
            rebuild_gpu_graph!(state.graph_data, g, schema, enc; backend); state.graph_dirty = false
        end
        gd = state.graph_data
    end
    n_presented = Int[can_pass ? sel_nsols[a] + 1 : sel_nsols[a] for a in eachindex(sel_ids)]
    idxs = select_action_gpu_batched(agent, g, enc, schema, sel_cands, n_presented, turn;
                                     graph_data = gd)

    # ── Phase 3: apply each agent's chosen move (box-entry slots stay valid —
    #    deletes tombstone, adds extend the high-water mark; no compaction here). ──
    for a in eachindex(sel_ids)
        ns  = sel_nsols[a]
        idx = idxs[a]
        (can_pass && idx > ns) && continue          # player passed on this agent
        chosen = sel_filtered[a][clamp(idx, 1, ns)]
        n_v = length(chosen)
        if length(scratch.buf_match) < n_v
            scratch.buf_match = CUDA.zeros(Int32, n_v * 2)
        end
        d_match = @view scratch.buf_match[1:n_v]
        copyto!(d_match, chosen)
        fired = _gpu_apply_inplace!(g, chosen, cube, rule, schema, enc, scratch;
                                    gpu_cube = gpu_cube, d_match = d_match)
        if fired
            push!(events, GpuRewriteEvent(Int32(turn), Int32(event_box_idx), true))
            rewrite_count[] += 1
            state.graph_dirty = true
        end
    end
    return true
end

function run_gpu_schedule!(state::GPUSchedulerState;
                           T_max::Int = 1000,
                           terminal_fn::Function = _DEFAULT_TERMINAL,
                           winner_wires::Dict{Symbol, Union{Symbol,Nothing}} = Dict{Symbol,Union{Symbol,Nothing}}())::Vector{GpuRewriteEvent}
    sched  = state.sched
    g      = state.g
    schema = state.schema
    enc    = state.enc

    wire_active = zeros(Bool, sched.n_wires)
    for w in sched.init_wires
        wire_active[w] = true
    end

    events        = GpuRewriteEvent[]
    rewrite_count = Ref(0)   # for compact_every triggering
    backend       = CUDA.functional() ? CUDA.CUDABackend() : CPU()

    for turn in 1:T_max
        # Terminal check: download world only when a custom predicate is provided
        if terminal_fn !== _DEFAULT_TERMINAL
            world_snap = download_acset(g, enc, state.world_type)
            done, _    = terminal_fn(world_snap)
            done && break
        end

        any_changed = false

        for (b_idx, box) in enumerate(sched.boxes)
            changed = _dispatch_gpu_box!(box, b_idx, sched, g, schema, enc,
                                         state, turn, wire_active, events,
                                         rewrite_count, backend)
            any_changed |= changed
        end

        # A true exit wire (a non-trace return) ends the episode.
        any(wire_active[w] for w in sched.exit_wires) && break

        # Otherwise loop the schedule: a trace-return wire re-activates its
        # trace-input wire for the next turn.  We rebuild wire_active from
        # scratch (like the CPU runner's fresh `_init_wires`) so stale
        # intermediate wires from this turn don't leak into the next one.  The
        # world `g` is mutated in place, so it threads forward automatically.
        if isempty(sched.trace_loops)
            # No trace structure: legacy single-pass behaviour.
            trace_active = any(wire_active[w] for w in sched.trace_wires)
            !trace_active && !any_changed && break
        else
            next_active = zeros(Bool, sched.n_wires)
            looped = false
            for (ret_w, in_w) in sched.trace_loops
                if wire_active[ret_w]
                    next_active[in_w] = true
                    looped = true
                end
            end
            looped || break          # body produced no continuing wire — done
            wire_active = next_active
        end
    end

    events
end
