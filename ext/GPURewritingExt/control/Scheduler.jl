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
    buf_violation     :: CuVector{Bool}         # [1] dangling-check flag
    buf_attr_mask     :: CuVector{UInt64}       # nc
    buf_pushout_slots :: CuVector{Int32}        # staging for slot indices (B5/B6, grown on demand)
    buf_pushout_vals  :: CuVector{Int32}        # staging for fk/attr values (B5/B6, grown on demand)
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
end

function GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                            log_trajectory=false, compact_every=100)
    traj = log_trajectory ? GPUTrajectoryLog(schema) : nothing

    scratch = if CUDA.functional()
        nc = isempty(sched.csps) ? 1 : sched.csps[1].n_chunks
        max_n_vars = isempty(sched.csps) ? 1 :
                     maximum(Int(csp.n_vars) for csp in sched.csps; init=1)
        max_n_bc   = isempty(sched.csps) ? 1 :
                     maximum(length(csp.bytecodes) for csp in sched.csps; init=1)
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
            CUDA.zeros(Int32,  max(max_n_vars, 1), 10_000),
            CUDA.zeros(Int32,  1),
            CUDA.zeros(UInt64, max(max_n_vars * MAX_CHUNKS, 1), 16),
            CUDA.zeros(UInt64, max(nc, 1)),
            CUDA.zeros(Bool,   max(total_alloc * 4, 1)),
            CUDA.zeros(Bool,   1),
            CUDA.zeros(UInt64, max(nc, 1)),
            CUDA.zeros(Int32,  256),   # buf_pushout_slots initial capacity
            CUDA.zeros(Int32,  256),   # buf_pushout_vals initial capacity
        )
    else
        nothing
    end

    GPUSchedulerState(sched, g, schema, enc, world_type, agents,
                      traj, compact_every, Xoshiro(42), Ref(1), NamedTuple[], scratch)
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

    KernelAbstractions.copyto!(backend, scratch.buf_hf_offs, hom_fwd_offs)

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
                              scratch::Union{GPUScratchBuffers, Nothing} = nothing)
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

    # 4. Add R\K elements via GPU scatter kernels
    r_to_local = apply_pushout!(g, sol, cube, rule, schema, enc; scratch = scratch)

    # 5. Patch preserved K elements that differ in R via GPU scatter kernels
    _update_preserved!(g, sol, cube, rule, schema, enc, r_to_local; scratch = scratch)

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
                            world_type, turn::Int)
    agent === nothing && return solutions[1]

    # GPU player path: pass candidates matrix directly, no world download
    if agent isa AbstractGPUPlayer
        n_sols = length(solutions)
        cands  = CuArray(reduce(hcat, solutions))   # [n_vars × n_sols] on GPU
        idx    = select_action_gpu(agent, g, enc, schema, cands, n_sols, turn)
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
    isempty(action_pairs) && return solutions[1]

    actions = [p.first for p in action_pairs]
    chosen  = select_action(agent, GameState(world_host, turn), actions)
    chosen  === nothing && return solutions[1]

    for (act, sol) in action_pairs
        act === chosen && return sol
    end
    solutions[1]
end

# ── Per-box solve-and-apply ───────────────────────────────────────────────────

function _gpu_solve_inplace!(g::GPUACSet, csp::CSPProblem, rule,
                              cube::AdhesiveCube,
                              schema::SchemaInfo, enc::AttributeEncoder,
                              box, b_idx::Int, sched, state, turn::Int)::Bool
    scratch = state.scratch

    solutions = if CUDA.functional() && scratch !== nothing
        backend = CUDA.CUDABackend()
        nc      = csp.n_chunks

        # Build hom_fwd and domains into pre-allocated scratch buffers (B1)
        hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, nc, scratch)
        d_gpu            = _build_domains_gpu!(backend, csp, g, schema, scratch)

        # Apply PROP_ATTR_EQ masks on-device (B2)
        _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc, backend, scratch)

        # Synchronize once before the solve (only mandatory sync before dive kernel)
        KernelAbstractions.synchronize(backend)
        gpu_dive_solve(backend, csp, d_gpu, hf_flat, hf_offs;
                       scratch = scratch)
    else
        if CUDA.functional()
            # CUDA available but no scratch (shouldn't normally happen)
            backend = CUDA.CUDABackend()
            nc      = csp.n_chunks
            hf_flat, hf_offs = _build_hom_fwd_gpu(backend, g, schema, nc)
            d_gpu            = _build_domains_gpu(backend, csp, g, schema)
            _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc)
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
            cpu_dive_solve(fresh_csp, domains)
        end
    end
    isempty(solutions) && return false

    chosen_sol = if box.box_type == BOX_PLAYER_RULE
        player_sym = sched.box_players[b_idx]
        agent      = get(state.agents, player_sym, nothing)
        _choose_gpu_match(solutions, agent, rule, csp, schema,
                          g, enc, state.world_type, turn)
    else
        solutions[1]
    end
    chosen_sol === nothing && return false

    rule !== nothing && _gpu_apply_inplace!(g, chosen_sol, cube, rule, schema, enc, scratch)
    true
end

# ── Main scheduler ────────────────────────────────────────────────────────────

const _DEFAULT_TERMINAL = (W) -> (false, nothing)

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
    rewrite_count = 0   # for compact_every triggering
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
            in_w = Int(box.in_wire)
            (in_w == 0 || !wire_active[in_w]) && continue

            if box.box_type == BOX_WEAKEN
                wire_active[in_w] = false
                for ow in box.out_wires
                    Int(ow) == 0 && break
                    wire_active[Int(ow)] = true
                end
                any_changed = true

            elseif box.box_type == BOX_COIN
                wire_active[in_w] = false
                p      = Float64(box.params[1])
                branch = rand(state.rng) < p ? 1 : 2
                ow     = Int(box.out_wires[branch])
                ow != 0 && (wire_active[ow] = true)
                any_changed = true

            elseif box.box_type == BOX_NATIVE_RULE || box.box_type == BOX_PLAYER_RULE
                wire_active[in_w] = false
                any_changed = true
                ridx = Int(box.csp_idx)
                csp  = sched.csps[ridx]
                rule = sched.rules[ridx]
                cube = sched.adhesive_cubes[Int(box.adh_idx)]

                fired = _gpu_solve_inplace!(g, csp, rule, cube, schema, enc,
                                            box, b_idx, sched, state, turn)
                ow = Int(box.out_wires[fired ? 1 : 2])
                ow != 0 && (wire_active[ow] = true)
                if fired
                    push!(events, GpuRewriteEvent(Int32(turn), Int32(b_idx), true))
                    rewrite_count += 1
                    if state.compact_every > 0 && rewrite_count % state.compact_every == 0
                        compact_gpu_acset!(g, schema, backend)
                    end
                end
            end
        end

        any(wire_active[w] for w in sched.exit_wires) && break
        trace_active = any(wire_active[w] for w in sched.trace_wires)
        !trace_active && !any_changed && break
    end

    events
end
