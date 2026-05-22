"""
GPU master scheduler — GPU-native rewriting.

Pattern matching uses the Turbo CSP solver (GPU path when CUDA.functional()).
Rewriting is applied in-place on the GPUACSet via deletion and addition
kernels, with no world round-trip through the CPU.

For PLAYER_RULE boxes the world is downloaded once per box execution so
the agent can see it; no re-upload is needed after rewriting.
"""

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
end

function GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                            log_trajectory=false, compact_every=100)
    traj = log_trajectory ? GPUTrajectoryLog(schema) : nothing
    GPUSchedulerState(sched, g, schema, enc, world_type, agents,
                      traj, compact_every, Xoshiro(42), Ref(1), NamedTuple[])
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
Build flat hom_fwd and offset arrays entirely on GPU.
Returns `(hf_flat_gpu, hf_offs_gpu)` ready for `gpu_dive_solve`.
`hf_offs_gpu[h_idx]` is the 0-based word offset into `hf_flat_gpu` for morphism h_idx.
"""
function _build_hom_fwd_gpu(backend, g::GPUACSet, schema::SchemaInfo, nc::Int)
    # Compute offsets on CPU (fast, no data transfer)
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

    type_bases = sort([(base, o) for (o, base) in pairs(csp.var_offset)], by=first)
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        n_vars_o  = next_base - base
        n         = g.n_alloc[o]
        n == 0 && continue

        # Build per-type bitmask on GPU (nc words), then copy to each variable slot
        type_mask = KernelAbstractions.allocate(backend, UInt64, nc)
        KernelAbstractions.fill!(type_mask, UInt64(0))
        _build_type_mask_kernel!(backend, 256)(
            type_mask, g.active[o], Int32(nc); ndrange = n)
        KernelAbstractions.synchronize(backend)

        # Copy bitmask into each variable's nc-word slot
        for v in base:(next_base - 1)
            off = (v - 1) * nc   # 0-based word offset
            copyto!(d, off + 1, type_mask, 1, nc)
        end
    end
    d
end

"""
Apply PROP_ATTR_EQ masks to a GPU-resident domain array.
Downloads only the attribute value array (not active flags — inactive elements
are already excluded by the domain initialisation from active flags).
"""
function _apply_attr_masks_gpu_device!(d_gpu, csp::CSPProblem,
                                        g::GPUACSet, schema::SchemaInfo,
                                        enc::AttributeEncoder)
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
        # AND mask into d_gpu: download nc words, AND, re-upload
        off = (v - 1) * nc
        dom_slice = Array(d_gpu[off+1:off+nc])
        for c in 1:nc; dom_slice[c] &= mask[c]; end
        copyto!(d_gpu, off+1, CuArray(dom_slice), 1, nc)
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
    type_bases = sort([(base, o) for (o, base) in pairs(csp.var_offset)], by=first)
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
    type_bases = sort([(base, o) for (o, base) in pairs(csp.var_offset)], by=first)
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
"""
function _gpu_apply_inplace!(g::GPUACSet, sol::Vector{Int32},
                              cube::AdhesiveCube, rule,
                              schema::SchemaInfo, enc::AttributeEncoder)
    backend = CUDA.functional() ? CUDA.CUDABackend() : CPU()

    # 1. Build deletion mask on host (no GPU download), upload to GPU
    to_del = build_to_del_mask(sol, cube, schema, g)

    # 2. Dangling check entirely on GPU
    gpu_dangling_ok(to_del, g, schema, backend) || return false

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
        to_del_o = to_del[off+1:off+n]
        n_del    = Int(sum(to_del_o))
        n_del == 0 && continue
        dpo_deletion_kernel!(backend, 256)(g.active[o], to_del_o; ndrange=n)
        g.n_live[o][] -= n_del
    end

    # 4. Add R\K elements via GPU scatter kernels
    r_to_local = apply_pushout!(g, sol, cube, rule, schema, enc)

    # 5. Patch preserved K elements that differ in R via GPU scatter kernels
    _update_preserved!(g, sol, cube, rule, schema, enc, r_to_local)

    KernelAbstractions.synchronize(backend)
    true
end

# ── Agent dispatch for PLAYER_RULE ────────────────────────────────────────────

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

    # Download compact world once for agent's view
    gpu_to_compact, _ = _gpu_to_compact_mapping(g, schema)
    world_host = download_acset(g, enc, world_type)

    action_pairs = Pair{Action, Vector{Int32}}[]
    for sol in solutions
        compact_sol = _sol_gpu_to_compact(sol, csp, schema, gpu_to_compact)
        hom         = _assignment_to_hom(compact_sol, L, world_host, csp, schema)
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
    solutions = if CUDA.functional()
        backend = CUDA.CUDABackend()
        nc      = csp.n_chunks
        # Build hom_fwd and domains entirely on GPU — no CPU downloads
        hf_flat, hf_offs = _build_hom_fwd_gpu(backend, g, schema, nc)
        d_gpu            = _build_domains_gpu(backend, csp, g, schema)
        _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc)
        KernelAbstractions.synchronize(backend)
        gpu_dive_solve(backend, csp, d_gpu, hf_flat, hf_offs)
    else
        hf        = _recompute_hom_forward_gpu(g, schema, csp.n_chunks)
        fresh_csp = CSPProblem(csp.n_vars, csp.var_offset, csp.domain_sizes,
                               csp.bytecodes, csp.nac_groups, csp.pac_groups,
                               csp.agent_var_map, hf, csp.n_chunks)
        domains   = _init_gpu_domains(fresh_csp, g, schema)
        _apply_attr_masks_gpu!(domains, fresh_csp, g, schema, enc)
        cpu_dive_solve(fresh_csp, domains)
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

    rule !== nothing && _gpu_apply_inplace!(g, chosen_sol, cube, rule, schema, enc)
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

    events = GpuRewriteEvent[]

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
                fired && push!(events, GpuRewriteEvent(Int32(turn), Int32(b_idx), true))
            end
        end

        any(wire_active[w] for w in sched.exit_wires) && break
        trace_active = any(wire_active[w] for w in sched.trace_wires)
        !trace_active && !any_changed && break
    end

    events
end
