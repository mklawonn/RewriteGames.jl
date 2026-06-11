"""
    GPURewritingExt

Optional extension for RewriteGames.jl that enables GPU-accelerated algebraic
rewriting via KernelAbstractions.jl and CUDA.jl.

Loaded automatically when both `KernelAbstractions` and `CUDA` are available
in the active environment.  Provides:

  - `gpu_run_game_sched!(gs, initial_world, agents; ...)` — GPU analog of
    `run_game_sched!`, returning the same `Vector{Experience}` type.
  - `turbo_homomorphisms(L, G; ...)` — GPU Turbo solver for use in equivalence
    tests against Catlab's CPU homomorphism search.

## Architecture

Five layers (each a subdirectory):

  lowering/      Host-side compilation: schema introspection, attribute
                 encoding, CSP / TCN bytecode generation, schedule compilation,
                 adhesive cube precomputation.

  solver/        Turbo pattern matching: TCN bytecode struct, AC-1 propagation
                 kernel, dive-and-solve search kernel.

  rewriting/     GPU DPO/SPO rewriting: GPUACSet representation, deletion
                 kernel (pushout complement), addition kernel (pushout via
                 parallel prefix-sum), incremental match update.

  control/       Schedule execution: master scheduler loop, Graph Process
                 trajectory logger, stream compaction.

  reconstruction/ Solution decoding: GPU → host transfer, attribute
                  integer → Julia value decoding, Experience reconstruction.
"""
module GPURewritingExt

using RewriteGames
using Catlab
using AlgebraicRewriting
using KernelAbstractions
using CUDA
using Atomix, StaticArrays
using Random: Xoshiro, AbstractRNG, default_rng, shuffle

import Catlab.CategoricalAlgebra:
    acset_schema, ob, hom, attr, dom, codom, nparts, parts, subpart,
    set_subpart!, add_parts!, add_part!, homomorphism, homomorphisms,
    ACSetTransformation, AttrVar

import Catlab.CategoricalAlgebra: left, right

import RewriteGames: GameSched, PlayerRuleApp, Experience, GameState, Action,
                     select_action, _collect_player_apps,
                     AbstractGPUPlayer, GPUFunctionPlayer, select_action_gpu

# ── GPU player dispatch ───────────────────────────────────────────────────────

function RewriteGames.select_action_gpu(p::GPUFunctionPlayer, g, enc, schema,
                                         cands::AbstractArray{Int32,2},
                                         n_sols::Int, turn::Int;
                                         graph_data=nothing)::Int
    clamp(Int(p.f(g, cands, n_sols, turn)), 1, max(n_sols, 1))
end

# ── Lowering layer ────────────────────────────────────────────────────────────
include("lowering/SchemaInfo.jl")
include("lowering/RuleRegistry.jl")
include("lowering/DeviceData.jl")
include("lowering/FlattenRegistry.jl")
include("lowering/AttributeEncoder.jl")
include("solver/TCNBytecode.jl")        # TCNBytecode needed by CSPLowering
include("solver/BitwiseDomain.jl")      # multi-chunk bitset helpers
include("lowering/CSPLowering.jl")
include("lowering/AdhesiveCubes.jl")
include("lowering/ScheduleCompiler.jl")

# ── Solver layer ──────────────────────────────────────────────────────────────
include("solver/PropagationKernel.jl")
include("solver/DiveSolveKernel.jl")
include("solver/CodomainDecomp.jl")   # compact per-agent codomain decomposition
include("solver/AnchorDecomp.jl")     # anchored-fiber decomposition of EPS solves

# ── Rewriting layer ───────────────────────────────────────────────────────────
include("rewriting/GPUACSet.jl")
include("rewriting/DeletionKernel.jl")
include("rewriting/AdditionKernel.jl")
include("rewriting/IncrementalUpdate.jl")
include("rewriting/GPUGraphData.jl")

# ── Control layer ─────────────────────────────────────────────────────────────
include("control/TrajectoryLogger.jl")
include("control/StreamCompaction.jl")
include("control/Scheduler.jl")
include("control/ZonePartition.jl")
include("control/MasterScheduler.jl")

# ── Reconstruction layer ──────────────────────────────────────────────────────
include("reconstruction/Decode.jl")

# ── Public API ────────────────────────────────────────────────────────────────

"""
    _agent_loop_body_player(sched, agent_loop_box) -> Symbol

The first acting player inside a `BOX_AGENT_LOOP`'s body sub-schedule (the move
player, e.g. `:blue`), recursing through nested agent loops.  Returns `:_none`
if the body has no player-rule box.  Used to attribute agent-loop body firings
to the correct player during experience reconstruction.
"""
function _agent_loop_body_player(sched::CompiledGPUSched, agent_loop_box)::Symbol
    sidx = Int(agent_loop_box.sub_sched_idx)
    (sidx < 1 || sidx > length(sched.sub_schedules)) && return :_none
    sub = sched.sub_schedules[sidx]
    for (i, sbox) in enumerate(sub.boxes)
        if sbox.box_type == BOX_AGENT_LOOP
            pp = _agent_loop_body_player(sub, sbox)
            pp !== :_none && return pp
        elseif sbox.box_type == BOX_PLAYER_RULE
            p = i <= length(sub.box_players) ? sub.box_players[i] : :_none
            p !== :_none && return p
        end
    end
    return :_none
end

"""
    gpu_run_game_sched!(gs, initial_world, agents; backend, T_max, terminal,
                        winner_wires, log_trajectory, compact_every)
        -> Vector{Experience}

GPU-accelerated analog of `run_game_sched!`.

Compiles `gs` and `initial_world` to a GPU-resident state machine, executes
the schedule on-device, and decodes the result back to a `Vector{Experience}`
with the same structure as the CPU implementation.

# Arguments
- `gs`:            A `GameSched` built with `mk_game_sched`.
- `initial_world`: Starting ACSet world.
- `agents`:        Dict mapping player `Symbol` → `AbstractAgent`.

# Keyword arguments
- `backend`:        KernelAbstractions backend (default: `CUDA.CUDABackend()`).
- `T_max`:          Maximum schedule iterations (default: 1000).
- `terminal`:       `world -> (done::Bool, winner)` predicate (default: never).
- `winner_wires`:   Dict mapping exit wire names → winning player.
- `log_trajectory`: Record the full Graph Process trajectory (default: false).
- `compact_every`:  Run stream compaction every N rewrites (default: 100).

# Returns
A `Vector{Experience}` identical in structure to `run_game_sched!` output.
"""
function RewriteGames.gpu_run_game_sched!(
    gs            :: GameSched,
    initial_world,
    agents        :: Dict;
    backend                                                    = CUDA.CUDABackend(),
    T_max         :: Int                                       = 1000,
    terminal      :: Function                                  = _DEFAULT_TERMINAL,
    winner_wires  :: Dict{Symbol, Union{Symbol,Nothing}}       =
                        Dict{Symbol,Union{Symbol,Nothing}}(),
    log_trajectory :: Bool                                     = false,
    compact_every  :: Int                                      = 100,
    discretizers   :: Dict{Symbol, Pair{Function,Function}}    = Dict{Symbol,Pair{Function,Function}}(),
    max_world_size :: Union{Int,Nothing}                       = nothing,
    zone_partition :: Union{ZonePartition, Nothing}            = nothing,
    take                                                       = nothing,
    sample_seed    :: Int                                      = 0,
)::Vector{Experience}

    # ── Phase 1: Host-side compilation ────────────────────────────────────────
    schema   = extract_schema_info(initial_world)
    enc      = build_encoder(initial_world, schema; discretizers = discretizers)
    # Extend encoder with all R-side ACSets from every rule in the schedule so
    # that attribute values introduced by rules (e.g. Bool flags on new Platforms)
    # are registered before CSP lowering and before GPU upload.
    for r_acs in _collect_rule_r_acssets(gs)
        extend_encoder!(enc, r_acs, schema)
    end
    max_n    = isempty(schema.obj_types) ? 1 :
               maximum(nparts(initial_world, o) for o in schema.obj_types; init=1)
    n_chunks = cld(max(max_world_size !== nothing ? max_world_size : max_n, 1), 64)
    sched    = compile_schedule(gs, initial_world, schema, enc; n_chunks=n_chunks)

    # ── Phase 2: Upload initial world to GPU ─────────────────────────────────
    g          = upload_acset(initial_world, schema, enc)
    world_type = typeof(initial_world)

    # ── Phase 3: Build scheduler state ───────────────────────────────────────
    state = GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                              log_trajectory = log_trajectory,
                              compact_every  = compact_every,
                              take           = take,
                              sample_seed    = sample_seed)
    state.zone_partition = zone_partition

    # ── Phase 4: Execute on GPU ───────────────────────────────────────────────
    gpu_events = run_gpu_schedule!(state;
                                   T_max        = T_max,
                                   terminal_fn  = terminal,
                                   winner_wires = winner_wires)

    # ── Phase 5: Decode results ───────────────────────────────────────────────
    # state.g is updated after each rewrite; download the final world from it
    final_world = download_acset(state.g, enc, world_type)

    exps = Experience[]
    for ev in gpu_events
        bidx = Int(ev.box_idx)
        1 <= bidx <= length(sched.boxes) || continue
        # Agent-loop body firings are recorded against the parent AGENT_LOOP box,
        # whose box_players entry is the agent *object* (e.g. :Platform), not the
        # acting player.  Resolve the real player from the body sub-schedule.
        player = sched.boxes[bidx].box_type == BOX_AGENT_LOOP ?
                 _agent_loop_body_player(sched, sched.boxes[bidx]) :
                 sched.box_players[bidx]
        player == :_none && continue

        turn_n     = Int(ev.turn)
        # NOTE (B13): state_pre is always the episode-start world, not the per-turn
        # pre-rewrite state.  For RL workflows that need the correct intermediate
        # pre-states, pass `track_pre_states=true` (unimplemented; see GPU_PLAN.md
        # Bottleneck 13 for the trajectory-replay approach).
        state_pre  = GameState(initial_world, turn_n)
        state_post = GameState(final_world, turn_n + 1)

        push!(exps, Experience(
            player, state_pre, Action[], nothing,
            state_post, false, nothing,
            Dict{Symbol,Any}(), Symbol[], nothing,
        ))
    end
    exps
end

"""
    turbo_homomorphisms(L, G; backend, monic, initial) -> Vector{ACSetTransformation}

GPU Turbo solver for homomorphism enumeration.  Returns the same set as
Catlab's `homomorphisms(L, G; monic=monic, initial=initial)` but computed
via the CSP propagation + dive-solve engine.

Used in the equivalence test suite: results are sorted by assignment tuple
and compared against the CPU ground truth.

Falls back to the CPU solver when `!CUDA.functional()`.
"""
function RewriteGames.turbo_homomorphisms(L, G;
                           backend = CUDA.functional() ? CUDA.CUDABackend() : nothing,
                           monic   = false,
                           initial :: Union{Nothing, NamedTuple, Dict} = nothing,
                           take    :: Union{Nothing, Int} = nothing,
                           thresholds = nothing,
                           seed    :: Integer = 0)

    schema   = extract_schema_info(G)
    enc      = build_encoder(G, schema)
    max_n    = isempty(schema.obj_types) ? 1 :
               maximum(nparts(G, o) for o in schema.obj_types)
    n_chunks = cld(max(max_n, 1), 64)

    # Build a mock rule wrapping L so lower_rule_to_csp can access left(rule)
    mock_rule = _make_identity_rule(L, monic)
    thresholds === nothing || RewriteGames.set_attr_thresholds!(mock_rule, thresholds)
    csp = lower_rule_to_csp(mock_rule, G, schema, enc; n_chunks=n_chunks)

    g_offset = Dict{Symbol,Int}()
    cursor = 0
    for o in schema.obj_types
        g_offset[o] = cursor
        cursor += nparts(G, o)
    end

    domains = _init_domains_from_world(csp, G, schema)
    _apply_attr_masks_world!(domains, csp, G, schema, enc)

    # Apply initial constraints (pinned search)
    if initial !== nothing
        _pin_initial!(domains, initial, csp, G, schema)
    end

    if take !== nothing
        # "take N": count-weighted random-descent sampling of up to N solutions.
        # GPU path goes through gpu_turbo_sample; CPU path (and the no-GPU
        # fallback) uses the cpu_sample_solve reference.
        solutions = backend === nothing ?
            cpu_sample_solve(csp, domains; take=take, rng=Xoshiro(seed)) :
            gpu_turbo_sample(backend, csp, domains; take=take, seed=seed)
    elseif backend === nothing
        solutions = cpu_dive_solve(csp, domains)
    else
        solutions = gpu_dive_solve(backend, csp, domains)
    end

    # Convert flat Int32 assignments to ACSetTransformations
    result = ACSetTransformation[]
    seen   = Set{Tuple}()
    for sol in solutions
        key = Tuple(sol)
        key ∈ seen && continue
        push!(seen, key)
        hom = _assignment_to_hom(sol, L, G, csp, schema)
        hom === nothing && continue
        push!(result, hom)
    end
    result
end

# ── Helpers for turbo_homomorphisms ─────────────────────────────────────────────

struct _MockRule
    _left  :: Any
    _right :: Any
    monic  :: Any
end

function _make_identity_rule(L, monic)
    S = acset_schema(L)
    cat = Catlab.CategoricalAlgebra.infer_acset_cat(L)
    init = Dict{Symbol, Any}(o => collect(1:nparts(L, o)) for o in ob(S))
    id_L = first(homomorphisms(L, L; cat=cat, initial=init))
    _MockRule(id_L, id_L, monic)
end

Catlab.CategoricalAlgebra.left(r::_MockRule)  = r._left
Catlab.CategoricalAlgebra.right(r::_MockRule) = r._right

function _init_domains_from_world(csp::CSPProblem, G, schema::SchemaInfo)
    nc = csp.n_chunks
    nv = Int(csp.n_vars)
    domains = zeros(UInt64, nv * nc)
    for v in 1:nv
        n   = Int(csp.domain_sizes[v])
        off = (v - 1) * nc
        for i in 1:min(n, nc * 64)
            ci, bi = elem_to_chunk(i)
            ci <= nc && (domains[off + ci] |= UInt64(1) << bi)
        end
    end
    domains
end

function _apply_attr_masks_world!(domains::Vector{UInt64}, csp, G, schema, enc)
    nc = csp.n_chunks
    for bc in csp.bytecodes
        cmp = _attr_cmp_code(bc.op)
        cmp < 0 && continue
        v     = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        n     = nparts(G, owner)
        mask  = zeros(UInt64, nc)
        for i in 1:min(n, nc * 64)
            raw = subpart(G, i, a)
            _attr_hit(encode_value(enc, a, raw), req, cmp) || continue
            ci, bi = elem_to_chunk(i)
            ci <= nc && (mask[ci] |= UInt64(1) << bi)
        end
        off = (v - 1) * nc
        for c in 1:nc; domains[off + c] &= mask[c]; end
    end
end

function _pin_initial!(domains::Vector{UInt64}, initial, csp, G, schema)
    nc = csp.n_chunks
    entries = initial isa NamedTuple ? pairs(initial) :
              initial isa Dict       ? pairs(initial) : ()
    for (obj, mapping) in entries
        haskey(csp.var_offset, obj) || continue
        base = csp.var_offset[obj]
        if mapping isa AbstractVector
            for (i, tgt) in enumerate(mapping)
                tgt == 0 && continue
                v = base + (i - 1)
                v > Int(csp.n_vars) && continue
                ci, bi = elem_to_chunk(tgt)
                ci > nc && continue
                off = (v - 1) * nc
                for c in 1:nc; domains[off + c] = UInt64(0); end
                domains[off + ci] = UInt64(1) << bi
            end
        elseif mapping isa Dict
            for (src, tgt) in mapping
                v = base + (src - 1)
                v > Int(csp.n_vars) && continue
                ci, bi = elem_to_chunk(tgt)
                ci > nc && continue
                off = (v - 1) * nc
                for c in 1:nc; domains[off + c] = UInt64(0); end
                domains[off + ci] = UInt64(1) << bi
            end
        end
    end
end

function _assignment_to_hom(sol::Vector{Int32}, L, G, csp::CSPProblem, schema::SchemaInfo)
    comps = Dict{Symbol, Vector{Int}}()
    S     = acset_schema(L)
    for o in ob(S)
        base = get(csp.var_offset, o, 0)
        base == 0 && continue
        n  = nparts(L, o)
        vs = Int[Int(sol[base + i - 1]) for i in 1:n]
        comps[o] = vs
    end
    try
        # Use homomorphism search to bind AttrVars based on the combinatorial match found by GPU
        𝒞 = Catlab.CategoricalAlgebra.infer_acset_cat(G)
        homs = homomorphisms(L, G; initial=comps, cat=𝒞)
        return isempty(homs) ? nothing : first(homs)
    catch
        return nothing
    end
end

end # module GPURewritingExt
