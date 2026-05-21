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
using Random: Xoshiro

import Catlab.CategoricalAlgebra:
    acset_schema, ob, hom, attr, dom, codom, nparts, parts, subpart,
    set_subpart!, add_parts!, add_part!, homomorphism, homomorphisms,
    ACSetTransformation, AttrVar

import Catlab.CategoricalAlgebra: left, right

import RewriteGames: GameSched, PlayerRuleApp, Experience, GameState, Action,
                     select_action, _collect_player_apps

# ── Lowering layer ────────────────────────────────────────────────────────────
include("lowering/SchemaInfo.jl")
include("lowering/RuleRegistry.jl")
include("lowering/DeviceData.jl")
include("lowering/FlattenRegistry.jl")
include("lowering/AttributeEncoder.jl")
include("solver/TCNBytecode.jl")        # TCNBytecode needed by CSPLowering
include("lowering/CSPLowering.jl")
include("lowering/AdhesiveCubes.jl")
include("lowering/ScheduleCompiler.jl")

# ── Solver layer ──────────────────────────────────────────────────────────────
include("solver/PropagationKernel.jl")
include("solver/DiveSolveKernel.jl")

# ── Rewriting layer ───────────────────────────────────────────────────────────
include("rewriting/GPUACSet.jl")
include("rewriting/DeletionKernel.jl")
include("rewriting/AdditionKernel.jl")
include("rewriting/IncrementalUpdate.jl")

# ── Control layer ─────────────────────────────────────────────────────────────
include("control/TrajectoryLogger.jl")
include("control/StreamCompaction.jl")
include("control/Scheduler.jl")
include("control/MasterScheduler.jl")

# ── Reconstruction layer ──────────────────────────────────────────────────────
include("reconstruction/Decode.jl")

# ── Public API ────────────────────────────────────────────────────────────────

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
    terminal      :: Function                                  = (W) -> (false, nothing),
    winner_wires  :: Dict{Symbol, Union{Symbol,Nothing}}       =
                        Dict{Symbol,Union{Symbol,Nothing}}(),
    log_trajectory :: Bool                                     = false,
    compact_every  :: Int                                      = 100,
)::Vector{Experience}

    # ── Phase 1: Host-side compilation ────────────────────────────────────────
    schema = extract_schema_info(initial_world)
    enc    = build_encoder(initial_world, schema)
    sched  = compile_schedule(gs, initial_world, schema, enc)

    # ── Upload initial world to GPU ───────────────────────────────────────────
    g = upload_acset(initial_world, schema, enc)

    world_type = typeof(initial_world)

    # ── Build scheduler state ─────────────────────────────────────────────────
    state = GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                              log_trajectory = log_trajectory,
                              compact_every  = compact_every)

    # ── Phase 4: Execute on GPU ───────────────────────────────────────────────
    run_gpu_schedule!(state;
                      T_max        = T_max,
                      terminal_fn  = terminal,
                      winner_wires = winner_wires)

    # ── Phase 5: Decode results ───────────────────────────────────────────────
    reconstruct_experiences(state.step_log, schema, enc, world_type, agents)
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
                           initial :: Union{Nothing, NamedTuple, Dict} = nothing)

    schema = extract_schema_info(G)
    enc    = build_encoder(G, schema)

    # Build a mock rule wrapping L so lower_rule_to_csp can access left(rule)
    mock_rule = _make_identity_rule(L, monic)
    csp = lower_rule_to_csp(mock_rule, G, schema, enc)

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

    if backend === nothing
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
    domains = zeros(UInt64, Int(csp.n_vars))
    for v in 1:Int(csp.n_vars)
        n = Int(csp.domain_sizes[v])
        mask = n < 64 ? (UInt64(1) << n) - UInt64(1) : typemax(UInt64)
        domains[v] = mask
    end
    domains
end

function _apply_attr_masks_world!(domains, csp, G, schema, enc)
    for bc in csp.bytecodes
        bc.op != PROP_ATTR_EQ && continue
        v     = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        n     = nparts(G, owner)
        mask  = UInt64(0)
        for i in 1:min(n, 63)
            raw = subpart(G, i, a)
            encode_value(enc, a, raw) == req && (mask |= UInt64(1) << (i-1))
        end
        domains[v] &= mask
    end
end

function _pin_initial!(domains, initial, csp, G, schema)
    entries = initial isa NamedTuple ? pairs(initial) :
              initial isa Dict       ? pairs(initial) : ()
    for (obj, mapping) in entries
        haskey(csp.var_offset, obj) || continue
        base = csp.var_offset[obj]
        if mapping isa AbstractVector
            for (i, tgt) in enumerate(mapping)
                tgt == 0 && continue
                tgt > 63 && continue
                v = base + (i - 1)
                v > Int(csp.n_vars) && continue
                domains[v] = UInt64(1) << (tgt - 1)
            end
        elseif mapping isa Dict
            for (src, tgt) in mapping
                tgt > 63 && continue
                v = base + (src - 1)
                v > Int(csp.n_vars) && continue
                domains[v] = UInt64(1) << (tgt - 1)
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
