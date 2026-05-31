"""
    RewriteGames

A framework for defining turn-based games via AlgebraicRewriting wiring-diagram
schedules on attributed C-sets, with a minimal harness for running agents and
collecting experience.

## Quick-start

```julia
using RewriteGames
using Catlab, AlgebraicRewriting

# 1. Define schema, world factory, and terminal predicate
game = Game(SchGraph;
    players  = [:alice, :bob],
    terminal = (W) -> (nparts(W,:V) >= 10, nothing),
    initial  = () -> Graph(),
)

# 2. Build wiring-diagram schedule with PlayerRuleApp boxes
alice_app = PlayerRuleApp(:add_vertex, rule_add_vertex, I, :alice; cat=𝒞)
bob_app   = PlayerRuleApp(:add_edge,   rule_add_edge,   I, :bob;   cat=𝒞)
sched = mk_game_sched(
    (trace_arg=:I,), (init=:I,), N,
    (a=alice_app, b=bob_app, mw=merge_wires(I)),
    quote
        a_moved, a_pass = a(init)
        b_moved, b_pass = b([a_moved, trace_arg])
        cont = mw(b_moved, b_pass)
        return cont, a_pass
    end)

# 3. Run with random agents
agents = Dict(:alice => FunctionAgent((s,a) -> rand(a)),
              :bob   => FunctionAgent((s,a) -> rand(a)))
exps = run_game_sched!(sched, game, agents; T_max=50)
```
"""
module RewriteGames

using Catlab
using AlgebraicRewriting

# ── Core ────────────────────────────────────────────────────────────────────
include("core/game.jl")

# ── Encoding utilities ────────────────────────────────────────────────────
include("encoding/encoding.jl")

# ── Agent interface ──────────────────────────────────────────────────────
include("agents/abstract.jl")
include("agents/function_agent.jl")

# ── Engine ──────────────────────────────────────────────────────────────────
include("engine/driver.jl")
include("engine/match_cache.jl")

# ── Wiring-diagram schedule layer ────────────────────────────────────────────
# (player_rule_app.jl defines _parse_body used by sched_runner.jl)
include("schedule/player_rule_app.jl")
include("engine/sched_runner.jl")

# ── Serialization ─────────────────────────────────────────────────────────
include("serialization/arrow.jl")
include("serialization/game_json.jl")

# ── Schema migration ─────────────────────────────────────────────────────
include("migration/game_migration.jl")

# ── Analysis utilities ───────────────────────────────────────────────────
include("analysis.jl")

# ── DSL ──────────────────────────────────────────────────────────────────
include("dsl.jl")

# ── GPU extension stubs ───────────────────────────────────────────────────────
# Implementations live in ext/GPURewritingExt/ and are loaded automatically
# when KernelAbstractions + CUDA are present in the environment.

"""
    gpu_run_game_sched!(gs, initial_world, agents; backend, T_max, ...)
        -> Vector{Experience}

GPU-accelerated version of `run_game_sched!`.  Requires `KernelAbstractions`
and `CUDA` to be loaded.  See the `GPURewritingExt` extension for details.
"""
function gpu_run_game_sched! end

"""
    turbo_homomorphisms(L, G; backend, monic, initial) -> Vector{ACSetTransformation}

GPU Turbo homomorphism enumerator.  Returns the same set as Catlab's
`homomorphisms(L, G)` computed via the CSP propagation + dive-solve engine.
Requires `KernelAbstractions` and `CUDA`.
"""
function turbo_homomorphisms end

"""
    select_action_gpu(player::AbstractGPUPlayer, g, enc, schema,
                      candidates, n_sols, turn) -> Int

Choose one candidate solution (1-based column index) from the GPU-resident
`candidates` matrix of shape `[n_vars × n_sols]`.  Implemented by concrete
`AbstractGPUPlayer` subtypes.  The default for `GPUFunctionPlayer` calls
`player.f(g, candidates, n_sols, turn)`.
"""
function select_action_gpu end

# ─── Public API ─────────────────────────────────────────────────────────────

export
    # Core
    Game, GameState,
    nplayers,

    # Agents
    AbstractAgent, Action,
    FunctionAgent,
    select_action,
    AbstractGPUPlayer, AbstractGNNPlayer, GPUFunctionPlayer,
    select_action_gpu,

    # Engine
    Experience,

    # Wiring-diagram schedule
    PlayerRuleApp, GameSched,
    mk_game_sched, player_migrate,
    merge_wires, coin,
    view_sched,
    run_game_sched!,
    _collect_rule_r_acssets,

    # Incremental match cache
    MatchCache,
    update_cache!,

    # Encoding utilities
    elements_graph,

    # Serialization
    write_experiences, read_experiences,
    write_game, read_game,

    # Migration
    GameMigration,
    migrate_world,

    # Analysis
    win_rate, episode_length, action_counts,

    # DSL
    @game,

    # GPU extension (available when KernelAbstractions + CUDA are loaded)
    gpu_run_game_sched!,
    turbo_homomorphisms,

    # GPU graph representation
    GPUGraphData,
    build_gpu_graph,
    rebuild_gpu_graph!,
    update_graph_deletions!,
    update_graph_additions!,
    live_coo,

    # Zone-partitioned CSP
    ZonePartition,
    build_zone_partition,
    update_zone_masks!,
    collect_zoned_solutions!

end # module RewriteGames
