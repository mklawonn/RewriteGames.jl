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

# ── Core types ──────────────────────────────────────────────────────────────
include("core/rule_entry.jl")
include("core/auto_rule.jl")

# ── Game struct ─────────────────────────────────────────────────────────────
include("core/game.jl")

# ── Encoding utilities ────────────────────────────────────────────────────
include("encoding/encoding.jl")

# ── Agent interface ──────────────────────────────────────────────────────
include("agents/abstract.jl")
include("agents/function_agent.jl")
# ONNXAgent is loaded via the ONNXAgentExt package extension when ONNXRunTime is available.

# ── Engine ──────────────────────────────────────────────────────────────────
include("engine/matches.jl")
include("engine/auto.jl")
include("engine/driver.jl")

# ── Wiring-diagram schedule layer ────────────────────────────────────────────
# (player_rule_app.jl defines _parse_body used by sched_runner.jl)
include("schedule/player_rule_app.jl")
include("engine/sched_runner.jl")

# ── Serialization ─────────────────────────────────────────────────────────
include("serialization/arrow.jl")

# ── Schema migration ─────────────────────────────────────────────────────
include("migration/game_migration.jl")

# ── Analysis utilities ───────────────────────────────────────────────────
include("analysis.jl")

# ── DSL ──────────────────────────────────────────────────────────────────
include("dsl.jl")

# ─── Public API ─────────────────────────────────────────────────────────────

export
    # Core
    RuleEntry, RuleLibrary,
    AutoRule,
    Game, GameState,
    nplayers,

    # Agents
    AbstractAgent, Action,
    FunctionAgent,
    select_action,

    # Engine
    enumerate_all_matches, enumerate_legal_actions,
    apply_rule!, rule_index,
    fire_auto_rules!,
    Experience,

    # Wiring-diagram schedule
    PlayerRuleApp, GameSched,
    mk_game_sched, player_migrate,
    view_sched,
    run_game_sched!,

    # Encoding utilities
    elements_graph,

    # Serialization
    write_experiences, read_experiences,

    # Migration
    GameMigration,
    migrate_world,

    # Analysis
    win_rate, episode_length, action_counts,

    # DSL
    @game

end # module RewriteGames
