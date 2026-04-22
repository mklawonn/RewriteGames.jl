"""
    RewriteGames

A framework for defining turn-based games via algebraic rewriting on
attributed C-sets, with a minimal harness for running agents and collecting
experience.

## Quick-start

```julia
using RewriteGames
using Catlab, AlgebraicRewriting

# 1. Define a schema and an initial world factory
@present SchGraph(FreeSchema) begin
    V::Ob; E::Ob
    src::Hom(E,V); tgt::Hom(E,V)
end
@acset_type Graph(SchGraph, index=[:src,:tgt])

# 2. Define rewrite rules (see AlgebraicRewriting.jl docs)
# r_add_vertex = Rule(...)
# r_add_edge   = Rule(...)

# 3. Construct a Game
game = Game(SchGraph;
    players  = [:alice, :bob],
    rules    = Dict(
        :alice => [RuleEntry(r_add_vertex)],
        :bob   => [RuleEntry(r_add_edge)],
    ),
    terminal = (W) -> (nparts(W,:V) >= 10, nothing),
    initial  = () -> Graph(),
)

# 4. Run with random agents — returns a GameHistory
agents = Dict(:alice => FunctionAgent((s,a) -> rand(a)),
              :bob   => FunctionAgent((s,a) -> rand(a)))
hist = run_game(game, agents; T_max=50)

# 5. Inspect the history
get_world(hist, 0)          # initial world
get_world(hist, 1)          # world after turn 1
get_player(hist, 0)         # who acted on turn 0
winner(hist)                # winning player or nothing
```
"""
module RewriteGames

using Catlab
using AlgebraicRewriting

# ── Core types (rule_entry + auto_rule must precede game_step which uses them) ─
include("core/rule_entry.jl")
include("core/auto_rule.jl")

# ── Schedule primitives (before core/game.jl which references GameStep) ───────
include("schedule/game_step.jl")

# ── Game struct ─────────────────────────────────────────────────────────────────
include("core/game.jl")

# ── Encoding (must come before engine so EncodedState is visible) ───────────
include("encoding/encoding.jl")

# ── Agent interface ────────────────────────────────────────────────────────────
include("agents/abstract.jl")
include("agents/function_agent.jl")
# ONNXAgent is loaded via the ONNXAgentExt package extension when ONNXRunTime is available.

# ── Engine ─────────────────────────────────────────────────────────────────────
include("engine/matches.jl")
include("engine/auto.jl")
include("engine/driver.jl")

# ── History (before scheduled_driver which records into it) ────────────────────
include("history/game_history.jl")

# ── Schedule context + scheduled driver (after engine so GameHistory is defined)
include("schedule/agent_context.jl")
include("engine/scheduled_driver.jl")

# ── Serialization ──────────────────────────────────────────────────────────────
include("serialization/json.jl")

# ── Schema migration ───────────────────────────────────────────────────────────
include("migration/game_migration.jl")

# ── Analysis utilities ─────────────────────────────────────────────────────────
include("analysis.jl")

# ── DSL ────────────────────────────────────────────────────────────────────────
include("dsl.jl")

# ─── Public API ────────────────────────────────────────────────────────────────

export
    # Core
    RuleEntry, RuleLibrary,
    AutoRule,
    Game, GameState,
    nplayers,

    # Agents (ONNXAgent is exported by the ONNXAgentExt extension when loaded)
    AbstractAgent, Action,
    FunctionAgent,
    select_action,

    # Engine
    enumerate_all_matches, enumerate_legal_actions,
    apply_rule!, rule_index,
    fire_auto_rules!,
    run_game,

    # Schedule
    GameStep,
    PlayerStep, AutoStep, Auto,
    Seq, Cond, WhileStep, ForEachStep,
    AgentContext, push_context,
    ScheduledGameDriver, run_schedule!,

    # History
    GameHistory,
    record_world!, record_step!,
    get_world, get_chosen, get_available,
    get_player, get_path, get_match, get_terminal,
    turns, history_length,

    # Encoding (kept for agent implementations)
    EncodedState, encode_state,

    # Serialization
    write_history, read_history,

    # Migration
    GameMigration,
    migrate_world, migrate_rules, migrate_game,

    # Analysis
    winner, win_rate, episode_length, action_counts,

    # DSL
    @game

end # module RewriteGames
