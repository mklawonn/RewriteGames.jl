"""
    GameStep

Abstract supertype for all schedule tree nodes.  A `GameStep` tree describes
one *round* of a game; the driver loops over rounds until the terminal
predicate fires.

Every concrete subtype carries a `name::Symbol` field used to build the
`schedule_path` in emitted `Experience` records.
"""
abstract type GameStep end

# ─── Leaf nodes ───────────────────────────────────────────────────────────────

"""
    PlayerStep(player::Symbol; name=player)

A leaf schedule node where the named player's agent picks an action from its
rule library.  When inside a `ForEachStep`, action enumeration is restricted to
matches that involve the current context instance.

Emits exactly one `Experience` per evaluation.
"""
struct PlayerStep <: GameStep
    player :: Symbol
    name   :: Symbol
end
PlayerStep(player::Symbol; name::Symbol = player) = PlayerStep(player, name)

"""
    AutoStep(rules=nothing; name=:auto)
    AutoStep(rules::Vector{AutoRule}; name=:auto)

A leaf schedule node that fires auto-rules without agent input, analogous to
the between-turn auto-rule firing in the legacy `GameDriver`.

When `rules` is `nothing` (the default), the driver uses `game.auto` at
execution time.  When an explicit `Vector{AutoRule}` is supplied those rules
fire instead, overriding `game.auto` at that point in the schedule.

Emits no `Experience` records.
"""
struct AutoStep <: GameStep
    rules :: Union{Vector{AutoRule}, Nothing}
    name  :: Symbol
end
AutoStep(; name::Symbol = :auto)                              = AutoStep(nothing, name)
AutoStep(rules::Vector{AutoRule}; name::Symbol = :auto)       = AutoStep(rules, name)

"""Convenience alias for `AutoStep()`."""
const Auto = AutoStep

# ─── Composite nodes ──────────────────────────────────────────────────────────

"""
    Seq(steps...; name=:seq)
    Seq(steps::Vector{GameStep}; name=:seq)

Execute `steps` in order.  Short-circuits (stops executing remaining steps)
as soon as the game terminal condition is triggered.
"""
struct Seq <: GameStep
    steps :: Vector{GameStep}
    name  :: Symbol
end
Seq(steps::GameStep...; name::Symbol = :seq)       = Seq(collect(GameStep, steps), name)
Seq(steps::Vector{GameStep}; name::Symbol = :seq)  = Seq(steps, name)

"""
    Cond(pred, branches...; name=:cond)
    Cond(pred, branches::Vector{GameStep}; name=:cond)

Branch on world state.  `pred(W) -> Int` returns a 1-based branch index;
the selected branch is executed and the others are skipped.

For a boolean condition use index 1 = true, 2 = false:
```julia
Cond(W -> nparts(W, :V) > 3 ? 1 : 2, big_graph_step, small_graph_step)
```
"""
struct Cond <: GameStep
    pred     :: Function          # W -> Int (1-based)
    branches :: Vector{GameStep}
    name     :: Symbol
end
Cond(pred::Function, branches::GameStep...; name::Symbol = :cond) =
    Cond(pred, collect(GameStep, branches), name)
Cond(pred::Function, branches::Vector{GameStep}; name::Symbol = :cond) =
    Cond(pred, branches, name)

"""
    WhileStep(cond, body; name=:while, max_iter=1000)

Iterate `body` while `cond(W)` is `true`.  Designed for intra-round sub-loops
(e.g. "wolf grazes until satiated") that are distinct from the inter-round
loop driven by the terminal predicate.

`max_iter` is a safety cap that throws an error if exceeded, preventing
runaway loops from non-terminating conditions.
"""
struct WhileStep <: GameStep
    cond     :: Function     # W -> Bool; iterate while true
    body     :: GameStep
    name     :: Symbol
    max_iter :: Int
end
WhileStep(cond::Function, body::GameStep;
          name::Symbol = :while, max_iter::Int = 1000) =
    WhileStep(cond, body, name, max_iter)

"""
    ForEachStep(ob, body; name=:foreach, order=:natural)

For every part of type `ob` in the world, execute `body` with an
`AgentContext(ob, id)` active.  `PlayerStep` nodes inside `body` will only
present matches that involve the specific instance being iterated.

`order` controls iteration order:
- `:natural`  — ascending part id (default)
- `:random`   — randomly shuffled each time
- `:reversed` — descending part id

## Instance deletion during iteration

Part ids are snapshotted at the start of the loop.  If a rewrite deletes an
instance mid-iteration, the corresponding id is skipped (checked against the
live world).  Due to DPO renumbering, surviving instances retain consecutive
ids, so the snapshot approach is safe for non-deleting rules and gives
reasonable (if shifted) results when deletions occur.

## Rule design note

A rule whose left-hand pattern has no parts of type `ob` will have all its
matches excluded when a context is active — its `ob` component image is empty,
so the context id cannot be found in it.  Design rules used inside
`ForEachStep(:Wolf, ...)` to include at least one `Wolf` part in their
left-hand pattern.
"""
struct ForEachStep <: GameStep
    ob    :: Symbol
    body  :: GameStep
    name  :: Symbol
    order :: Symbol
end
ForEachStep(ob::Symbol, body::GameStep;
            name::Symbol = :foreach, order::Symbol = :natural) =
    ForEachStep(ob, body, name, order)

# ─── show methods ─────────────────────────────────────────────────────────────

Base.show(io::IO, s::PlayerStep)  = print(io, "PlayerStep(:$(s.player))")
Base.show(io::IO, s::AutoStep)    =
    print(io, "AutoStep($(s.rules === nothing ? "game.auto" : "$(length(s.rules)) explicit rule(s)"))")
Base.show(io::IO, s::Seq)         = print(io, "Seq($(length(s.steps)) step(s))")
Base.show(io::IO, s::Cond)        = print(io, "Cond($(length(s.branches)) branch(es))")
Base.show(io::IO, s::WhileStep)   = print(io, "WhileStep(max_iter=$(s.max_iter))")
Base.show(io::IO, s::ForEachStep) = print(io, "ForEachStep(:$(s.ob), order=:$(s.order))")
