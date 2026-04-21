"""
    Experience

A single step of interaction between a player and the game environment.

# Fields
- `player`:        Symbol naming the active player.
- `state`:         `EncodedState` capturing the world *before* the action.
- `legal_actions`: All `Action`s available to the player this turn.
- `action`:        The chosen `Action`, or `nothing` if the player passed.
- `next_state`:    `EncodedState` capturing the world *after* the action
                   and after any `AutoStep` nodes that followed it in the schedule.
- `done`:          Whether the game terminated after this step.
- `winner`:        Winning player symbol, or `nothing`.
- `info`:          Metadata dict (budget snapshot, context, …).
- `schedule_path`: Path through the `GameStep` tree where this experience was
                   emitted, as a `Vector{Symbol}` of node names.
"""
struct Experience
    player        :: Symbol
    state         :: EncodedState
    legal_actions :: Vector{Action}
    action        :: Union{Action, Nothing}
    next_state    :: EncodedState
    done          :: Bool
    winner        :: Union{Symbol, Nothing}
    info          :: Dict{Symbol, Any}
    schedule_path :: Vector{Symbol}
end

Base.show(io::IO, e::Experience) =
    print(io, "Experience(player=:$(e.player), action=$(e.action), done=$(e.done), winner=$(e.winner))")

# ─── run_game ─────────────────────────────────────────────────────────────────

"""
    run_game(game::Game, agents::Dict{Symbol,<:AbstractAgent}; T_max=1000)
        -> Vector{Experience}

Run a single complete episode and return all experiences.

Delegates to `ScheduledGameDriver` (defined in `engine/scheduled_driver.jl`),
which executes `game.schedule` once per round until the terminal predicate fires
or `T_max` steps are reached.
"""
function run_game(game::Game, agents::Dict{Symbol, <:AbstractAgent}; T_max::Int=1000)
    # _run_scheduled_game is defined in engine/scheduled_driver.jl, loaded after
    # this file; the forward reference is resolved at call-time, not load-time.
    return _run_scheduled_game(game, agents; T_max=T_max)
end
