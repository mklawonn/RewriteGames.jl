"""
    Experience

A single step of interaction between a player and the game environment.

# Fields
- `player`:        Symbol naming the active player.
- `state`:         `GameState` capturing the full world *before* the action.
- `legal_actions`: All `Action`s available to the player this turn.
- `action`:        The chosen `Action`, or `nothing` if the player passed.
- `next_state`:    `GameState` capturing the full world *after* the action.
- `done`:          Whether the game terminated after this step.
- `winner`:        Winning player symbol, or `nothing`.
- `info`:          Metadata dict.
- `schedule_path`: Reserved for compatibility; always `Symbol[]`.
- `view`:          The subworld the player observed at decision time, or
                   `nothing` when the box has no `view_fn` (full information).
"""
struct Experience
    player        :: Symbol
    state         :: GameState
    legal_actions :: Vector{Action}
    action        :: Union{Action, Nothing}
    next_state    :: GameState
    done          :: Bool
    winner        :: Union{Symbol, Nothing}
    info          :: Dict{Symbol, Any}
    schedule_path :: Vector{Symbol}
    view          :: Any   # subworld ACSet or nothing
end

Experience(player, state, legal_actions, action, next_state, done, winner, info, schedule_path) =
    Experience(player, state, legal_actions, action, next_state, done, winner, info, schedule_path, nothing)

Experience(player, state, legal_actions, action, next_state, done, winner, info) =
    Experience(player, state, legal_actions, action, next_state, done, winner, info, Symbol[], nothing)

Base.show(io::IO, e::Experience) =
    print(io, "Experience(player=:$(e.player), action=$(e.action), done=$(e.done), winner=$(e.winner))")
