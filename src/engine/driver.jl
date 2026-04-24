"""
    Experience

A single step of interaction between a player and the game environment.

# Fields
- `player`:        Symbol naming the active player.
- `state`:         `EncodedState` capturing the world *before* the action.
- `legal_actions`: All `Action`s available to the player this turn.
- `action`:        The chosen `Action`, or `nothing` if the player passed.
- `next_state`:    `EncodedState` capturing the world *after* the action.
- `done`:          Whether the game terminated after this step.
- `winner`:        Winning player symbol, or `nothing`.
- `info`:          Metadata dict.
- `schedule_path`: Reserved for compatibility; always `Symbol[]`.
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
