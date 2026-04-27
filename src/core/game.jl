"""
    Game

Defines a turn-based game over ACSet world states.

# Fields
- `players`:        Ordered list of player names (Symbols).
- `terminal`:       `(W) -> (done::Bool, winner::Union{Symbol,Nothing})` predicate,
                    or `nothing` to rely entirely on schedule exit wires.
- `initial`:        `() -> ACSet` factory producing fresh world states.
- `schema`:         The presentation / schema object (optional metadata).
- `win_conditions`: Optional `Dict{Symbol,Any}` mapping exit wire names to the
                    winner identity (`Symbol` or `nothing` for a draw).  When
                    provided, `run_game_sched!` uses it to resolve the winner
                    from the active exit wire instead of calling `terminal`.
"""
struct Game
    players        :: Vector{Symbol}
    terminal       :: Union{Function, Nothing}  # W -> (Bool, Union{Symbol,Nothing}), or nothing
    initial        :: Function                  # () -> ACSet
    schema         :: Any                       # optional schema metadata
    win_conditions :: Union{Dict{Symbol,Any}, Nothing}
end

function Game(
    schema = nothing;
    players        :: Vector{Symbol}                    = [:player],
    terminal       :: Union{Function, Nothing}          = nothing,
    initial        :: Function                          = () -> error("No initial world factory provided"),
    win_conditions :: Union{Dict{Symbol,Any}, Nothing}  = nothing,
)
    Game(players, terminal, initial, schema, win_conditions)
end

# ─── GameState ────────────────────────────────────────────────────────────────

"""
    GameState

Mutable container for the live state of an in-progress game.

# Fields
- `world`:  The current ACSet world.
- `turn`:   Current turn number (starts at 1).
"""
mutable struct GameState
    world :: Any   # ACSet
    turn  :: Int
end

Base.copy(s::GameState) = GameState(copy(s.world), s.turn)

# ─── Game helpers ─────────────────────────────────────────────────────────────

"""Return the number of players in the game."""
nplayers(g::Game) = length(g.players)
Base.length(g::Game) = length(g.players)

Base.show(io::IO, g::Game) = print(io,
    "Game(players=$(g.players))")

Base.show(io::IO, s::GameState) =
    print(io, "GameState(turn=$(s.turn))")
