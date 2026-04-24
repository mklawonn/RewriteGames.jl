"""
    Game

Defines a turn-based game over ACSet world states.

# Fields
- `players`:   Ordered list of player names (Symbols).
- `terminal`:  `(W) -> (done::Bool, winner::Union{Symbol,Nothing})` predicate.
- `initial`:   `() -> ACSet` factory producing fresh world states.
- `schema`:    The presentation / schema object (optional metadata).
"""
struct Game
    players  :: Vector{Symbol}
    terminal :: Function            # W -> (Bool, Union{Symbol,Nothing})
    initial  :: Function            # () -> ACSet
    schema   :: Any                 # optional schema metadata
end

function Game(
    schema = nothing;
    players  :: Vector{Symbol}    = [:player],
    terminal :: Function          = (W) -> (false, nothing),
    initial  :: Function          = () -> error("No initial world factory provided"),
)
    Game(players, terminal, initial, schema)
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
