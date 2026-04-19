"""
    Game

Defines a turn-based game over ACSet world states.

# Fields
- `players`:   Ordered list of player names (Symbols).
- `rules`:     Dict mapping each player Symbol to its `RuleLibrary`.
- `auto`:      Ordered collection of `AutoRule`s that fire between turns.
- `terminal`:  `(W) -> (done::Bool, winner::Union{Symbol,Nothing})` predicate.
- `initial`:   `() -> ACSet` factory producing fresh world states.
- `schema`:    The presentation / schema object (optional metadata).
"""
struct Game
    players  :: Vector{Symbol}
    rules    :: Dict{Symbol, RuleLibrary}
    auto     :: Vector{AutoRule}
    terminal :: Function            # W -> (Bool, Union{Symbol,Nothing})
    initial  :: Function            # () -> ACSet
    schema   :: Any                 # optional schema metadata
end

function Game(
    schema=nothing;
    players::Vector{Symbol} = [:player],
    rules::Dict   = Dict{Symbol,RuleLibrary}(),
    auto::Vector  = AutoRule[],
    terminal::Function = (W) -> (false, nothing),
    initial::Function  = () -> error("No initial world factory provided"),
)
    # Normalise rules: accept Vector{RuleEntry} values
    norm_rules = Dict{Symbol, RuleLibrary}()
    for p in players
        lib = get(rules, p, RuleEntry[])
        norm_rules[p] = lib isa RuleLibrary ? lib : RuleLibrary(collect(RuleEntry, lib))
    end
    Game(players, norm_rules, collect(AutoRule, auto), terminal, initial, schema)
end

# ─── GameState ────────────────────────────────────────────────────────────────

"""
    GameState

Mutable container for the live state of an in-progress game.

# Fields
- `world`:    The current ACSet world.
- `counters`: Per-rule remaining budget counters, indexed by (player, rule_index).
- `turn`:     Current turn number (starts at 1).
"""
mutable struct GameState
    world    :: Any                         # ACSet
    counters :: Dict{Tuple{Symbol,Int}, Int}  # (player, rule_idx) -> remaining uses
    turn     :: Int
end

function GameState(world, game::Game)
    counters = Dict{Tuple{Symbol,Int}, Int}()
    for p in game.players
        for (i, entry) in enumerate(game.rules[p].entries)
            if entry.budget !== nothing
                counters[(p, i)] = entry.budget
            end
        end
    end
    GameState(world, counters, 1)
end

Base.copy(s::GameState) = GameState(copy(s.world), copy(s.counters), s.turn)

# ─── Game helpers ─────────────────────────────────────────────────────────────

"""Return the number of players in the game."""
nplayers(g::Game) = length(g.players)
Base.length(g::Game) = length(g.players)

Base.show(io::IO, g::Game) =
    print(io, "Game(players=$(g.players), auto=$(length(g.auto)) rule(s))")

Base.show(io::IO, s::GameState) =
    print(io, "GameState(turn=$(s.turn), budgets=$(length(s.counters)))")
