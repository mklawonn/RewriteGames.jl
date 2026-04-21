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
- `schedule`:  `GameStep` tree describing the round structure.  When the
               keyword constructor receives `schedule=nothing` (the default),
               a round-robin schedule is generated automatically:
               `Seq([PlayerStep(p), AutoStep()]` for each player in order.
"""
struct Game
    players  :: Vector{Symbol}
    rules    :: Dict{Symbol, RuleLibrary}
    auto     :: Vector{AutoRule}
    terminal :: Function            # W -> (Bool, Union{Symbol,Nothing})
    initial  :: Function            # () -> ACSet
    schema   :: Any                 # optional schema metadata
    schedule :: GameStep
end

function Game(
    schema=nothing;
    players::Vector{Symbol} = [:player],
    rules::Dict   = Dict{Symbol,RuleLibrary}(),
    auto::Vector  = AutoRule[],
    terminal::Function = (W) -> (false, nothing),
    initial::Function  = () -> error("No initial world factory provided"),
    schedule::Union{GameStep, Nothing} = nothing,
)
    # Normalise rules: accept Vector{RuleEntry} values
    norm_rules = Dict{Symbol, RuleLibrary}()
    for p in players
        lib = get(rules, p, RuleEntry[])
        norm_rules[p] = lib isa RuleLibrary ? lib : RuleLibrary(collect(RuleEntry, lib))
    end
    # Auto-generate a round-robin schedule when none is provided.
    # Each player gets a PlayerStep followed by an AutoStep so that auto-rules
    # fire after every move, exactly as the legacy GameDriver did.
    sched = schedule === nothing ?
        Seq(GameStep[s for p in players for s in (PlayerStep(p), AutoStep())]) :
        schedule
    Game(players, norm_rules, collect(AutoRule, auto), terminal, initial, schema, sched)
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

Base.show(io::IO, g::Game) = print(io,
    "Game(players=$(g.players), auto=$(length(g.auto)) rule(s), schedule=$(g.schedule))")

Base.show(io::IO, s::GameState) =
    print(io, "GameState(turn=$(s.turn), budgets=$(length(s.counters)))")
