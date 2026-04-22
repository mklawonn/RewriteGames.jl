using TemporalData
using TemporalData: Sheaf, DiscreteTime, Interval, add!
using Catlab: coproduct, apex, dom, codom

# ─── GameHistory ──────────────────────────────────────────────────────────────

"""
    GameHistory

Seven parallel narratives capturing a complete game episode:

- **world_narrative**: `Sheaf` storing world ACSet at each integer time point `t`.
- **chosen_spans**: `Dict{Int, NamedTuple}` — chosen rule's (rule_name, L, K, R) at turn `t`.
- **avail_spans**: `Dict{Int, NamedTuple}` — coproduct of all applicable rules' (L, K, R) at turn `t`.
- **player_narrative**: `Dict{Int, Symbol}` — active player at turn `t`.
- **path_narrative**: `Dict{Int, Vector{Symbol}}` — schedule path taken at turn `t`.
- **match_narrative**: `Dict{Int, Any}` — chosen match morphism at turn `t`.
- **terminal_narrative**: `Dict{Int, Union{Symbol,Nothing}}` — winner when the game ended, else `nothing`.
"""
mutable struct GameHistory
    world_narrative    :: Any  # Sheaf{DiscreteTime, W} — W_t at Interval(t)
    chosen_spans       :: Dict{Int, Any}  # t => (rule_name, L, K, R) or nothing
    avail_spans        :: Dict{Int, Any}  # t => (L, K, R) coproducts or nothing
    player_narrative   :: Dict{Int, Symbol}
    path_narrative     :: Dict{Int, Vector{Symbol}}
    match_narrative    :: Dict{Int, Any}
    terminal_narrative :: Dict{Int, Union{Symbol, Nothing}}
    _world_turns       :: Vector{Int}  # sorted integer times with a recorded world
    _step_turns        :: Vector{Int}  # sorted integer times with a recorded action
end

export GameHistory

"""
    GameHistory(initial_world)

Construct a `GameHistory` for a game whose world ACSet has the same type as
`initial_world`.  Records the initial world at time 0.
"""
function GameHistory(initial_world)
    T = typeof(initial_world)
    h = GameHistory(
        Sheaf(DiscreteTime(), T),
        Dict{Int, Any}(),
        Dict{Int, Any}(),
        Dict{Int, Symbol}(),
        Dict{Int, Vector{Symbol}}(),
        Dict{Int, Any}(),
        Dict{Int, Union{Symbol, Nothing}}(),
        Int[],
        Int[],
    )
    record_world!(h, initial_world, 0)
    return h
end

# ─── Recording functions ──────────────────────────────────────────────────────

"""
    record_world!(h::GameHistory, world, t::Int)

Store a copy of `world` at integer time `t` in the world narrative.
"""
function record_world!(h::GameHistory, world, t::Int)
    add!(h.world_narrative, Interval(t), copy(world))
    push!(h._world_turns, t)
end

export record_world!

"""
    record_step!(h::GameHistory; chosen_action, legal_actions, player, path, winner, t)

Record all narrative data for the action taken at turn `t`.

- `chosen_action`: the `Action` that was selected, or `nothing` if the player passed.
- `legal_actions`: all `Action`s that were legal this turn.
- `player`: the active player's `Symbol`.
- `path`: schedule path as `Vector{Symbol}`.
- `winner`: winning player symbol if the game terminated this turn, else `nothing`.
- `t`: the transition index (rule span spans interval `[t, t+1]`).
"""
function record_step!(h::GameHistory;
                      chosen_action,
                      legal_actions,
                      player::Symbol,
                      path::Vector{Symbol},
                      winner::Union{Symbol, Nothing},
                      t::Int)
    # Chosen rule span
    if chosen_action !== nothing
        rule = chosen_action.entry.rule
        h.chosen_spans[t] = (
            rule_name = chosen_action.entry.name,
            L = codom(rule.L),  # left foot: pre-rewrite interface
            K = dom(rule.L),    # apex: preserved interface
            R = codom(rule.R),  # right foot: post-rewrite interface
        )
        h.match_narrative[t] = chosen_action.match
    else
        h.chosen_spans[t] = nothing
        h.match_narrative[t] = nothing
    end

    # Available rules: one copy of each distinct rule with ≥1 legal match
    distinct_rules = unique(a.entry.rule for a in legal_actions)
    if isempty(distinct_rules)
        h.avail_spans[t] = nothing
    else
        Ks = [dom(r.L)   for r in distinct_rules]
        Ls = [codom(r.L) for r in distinct_rules]
        Rs = [codom(r.R) for r in distinct_rules]
        h.avail_spans[t] = (
            L = _acset_coproduct(Ls),
            K = _acset_coproduct(Ks),
            R = _acset_coproduct(Rs),
        )
    end

    h.player_narrative[t]   = player
    h.path_narrative[t]     = path
    h.terminal_narrative[t] = winner
    push!(h._step_turns, t)
end

export record_step!

# Compute the coproduct of a non-empty vector of ACSets.
# Returns the single element for length-1 vectors (no Catlab call needed).
function _acset_coproduct(xs::Vector)
    length(xs) == 1 && return only(xs)
    apex(coproduct(xs...))
end

# ─── Accessors ────────────────────────────────────────────────────────────────

"""
    get_world(h::GameHistory, t::Int)

Return the world ACSet at integer time `t`, or `nothing` if not recorded.
"""
get_world(h::GameHistory, t::Int) = get(h.world_narrative, Interval(t))

"""
    get_chosen(h::GameHistory, t::Int)

Return the chosen rule's named tuple `(rule_name, L, K, R)` at turn `t`,
or `nothing` if the player passed.
"""
get_chosen(h::GameHistory, t::Int) = h.chosen_spans[t]

"""
    get_available(h::GameHistory, t::Int)

Return the available-rules coproduct named tuple `(L, K, R)` at turn `t`,
or `nothing` if there were no legal moves.
"""
get_available(h::GameHistory, t::Int) = h.avail_spans[t]

"""
    get_player(h::GameHistory, t::Int) -> Symbol
"""
get_player(h::GameHistory, t::Int) = h.player_narrative[t]

"""
    get_path(h::GameHistory, t::Int) -> Vector{Symbol}
"""
get_path(h::GameHistory, t::Int) = h.path_narrative[t]

"""
    get_match(h::GameHistory, t::Int)

Return the chosen match morphism at turn `t`, or `nothing` if the player passed.
"""
get_match(h::GameHistory, t::Int) = h.match_narrative[t]

"""
    get_terminal(h::GameHistory, t::Int) -> Union{Symbol, Nothing}

Return the winner recorded at turn `t`, or `nothing` if the game was still ongoing.
"""
get_terminal(h::GameHistory, t::Int) = h.terminal_narrative[t]

export get_world, get_chosen, get_available, get_player, get_path, get_match, get_terminal

"""
    turns(h::GameHistory) -> Vector{Int}

Return sorted integer times at which a world state was recorded (including t=0).
"""
turns(h::GameHistory) = sort(h._world_turns)

"""
    history_length(h::GameHistory) -> Int

Return the number of player-action turns recorded (excludes the initial world at t=0).
"""
history_length(h::GameHistory) = length(h._step_turns)

export turns, history_length

Base.show(io::IO, h::GameHistory) =
    print(io, "GameHistory(turns=$(history_length(h)), world_snapshots=$(length(h._world_turns)))")
