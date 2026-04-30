"""
    PlayerRuleApp

Wraps an AlgebraicRewriting `RuleApp` and tags it with a player identity so
that `run_game_sched!` can intercept it and ask the appropriate agent to
choose a match.

Mirrors the `RuleApp(name, rule, init; cat=...)` constructor signature.
The inner `RuleApp` is stored in `_inner` and forwarded to `mk_sched` /
`view_sched`; `PlayerRuleApp` itself does not need to subtype `AgentBox`.

# Fields
- `name`:          Box name (Symbol), used for labels and `Action` display.
- `rule`:          The AlgebraicRewriting `Rule` to apply.
- `init`:          Interface ACSet (same role as `RuleApp`'s third argument).
- `player`:        Key into the `agents` dict passed to `run_game_sched!`.
- `cat`:           AC-category (passed through to the inner `RuleApp`).
- `_inner`:        Inner `RuleApp` forwarded to `mk_sched` / `view_sched`.
- `fast_match_fn`: Optional user-provided function
                   `(rule, world, cat) -> Vector{ACSetTransformation}`
                   that replaces `get_matches` entirely.  Use this when
                   domain knowledge allows a faster enumeration than the
                   general homomorphism search (e.g. index lookups).
                   When set, `use_cache` is ignored.
- `use_cache`:     When `true`, `run_game_sched!` maintains a `MatchCache`
                   for this box and updates it incrementally after every DPO
                   rewrite, avoiding a full re-search each turn.
- `match_limit`:   When set to an `Int`, caps the number of matches
                   enumerated to at most that many per turn.  Passes a
                   lazy `take` limit to the underlying homomorphism search
                   so that work stops once enough matches are found.  Has
                   no effect when `fast_match_fn` is set (the user-supplied
                   function controls enumeration).
"""
struct PlayerRuleApp
    name          :: Symbol
    rule          :: Any      # AbsRule
    init          :: Any      # interface ACSet
    player        :: Symbol
    cat           :: Any
    _inner        :: Any      # inner RuleApp
    fast_match_fn :: Union{Function, Nothing}
    use_cache     :: Bool
    match_limit   :: Union{Int, Nothing}
end

function PlayerRuleApp(name::Symbol, rule, init, player::Symbol;
                        cat=nothing,
                        fast_match_fn::Union{Function,Nothing}=nothing,
                        use_cache::Bool=false,
                        match_limit::Union{Int,Nothing}=nothing)
    inner = cat === nothing ? RuleApp(name, rule, init) :
                              RuleApp(name, rule, init; cat=cat)
    PlayerRuleApp(name, rule, init, player, cat, inner, fast_match_fn, use_cache,
                  match_limit)
end

Base.show(io::IO, p::PlayerRuleApp) =
    print(io, "PlayerRuleApp(:$(p.name), player=:$(p.player))")

# ─── GameSched ────────────────────────────────────────────────────────────────

"""
    GameSched

Wraps an AlgebraicRewriting `Schedule` together with the metadata needed by
`run_game_sched!` to intercept `PlayerRuleApp` boxes and ask agents for match
selections.  Also stores enough data to rebuild the schedule after schema
migration via `player_migrate`.

Construct with `mk_game_sched` rather than directly.
"""
struct GameSched
    _inner       :: Any                   # AR Schedule (view_sched target)
    _player_map  :: Dict{Symbol, PlayerRuleApp}
    _all_boxes   :: Any                   # original NamedTuple of boxes
    _steps       :: Vector{Any}           # parsed ExecStep vector
    _init_names  :: Vector{Symbol}
    _trace_names :: Vector{Symbol}
    _ret_names   :: Vector{Symbol}
    _trace_args  :: Any                   # original trace_args NamedTuple (for rebuild)
    _init_args   :: Any                   # original init_args NamedTuple (for rebuild)
    _body        :: Any                   # original body Expr (for rebuild)
    _N           :: Any                   # Names object (for rebuild)
end

Base.show(io::IO, gs::GameSched) =
    print(io, "GameSched(players=$(collect(keys(gs._player_map))), " *
              "init=$(gs._init_names), trace=$(gs._trace_names))")

# ─── mk_game_sched ────────────────────────────────────────────────────────────

"""
    mk_game_sched(trace_args, init_args, N, boxes, body; kwargs...)
        -> GameSched

Build a `GameSched` with the same signature as `mk_sched`.

`boxes` is a `NamedTuple` whose values may be:
- `PlayerRuleApp` — a player-controlled rule application; its inner `RuleApp`
  is forwarded to `mk_sched` and it is registered in the schedule's player map.
- `GameSched` — a nested schedule; its inner AR `Schedule` is forwarded.
- Any other `AgentBox` or `Schedule` value — forwarded as-is.

`view_sched(gs)` delegates to `view_sched(gs._inner)`.
"""
function mk_game_sched(trace_args, init_args, N, boxes, body; kwargs...)
    # Translate boxes for AR's mk_sched:
    #   PlayerRuleApp → its inner RuleApp
    #   GameSched     → its inner AR Schedule
    #   anything else → pass through unchanged
    ar_boxes = map(boxes) do v
        v isa PlayerRuleApp ? v._inner :
        v isa GameSched     ? v._inner : v
    end

    inner = mk_sched(trace_args, init_args, N, ar_boxes, body; kwargs...)

    player_map = Dict{Symbol, PlayerRuleApp}(
        k => v for (k, v) in pairs(boxes) if v isa PlayerRuleApp
    )

    steps, ret_names = _parse_body(body)

    GameSched(
        inner, player_map, boxes, steps,
        collect(Symbol, keys(init_args)),
        collect(Symbol, keys(trace_args)),
        ret_names,
        trace_args, init_args, body, N,
    )
end

# ─── view_sched extension ─────────────────────────────────────────────────────

import AlgebraicRewriting: view_sched

"""
    view_sched(gs::GameSched; kw...) -> Graphviz diagram

Render the underlying AlgebraicRewriting schedule diagram.  Identical output
to calling `view_sched` on the schedule built by `mk_sched` with the same
arguments (since `PlayerRuleApp` boxes display identically to `RuleApp` boxes).
"""
view_sched(gs::GameSched; kw...) = view_sched(gs._inner; kw...)

# ─── player_migrate ───────────────────────────────────────────────────────────

"""
    player_migrate(F, gs::GameSched, player_map::Dict{Symbol,Symbol}) -> GameSched

Migrate a `GameSched` to a new schema via the functor `F` (an
`AlgebraicRewriting.Migrate` object) and re-assign player identities.

Traverses all boxes in the schedule:
- `PlayerRuleApp` boxes have their rule and interface migrated via `F`, and
  their `player` field remapped through `player_map`.
- Nested `GameSched` boxes are migrated recursively.
- Other boxes (plain `RuleApp`, `Schedule`) are migrated by calling `F(box)`.
"""
function player_migrate(F, gs::GameSched, player_map::Dict{Symbol, Symbol};
                        name_map::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}())
    new_boxes = map(gs._all_boxes) do v
        if v isa PlayerRuleApp
            new_player = get(player_map, v.player, v.player)
            new_name   = get(name_map,   v.name,   v.name)
            PlayerRuleApp(new_name, F(v.rule), F(v.init), new_player;
                          cat           = v.cat === nothing ? nothing : v.cat,
                          fast_match_fn = v.fast_match_fn,
                          use_cache     = v.use_cache,
                          match_limit   = v.match_limit)
        elseif v isa GameSched
            player_migrate(F, v, player_map; name_map)
        else
            F(v)
        end
    end
    mk_game_sched(gs._trace_args, gs._init_args, gs._N, new_boxes, gs._body)
end
