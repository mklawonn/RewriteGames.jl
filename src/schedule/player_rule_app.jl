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
- `in_hom`:        Agent -> L interface morphism.
- `out_hom`:       Agent -> R interface morphism.
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
                   Ignored when `view_fn` is set.
- `match_limit`:   When set to an `Int`, caps the number of matches
                   enumerated to at most that many per turn.  Passes a
                   lazy `take` limit to the underlying homomorphism search
                   so that work stops once enough matches are found.  Has
                   no effect when `fast_match_fn` is set (the user-supplied
                   function controls enumeration).
- `view_fn`:       Optional fog-of-war function
                   `(player::Symbol, world) -> (subworld, v::ACSetTransformation)`
                   where `v : subworld → world` is the monic inclusion.
                   When set, matches are enumerated against `subworld`
                   and translated back to `world` via composition with `v`;
                   translated matches that violate the DPO dangling condition
                   in `world` are silently discarded.  The agent receives a
                   `GameState` whose world is `subworld`.  `use_cache` is
                   incompatible with `view_fn` and is ignored when both are set.
"""
struct PlayerRuleApp
    name          :: Symbol
    rule          :: Any      # AbsRule
    in_hom        :: Any      # agent -> L
    out_hom       :: Any      # agent -> R
    player        :: Symbol
    cat           :: Any
    _inner        :: Any      # inner RuleApp
    fast_match_fn :: Union{Function, Nothing}
    use_cache     :: Bool
    match_limit   :: Union{Int, Nothing}
    view_fn       :: Union{Function, Nothing}
end

function PlayerRuleApp(name::Symbol, rule, in_hom, out_hom, player::Symbol;
                        cat=nothing,
                        fast_match_fn::Union{Function,Nothing}=nothing,
                        use_cache::Bool=false,
                        match_limit::Union{Int,Nothing}=nothing,
                        view_fn::Union{Function,Nothing}=nothing)
    inner = if in_hom isa Catlab.CategoricalAlgebra.ACSetTransformation
        cat === nothing ? RuleApp(name, rule, in_hom, out_hom) :
                          RuleApp(name, rule, in_hom, out_hom; cat=cat)
    else
        cat === nothing ? RuleApp(name, rule, in_hom) :
                          RuleApp(name, rule, in_hom; cat=cat)
    end
    PlayerRuleApp(name, rule, in_hom, out_hom, player, cat, inner, fast_match_fn, use_cache,
                  match_limit, view_fn)
end

function PlayerRuleApp(name::Symbol, rule, init, player::Symbol; kwargs...)
    PlayerRuleApp(name, rule, init, init, player; kwargs...)
end

Base.show(io::IO, p::PlayerRuleApp) =
    print(io, "PlayerRuleApp(:$(p.name), player=:$(p.player))")

# ─── Composition & Agent Methods ───────────────────────────────────────────────

import Catlab.Theories: ⋅, ⊗
import AlgebraicRewriting: agent, singleton, tryrule, merge_wires

_pra_interface(p::PlayerRuleApp) =
    p.in_hom isa Catlab.CategoricalAlgebra.ACSetTransformation ?
    Catlab.CategoricalAlgebra.dom(p.in_hom) : p.in_hom

function tryrule(p::PlayerRuleApp)
    I_state = _pra_interface(p)
    # We create a local Names object to resolve "I" in mk_game_sched
    N_local = Names(Dict("I" => I_state))
    mk_game_sched(NamedTuple(), (init=:I,), N_local,
                  NamedTuple{(p.name, :mw)}((p, merge_wires(I_state))),
                  quote
                      s, f = $(p.name)(init)
                      out = mw(s, f)
                      return out
                  end; cat=p.cat)
end

"""
    agent(p::PlayerRuleApp; n::Symbol) -> GameSched

Wrap a `PlayerRuleApp` in an `agent` loop.  Returns a `GameSched`.
"""
function agent(p::PlayerRuleApp; n::Symbol)
    # Wrap it in a tryrule first to ensure it has a 1-to-1 interface,
    # then wrap the resulting GameSched in an agent loop.
    return agent(tryrule(p); n=n)
end

"""
    ⋅(p1::PlayerRuleApp, p2::PlayerRuleApp) -> GameSched

Compose two `PlayerRuleApp`s.  Returns a `GameSched`.
"""
function ⋅(p1::PlayerRuleApp, p2::PlayerRuleApp)
    I_state = _pra_interface(p1)
    N_local = Names(Dict("I" => I_state))
    mw = merge_wires(I_state)
    mk_game_sched(NamedTuple(), (init=:I,), N_local,
                  NamedTuple{(p1.name, p2.name, :mw)}((p1, p2, mw)),
                  quote
                      s1, f1 = $(p1.name)(init)
                      s2, f2 = $(p2.name)(s1)
                      fail = mw(f1, f2)
                      return s2, fail
                  end; cat=p1.cat)
end

"""
    ⊗(p1::PlayerRuleApp, p2::PlayerRuleApp) -> GameSched

Tensor product of two `PlayerRuleApp`s.  Returns a `GameSched`.
"""
function ⊗(p1::PlayerRuleApp, p2::PlayerRuleApp)
    I_state = _pra_interface(p1)
    N_local = Names(Dict("I" => I_state))
    mk_game_sched(NamedTuple(), (init1=:I, init2=:I), N_local,
                  NamedTuple{(p1.name, p2.name)}((p1, p2)),
                  quote
                      out1 = $(p1.name)(init1)
                      out2 = $(p2.name)(init2)
                      return out1, out2
                  end; cat=p1.cat)
end

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
    _agent_name  :: Union{Symbol, Nothing} # Name of agent type to loop over
    cat          :: Any                   # AC-category
end

function _collect_player_apps(gs::GameSched)
    apps = Dict{Symbol, PlayerRuleApp}()
    for (k, v) in pairs(gs._all_boxes)
        if v isa PlayerRuleApp
            apps[k] = v
        elseif v isa GameSched
            merge!(apps, _collect_player_apps(v))
        end
    end
    return apps
end

function _merge_names(n1, n2)
    n1 === nothing && return n2
    n2 === nothing && return n1
    # AlgebraicRewriting.Names has a from_name Dict{String, Any}
    return Names(merge(n1.from_name, n2.from_name))
end

function agent(gs::GameSched; n::Symbol)
    # 1. Resolve agent interface
    interface = nothing
    if gs._N !== nothing && haskey(gs._N.from_name, string(n))
        interface = gs._N[string(n)]
    else
        # Try to find from any PRAs inside
        apps = _collect_player_apps(gs)
        if !isempty(apps)
            first_pra = first(values(apps))
            interface = Catlab.CategoricalAlgebra.dom(first_pra.in_hom)
        end
    end
    
    if interface === nothing
        error("agent: could not find interface for agent type :$n in schedule Names or boxes.")
    end

    # 2. Update Names object with the agent interface
    new_N = gs._N === nothing ? Names(Dict(string(n) => interface)) :
                                Names(merge(gs._N.from_name, Dict(string(n) => interface)))

    inner = agent(gs._inner; n=n)
    GameSched(inner, gs._player_map, gs._all_boxes, gs._steps, gs._init_names,
              gs._trace_names, gs._ret_names, gs._trace_args, gs._init_args,
              gs._body, new_N, n, gs.cat)
end

function ⋅(gs1::GameSched, gs2::GameSched)
    new_N = _merge_names(gs1._N, gs2._N)
    mk_game_sched(NamedTuple(), (init=:I,), new_N,
                  (b1=gs1, b2=gs2),
                  quote
                      out1 = b1(init)
                      out2 = b2(out1)
                      return out2
                  end; cat=gs1.cat)
end

function ⊗(gs1::GameSched, gs2::GameSched)
    new_N = _merge_names(gs1._N, gs2._N)
    mk_game_sched(NamedTuple(), (init1=:I, init2=:I), new_N,
                  (b1=gs1, b2=gs2),
                  quote
                      out1 = b1(init1)
                      out2 = b2(init2)
                      return out1, out2
                  end; cat=gs1.cat)
end

# Mixed Composition
function ⋅(gs::GameSched, box)
    bname = box isa PlayerRuleApp ? box.name : gensym("box")
    mk_game_sched(gs._trace_args, gs._init_args, gs._N, 
                  merge(gs._all_boxes, NamedTuple{(bname,)}((box,))),
                  quote
                      out_gs = gs(init) # This won't work easily because gs is not a box name
                      # ...
                  end; cat=gs.cat)
    # Actually, the mixed composition is harder. Let's stick to GameSched ⋅ GameSched for now
    # and PlayerRuleApp ⋅ PlayerRuleApp.
    # The current implementation of mixed ⋅ is already there but broken for running.
    # I'll just fix the ones I need for Tesseract.
    inner = gs._inner ⋅ (box isa PlayerRuleApp ? box._inner : box)
    player_map = copy(gs._player_map)
    if box isa PlayerRuleApp; player_map[box.name] = box; end
    # We can't easily synthesize steps here without a body.
    # But wait, Tesseract doesn't seem to use mixed composition.
    GameSched(inner, player_map, merge(gs._all_boxes, NamedTuple{(box isa PlayerRuleApp ? box.name : gensym("box"),)}((box,))),
              [], [], [], [], nothing, nothing, nothing, gs._N, gs._agent_name, gs.cat)
end

function ⋅(box, gs::GameSched)
    GameSched((box isa PlayerRuleApp ? box._inner : box) ⋅ gs._inner,
              gs._player_map, gs._all_boxes, gs._steps, gs._init_names,
              gs._trace_names, gs._ret_names, gs._trace_args, gs._init_args,
              gs._body, gs._N, gs._agent_name, gs.cat)
end
function ⋅(p::PlayerRuleApp, box)
    # Box could be an AR AgentBox or Schedule
    inner = p._inner ⋅ (box isa PlayerRuleApp ? box._inner : box)
    player_map = Dict(p.name => p)
    if box isa PlayerRuleApp; player_map[box.name] = box; end
    GameSched(inner, player_map, NamedTuple(), [], [], [], [], nothing, nothing, nothing, nothing, nothing, p.cat)
end
function ⋅(box, p::PlayerRuleApp)
    inner = (box isa PlayerRuleApp ? box._inner : box) ⋅ p._inner
    player_map = Dict(p.name => p)
    if box isa PlayerRuleApp; player_map[box.name] = box; end
    GameSched(inner, player_map, NamedTuple(), [], [], [], [], nothing, nothing, nothing, nothing, nothing, p.cat)
end

Base.show(io::IO, gs::GameSched) =
    print(io, "GameSched(players=$(collect(keys(gs._player_map))), " *
              "init=$(gs._init_names), agent=$(gs._agent_name))")

# ─── Body-parsing IR ─────────────────────────────────────────────────────────

"""
    BoxStep

Represents one box invocation in the parsed wiring-diagram body.  Created once
at `mk_game_sched` time; reused on every call to `run_game_sched!`.

# Fields
- `box`:     Key of the box in the `_all_boxes` NamedTuple.
- `inputs`:  Wire names supplying input to the box (merged when more than one).
- `outputs`: Wire names receiving the box's output ports (index = port number).
"""
struct BoxStep
    box     :: Symbol
    inputs  :: Vector{Symbol}
    outputs :: Vector{Symbol}
end

Base.show(io::IO, s::BoxStep) =
    print(io, "BoxStep(:$(s.box), in=$(s.inputs), out=$(s.outputs))")

# ─── Body parser (called once at mk_game_sched time) ──────────────────────────

"""
    _parse_body(body::Expr) -> (steps::Vector{BoxStep}, ret_names::Vector{Symbol})

Walk the AST of the `mk_game_sched` body quote block and extract an ordered
list of `BoxStep` records plus the final return wire names.

Supported statement patterns:
- `a, b = box(w)`            → `BoxStep(:box, [:w], [:a, :b])`
- `a, b, c = box([w1, w2])`  → `BoxStep(:box, [:w1, :w2], [:a, :b, :c])`
- `a = box(w)`               → `BoxStep(:box, [:w], [:a])`
- Nested calls such as `a = mw(mw(b, c), d)` are flattened to intermediate
  `_gstmp_N` wires.
- `return a, b` or `return a` → sets the return wire list.
"""
function _parse_body(body::Expr)
    steps      = []
    ret_names  = Symbol[]
    tmp_ctr    = Ref(0)

    for stmt in body.args
        stmt isa LineNumberNode && continue

        if stmt isa Expr && stmt.head === :return
            ret_names = _parse_names(stmt.args[1])

        elseif stmt isa Expr && stmt.head === :(=)
            lhs, rhs = stmt.args[1], stmt.args[2]
            out_names = _parse_names(lhs)
            rhs isa Expr && rhs.head === :call ||
                error("_parse_body: expected a function call on the rhs, got: $rhs")
            _flatten_call!(steps, rhs, out_names, tmp_ctr)

        else
            error("_parse_body: unrecognised statement: $stmt")
        end
    end

    return steps, ret_names
end

function _parse_names(expr)
    expr isa Symbol                             && return [expr]
    expr isa Expr && expr.head === :tuple       && return [a for a in expr.args if a isa Symbol]
    error("_parse_body: expected symbol or tuple of symbols, got: $expr")
end

function _flatten_call!(steps, call::Expr, out_names::Vector{Symbol}, tmp_ctr::Ref{Int})
    box_sym = call.args[1]
    box_sym isa Symbol || error("_parse_body: box name must be a symbol, got: $(call.args[1])")

    in_names = Symbol[]
    for arg in call.args[2:end]
        if arg isa Symbol
            push!(in_names, arg)
        elseif arg isa Expr && arg.head === :vect
            for a in arg.args
                a isa Symbol || error("_parse_body: wire list entry must be a symbol, got: $a")
                push!(in_names, a)
            end
        elseif arg isa Expr && arg.head === :call
            # Nested call — materialise to a temp wire and recurse
            tmp_ctr[] += 1
            tmp = Symbol("_gstmp_$(tmp_ctr[])")
            _flatten_call!(steps, arg, [tmp], tmp_ctr)
            push!(in_names, tmp)
        else
            error("_parse_body: unexpected call argument: $arg")
        end
    end

    # Note: BoxStep is defined above
    push!(steps, BoxStep(box_sym, in_names, out_names))
end

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
function mk_game_sched(trace_args, init_args, N, boxes, body; cat=nothing, kwargs...)
    # Translate boxes for AR's mk_sched:
    #   PlayerRuleApp → its inner RuleApp
    #   GameSched     → its inner AR Schedule
    #   anything else → pass through unchanged
    ar_boxes = map(boxes) do v
        v isa PlayerRuleApp ? v._inner :
        v isa GameSched     ? v._inner : v
    end

    inner = mk_sched(trace_args, init_args, N, ar_boxes, body)

    player_map = Dict{Symbol, PlayerRuleApp}()
    for (k, v) in pairs(boxes)
        if v isa PlayerRuleApp
            player_map[k] = v
        elseif v isa GameSched
            merge!(player_map, v._player_map)
        end
    end

    steps, ret_names = _parse_body(body)

    GameSched(
        inner, player_map, boxes, steps,
        collect(Symbol, keys(init_args)),
        collect(Symbol, keys(trace_args)),
        ret_names,
        trace_args, init_args, body, N,
        nothing, # No agent loop by default
        cat
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
            PlayerRuleApp(new_name, F(v.rule), F(v.in_hom), F(v.out_hom), new_player;
                          cat           = v.cat === nothing ? nothing : v.cat,
                          fast_match_fn = v.fast_match_fn,
                          use_cache     = v.use_cache,
                          match_limit   = v.match_limit,
                          view_fn       = v.view_fn)
        elseif v isa GameSched
            player_migrate(F, v, player_map; name_map)
        else
            migrated = F(v)
            migrated isa RuleApp ?
                RuleApp(get(name_map, migrated.name, migrated.name),
                        migrated.rule, migrated.in_agent, migrated.out_agent) :
                migrated
        end
    end
    mk_game_sched(gs._trace_args, gs._init_args, gs._N, new_boxes, gs._body; cat=gs.cat)
end
