"""
    PlayerRuleApp

Wraps an AlgebraicRewriting `RuleApp` and tags it with a player identity.
"""
struct PlayerRuleApp
    name          :: Symbol
    rule          :: Any
    in_hom        :: Any
    out_hom       :: Any
    player        :: Symbol
    cat           :: Any
    _inner        :: Any
    fast_match_fn :: Union{Function, Nothing}
    use_cache     :: Bool
    match_limit   :: Union{Int, Nothing}
    view_fn       :: Union{Function, Nothing}
end

function PlayerRuleApp(name::Symbol, rule, in_hom, out_hom, player::Symbol;
                        cat=nothing, fast_match_fn=nothing, use_cache=false,
                        match_limit=nothing, view_fn=nothing)
    inner = if in_hom isa Catlab.CategoricalAlgebra.ACSetTransformation
        cat === nothing ? RuleApp(name, rule, in_hom) :
                          RuleApp(name, rule, in_hom; cat=cat)
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

import Catlab.Theories: ⋅, ⊗
import AlgebraicRewriting: agent, singleton, tryrule, merge_wires

_pra_interface(p::PlayerRuleApp) =
    p.in_hom isa Catlab.CategoricalAlgebra.ACSetTransformation ?
    Catlab.CategoricalAlgebra.dom(p.in_hom) : p.in_hom

function tryrule(p::PlayerRuleApp)
    I_state = _pra_interface(p)
    N_local = Names(Dict("I" => I_state))
    mk_game_sched(NamedTuple(), (init=:I,), N_local,
                  NamedTuple{(p.name, :mw)}((p, merge_wires(I_state))),
                  quote
                      s, f = $(p.name)(init)
                      out = mw(s, f)
                      return out
                  end; cat=p.cat)
end

function agent(p::PlayerRuleApp; n::Symbol)
    return agent(tryrule(p); n=n)
end

function ⊗(p1::PlayerRuleApp, p2::PlayerRuleApp)
    I_state = _pra_interface(p1)
    N_local = Names(Dict("I" => I_state))
    mk_game_sched(NamedTuple(), (init1=:I, init2=:I), N_local,
                  NamedTuple{(p1.name, p2.name)}((p1, p2)),
                  quote
                      s1, f1 = $(p1.name)(init1)
                      s2, f2 = $(p2.name)(init2)
                      return s1, s2, f1, f2
                  end; cat=p1.cat)
end

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

struct GameSched
    _inner       :: Any
    _player_map  :: Dict{Symbol, PlayerRuleApp}
    _all_boxes   :: Any
    _steps       :: Vector{Any}
    _init_names  :: Vector{Symbol}
    _trace_names :: Vector{Symbol}
    _ret_names   :: Vector{Symbol}
    _trace_args  :: Any
    _init_args   :: Any
    _body        :: Any
    _N           :: Any
    _agent_name  :: Union{Symbol, Nothing}
    cat          :: Any
end

function _collect_player_apps(gs::GameSched)
    apps = Dict{Symbol, PlayerRuleApp}()
    for (k, v) in pairs(gs._all_boxes)
        if v isa PlayerRuleApp; apps[k] = v;
        elseif v isa GameSched; merge!(apps, _collect_player_apps(v)); end
    end
    return apps
end

function agent(gs::GameSched; n::Symbol)
    gs._agent_name !== nothing && return gs
    interface = nothing
    if gs._N !== nothing && haskey(gs._N.from_name, string(n))
        interface = gs._N[string(n)]
    else
        apps = _collect_player_apps(gs)
        if !isempty(apps)
            interface = _pra_interface(first(values(apps)))
        end
    end
    if interface === nothing
        error("agent: could not find interface for agent type :$n in schedule Names or boxes.")
    end
    new_N = gs._N === nothing ? Names(Dict(string(n) => interface)) :
                                Names(merge(gs._N.from_name, Dict(string(n) => interface)))
    
    # Build AR agent loop
    inner = agent(gs._inner; n=n)

    # Return a GameSched that has the loop but keeps the metadata.
    # Note: we use gs._steps here so that compile_schedule sees the inner steps.
    # THIS IS THE PROBLEM FOR GPU.
    GameSched(inner, gs._player_map, gs._all_boxes, gs._steps, gs._init_names,
              gs._trace_names, gs._ret_names, gs._trace_args, gs._init_args,
              gs._body, new_N, n, gs.cat)
end
function ⋅(gs1::GameSched, gs2::GameSched)
    # Chain outputs of gs1 to inputs of gs2
    N_local = gs1._N
    # Use inner schedules directly for mk_sched
    mk_game_sched(gs1._trace_args, gs1._init_args, N_local,
                  NamedTuple{(:gs1, :gs2)}((gs1, gs2)),
                  quote
                      out1 = gs1($(gs1._init_names...))
                      out2 = gs2(out1)
                      return out2
                  end; cat=gs1.cat)
end


function mk_game_sched(trace_args, init_args, N, boxes, body; cat=nothing, kwargs...)
    # Infer interface from Names if possible
    _I = first(values(N.from_name))
    ar_boxes = map(boxes) do v
        if v isa PlayerRuleApp; v._inner
        elseif v isa GameSched; v._inner
        elseif v isa AlgebraicRewriting.Schedules.Queries.Query; v
        elseif v isa AlgebraicRewriting.Schedules.RuleApps.RuleApp; v
        elseif v isa AlgebraicRewriting.Schedules.Wiring.Schedule; v
        elseif hasproperty(v, :rule); RuleApp(:_tmp, v, _I; cat=cat)
        else; AlgebraicRewriting.Schedules.Queries.Query(:_dummy, _I)
        end
    end
    inner = mk_sched(trace_args, init_args, N, ar_boxes, body)
    player_map = Dict{Symbol, PlayerRuleApp}()
    for (k, v) in pairs(boxes)
        if v isa PlayerRuleApp; player_map[k] = v;
        elseif v isa GameSched; merge!(player_map, v._player_map); end
    end
    steps, ret_names = RewriteGames._parse_body(body)
    GameSched(inner, player_map, boxes, steps, collect(Symbol, keys(init_args)),
              collect(Symbol, keys(trace_args)), ret_names, trace_args, init_args,
              body, N, nothing, cat)
end

function _parse_body(body::Expr)
    steps = []; ret_names = Symbol[]; tmp_ctr = Ref(0)
    for stmt in body.args
        stmt isa LineNumberNode && continue
        if stmt isa Expr && stmt.head === :return; ret_names = [a for a in (stmt.args[1] isa Expr ? stmt.args[1].args : [stmt.args[1]]) if a isa Symbol]
        elseif stmt isa Expr && stmt.head === :(=); lhs, rhs = stmt.args[1], stmt.args[2]; out_names = lhs isa Symbol ? [lhs] : [a for a in lhs.args if a isa Symbol]
            _flatten_call!(steps, rhs, out_names, tmp_ctr)
        end
    end
    return steps, ret_names
end

function _flatten_call!(steps, call::Expr, out_names, tmp_ctr)
    box_sym = call.args[1]; in_names = Symbol[]
    for arg in call.args[2:end]
        if arg isa Symbol; push!(in_names, arg)
        elseif arg isa Expr && arg.head === :vect; append!(in_names, [a for a in arg.args if a isa Symbol])
        elseif arg isa Expr && arg.head === :call
            tmp_ctr[] += 1; tmp = Symbol("_gstmp_$(tmp_ctr[])")
            _flatten_call!(steps, arg, [tmp], tmp_ctr)
            push!(in_names, tmp)
        end
    end
    push!(steps, BoxStep(box_sym, in_names, out_names))
end

struct BoxStep; box::Symbol; inputs::Vector{Symbol}; outputs::Vector{Symbol}; end

import AlgebraicRewriting: view_sched
view_sched(gs::GameSched; kw...) = view_sched(gs._inner; kw...)

function player_migrate(F, gs::GameSched, player_map; name_map=Dict())
    new_boxes = map(gs._all_boxes) do v
        if v isa PlayerRuleApp; PlayerRuleApp(get(name_map, v.name, v.name), F(v.rule), F(v.in_hom), F(v.out_hom), get(player_map, v.player, v.player); cat=v.cat, use_cache=v.use_cache)
        elseif v isa GameSched; player_migrate(F, v, player_map; name_map)
        else F(v) end
    end
    mk_game_sched(gs._trace_args, gs._init_args, gs._N, new_boxes, gs._body; cat=gs.cat)
end

function agent(box::AlgebraicRewriting.Schedules.RuleApps.RuleApp; n::Symbol)
    in_hom = box.in_agent isa Catlab.CategoricalAlgebra.ACSetTransformation ? box.in_agent : homomorphism(box.in_agent, codom(Catlab.CategoricalAlgebra.left(box.rule)))
    I_state = dom(in_hom)
    N_local = Names(Dict("I" => I_state, string(n) => I_state))
    gs = mk_game_sched(NamedTuple(), (init=:I,), N_local, (box=box,), quote out = box(init); return out end; cat=nothing)
    return agent(gs; n=n)
end