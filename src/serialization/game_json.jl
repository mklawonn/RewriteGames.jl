using JSON3
using Catlab
using Catlab.CategoricalAlgebra
using Catlab.WiringDiagrams
using AlgebraicRewriting

# --- ACSet Serialization ---

function acset_to_dict(X::StructACSet)
    d = Dict{String, Any}()
    S = acset_schema(X)
    for ob in S.obs
        n = nparts(X, ob)
        n == 0 && continue
        row_d = Dict{String, Any}("_n" => n)
        for (h, dom_h, codom_h) in S.homs
            if dom_h == ob
                row_d[string(h)] = collect(subpart(X, h))
            end
        end
        for (a, dom_a, codom_a) in S.attrs
            if dom_a == ob
                row_d[string(a)] = [string(v) for v in subpart(X, a)]
            end
        end
        d[string(ob)] = row_d
    end
    return d
end

function dict_to_acset(T::Type{<:StructACSet}, d::AbstractDict)
    X = T()
    S = acset_schema(X)
    
    # Sort objects by dependencies
    obs = S.obs
    deps = Dict(o => Set{Symbol}() for o in obs)
    for (h, dom_h, codom_h) in S.homs
        push!(deps[dom_h], codom_h)
    end
    
    ordered_obs = Symbol[]
    remaining = Set(obs)
    while !isempty(remaining)
        progress = false
        for o in remaining
            if all(d -> d ∉ remaining, deps[o])
                push!(ordered_obs, o)
                delete!(remaining, o)
                progress = true
                break
            end
        end
        if !progress
            for o in remaining; push!(ordered_obs, o); end
            break
        end
    end

    for ob in ordered_obs
        ob_str = string(ob)
        haskey(d, ob_str) || continue
        tbl = d[ob_str]
        n_val = get(tbl, "_n", get(tbl, :_n, nothing))
        n = n_val !== nothing ? Int(n_val) : 0
        for (col, vals) in tbl
            String(col) == "_n" && continue
            n = max(n, length(vals))
        end
        add_parts!(X, ob, n)
        for (col_str, vals) in tbl
            String(col_str) == "_n" && continue
            col = Symbol(col_str)
            if !isempty(vals) && (vals[1] isa Integer || (vals[1] isa String && tryparse(Int, vals[1]) !== nothing))
                set_subpart!(X, :, col, Int.(vals))
            else
                set_subpart!(X, :, col, Symbol.(string.(vals)))
            end
        end
    end
    return X
end

# --- Transformation Serialization ---

function transformation_to_dict(phi::ACSetTransformation)
    return Dict(
        "dom" => acset_to_dict(dom(phi)),
        "codom" => acset_to_dict(codom(phi)),
        "components" => Dict(string(ob) => collect(phi.components[ob]) for ob in acset_schema(dom(phi)).obs)
    )
end

function dict_to_transformation(d::AbstractDict, T_world::Type{<:StructACSet})
    dom_X = dict_to_acset(T_world, d["dom"])
    codom_X = dict_to_acset(T_world, d["codom"])
    comps = Pair{Symbol, Any}[Symbol(k) => Int.(v) for (k, v) in d["components"]]
    return ACSetTransformation(dom_X, codom_X; comps...)
end

# --- Rule Serialization ---

function rule_to_dict(r::Rule)
    return Dict(
        "l" => transformation_to_dict(r.L),
        "r" => transformation_to_dict(r.R)
    )
end

function dict_to_rule(d::AbstractDict, T_world::Type{<:StructACSet})
    l = dict_to_transformation(d["l"], T_world)
    r = dict_to_transformation(d["r"], T_world)
    return Rule(l, r)
end

# --- Box Serialization ---

function box_to_dict(b)
    if b isa PlayerRuleApp
        return Dict(
            "type" => "PlayerRuleApp",
            "name" => string(b.name),
            "player" => string(b.player),
            "rule" => rule_to_dict(b.rule),
            "use_cache" => b.use_cache,
            "match_limit" => b.match_limit
        )
    elseif b isa RuleApp
        return Dict(
            "type" => "RuleApp",
            "name" => string(b.name),
            "rule" => rule_to_dict(b.rule)
        )
    elseif b isa GameSched
        return Dict("type" => "GameSched")
    elseif b isa Schedule && hasproperty(b, :x) && occursin("mmerge", string(b.x))
        return Dict("type" => "MergeWires")
    elseif b isa Schedule
        return Dict("type" => "Schedule")
    else
        return Dict("type" => string(typeof(b)))
    end
end

function dict_to_box(d::AbstractDict, T_world::Type{<:StructACSet}, I_state::ACSet)
    type = d["type"]
    if type == "PlayerRuleApp"
        rule = dict_to_rule(d["rule"], T_world)
        return PlayerRuleApp(Symbol(d["name"]), rule, I_state, Symbol(d["player"]);
                             use_cache=get(d, "use_cache", false),
                             match_limit=get(d, "match_limit", nothing))
    elseif type == "RuleApp"
        rule = dict_to_rule(d["rule"], T_world)
        return RuleApp(Symbol(d["name"]), rule, I_state)
    elseif type == "MergeWires"
        return merge_wires(I_state)
    else
        error("Unknown box type: $type")
    end
end

# --- Game & Schedule Serialization ---

function write_game(path::String, game::Game, gs::GameSched)
    init_acset = game.initial()
    
    data = Dict(
        "players" => string.(game.players),
        "initial_state" => acset_to_dict(init_acset),
        "win_conditions" => game.win_conditions === nothing ? nothing : 
                            Dict(string(k) => string(v) for (k, v) in game.win_conditions),
        "boxes" => Dict(string(k) => box_to_dict(v) for (k, v) in pairs(gs._all_boxes)),
        "trace_args" => Dict(string(k) => string(v) for (k, v) in pairs(gs._trace_args)),
        "init_args" => Dict(string(k) => string(v) for (k, v) in pairs(gs._init_args)),
        "body_expr" => string(gs._body)
    )
    
    open(path, "w") do f
        JSON3.write(f, data)
    end
end

function read_game(path::String, T_world::Type{<:StructACSet}, N_obj::Names)
    data = JSON3.read(read(path, String))
    
    players = Symbol.(data["players"])
    init_acset = dict_to_acset(T_world, data["initial_state"])
    win_conditions = data["win_conditions"] === nothing ? nothing :
                     Dict{Symbol, Any}(Symbol(k) => (string(v) == "nothing" ? nothing : Symbol(v))
                                       for (k, v) in data["win_conditions"])
    
    I_state = N_obj["I"]
    
    boxes_dict = Dict{Symbol, Any}()
    for (k, v) in data["boxes"]
        boxes_dict[Symbol(k)] = dict_to_box(v, T_world, I_state)
    end
    
    body = Meta.parse(data["body_expr"])
    
    trace_args = NamedTuple(Symbol(k) => Symbol(v) for (k, v) in pairs(data["trace_args"]))
    init_args = NamedTuple(Symbol(k) => Symbol(v) for (k, v) in pairs(data["init_args"]))
    
    gs = mk_game_sched(trace_args, init_args, N_obj, NamedTuple(boxes_dict), body)
    
    game = Game(T_world; players=players, initial=() -> init_acset, win_conditions=win_conditions)
    
    return game, gs
end
