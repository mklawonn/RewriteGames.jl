import Catlab.CategoricalAlgebra
import AlgebraicRewriting: get_matches, can_match, can_pushout_complement

function _compose_hom(f::Catlab.CategoricalAlgebra.ACSetTransformation,
                      g::Catlab.CategoricalAlgebra.ACSetTransformation)
    S = Catlab.CategoricalAlgebra.acset_schema(Catlab.CategoricalAlgebra.dom(f))
    obs = Catlab.CategoricalAlgebra.ob(S)
    comps = Pair{Symbol, Vector{Int}}[
        o => [g.components[o](f.components[o](i))
              for i in 1:Catlab.CategoricalAlgebra.nparts(Catlab.CategoricalAlgebra.dom(f), o)]
        for o in obs
    ]
    Catlab.CategoricalAlgebra.ACSetTransformation(
        Catlab.CategoricalAlgebra.dom(f),
        Catlab.CategoricalAlgebra.codom(g);
        comps...)
end

# ─── Game execution engine ────────────────────────────────────────────────────

"""
    run_game_sched!(gs::GameSched, initial_world, agents; T_max=1000, ...)
        -> Vector{Experience}

Run a `GameSched` starting from `initial_world` using the provided `agents` dict.
Continues for `T_max` iterations of the top-level schedule or until `terminal(world)`
returns `true`.

Returns a vector of `Experience` records from all `PlayerRuleApp` boxes encountered.
"""
function run_game_sched!(gs::GameSched, initial_world::ACSet, agents::Dict;
                         T_max::Int = 1000,
                         terminal::Function = (W) -> (false, nothing),
                         winner_wires::Dict{Symbol, Union{Symbol, Nothing}} = Dict{Symbol, Union{Symbol, Nothing}}())
    all_exps   = Experience[]
    turn       = Ref(1)
    n_trace    = length(gs._trace_names)
    _terminal  = terminal === nothing ? (W) -> (false, nothing) : terminal

    # Build incremental match caches for all PlayerRuleApp boxes that request one.
    cache_dict = Dict{Symbol, MatchCache}()
    all_pra    = _collect_player_apps(gs)
    for (name, pra) in all_pra
        if pra.use_cache && pra.fast_match_fn === nothing && pra.view_fn === nothing
            cache_dict[name] = MatchCache(pra.rule, pra.cat, initial_world;
                                          match_limit = pra.match_limit)
        end
    end

    # Initialise wires for the first iteration
    wires = _init_wires(gs, initial_world, nothing)

    fired_exit_name = nothing  # name of the exit wire that ended the game

    for _ in 1:(T_max + 1)
        round_exps = Experience[]
        _run_body!(gs._steps, wires, gs._all_boxes, agents, _terminal, turn, T_max, round_exps, cache_dict)
        append!(all_exps, round_exps)

        # Extract trace and exit return values
        trace_worlds = [get(wires, gs._ret_names[i], nothing)
                        for i in 1:min(n_trace, length(gs._ret_names))]
        exit_world   = nothing
        for i in (n_trace + 1):length(gs._ret_names)
            w = get(wires, gs._ret_names[i], nothing)
            if w !== nothing
                exit_world = w
                fired_exit_name = gs._ret_names[i]
                break
            end
        end

        # Stop if any exit wire is active or turn limit reached
        done_flag = exit_world !== nothing ||
                    (!isempty(all_exps) && all_exps[end].done) ||
                    turn[] > T_max
        done_flag && break

        # Feed trace wire back for next iteration
        active_trace = findfirst(!isnothing, trace_worlds)
        active_trace === nothing && break   # no continuing wire — episode over
        trace_world = trace_worlds[active_trace]

        wires = _init_wires(gs, nothing, trace_world)
    end

    # Post-process: if winner_wires is provided and an exit wire fired, update
    # the final Experience with the correct winner and done = true.
    if !isempty(winner_wires) && fired_exit_name !== nothing && !isempty(all_exps)
        wire_winner = get(winner_wires, fired_exit_name, nothing)
        last_exp = all_exps[end]
        if terminal === nothing || !last_exp.done
            all_exps[end] = Experience(
                last_exp.player, last_exp.state, last_exp.legal_actions,
                last_exp.action, last_exp.next_state,
                true, wire_winner,
                last_exp.info, last_exp.schedule_path, last_exp.view,
            )
        end
    end

    return all_exps
end

function run_game_sched!(gs::GameSched, game::Game, agents::Dict; T_max::Int = 1000)
    if game.win_conditions !== nothing
        run_game_sched!(gs, game.initial(), agents;
                        T_max         = T_max,
                        winner_wires  = Dict{Symbol, Union{Symbol, Nothing}}(
                            k => v for (k, v) in game.win_conditions))
    else
        run_game_sched!(gs, game.initial(), agents;
                        T_max    = T_max,
                        terminal = game.terminal)
    end
end

# ── helpers ───────────────────────────────────────────────────────────────────

function _init_wires(gs::GameSched, initial_world, trace_world)
    wires = Dict{Symbol, Any}()
    for name in gs._init_names
        wires[name] = initial_world
    end
    for name in gs._trace_names
        wires[name] = trace_world
    end
    return wires
end

# _collect_player_apps moved to player_rule_app.jl

# ─── _run_body! ───────────────────────────────────────────────────────────────

function _run_body!(steps, wires, boxes, agents, terminal, turn::Ref{Int}, T_max, exps,
                    cache_dict::Dict{Symbol, MatchCache}, agent_match=nothing)
    for step in steps
        box = boxes[step.box]

        input_world = nothing
        for wname in step.inputs
            w = get(wires, wname, nothing)
            w !== nothing && (input_world = w; break)
        end

        if input_world === nothing
            for out in step.outputs; wires[out] = nothing; end
            continue
        end

        if box isa PlayerRuleApp
            _exec_player!(step, box, input_world, wires, agents, terminal, turn, T_max, exps, cache_dict, agent_match)
        elseif box isa GameSched
            _exec_subsched!(step, box, input_world, wires, boxes, agents, terminal, turn, T_max, exps, cache_dict, agent_match)
        elseif hasproperty(box, :rule)
            _exec_native_rule!(step, box, input_world, wires, agent_match)
        elseif box isa MergeWires
            wires[step.outputs[1]] = box([get(wires, w, nothing) for w in step.inputs]...)
        elseif box isa Schedule
            # AlgebraicRewriting.Schedule acting as merge_wires: take first active input
            result = nothing
            for wname in step.inputs
                w = get(wires, wname, nothing)
                if w !== nothing; result = w; break; end
            end
            length(step.outputs) >= 1 && (wires[step.outputs[1]] = result)
        elseif box isa Coin
            out = box(input_world)
            for i in 1:length(step.outputs); wires[step.outputs[i]] = i <= length(out) ? out[i] : nothing; end
        else
        end
    end
end

# ─── PlayerRuleApp execution ──────────────────────────────────────────────────

function _exec_player!(step, box::PlayerRuleApp, world, wires, agents, terminal,
                        turn::Ref{Int}, T_max, exps,
                        cache_dict::Dict{Symbol, MatchCache}, agent_match=nothing)
    _cat = isnothing(box.cat) ? infer_acset_cat(world) : box.cat

    initial_map = Dict{Symbol, Any}()
    if agent_match !== nothing
        S = acset_schema(world)
        # only bind combinatorial parts, let Catlab find variables from them
        for o in Catlab.CategoricalAlgebra.ob(S)
            d = Dict{Int, Int}()
            for i in parts(dom(box.in_hom), o)
                d[box.in_hom[o](i)] = agent_match[o](i)
            end
            if !isempty(d); initial_map[o] = d; end
        end
    end

    if box.view_fn !== nothing
        subworld, v = box.view_fn(box.player, world)
        raw_sub = box.fast_match_fn !== nothing ?
            box.fast_match_fn(box.rule, subworld, _cat) :
            collect(get_matches(box.rule, subworld; cat=box.cat))
        raw = [_compose_hom(m, v) for m in raw_sub]
    elseif box.fast_match_fn !== nothing
        raw = box.fast_match_fn(box.rule, world, _cat)
    elseif haskey(cache_dict, box.name) && agent_match === nothing
        ms  = cache_dict[box.name].matches
        raw = box.match_limit === nothing ? ms : @view ms[1:min(end, box.match_limit)]
    else
        gen = isempty(initial_map) ?
              get_matches(box.rule, world; cat=box.cat) :
              Catlab.CategoricalAlgebra.homomorphisms(Catlab.CategoricalAlgebra.codom(Catlab.CategoricalAlgebra.left(box.rule)), world; cat=_cat, initial=initial_map, monic=box.rule.monic)
        raw = box.match_limit === nothing ? collect(gen) : collect(Iterators.take(gen, box.match_limit))
    end

    actions = [Action(box, m) for m in raw]
    state_pre  = GameState(world, turn[])
    chosen     = isempty(actions) ? nothing : select_action(agents[box.player], state_pre, actions)

    if chosen !== nothing
        maps      = AlgebraicRewriting.rewrite_match_maps(box.rule, chosen.match; cat=_cat)
        new_world = Catlab.CategoricalAlgebra.codom(maps[:rh])
        if agent_match === nothing
            for cache in values(cache_dict); update_cache!(cache, maps); end
        end
        done, winner = terminal(new_world)
        turn[] += 1
        state_post = GameState(new_world, turn[])
        push!(exps, Experience(box.player, state_pre, actions, chosen,
                               state_post, done || turn[] > T_max, winner,
                               Dict{Symbol, Any}(), Symbol[], nothing))
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = new_world)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = nothing)
    else
        done, winner = terminal(world)
        turn[] += 1
        state_post = GameState(world, turn[])
        push!(exps, Experience(box.player, state_pre, actions, nothing,
                               state_post, done || turn[] > T_max, winner,
                               Dict{Symbol, Any}(), Symbol[], nothing))
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = nothing)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = world)
    end
end

# ─── Nested GameSched execution ───────────────────────────────────────────────

function _exec_subsched!(step, sub_gs::GameSched, world, wires, _boxes, agents,
                          terminal, turn::Ref{Int}, T_max, exps,
                          cache_dict::Dict{Symbol, MatchCache}, agent_match=nothing)
    if sub_gs._agent_name !== nothing
        _cat = isnothing(sub_gs.cat) ? infer_acset_cat(world) : sub_gs.cat
        agent_interface = sub_gs._N[string(sub_gs._agent_name)]
        agent_matches = collect(Catlab.CategoricalAlgebra.homomorphisms(agent_interface, world; cat=_cat))
        
        current_world = world
        for am in agent_matches
            sub_wires = Dict{Symbol, Any}()
            for (i, name) in enumerate(sub_gs._init_names); sub_wires[name] = i == 1 ? current_world : nothing; end
            for name in sub_gs._trace_names; sub_wires[name] = nothing; end
            _run_body!(sub_gs._steps, sub_wires, sub_gs._all_boxes, agents, terminal, turn, T_max, exps, cache_dict, am)
            if !isempty(sub_gs._ret_names); current_world = get(sub_wires, sub_gs._ret_names[1], current_world); end
            if !isempty(exps) && exps[end].done; break; end
        end
        wires[step.outputs[1]] = current_world
        for i in 2:length(step.outputs); wires[step.outputs[i]] = nothing; end
    else
        sub_wires = Dict{Symbol, Any}()
        for (i, name) in enumerate(sub_gs._init_names); sub_wires[name] = i == 1 ? world : nothing; end
        for name in sub_gs._trace_names; sub_wires[name] = nothing; end
        _run_body!(sub_gs._steps, sub_wires, sub_gs._all_boxes, agents, terminal, turn, T_max, exps, cache_dict, agent_match)
        for (i, out_name) in enumerate(step.outputs)
            wires[out_name] = i <= length(sub_gs._ret_names) ? get(sub_wires, sub_gs._ret_names[i], nothing) : nothing
        end
    end
end

# ─── Native RuleApp execution ─────────────────────────────────────────────────

function _exec_native_rule!(step, box, world, wires, agent_match=nothing)
    _cat    = hasproperty(box, :cat) ? box.cat : infer_acset_cat(Catlab.CategoricalAlgebra.codom(Catlab.CategoricalAlgebra.left(box.rule)))
    initial_map = Dict{Symbol, Any}()
    if agent_match !== nothing && hasproperty(box, :in_agent) && box.in_agent !== nothing
        S = acset_schema(world)
        for o in Catlab.CategoricalAlgebra.ob(S)
            d = Dict{Int, Int}()
            for i in parts(dom(box.in_agent), o); d[box.in_agent[o](i)] = agent_match[o](i); end
            if !isempty(d); initial_map[o] = d; end
        end
    end
    gen = isempty(initial_map) ? get_matches(box.rule, world; cat=_cat) :
          Catlab.CategoricalAlgebra.homomorphisms(Catlab.CategoricalAlgebra.codom(Catlab.CategoricalAlgebra.left(box.rule)), world; cat=_cat, initial=initial_map, monic=box.rule.monic)
    matches = collect(gen)
    if !isempty(matches)
        new_world = AlgebraicRewriting.rewrite_match(box.rule, first(matches))
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = new_world)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = nothing)
    else
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = nothing)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = world)
    end
end

# ─── Utility Boxes ────────────────────────────────────────────────────────────

struct MergeWires; I_state::Any; end
merge_wires(I_state) = MergeWires(I_state)

function (box::MergeWires)(worlds...)
    for w in worlds
        w !== nothing && return w
    end
    return nothing
end

struct Coin; p::Float64; end
coin(p) = Coin(p)

function (box::Coin)(world)
    if rand() < box.p
        return world, nothing
    else
        return nothing, world
    end
end
