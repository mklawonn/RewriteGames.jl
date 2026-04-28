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
    steps      = BoxStep[]
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

    push!(steps, BoxStep(box_sym, in_names, out_names))
end

# ─── Runtime executor ─────────────────────────────────────────────────────────

"""
    run_game_sched!(gs::GameSched, initial_world, agents::Dict;
                    T_max::Int=1000,
                    terminal::Union{Function,Nothing}=nothing,
                    winner_wires::Dict{Symbol,Union{Symbol,Nothing}}=Dict())
        -> Vector{Experience}

Execute a complete game episode using the wiring-diagram schedule `gs`.

Wire semantics: each named wire holds either a live ACSet world (active) or
`nothing` (inactive).  Only one wire should be active at any time.

Loop structure:
- `init` wires are active on the first iteration.
- The first `length(gs._trace_names)` return wires loop back to the trace
  inputs on subsequent iterations; the rest are exit wires.
- The episode terminates when an exit wire becomes active, when
  `terminal(world)` returns `true` (if `terminal` is provided), or when
  `T_max` turns are exhausted.

Winner resolution (in priority order):
1. If `terminal` is non-`nothing`, it is called after each move and its result
   populates `Experience.winner` immediately (backward-compatible path).
2. If `winner_wires` is non-empty and an exit wire fires, the winner is looked
   up from `winner_wires` and the final `Experience` record is updated.
3. If `T_max` is reached without an exit wire, `winner = nothing` (draw/timeout).
"""
function run_game_sched!(gs::GameSched, initial_world, agents::Dict;
                         T_max::Int = 1000,
                         terminal::Union{Function, Nothing} = nothing,
                         winner_wires::Dict{Symbol, Union{Symbol, Nothing}} = Dict{Symbol, Union{Symbol, Nothing}}())
    all_exps   = Experience[]
    turn       = Ref(1)
    n_trace    = length(gs._trace_names)
    # Fallback terminal: never done, no winner (used when terminal === nothing)
    _terminal  = terminal === nothing ? (W) -> (false, nothing) : terminal

    # Initialise wires for the first iteration
    wires = _init_wires(gs, initial_world, nothing)

    fired_exit_name = nothing  # name of the exit wire that ended the game

    for _ in 1:(T_max + 1)
        round_exps = Experience[]
        _run_body!(gs._steps, wires, gs._all_boxes, agents, _terminal, turn, T_max, round_exps)
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
        # Only update if terminal() did not already resolve the winner
        if terminal === nothing || !last_exp.done
            all_exps[end] = Experience(
                last_exp.player, last_exp.state, last_exp.legal_actions,
                last_exp.action, last_exp.next_state,
                true, wire_winner,
                last_exp.info, last_exp.schedule_path,
            )
        end
    end

    return all_exps
end

"""
Convenience overload: use `game.initial()` as the starting world.

If `game.win_conditions` is non-`nothing`, it is used as `winner_wires` and
`terminal` is ignored (categorical exit-wire mode).  Otherwise `game.terminal`
is used as the termination predicate (backward-compatible mode).
"""
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

# ─── _run_body! ───────────────────────────────────────────────────────────────

"""
    _run_body!(steps, wires, boxes, agents, terminal, turn, T_max, exps)

Execute a parsed body `steps` against the live wire state `wires`.

Dispatches on the concrete box type stored in `boxes`:
- `PlayerRuleApp` — enumerates matches, calls the agent, applies the chosen
  rewrite, emits an `Experience`.
- `GameSched`     — recurses into the sub-schedule's body.
- `RuleApp` (AR native) — tries one match; routes to success/failure output.
- Anything else   — merge semantics: first active input wire → first output.
"""
function _run_body!(steps, wires, boxes, agents, terminal, turn::Ref{Int}, T_max, exps)
    for step in steps
        box = boxes[step.box]

        # Collect active input worlds (first non-nothing)
        input_world = nothing
        for wname in step.inputs
            w = get(wires, wname, nothing)
            w !== nothing && (input_world = w; break)
        end

        if input_world === nothing
            # No active input → all outputs inactive
            for out in step.outputs
                wires[out] = nothing
            end
            continue
        end

        if box isa PlayerRuleApp
            _exec_player!(step, box, input_world, wires, agents, terminal, turn, T_max, exps)

        elseif box isa GameSched
            _exec_subsched!(step, box, input_world, wires, boxes, agents, terminal, turn, T_max, exps)

        elseif hasproperty(box, :rule)
            # Native RuleApp (or similar) — try-apply semantics, 2 output ports
            _exec_native_rule!(step, box, input_world, wires)

        else
            # Schedule / merge_wires / other — pass first active input to first output
            wires[step.outputs[1]] = input_world
            for i in 2:length(step.outputs)
                wires[step.outputs[i]] = nothing
            end
        end
    end
end

# ─── PlayerRuleApp execution ──────────────────────────────────────────────────

function _exec_player!(step, box::PlayerRuleApp, world, wires, agents, terminal,
                        turn::Ref{Int}, T_max, exps)
    matches    = collect(get_matches(box.rule, world; cat=box.cat))
    actions    = [Action(box, m) for m in matches]
    state_pre  = GameState(world, turn[])
    agent      = agents[box.player]

    chosen = isempty(actions) ? nothing : select_action(agent, state_pre, actions)

    if chosen !== nothing
        new_world  = rewrite_match(box.rule, chosen.match)
        done, winner = terminal(new_world)
        turn[] += 1
        state_post = GameState(new_world, turn[])
        push!(exps, Experience(box.player, state_pre, actions, chosen,
                               state_post, done || turn[] > T_max, winner,
                               Dict{Symbol, Any}(), Symbol[]))
        # Route to success port (1) and clear failure port (2)
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = new_world)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = nothing)
    else
        # No matches — route to failure port (2), success port (1) inactive
        done, winner = terminal(world)
        turn[] += 1
        state_post = GameState(world, turn[])
        push!(exps, Experience(box.player, state_pre, actions, nothing,
                               state_post, done || turn[] > T_max, winner,
                               Dict{Symbol, Any}(), Symbol[]))
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = nothing)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = world)
    end
end

# ─── Nested GameSched execution ───────────────────────────────────────────────

function _exec_subsched!(step, sub_gs::GameSched, world, wires, _boxes, agents,
                          terminal, turn::Ref{Int}, T_max, exps)
    # Build sub-wire state: map parent input wires positionally to sub init wires
    sub_wires = Dict{Symbol, Any}()
    for (i, name) in enumerate(sub_gs._init_names)
        sub_wires[name] = i == 1 ? world : nothing
    end
    for name in sub_gs._trace_names
        sub_wires[name] = nothing
    end

    sub_exps = Experience[]
    _run_body!(sub_gs._steps, sub_wires, sub_gs._all_boxes,
               agents, terminal, turn, T_max, sub_exps)
    append!(exps, sub_exps)

    # Map sub-schedule return wires (by position) to parent output wires
    for (i, out_name) in enumerate(step.outputs)
        if i <= length(sub_gs._ret_names)
            wires[out_name] = get(sub_wires, sub_gs._ret_names[i], nothing)
        else
            wires[out_name] = nothing
        end
    end
end

# ─── Native RuleApp execution ─────────────────────────────────────────────────

function _exec_native_rule!(step, box, world, wires)
    _cat    = hasproperty(box, :cat) ? box.cat :
              infer_acset_cat(codom(left(box.rule)))
    matches = collect(get_matches(box.rule, world; cat=_cat))
    if !isempty(matches)
        new_world = rewrite_match(box.rule, first(matches))
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = new_world)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = nothing)
    else
        length(step.outputs) >= 1 && (wires[step.outputs[1]] = nothing)
        length(step.outputs) >= 2 && (wires[step.outputs[2]] = world)
    end
end
