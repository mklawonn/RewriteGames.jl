"""
    CompiledBox

GPU-executable descriptor for a single wiring-diagram box.

box_type values:
  0 = PLAYER_RULE  (PlayerRuleApp: agent selects match, DPO applied)
  1 = QUERY        (read-only pattern search, no rewrite)
  2 = WEAKEN       (pass-through; no matching needed)
  3 = COIN         (stochastic split; params[1] = probability of output wire 1)
  4 = NATIVE_RULE  (automatic rule application, no agent)

`csp_idx` indexes into `CompiledGPUSched.csps` (0 = no CSP).
`adh_idx` indexes into `CompiledGPUSched.adhesive_cubes` (0 = no cube).
`out_wires` holds up to 4 output wire indices; unused slots are 0.
`params` holds box-specific Float32 parameters (e.g. coin probability).
"""
struct CompiledBox
    box_type      :: UInt8
    csp_idx       :: UInt16
    adh_idx       :: UInt16
    player        :: Symbol                  # :_none for non-player boxes
    in_wire       :: UInt16
    out_wires     :: NTuple{4, UInt16}
    params        :: NTuple{4, Float32}
    sub_sched_idx :: UInt16                  # index into sub_schedules
end

const BOX_PLAYER_RULE = UInt8(0)
const BOX_QUERY       = UInt8(1)
const BOX_WEAKEN      = UInt8(2)
const BOX_COIN        = UInt8(3)
const BOX_NATIVE_RULE = UInt8(4)
const BOX_AGENT_LOOP = UInt8(5)
const BOX_NESTED_SCHED = UInt8(6)

"""
    CompiledGPUSched

GPU state machine compiled from a `GameSched`.

- `boxes`:           ordered list of boxes (executed sequentially per turn).
- `wire_names`:      wire index → original wire name (for debugging/decoding).
- `csps`:            one `CSPProblem` per unique rule.
- `adhesive_cubes`:  one `AdhesiveCube` per unique rule.
- `init_wires`:      wire indices that receive the initial world.
- `trace_wires`:     wire indices that carry the world across iterations.
- `exit_wires`:      wire indices that signal episode termination.
"""
struct CompiledGPUSched
    boxes           :: Vector{CompiledBox}
    wire_names      :: Vector{Symbol}
    wire_index      :: Dict{Symbol, Int}
    csps            :: Vector{CSPProblem}
    adhesive_cubes  :: Vector{AdhesiveCube}
    init_wires      :: Vector{Int}
    trace_wires     :: Vector{Int}
    exit_wires      :: Vector{Int}
    n_wires         :: Int
    sub_schedules   :: Vector{CompiledGPUSched}
    rules           :: Vector{Any}
end

"""
    compile_schedule(gs, world, schema, enc) -> CompiledGPUSched

Walk `gs._steps` (the parsed wiring-diagram body) and produce a
`CompiledGPUSched`.  Rules are deduplicated by object identity so that a
single CSP / adhesive cube serves all boxes sharing the same rule.
"""

function _register_agent_interface!(registry, csps, cubes, rules_list, L, world, schema, enc)
    key = (:interface, objectid(L))
    if !haskey(registry, key)
        # Build identity rule wrapping L
        S = acset_schema(L)
        cat = Catlab.CategoricalAlgebra.infer_acset_cat(L)
        init = Dict{Symbol, Any}(o => collect(1:nparts(L, o)) for o in ob(S))
        homs = homomorphisms(L, L; cat=cat, initial=init)
        id_L = isempty(homs) ? nothing : first(homs)
        if id_L === nothing
             # fallback for variable categories where identity is hard to construct
             # just use codom(left(rule)) logic by mocking a rule
             # Use a mock that just provides L
             push!(csps, lower_rule_to_csp((L=L, monic=true), world, schema, enc))
        push!(rules_list, nothing) # Interface has no DPO rule object
        else
             # In this context _MockRule and lower_rule_to_csp expect L
             # but we can just pass L if we update lower_rule_to_csp or a mock
             push!(csps, lower_rule_to_csp(_MockRule(id_L, id_L, true), world, schema, enc))
        end
        push!(cubes, precompute_adhesive_cube(nothing, schema)) # Dummy cube
        registry[key] = length(csps)
    end
    registry[key]
end

function compile_schedule(gs::GameSched, world,
                          schema::SchemaInfo,
                          enc::AttributeEncoder)::CompiledGPUSched
    if gs._agent_name !== nothing
        # Wrap the steps in an AGENT_LOOP by creating a virtual schedule
        # First compile the inner schedule (without the agent name)
        inner_gs = GameSched(gs._inner, gs._player_map, gs._all_boxes, gs._steps,
                             gs._init_names, gs._trace_names, gs._ret_names,
                             gs._trace_args, gs._init_args, gs._body, gs._N, nothing, gs.cat)
        sub = compile_schedule(inner_gs, world, schema, enc)
        
        # Now build the wrapper
        wire_set = [:_init, :_exit]
        wire_index = Dict(:_init => 1, :_exit => 2)
        
        interface = gs._N[string(gs._agent_name)]
        csps = CSPProblem[]
        adhesive_cubes = AdhesiveCube[]
        rules_list = Any[]
        iface_idx = _register_agent_interface!(Dict(), csps, adhesive_cubes, rules_list,
                                               interface, world, schema, enc)
        
        # Note: sub-schedule out-wires need to map back to parent if needed,
        # but for top-level agent loop we just fire exit.
        box = CompiledBox(BOX_AGENT_LOOP, UInt16(iface_idx), UInt16(0),
                          gs._agent_name, UInt16(1), (UInt16(2), UInt16(0), UInt16(0), UInt16(0)),
                          (0f0, 0f0, 0f0, 0f0), UInt16(1))
        
        return CompiledGPUSched([box], wire_set, wire_index, csps, adhesive_cubes,
                                 [1], Int[], [2], 2, [sub], rules_list)
    end
    # ── 1. Collect and index all wires ────────────────────────────────────────
    wire_set = Symbol[]
    wire_index = Dict{Symbol,Int}()

    function _wire!(name::Symbol)
        haskey(wire_index, name) && return wire_index[name]
        push!(wire_set, name)
        wire_index[name] = length(wire_set)
        return wire_index[name]
    end

    for step in gs._steps
        for w in step.inputs;  _wire!(w); end
        for w in step.outputs; _wire!(w); end
    end
    for w in gs._init_names;  _wire!(w); end
    for w in gs._trace_names; _wire!(w); end
    for w in gs._ret_names;   _wire!(w); end

    # ── 2. Compile unique rules (CSP + adhesive cube) ─────────────────────────
    rule_registry = Dict{Any, Int}()   # rule object_id → csp index
    csps           = CSPProblem[]
    rules_list     = Any[]
    adhesive_cubes = AdhesiveCube[]

    function _register_rule!(app)
        rule = hasproperty(app, :rule) ? app.rule : app
        key = (objectid(rule), hasproperty(app, :in_hom) ? objectid(app.in_hom) : (hasproperty(app, :in_agent) ? objectid(app.in_agent) : 0))
        if !haskey(rule_registry, key)
            push!(csps,           lower_rule_to_csp(app, world, schema, enc))
            push!(rules_list,     rule)
            push!(adhesive_cubes, precompute_adhesive_cube(rule, schema))
            rule_registry[key] = length(csps)
        end
        rule_registry[key]
    end

    # Pre-register all PlayerRuleApp rules
    all_pras = _collect_player_apps(gs)
    for (_, pra) in all_pras
        _register_rule!(pra)
    end

    # ── 3. Compile each BoxStep ───────────────────────────────────────────────
    boxes = CompiledBox[]
    sub_schedules = CompiledGPUSched[]

    for step in gs._steps
        box    = gs._all_boxes[step.box]
        in_w   = UInt16(_wire!(first(step.inputs)))
        out_ws = ntuple(i -> i <= length(step.outputs) ?
                             UInt16(_wire!(step.outputs[i])) : UInt16(0), 4)

        if box isa PlayerRuleApp
            ridx = _register_rule!(box)
            push!(boxes, CompiledBox(BOX_PLAYER_RULE, UInt16(ridx), UInt16(ridx),
                                     box.player, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0), UInt16(0)))

        elseif box isa GameSched
            if box._agent_name !== nothing
                # Agent loop: compile recursively as a sub-schedule
                sub = compile_schedule(box, world, schema, enc)
                push!(sub_schedules, sub)
                
                # Find interface for agent loop
                interface = nothing
                if box._N !== nothing && haskey(box._N.from_name, string(box._agent_name))
                    interface = box._N[string(box._agent_name)]
                end
                
                if interface !== nothing
                    iface_idx = _register_agent_interface!(rule_registry, csps, adhesive_cubes, rules_list, 
                                                           interface, world, schema, enc)
                    push!(boxes, CompiledBox(BOX_AGENT_LOOP, UInt16(iface_idx), UInt16(0),
                                             box._agent_name, in_w, out_ws,
                                             (0f0, 0f0, 0f0, 0f0),
                                             UInt16(length(sub_schedules))))
                else
                    push!(boxes, CompiledBox(BOX_AGENT_LOOP, UInt16(0), UInt16(0),
                                             box._agent_name, in_w, out_ws,
                                             (0f0, 0f0, 0f0, 0f0),
                                             UInt16(length(sub_schedules))))
                end
            else
                # Plain nested schedule: compile as sub-schedule
                sub = compile_schedule(box, world, schema, enc)
                push!(sub_schedules, sub)
                push!(boxes, CompiledBox(BOX_NESTED_SCHED, UInt16(0), UInt16(0),
                                         :_none, in_w, out_ws,
                                         (0f0, 0f0, 0f0, 0f0),
                                         UInt16(length(sub_schedules))))
            end
        elseif hasproperty(box, :rule) || (box isa AlgebraicRewriting.Schedules.RuleApps.RuleApp)
            ridx = _register_rule!(box)
            push!(boxes, CompiledBox(BOX_NATIVE_RULE, UInt16(ridx), UInt16(ridx),
                                     :_none, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0), UInt16(0)))
        else
            # Utility box (merge_wires, etc.) — wire pass-through
            push!(boxes, CompiledBox(BOX_WEAKEN, UInt16(0), UInt16(0),
                                     :_none, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0), UInt16(0)))
        end
    end

    init_wires  = [wire_index[w] for w in gs._init_names  if haskey(wire_index, w)]
    trace_wires = [wire_index[w] for w in gs._trace_names if haskey(wire_index, w)]
    n_ret       = length(gs._ret_names)
    n_trace     = length(gs._trace_names)
    exit_wires  = [wire_index[gs._ret_names[i]]
                   for i in (n_trace+1):n_ret
                   if haskey(wire_index, gs._ret_names[i])]

    CompiledGPUSched(boxes, wire_set, wire_index,
                     csps, adhesive_cubes,
                     init_wires, trace_wires, exit_wires,
                     length(wire_set), sub_schedules,
                     rules_list)
end
