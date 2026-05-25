"""
    CompiledBox

GPU-executable descriptor for a single wiring-diagram box.

box_type values:
  0 = PLAYER_RULE  (PlayerRuleApp: agent selects match, DPO applied)
  1 = QUERY        (read-only pattern search, no rewrite)
  2 = WEAKEN       (pass-through; no matching needed)
  3 = COIN         (stochastic split; params[1] = probability of output wire 1)
  4 = NATIVE_RULE  (automatic rule application, no agent)
  5 = AGENT_LOOP   (loop over agent interface matches, dispatch sub-schedule)
  6 = NESTED_SCHED (nested schedule — only used as fallback; prefer inlining)

`csp_idx` indexes into `CompiledGPUSched.csps` (0 = no CSP).
`adh_idx` indexes into `CompiledGPUSched.adhesive_cubes` (0 = no cube).
`out_wires` holds up to 4 output wire indices; unused slots are 0.
`params` holds box-specific Float32 parameters (e.g. coin probability).
"""
function _p_idx(s::Symbol)
    s === :blue && return UInt8(1)
    s === :red  && return UInt8(2)
    return UInt8(0)
end
_p_idx(::Nothing) = UInt8(0)

struct CompiledBox
    box_type      :: UInt8
    csp_idx       :: UInt16
    adh_idx       :: UInt16
    player_idx    :: UInt8                  # _p_idx(:_none) for non-player boxes
    in_wire       :: UInt16
    out_wires     :: NTuple{4, UInt16}
    params        :: NTuple{4, Float32}
    sub_sched_idx :: UInt16                  # index into sub_schedules
end

const BOX_PLAYER_RULE  = UInt8(0)
const BOX_QUERY        = UInt8(1)
const BOX_WEAKEN       = UInt8(2)
const BOX_COIN         = UInt8(3)
const BOX_NATIVE_RULE  = UInt8(4)
const BOX_AGENT_LOOP   = UInt8(5)
const BOX_NESTED_SCHED = UInt8(6)

"""
    CompiledGPUSched

GPU state machine compiled from a `GameSched`.

- `boxes`:          flat ordered list of boxes (executed sequentially per turn).
- `wire_names`:     wire index → original wire name (for debugging).
- `csps`:           one `CSPProblem` per unique rule.
- `adhesive_cubes`: one `AdhesiveCube` per unique rule.
- `init_wires`:     wire indices that receive the initial world.
- `trace_wires`:    wire indices that carry the world across iterations.
- `exit_wires`:     wire indices that signal episode termination.
- `box_players`:    player symbol per box (`:_none` for non-player boxes).
- `sub_schedules`:  sub-schedules used by AGENT_LOOP boxes.
"""
struct CompiledGPUSched
    boxes           :: Vector{CompiledBox}
    wire_names      :: Vector{Symbol}
    wire_index      :: Dict{Symbol, Int}
    csps            :: Vector{CSPProblem}
    adhesive_cubes  :: Vector{AdhesiveCube}
    gpu_cubes       :: Vector{GPUAdhesiveCube}   # GPU-resident copies of adhesive_cubes
    init_wires      :: Vector{Int}
    trace_wires     :: Vector{Int}
    exit_wires      :: Vector{Int}
    n_wires         :: Int
    sub_schedules   :: Vector{CompiledGPUSched}
    rules           :: Vector{Any}
    box_players     :: Vector{Symbol}
    device_boxes    :: Any
    registry        :: Any   # DeviceRuleRegistry when CUDA functional, nothing otherwise
end

# ── Agent-interface helper ──────────────────────────────────────────────────

function _register_agent_interface!(rule_registry, csps, cubes, rules_list,
                                    L, world, schema, enc; n_chunks::Int=1)
    key = (:interface, objectid(L))
    if !haskey(rule_registry, key)
        S   = acset_schema(L)
        cat = Catlab.CategoricalAlgebra.infer_acset_cat(L)
        init = Dict{Symbol, Any}(o => collect(1:nparts(L, o)) for o in ob(S))
        homs = homomorphisms(L, L; cat=cat, initial=init)
        id_L = isempty(homs) ? nothing : first(homs)
        if id_L === nothing
            push!(csps, lower_rule_to_csp((L=L, monic=true), world, schema, enc;
                                          n_chunks=n_chunks))
        else
            push!(csps, lower_rule_to_csp(_MockRule(id_L, id_L, true), world, schema, enc;
                                          n_chunks=n_chunks))
        end
        push!(rules_list, nothing)
        push!(cubes, precompute_adhesive_cube(nothing, schema; enc=enc))
        rule_registry[key] = length(csps)
    end
    rule_registry[key]
end

# ── Recursive step processor with inlining ─────────────────────────────────

"""
    _process_steps!(steps, all_boxes, boxes, csps, rules_list, adhesive_cubes,
                    rule_registry, wire_index, wire_set, box_players, sub_schedules,
                    world, schema, enc; wire_subst, wire_prefix)

Compile `steps` into `boxes`, inlining any non-agent-loop nested `GameSched`
boxes rather than emitting `BOX_NESTED_SCHED`.

`wire_subst`  maps a wire name used at this level to the global wire name it
              should resolve to (used for inlining init/ret wire connections).
`wire_prefix` is prepended to internal wire names to avoid collisions.
"""
function _process_steps!(steps, all_boxes, boxes, csps, rules_list, adhesive_cubes,
                         rule_registry, wire_index, wire_set, box_players, sub_schedules,
                         world, schema, enc;
                         wire_subst  :: Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                         wire_prefix :: String              = "",
                         n_chunks    :: Int                 = 1)

    function _w!(name::Symbol)
        resolved = get(wire_subst, name, nothing)
        if resolved !== nothing
            name = resolved
        elseif !isempty(wire_prefix)
            name = Symbol(wire_prefix * string(name))
        end
        haskey(wire_index, name) && return wire_index[name]
        push!(wire_set, name)
        wire_index[name] = length(wire_set)
        return wire_index[name]
    end

    function _reg!(app)
        rule = hasproperty(app, :rule) ? app.rule : app
        key  = (objectid(rule),
                hasproperty(app, :in_hom)   ? objectid(app.in_hom)   :
                hasproperty(app, :in_agent) ? objectid(app.in_agent) : 0)
        if !haskey(rule_registry, key)
            push!(csps,           lower_rule_to_csp(app, world, schema, enc;
                                                    n_chunks=n_chunks))
            push!(rules_list,     rule)
            push!(adhesive_cubes, precompute_adhesive_cube(rule, schema; enc=enc))
            rule_registry[key] = length(csps)
        end
        rule_registry[key]
    end

    # Helper to resolve a step wire name via current substitution / prefix
    function _resolve_wire(name::Symbol)
        resolved = get(wire_subst, name, nothing)
        if resolved !== nothing; return resolved; end
        isempty(wire_prefix) ? name : Symbol(wire_prefix * string(name))
    end

    for step in steps
        box    = all_boxes[step.box]
        in_w   = UInt16(_w!(first(step.inputs)))
        out_ws = ntuple(i -> i <= length(step.outputs) ?
                             UInt16(_w!(step.outputs[i])) : UInt16(0), 4)

        if box isa PlayerRuleApp
            ridx = _reg!(box)
            push!(boxes, CompiledBox(BOX_PLAYER_RULE, UInt16(ridx), UInt16(ridx),
                                     _p_idx(box.player), in_w, out_ws,
                                     (0f0,0f0,0f0,0f0), UInt16(0)))
            push!(box_players, box.player)

        elseif box isa GameSched && box._agent_name !== nothing
            # Agent loop: compile recursively as a sub-schedule
            sub = compile_schedule(box, world, schema, enc; n_chunks=n_chunks)
            push!(sub_schedules, sub)
            sub_idx = UInt16(length(sub_schedules))

            iface_idx = UInt16(0)
            if box._N !== nothing && haskey(box._N.from_name, string(box._agent_name))
                iface = box._N[string(box._agent_name)]
                iface_idx = UInt16(_register_agent_interface!(rule_registry, csps,
                                                              adhesive_cubes, rules_list,
                                                              iface, world, schema, enc;
                                                              n_chunks=n_chunks))
            end

            push!(boxes, CompiledBox(BOX_AGENT_LOOP, iface_idx, UInt16(0),
                                     _p_idx(box._agent_name), in_w, out_ws,
                                     (0f0,0f0,0f0,0f0), sub_idx))
            push!(box_players, box._agent_name)

        elseif box isa GameSched && box._agent_name === nothing
            # Inline the sub-schedule by building a wire substitution map
            sub_subst = copy(wire_subst)

            for (i, init_n) in enumerate(box._init_names)
                parent_w = i <= length(step.inputs) ? step.inputs[i] : init_n
                sub_subst[init_n] = _resolve_wire(parent_w)
            end
            for (i, ret_n) in enumerate(box._ret_names)
                parent_w = i <= length(step.outputs) ? step.outputs[i] : ret_n
                sub_subst[ret_n] = _resolve_wire(parent_w)
            end
            for trace_n in box._trace_names
                sub_subst[trace_n] = Symbol(wire_prefix * string(step.box) * "__trace_" * string(trace_n))
            end

            new_prefix = wire_prefix * string(step.box) * "__"
            _process_steps!(box._steps, box._all_boxes, boxes, csps, rules_list,
                             adhesive_cubes, rule_registry, wire_index, wire_set,
                             box_players, sub_schedules, world, schema, enc;
                             wire_subst=sub_subst, wire_prefix=new_prefix,
                             n_chunks=n_chunks)

        elseif hasproperty(box, :rule) ||
               (box isa AlgebraicRewriting.Schedules.RuleApps.RuleApp)
            ridx = _reg!(box)
            push!(boxes, CompiledBox(BOX_NATIVE_RULE, UInt16(ridx), UInt16(ridx),
                                     _p_idx(:_none), in_w, out_ws,
                                     (0f0,0f0,0f0,0f0), UInt16(0)))
            push!(box_players, :_none)

        else
            # Utility box (MergeWires, Coin, etc.): emit one BOX_WEAKEN per input
            # to correctly implement "take first active input" semantics.
            for in_name in step.inputs
                iw = UInt16(_w!(in_name))
                push!(boxes, CompiledBox(BOX_WEAKEN, UInt16(0), UInt16(0),
                                         _p_idx(:_none), iw, out_ws,
                                         (0f0,0f0,0f0,0f0), UInt16(0)))
                push!(box_players, :_none)
            end
        end
    end
end

# ── Top-level compile_schedule ──────────────────────────────────────────────

"""
    compile_schedule(gs, world, schema, enc) -> CompiledGPUSched

Compile a `GameSched` to a flat `CompiledGPUSched` suitable for GPU execution.
Nested non-agent-loop sub-schedules are inlined; agent-loop sub-schedules are
kept as recursive sub-schedules.
"""
function compile_schedule(gs::GameSched, world,
                           schema::SchemaInfo,
                           enc::AttributeEncoder;
                           n_chunks::Int = 1)::CompiledGPUSched

    # ── Agent-loop top-level wrapping ────────────────────────────────────────
    if gs._agent_name !== nothing
        inner_gs = GameSched(gs._inner, gs._player_map, gs._all_boxes, gs._steps,
                             gs._init_names, gs._trace_names, gs._ret_names,
                             gs._trace_args, gs._init_args, gs._body, gs._N,
                             nothing, gs.cat)
        sub = compile_schedule(inner_gs, world, schema, enc; n_chunks=n_chunks)

        wire_set   = [:_init, :_exit]
        wire_index = Dict(:_init => 1, :_exit => 2)

        csps           = CSPProblem[]
        adhesive_cubes = AdhesiveCube[]
        rules_list     = Any[]
        rule_registry  = Dict{Any, Int}()

        iface_idx = UInt16(0)
        if gs._N !== nothing && haskey(gs._N.from_name, string(gs._agent_name))
            iface = gs._N[string(gs._agent_name)]
            iface_idx = UInt16(_register_agent_interface!(rule_registry, csps,
                                                          adhesive_cubes, rules_list,
                                                          iface, world, schema, enc;
                                                          n_chunks=n_chunks))
        end

        box = CompiledBox(BOX_AGENT_LOOP, iface_idx, UInt16(0),
                          _p_idx(gs._agent_name), UInt16(1),
                          (UInt16(2), UInt16(0), UInt16(0), UInt16(0)),
                          (0f0,0f0,0f0,0f0), UInt16(1))

        device_boxes = CUDA.functional() ? CuArray([box]) : nothing
        return CompiledGPUSched(
            [box], wire_set, wire_index, csps, adhesive_cubes,
            [gpu_upload_cube(c) for c in adhesive_cubes],
            [1], Int[], [2], 2, [sub], rules_list, [gs._agent_name],
            device_boxes,
            CUDA.functional() ? _build_device_registry(rules_list, csps, adhesive_cubes, schema, enc) : nothing)
    end

    # ── 1. Global wire index ─────────────────────────────────────────────────
    wire_set   = Symbol[]
    wire_index = Dict{Symbol, Int}()

    function _wire!(name::Symbol)
        haskey(wire_index, name) && return wire_index[name]
        push!(wire_set, name)
        wire_index[name] = length(wire_set)
        return wire_index[name]
    end

    for w in gs._init_names;  _wire!(w); end
    for w in gs._trace_names; _wire!(w); end
    for w in gs._ret_names;   _wire!(w); end

    # ── 2. Rule registry ──────────────────────────────────────────────────────
    rule_registry  = Dict{Any, Int}()
    csps           = CSPProblem[]
    rules_list     = Any[]
    adhesive_cubes = AdhesiveCube[]

    # Pre-register all PlayerRuleApp rules (ensures stable ordering)
    all_pras = _collect_player_apps(gs)
    for (_, pra) in all_pras
        rule = pra.rule
        key  = (objectid(rule), objectid(pra.in_hom))
        if !haskey(rule_registry, key)
            push!(csps,           lower_rule_to_csp(pra, world, schema, enc;
                                                    n_chunks=n_chunks))
            push!(rules_list,     rule)
            push!(adhesive_cubes, precompute_adhesive_cube(rule, schema; enc=enc))
            rule_registry[key] = length(csps)
        end
    end

    # ── 3. Compile steps (with inlining) ──────────────────────────────────────
    boxes         = CompiledBox[]
    box_players   = Symbol[]
    sub_schedules = CompiledGPUSched[]

    _process_steps!(gs._steps, gs._all_boxes, boxes, csps, rules_list, adhesive_cubes,
                    rule_registry, wire_index, wire_set, box_players, sub_schedules,
                    world, schema, enc; n_chunks=n_chunks)

    # ── 4. Compute control wire sets ──────────────────────────────────────────
    init_wires  = [wire_index[w] for w in gs._init_names  if haskey(wire_index, w)]
    trace_wires = [wire_index[w] for w in gs._trace_names if haskey(wire_index, w)]
    n_ret       = length(gs._ret_names)
    n_trace     = length(gs._trace_names)
    exit_wires  = [wire_index[gs._ret_names[i]]
                   for i in (n_trace+1):n_ret
                   if haskey(wire_index, gs._ret_names[i])]

    device_boxes = CUDA.functional() ? CuArray(boxes) : nothing
    CompiledGPUSched(
        boxes, wire_set, wire_index,
        csps, adhesive_cubes,
        [gpu_upload_cube(c) for c in adhesive_cubes],
        init_wires, trace_wires, exit_wires,
        length(wire_set), sub_schedules,
        rules_list, box_players,
        device_boxes,
        CUDA.functional() ? _build_device_registry(rules_list, csps, adhesive_cubes, schema, enc) : nothing)
end
