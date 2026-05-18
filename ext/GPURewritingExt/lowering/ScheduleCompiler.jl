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
    box_type  :: UInt8
    csp_idx   :: UInt16
    adh_idx   :: UInt16
    player    :: Symbol                  # :_none for non-player boxes
    in_wire   :: UInt16
    out_wires :: NTuple{4, UInt16}
    params    :: NTuple{4, Float32}
end

const BOX_PLAYER_RULE = UInt8(0)
const BOX_QUERY       = UInt8(1)
const BOX_WEAKEN      = UInt8(2)
const BOX_COIN        = UInt8(3)
const BOX_NATIVE_RULE = UInt8(4)

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
end

"""
    compile_schedule(gs, world, schema, enc) -> CompiledGPUSched

Walk `gs._steps` (the parsed wiring-diagram body) and produce a
`CompiledGPUSched`.  Rules are deduplicated by object identity so that a
single CSP / adhesive cube serves all boxes sharing the same rule.
"""
function compile_schedule(gs::GameSched, world,
                          schema::SchemaInfo,
                          enc::AttributeEncoder)::CompiledGPUSched
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
    adhesive_cubes = AdhesiveCube[]

    function _register_rule!(rule)
        key = objectid(rule)
        if !haskey(rule_registry, key)
            push!(csps,           lower_rule_to_csp(rule, world, schema, enc))
            push!(adhesive_cubes, precompute_adhesive_cube(rule, schema))
            rule_registry[key] = length(csps)
        end
        rule_registry[key]
    end

    # Pre-register all PlayerRuleApp rules
    all_pras = _collect_player_apps(gs)
    for (_, pra) in all_pras
        _register_rule!(pra.rule)
    end

    # ── 3. Compile each BoxStep ───────────────────────────────────────────────
    boxes = CompiledBox[]

    for step in gs._steps
        box    = gs._all_boxes[step.box]
        in_w   = UInt16(_wire!(first(step.inputs)))
        out_ws = ntuple(i -> i <= length(step.outputs) ?
                             UInt16(_wire!(step.outputs[i])) : UInt16(0), 4)

        if box isa PlayerRuleApp
            ridx = _register_rule!(box.rule)
            push!(boxes, CompiledBox(BOX_PLAYER_RULE, UInt16(ridx), UInt16(ridx),
                                     box.player, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0)))

        elseif box isa GameSched
            # Nested schedule: recursively compile and inline
            sub = compile_schedule(box, world, schema, enc)
            append!(boxes, sub.boxes)

        elseif hasproperty(box, :rule)
            ridx = _register_rule!(box.rule)
            push!(boxes, CompiledBox(BOX_NATIVE_RULE, UInt16(ridx), UInt16(ridx),
                                     :_none, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0)))
        else
            # Utility box (merge_wires, etc.) — wire pass-through
            push!(boxes, CompiledBox(BOX_WEAKEN, UInt16(0), UInt16(0),
                                     :_none, in_w, out_ws,
                                     (0f0, 0f0, 0f0, 0f0)))
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
                     length(wire_set))
end
