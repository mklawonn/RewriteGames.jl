"""
Solution reconstruction — convert GPU outputs back to user-facing Julia objects.

After the GPU schedule terminates:
1. `download_acset` transfers the compacted `GPUACSet` to host memory and
   decodes integer-encoded attributes back to their original Julia values.
   (Defined in `rewriting/GPUACSet.jl`.)

2. `reconstruct_experiences` rebuilds the `Vector{Experience}` from the
   per-step event log maintained during GPU execution.  This mirrors the
   Experience records emitted by `_exec_player!` in `sched_runner.jl`.

3. `decode_trajectory` converts the `GPUTrajectoryLog` into a list of
   `NamedTuple` records (turn, type, history_id, added/deleted) for
   downstream analysis — the foundation of the Graph Process trajectory.
"""

"""
    reconstruct_experiences(step_log, schema, enc, agents, sched) -> Vector{Experience}

Reconstruct `Experience` records from the `step_log` collected during GPU
execution.  Each entry in `step_log` captures the minimal data needed:
world snapshots (as encoded GPUACSets), the match assignment, and the
player / box identity.
"""
function reconstruct_experiences(step_log::Vector{<:NamedTuple},
                                 schema::SchemaInfo,
                                 enc::AttributeEncoder,
                                 world_type,
                                 agents::Dict)::Vector{Experience}
    exps = Experience[]
    for entry in step_log
        pre_world  = entry.pre_state isa GPUACSet ? download_acset(entry.pre_state,  enc, world_type) : entry.pre_state
        post_world = entry.post_state isa GPUACSet ? download_acset(entry.post_state, enc, world_type) : entry.post_state

        state_pre  = GameState(pre_world,  entry.turn)
        state_post = GameState(post_world, entry.turn + 1)

        # Re-build Action from match assignment (flat Int32 vector → ACSetTransformation)
        match_hom = _decode_match(entry.match, entry.csp, pre_world, schema)
        chosen    = match_hom === nothing ? nothing :
                    Action(entry.box_descriptor, match_hom)

        push!(exps, Experience(
            entry.player,
            state_pre,
            Action[],            # legal_actions not re-enumerated post-hoc
            chosen,
            state_post,
            entry.done,
            entry.winner,
            Dict{Symbol,Any}(),
            Symbol[],
            nothing,
        ))
    end
    exps
end

"""
    decode_trajectory(log, schema) -> Vector{NamedTuple}

Convert a `GPUTrajectoryLog` into a flat list of events for analysis.
"""
function decode_trajectory(log::GPUTrajectoryLog,
                           schema::SchemaInfo)::Vector{NamedTuple{(:turn,:obj_type,:hist_id,:added),Tuple{Int,Symbol,Int,Bool}}}
    events = trajectory_events(log)
    map(e -> (
        turn     = Int(e.turn),
        obj_type = schema.obj_types[Int(e.obj_type)],
        hist_id  = Int(e.elem_id),
        added    = e.is_add,
    ), events)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _decode_match(assignment::Vector{Int32},
                       csp::CSPProblem,
                       world,
                       schema::SchemaInfo)
    assignment === nothing && return nothing
    length(assignment) == 0 && return nothing

    comps = Dict{Symbol, Dict{Int,Int}}()
    for o in schema.obj_types
        base  = get(csp.var_offset, o, 0)
        base == 0 && continue
        d = Dict{Int,Int}()
        # Determine variable block size for this type
        n_vars_o = count(v -> begin
            # Check: variable v belongs to type o if it's in [base, next_base)
            in_range = v >= base
            for other in schema.obj_types
                ob = get(csp.var_offset, other, 0)
                other != o && ob > base && ob <= v && (in_range = false)
            end
            in_range
        end, 1:Int(csp.n_vars))
        for i in 1:n_vars_o
            v   = base + (i - 1)
            v > length(assignment) && break
            val = Int(assignment[v])
            val > 0 && (d[i] = val)
        end
        isempty(d) || (comps[o] = d)
    end

    try
        return homomorphism(world, world;   # placeholder — actual pattern world needed
                            initial=comps)
    catch
        return nothing
    end
end
