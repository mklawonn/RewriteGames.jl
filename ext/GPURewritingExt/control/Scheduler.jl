
"""
GPU master scheduler.

Executes a `CompiledGPUSched` box-by-box, dispatching the Turbo pattern
matcher and DPO rewriting kernels for each active box.  Control flow mirrors
the CPU `run_game_sched!` loop in `src/engine/sched_runner.jl` but all
world-mutation happens on the GPU.

For each turn:
  1. For each CompiledBox in order, check whether the input wire has a world.
  2. PLAYER_RULE:  enumerate matches (Turbo solver) → agent selects → DPO rewrite.
  3. NATIVE_RULE:  enumerate matches → pick first → DPO rewrite (no agent).
  4. WEAKEN:       pass the world through unchanged.
  5. COIN:         stochastic routing via Xoshiro128 PRNG.
  6. Check exit wires; if fired, stop.
  7. Propagate trace wire for next iteration.
"""

"""
    GPUSchedulerState

All mutable state required during a GPU schedule run.
"""
mutable struct GPUSchedulerState
    sched       :: CompiledGPUSched
    g           :: GPUACSet
    schema      :: SchemaInfo
    enc         :: AttributeEncoder
    world_type  :: Any              # Julia type of the original ACSet (for download)
    agents      :: Dict
    trajectory  :: Union{GPUTrajectoryLog, Nothing}
    compact_every :: Int
    rng         :: Xoshiro          # host-side PRNG for stochastic boxes
    turn        :: Ref{Int}
    step_log    :: Vector{NamedTuple}  # per-rewrite event records for Experience reconstruction
end

function GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                            log_trajectory=false, compact_every=100)
    traj = log_trajectory ? GPUTrajectoryLog(schema) : nothing
    GPUSchedulerState(sched, g, schema, enc, world_type, agents,
                      traj, compact_every, Xoshiro(42), Ref(1), NamedTuple[])
end

"""
    run_gpu_schedule!(state; T_max, terminal_fn, winner_wires) -> Bool

Main execution loop.  Returns `true` if the episode terminated normally
(exit wire fired or terminal condition met), `false` if T_max was hit.
"""
function run_gpu_schedule!(state::GPUSchedulerState;
                           T_max::Int = 1000,
                           terminal_fn::Function = (W) -> (false, nothing),
                           winner_wires::Dict{Symbol, Union{Symbol,Nothing}} = Dict{Symbol,Union{Symbol,Nothing}}())::Bool
    sched  = state.sched
    g      = state.g
    schema = state.schema
    enc    = state.enc

    # Wire state: each wire either holds a world (true) or is empty (false)
    wire_active = falses(sched.n_wires)
    for iw in sched.init_wires
        wire_active[iw] = true
    end

    for iter in 1:(T_max + 1)
        iter_changed = _run_gpu_boxes!(state, sched, wire_active, terminal_fn, winner_wires, nothing)
        
        # Check exit wires
        for ew in sched.exit_wires
            if wire_active[ew]
                return true
            end
        end

        # Check trace wires — feed back for next iteration
        trace_active = any(wire_active[tw] for tw in sched.trace_wires)
        trace_active || break

        # Reset trace wires to active for next iteration
        for tw in sched.trace_wires
            wire_active[tw] = true
        end

        !iter_changed && break   # quiescent — stop
    end

    state.turn[] > T_max
end

function _run_gpu_boxes!(state, sched, wire_active, terminal_fn, winner_wires, pinned_agent)::Bool
    iter_changed = false
    g      = state.g
    schema = state.schema
    enc    = state.enc

    for (box_idx, box) in enumerate(sched.boxes)
        wire_active[Int(box.in_wire)] || continue

        if box.box_type == BOX_WEAKEN
            for ow in box.out_wires
                ow == 0 && break
                wire_active[Int(ow)] = true
            end
            wire_active[Int(box.in_wire)] = false

        elseif box.box_type == BOX_COIN
            p = Float64(box.params[1])
            branch = rand(state.rng) < p ? 1 : 2
            ow = box.out_wires[branch]
            ow != 0 && (wire_active[Int(ow)] = true)
            wire_active[Int(box.in_wire)] = false

        elseif box.box_type == BOX_AGENT_LOOP
            # Enumerate agent matches
            if box.csp_idx != 0
                iface_csp = sched.csps[Int(box.csp_idx)]
                agent_matches = _turbo_cpu_solve(iface_csp, g, schema, enc)
                
                sub_sched_idx = Int(box.sub_sched_idx)
                if sub_sched_idx != 0 && sub_sched_idx <= length(sched.sub_schedules)
                    sub_sched = sched.sub_schedules[sub_sched_idx]
                    
                    for am in agent_matches
                        # Create a local wire state for the sub-schedule
                        sub_wire_active = falses(sub_sched.n_wires)
                        for iw in sub_sched.init_wires
                            sub_wire_active[iw] = true
                        end
                        
                        # Run sub-schedule for this agent match
                        changed = _run_gpu_boxes!(state, sub_sched, sub_wire_active, terminal_fn, winner_wires, am)
                        changed && (iter_changed = true)
                    end
                end
            end
            
            # After loop, propagate world to success output
            ow = box.out_wires[1]
            ow != 0 && (wire_active[Int(ow)] = true)
            wire_active[Int(box.in_wire)] = false

        elseif box.box_type ∈ (BOX_PLAYER_RULE, BOX_NATIVE_RULE)
            csp  = sched.csps[Int(box.csp_idx)]
            cube = sched.adhesive_cubes[Int(box.adh_idx)]

            # Enumerate matches (optionally pinned to current agent)
            matches = _turbo_cpu_solve(csp, g, schema, enc, pinned_agent)

            if isempty(matches)
                # No matches: fire pass/fail wire (out_wires[2])
                ow = box.out_wires[2]
                ow != 0 && (wire_active[Int(ow)] = true)
                wire_active[Int(box.in_wire)] = false
                continue
            end

            # Agent or auto selection
            chosen_match = if box.box_type == BOX_PLAYER_RULE && box.player != :_none
                agent = get(state.agents, box.player, nothing)
                if agent !== nothing
                    # Download current world snapshot for agent
                    snap = download_acset(g, enc, state.world_type)
                    gs   = GameState(snap, state.turn[])
                    actions = [Action(nothing, m) for m in matches]
                    chosen  = select_action(agent, gs, actions)
                    chosen === nothing ? nothing : chosen.match
                else
                    first(matches)
                end
            else
                first(matches)
            end

            if chosen_match !== nothing
                pre_snap = g   # reference before mutation (for logging)

                # DPO deletion
                to_del, dangling_ok = build_to_del_mask(chosen_match, cube, schema, g)
                if dangling_ok
                    deleted_g = _collect_deleted(chosen_match, cube, schema, g)
                    _apply_deletion!(g, to_del, schema)

                    # DPO addition
                    added_g = apply_pushout!(g, chosen_match, cube,
                                             _get_rule(sched, box),
                                             schema, enc, nothing)

                    # Trajectory logging
                    if state.trajectory !== nothing
                        log_deletions!(state.trajectory, schema, deleted_g,
                                       state.turn[], box_idx)
                        log_additions!(state.trajectory, schema, added_g,
                                       state.turn[], box_idx)
                    end

                    # Terminal check
                    snap = download_acset(g, enc, state.world_type)
                    done, winner = terminal_fn(snap)
                    state.turn[] += 1

                    # Record step for Experience reconstruction
                    push!(state.step_log, (
                        pre_state      = pre_snap,
                        post_state     = g,
                        match          = chosen_match,
                        csp            = csp,
                        player         = box.player,
                        box_descriptor = nothing,
                        turn           = state.turn[] - 1,
                        done           = done,
                        winner         = winner,
                    ))

                    # Stream compaction
                    if state.turn[] % state.compact_every == 0
                        compact_gpu_acset!(g, schema, nothing)
                    end

                    iter_changed = true
                    ow = box.out_wires[1]
                    ow != 0 && (wire_active[Int(ow)] = true)
                    wire_active[Int(box.in_wire)] = false
                end
            else
                ow = box.out_wires[2]
                ow != 0 && (wire_active[Int(ow)] = true)
                wire_active[Int(box.in_wire)] = false
            end
        end # box dispatch
    end # boxes
    return iter_changed
end

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    _turbo_cpu_solve(csp, g, schema, enc, [pinned_match]) -> Vector{Vector{Int32}}

Run the CPU Turbo solver (propagation + dive-solve) against the current
GPU world state.
"""
function _turbo_cpu_solve(csp::CSPProblem, g::GPUACSet,
                          schema::SchemaInfo,
                          enc::AttributeEncoder,
                          pinned_match::Union{Nothing, Vector{Int32}}=nothing)::Vector{Vector{Int32}}
    g_offset = _global_offset(g, schema)
    domains  = _init_domains(csp, g, schema, g_offset)

    if pinned_match !== nothing && !isempty(csp.agent_var_map)
        for (i, v_idx) in enumerate(csp.agent_var_map)
            tgt = pinned_match[i]
            tgt == 0 && continue
            v_idx > length(domains) && continue
            domains[v_idx] = UInt64(1) << (tgt - 1)
        end
    end

    _apply_attr_masks!(domains, csp, g, schema, enc, g_offset)
    cpu_propagate!(domains, csp.bytecodes) || return Vector{Int32}[]
    cpu_dive_solve(csp, domains)
end

function _apply_deletion!(g::GPUACSet, to_del::CuVector{Bool}, schema::SchemaInfo)
    offset = 0
    for o in schema.obj_types
        n = g.n_alloc[o]
        host_del = Array(to_del)[offset+1 : offset+n]
        host_act = Array(g.active[o])
        for i in 1:n
            host_del[i] && (host_act[i] = false)
        end
        g.active[o] = CuArray(host_act)
        g.n_live[o][] -= sum(host_del)
        offset += n
    end
end

function _collect_deleted(match::Vector{Int32}, cube::AdhesiveCube,
                          schema::SchemaInfo, g::GPUACSet)
    g_offset = _global_offset(g, schema)
    k_img    = Set{Int}(Int(x) for x in cube.k_to_l)
    deleted  = Dict(o => Int32[] for o in schema.obj_types)
    for (flat_l, type_idx) in enumerate(cube.l_types)
        flat_l ∈ k_img && continue
        g_elem = Int(match[flat_l])
        g_elem == 0 && continue
        o = schema.obj_types[Int(type_idx)]
        push!(deleted[o], Int32(g_elem - g_offset[o]))
    end
    deleted
end

function _get_rule(sched::CompiledGPUSched, box::CompiledBox)
    nothing
end
