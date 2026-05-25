using CUDA
using Random

# BOX_* constants are defined in ScheduleCompiler.jl

# Device-side Xoshiro128Plus state
struct XoshiroState
    s0::UInt64
    s1::UInt64
    s2::UInt64
    s3::UInt64
end

@inline function next_rand(s::XoshiroState)
    s0, s1, s2, s3 = s.s0, s.s1, s.s2, s.s3
    res = s0 + s3
    t = s1 << 17
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = (s3 << 45) | (s3 >> 19)
    return res, XoshiroState(s0, s1, s2, s3)
end

@inline function rand_f32(s::XoshiroState)
    res, next_s = next_rand(s)
    val = Float32(res >> 40) / Float32(1 << 24)
    return val, next_s
end

struct GpuRewriteEvent
    turn    :: Int32
    box_idx :: Int32
    success :: Bool
end

Base.zero(::Type{GpuRewriteEvent}) = GpuRewriteEvent(Int32(0), Int32(0), false)

# --- Native Matching & Rewriting Helpers ---

@inline function device_match(registry, rule_idx, world, sol_count)
    n_vars = registry.csp_n_vars[rule_idx]
    if n_vars == 0
        CUDA.atomic_add!(pointer(sol_count, 1), Int32(1))
        return true
    end
    # TODO: Full matcher integration
    return false
end

@inline function device_rewrite(registry, rule_idx, world, match_idx, turn, rng)
    n_obj = length(world.n_live)
    # 1. Addition
    for t_idx in 1:n_obj
        n_add = registry.rhs_n_add_flat[(rule_idx-1)*n_obj + t_idx]
        if n_add > 0
            # Claim slots
            old_n = CUDA.atomic_add!(pointer(world.n_live, t_idx), Int32(n_add))
            for i in 1:n_add
                local_id = old_n + i
                if local_id <= world.n_alloc[t_idx]
                    flat_id = world.obj_offsets[t_idx] + local_id
                    world.active[flat_id] = true
                end
            end
        end
    end
    return true
end

# --- Master Kernel ---
function master_scheduler_kernel(boxes, wire_active, n_wires,
                                 init_wires, n_init,
                                 trace_wires, n_trace,
                                 exit_wires, n_exit,
                                 registry, world,
                                 sol_count_buf,
                                 event_log, n_events_ref,
                                 T_max, seed)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx == 1 || return

    rng = XoshiroState(seed, seed ^ 0xBEADBEADBEADBEAD, seed ^ 0xFACEFACEFACEFACE, seed ^ 0xCAFECAFECAFECAFE)

    # Initialize init wires
    for i in 1:n_init
        wire_active[Int(init_wires[i])] = true
    end

    for iter in 1:T_max
        any_changed = false
        
        for b_idx in 1:length(boxes)
            box = boxes[b_idx]
            in_w = Int(box.in_wire)
            if in_w > 0 && wire_active[in_w]
                if box.box_type == BOX_WEAKEN
                    wire_active[in_w] = false
                    for i in 1:4
                        ow = box.out_wires[i]
                        ow == 0 && break
                        wire_active[Int(ow)] = true
                    end
                    any_changed = true
                elseif box.box_type == BOX_COIN
                    wire_active[in_w] = false
                    val, rng = rand_f32(rng)
                    branch = val < box.params[1] ? 1 : 2
                    ow = box.out_wires[branch]
                    if ow != 0
                        wire_active[Int(ow)] = true
                    end
                    any_changed = true
                elseif box.box_type == BOX_PLAYER_RULE || box.box_type == BOX_NATIVE_RULE
                    wire_active[in_w] = false
                    
                    # Native Match
                    sol_count_buf[1] = 0
                    found = device_match(registry, Int(box.csp_idx), world, sol_count_buf)
                    
                    if found
                        # Native Rewrite (pick first match)
                        device_rewrite(registry, Int(box.csp_idx), world, 1, iter, rng)
                        
                        ow = box.out_wires[1] # success
                        if ow != 0
                            wire_active[Int(ow)] = true
                        end
                        
                        # Log event
                        ev_idx = CUDA.atomic_add!(pointer(n_events_ref, 1), Int32(1)) + Int32(1)
                        if ev_idx <= length(event_log)
                            event_log[ev_idx] = GpuRewriteEvent(Int32(iter), Int32(b_idx), true)
                        end
                    else
                        ow = box.out_wires[2] # fail
                        if ow != 0
                            wire_active[Int(ow)] = true
                        end
                    end
                    any_changed = true
                end
            end
        end

        # Termination checks
        terminated = false
        for i in 1:n_exit
            if wire_active[Int(exit_wires[i])]; terminated = true; break; end
        end
        terminated && break

        trace_found = false
        for i in 1:n_trace
            if wire_active[Int(trace_wires[i])]; trace_found = true; break; end
        end
        if !trace_found && !any_changed; break; end
    end
    return
end
