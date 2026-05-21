"""
GPU master scheduler.

Executes a `CompiledGPUSched` box-by-box, dispatching the Turbo pattern
matcher and DPO rewriting kernels for each active box.
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
    world_type  :: Any
    agents      :: Dict
    trajectory  :: Union{GPUTrajectoryLog, Nothing}
    compact_every :: Int
    rng         :: Xoshiro
    turn        :: Ref{Int}
    step_log    :: Vector{NamedTuple}
end

function GPUSchedulerState(sched, g, schema, enc, world_type, agents;
                            log_trajectory=false, compact_every=100)
    traj = log_trajectory ? GPUTrajectoryLog(schema) : nothing
    GPUSchedulerState(sched, g, schema, enc, world_type, agents,
                      traj, compact_every, Xoshiro(42), Ref(1), NamedTuple[])
end

function run_gpu_schedule!(state::GPUSchedulerState;
                           T_max::Int = 1000,
                           terminal_fn::Function = (W) -> (false, nothing),
                           winner_wires::Dict{Symbol, Union{Symbol,Nothing}} = Dict{Symbol,Union{Symbol,Nothing}}())::Bool
    sched  = state.sched
    g      = state.g
    schema = state.schema

    # 1. Flatten World
    n_obj = length(schema.obj_types)
    total_alloc = sum(g.n_alloc[o] for o in schema.obj_types)
    d_active = CUDA.zeros(Bool, total_alloc)
    d_n_alloc = CuArray(Int32[g.n_alloc[o] for o in schema.obj_types])
    d_n_live  = CuArray(Int32[g.n_live[o][] for o in schema.obj_types])
    
    obj_offsets = Int32[]
    curr = 0
    for o in schema.obj_types
        push!(obj_offsets, curr)
        n = g.n_alloc[o]
        if n > 0
            copyto!(d_active, curr+1, g.active[o], 1, n)
        end
        curr += n
    end
    d_obj_offsets = CuArray(obj_offsets)

    # Flatten Homs
    h_offsets = Int32[]
    h_cod_offsets = Int32[]
    total_hom_alloc = sum(g.n_alloc[schema.hom_dom[h]] for h in schema.homs; init=0)
    d_homs = CUDA.zeros(Int32, total_hom_alloc)
    curr_h = 0
    for h in schema.homs
        push!(h_offsets, curr_h)
        push!(h_cod_offsets, obj_offsets[schema.obj_index[schema.hom_cod[h]]])
        n = g.n_alloc[schema.hom_dom[h]]
        if n > 0
            copyto!(d_homs, curr_h+1, g.homs[h], 1, n)
        end
        curr_h += n
    end
    d_hom_offsets = CuArray(h_offsets)
    d_hom_cod_offsets = CuArray(h_cod_offsets)

    # Flatten Attrs
    a_offsets = Int32[]
    total_attr_alloc = sum(g.n_alloc[schema.attr_dom[a]] for a in schema.attrs; init=0)
    d_attrs = CUDA.zeros(Int32, total_attr_alloc)
    curr_a = 0
    for a in schema.attrs
        push!(a_offsets, curr_a)
        n = g.n_alloc[schema.attr_dom[a]]
        if n > 0
            copyto!(d_attrs, curr_a+1, g.attrs[a], 1, n)
        end
        curr_a += n
    end
    d_attr_offsets = CuArray(a_offsets)

    world_device = DeviceACSet(d_active, d_n_live, d_obj_offsets, d_n_alloc,
                               d_homs, d_hom_offsets, d_hom_cod_offsets,
                               d_attrs, d_attr_offsets)

    # 2. Control Wires
    d_wire_active = CUDA.zeros(Bool, sched.n_wires)
    d_init_wires  = CuArray(Int32.(sched.init_wires))
    d_trace_wires = CuArray(Int32.(sched.trace_wires))
    d_exit_wires  = CuArray(Int32.(sched.exit_wires))
    
    # 3. Matcher Buffers
    d_sol_count_buf = CUDA.zeros(Int32, 1)

    # 4. Event Log
    max_events = T_max * length(sched.boxes)
    d_event_log = CUDA.zeros(GpuRewriteEvent, max_events)
    d_n_events  = CUDA.zeros(Int32, 1)
    
    seed = rand(UInt64)

    # 5. Launch Master Kernel
    @cuda master_scheduler_kernel(sched.device_boxes, d_wire_active, Int32(sched.n_wires),
                                  d_init_wires, Int32(length(d_init_wires)),
                                  d_trace_wires, Int32(length(d_trace_wires)),
                                  d_exit_wires, Int32(length(d_exit_wires)),
                                  sched.registry, world_device,
                                  d_sol_count_buf,
                                  d_event_log, d_n_events,
                                  Int32(T_max), seed)
    
    CUDA.synchronize()
    
    # 6. Copy results back
    h_n_live = Array(d_n_live)
    h_active = Array(d_active)
    h_homs   = Array(d_homs)
    h_attrs  = Array(d_attrs)
    
    for (i, o) in enumerate(schema.obj_types)
        g.n_live[o][] = Int(h_n_live[i])
        off = obj_offsets[i]
        n = g.n_alloc[o]
        copyto!(g.active[o], 1, h_active, off+1, n)
    end
    
    for (i, h) in enumerate(schema.homs)
        off = h_offsets[i]
        n = g.n_alloc[schema.hom_dom[h]]
        copyto!(g.homs[h], 1, h_homs, off+1, n)
    end
    
    for (i, a) in enumerate(schema.attrs)
        off = a_offsets[i]
        n = g.n_alloc[schema.attr_dom[a]]
        copyto!(g.attrs[a], 1, h_attrs, off+1, n)
    end
    
    wire_active = Array(d_wire_active)
    for ew in sched.exit_wires
        if wire_active[ew]
            return true
        end
    end
    
    return false
end
