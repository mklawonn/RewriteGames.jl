"""
Graph Process Trajectory Logger.

On every DPO rewrite, the GPU appends the delta (added/deleted element IDs)
to a pre-allocated event log and maintains a mapping from current element IDs
to their permanent history IDs.  This records the full causal history of the
simulation without requiring host communication mid-episode.
"""

struct DeltaEvent
    turn      :: Int32
    box_sym   :: UInt16     # index into a symbol table
    obj_type  :: UInt16     # schema.obj_index value
    elem_id   :: Int32      # world element (history ID)
    is_add    :: Bool       # true = addition, false = deletion
end

"""
    GPUTrajectoryLog

Pre-allocated trajectory recording structure.

- `events`:     Ring buffer of `DeltaEvent` records.
- `n_events`:   Current event count (host-side counter, updated after each step).
- `hist_id`:    Per-object-type mapping from current world element ID to the
                permanent history ID assigned when the element was first created.
"""
mutable struct GPUTrajectoryLog
    events   :: Vector{DeltaEvent}   # host-side buffer (GPU episodes are short)
    n_events :: Int
    hist_id  :: Dict{Symbol, Vector{Int32}}   # obj_type → current_id → hist_id
    n_hist   :: Dict{Symbol, Ref{Int}}         # history element count per type
    capacity :: Int
end

function GPUTrajectoryLog(schema::SchemaInfo, capacity::Int=100_000)
    hist_id = Dict(o => Int32[] for o in schema.obj_types)
    n_hist  = Dict(o => Ref(0) for o in schema.obj_types)
    GPUTrajectoryLog(Vector{DeltaEvent}(undef, capacity), 0, hist_id, n_hist, capacity)
end

"""
    log_additions!(log, schema, added_g, turn, box_idx)

Record newly added world elements and assign permanent history IDs.
`added_g`: Dict{Symbol, Vector{Int32}} mapping obj type → new world element indices.
"""
function log_additions!(log::GPUTrajectoryLog, schema::SchemaInfo,
                        added_g::Dict{Symbol, Vector{Int32}},
                        turn::Int, box_idx::Int)
    for o in schema.obj_types
        new_elems = get(added_g, o, Int32[])
        isempty(new_elems) && continue
        type_idx = UInt16(schema.obj_index[o])

        for elem in new_elems
            # Grow history mapping if needed
            elem_int = Int(elem)
            while length(log.hist_id[o]) < elem_int
                push!(log.hist_id[o], Int32(0))
            end

            log.n_hist[o][] += 1
            hist = Int32(log.n_hist[o][])
            log.hist_id[o][elem_int] = hist

            log.n_events < log.capacity || continue
            log.n_events += 1
            log.events[log.n_events] = DeltaEvent(Int32(turn), UInt16(box_idx),
                                                   type_idx, hist, true)
        end
    end
end

"""
    log_deletions!(log, schema, deleted_g, turn, box_idx)

Record deleted world elements using their permanent history IDs.
"""
function log_deletions!(log::GPUTrajectoryLog, schema::SchemaInfo,
                        deleted_g::Dict{Symbol, Vector{Int32}},
                        turn::Int, box_idx::Int)
    for o in schema.obj_types
        del_elems = get(deleted_g, o, Int32[])
        isempty(del_elems) && continue
        type_idx = UInt16(schema.obj_index[o])

        for elem in del_elems
            elem_int = Int(elem)
            hist = (elem_int <= length(log.hist_id[o])) ?
                   log.hist_id[o][elem_int] : Int32(0)
            hist == 0 && continue

            log.n_events < log.capacity || continue
            log.n_events += 1
            log.events[log.n_events] = DeltaEvent(Int32(turn), UInt16(box_idx),
                                                   type_idx, hist, false)
        end
    end
end

"""
    trajectory_events(log) -> Vector{DeltaEvent}

Return the recorded events in order.
"""
trajectory_events(log::GPUTrajectoryLog) = log.events[1:log.n_events]
