"""
    GPUACSet

GPU-resident representation of an ACSet.  Every column is a `CuArray` so
that rewriting kernels operate without host round-trips.

Layout per object type `o`:
  active[o]  :: CuVector{Bool}   — live element flag (tombstone = false)
  homs[h]    :: CuVector{Int32}  — foreign key for morphism h (1-based, 0=unset)
  attrs[a]   :: CuVector{Int32}  — encoded integer attribute value (0=wildcard)

`n_alloc[o]` is the high-water mark: slots 1..n_alloc[o] have been assigned
(live or tombstoned).  The CuArray capacity is `length(g.active[o])` which
may exceed `n_alloc[o]` due to 2× over-allocation; spare slots have
`active = false` and are ready for the next addition without reallocation.
`n_live[o]` is the count of currently active (non-tombstoned) elements.
"""
struct GPUACSet
    schema  :: SchemaInfo
    active  :: Dict{Symbol, Any}   # CuVector{Bool} on GPU, Vector{Bool} on CPU
    homs    :: Dict{Symbol, Any}   # CuVector{Int32} on GPU, Vector{Int32} on CPU
    attrs   :: Dict{Symbol, Any}   # CuVector{Int32} on GPU, Vector{Int32} on CPU
    n_alloc :: Dict{Symbol, Int}   # high-water mark per obj type
    n_live  :: Dict{Symbol, Ref{Int}}
end

"""
    upload_acset(world, schema, enc) -> GPUACSet

Encode `world` and upload every column to the GPU.
"""
function upload_acset(world, schema::SchemaInfo, enc::AttributeEncoder; headspace=1000)::GPUACSet
    active  = Dict{Symbol, Any}()
    homs    = Dict{Symbol, Any}()
    attrs   = Dict{Symbol, Any}()
    n_alloc = Dict{Symbol, Int}()
    n_live  = Dict{Symbol, Ref{Int}}()

    use_cuda = CUDA.functional()
    _zeros(T, n) = use_cuda ? CUDA.zeros(T, n) : zeros(T, n)

    for o in schema.obj_types
        n = nparts(world, o)
        n_alloc[o] = n
        n_live[o]  = Ref(n)
        act = _zeros(Bool, n + headspace)
        if n > 0; act[1:n] .= true; end
        active[o] = act
    end

    for h in schema.homs
        owner = schema.hom_dom[h]
        n     = nparts(world, owner)
        if n == 0
            homs[h] = _zeros(Int32, headspace)
        else
            host_fk = Int32[subpart(world, i, h) for i in 1:n]
            fk = _zeros(Int32, n + headspace)
            copyto!(fk, 1, host_fk, 1, n)
            homs[h] = fk
        end
    end

    for a in schema.attrs
        owner = schema.attr_dom[a]
        n     = nparts(world, owner)
        if n == 0
            attrs[a] = _zeros(Int32, headspace)
        else
            host_av = Int32[encode_value(enc, a, subpart(world, i, a)) for i in 1:n]
            av = _zeros(Int32, n + headspace)
            copyto!(av, 1, host_av, 1, n)
            attrs[a] = av
        end
    end

    GPUACSet(schema, active, homs, attrs, n_alloc, n_live)
end

"""
    download_acset(g, enc, schema) -> ACSet

Transfer GPU arrays to host memory and decode attribute values back to their
original Julia types.  Returns a fresh ACSet of the same schema as the
original world.
"""
function download_acset(g::GPUACSet, enc::AttributeEncoder, world_type)
    schema   = g.schema
    host_act = Dict(o => Array(g.active[o]) for o in schema.obj_types)
    host_hom = Dict(h => Array(g.homs[h])   for h in schema.homs)
    host_att = Dict(a => Array(g.attrs[a])  for a in schema.attrs)

    # Build a compact (no-tombstone) new ACSet
    result = world_type()

    # New IDs after compaction: old_id → new_id (0 = deleted).  Vectorized:
    # the new id of the k-th live part is its running count of live flags; dead
    # parts map to 0.  `live_ids[o]` holds the old ids of live parts in compacted
    # (new-id) order, so columns can be gathered and written in bulk below.
    new_id   = Dict{Symbol, Vector{Int}}()
    live_ids = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        flags   = host_act[o]
        mapping = cumsum(flags)             # Bool → Int running count
        @inbounds for i in eachindex(flags); flags[i] || (mapping[i] = 0); end
        new_id[o]   = mapping
        live_ids[o] = findall(flags)        # old ids of live parts, ascending
        add_parts!(result, o, length(live_ids[o]))
    end

    # Set morphisms (skip deleted sources/targets).  Gather the target column for
    # live sources, remap through the codomain's compaction, then write the whole
    # column at once when it is fully defined; fall back to per-element only when
    # some targets are missing/deleted (0), preserving the original "skip 0"
    # semantics exactly.
    for h in schema.homs
        owner = schema.hom_dom[h]
        cod   = schema.hom_cod[h]
        live  = live_ids[owner]
        isempty(live) && continue
        fks   = host_hom[h]
        cmap  = new_id[cod]
        col   = Vector{Int}(undef, length(live))
        @inbounds for k in eachindex(live)
            t = fks[live[k]]
            col[k] = t > 0 ? cmap[t] : 0
        end
        if all(>(0), col)
            set_subpart!(result, :, h, col)
        else
            @inbounds for k in eachindex(col)
                col[k] > 0 && set_subpart!(result, k, h, col[k])
            end
        end
    end

    # Set attributes.  Decode the live column, then bulk-write when no value is
    # `nothing`; otherwise per-element (preserving the original "skip nothing").
    for a in schema.attrs
        owner = schema.attr_dom[a]
        live  = live_ids[owner]
        isempty(live) && continue
        avs     = host_att[a]
        decoded = [decode_value(enc, a, avs[i]) for i in live]
        if !any(isnothing, decoded)
            set_subpart!(result, :, a, decoded)
        else
            @inbounds for k in eachindex(decoded)
                decoded[k] !== nothing && set_subpart!(result, k, a, decoded[k])
            end
        end
    end

    result
end

"""
    download_acset_compact(g, enc, world_type, backend) -> ACSet

Compact `g` in-place (GPU-native prefix-sum, no tombstones remain), then
download without the tombstone-skipping loop.  Use only when it is acceptable
to mutate `g` (e.g., at episode end before discarding `g`).
"""
function download_acset_compact(g::GPUACSet, enc::AttributeEncoder, world_type, backend)
    compact_gpu_acset!(g, g.schema, backend)
    schema = g.schema
    result = world_type()

    for o in schema.obj_types
        add_parts!(result, o, g.n_alloc[o])
    end

    for h in schema.homs
        dom = schema.hom_dom[h]
        n   = g.n_alloc[dom]
        n == 0 && continue
        host_fk = Array(g.homs[h])
        for i in 1:n
            host_fk[i] > 0 && set_subpart!(result, i, h, Int(host_fk[i]))
        end
    end

    for a in schema.attrs
        dom = schema.attr_dom[a]
        n   = g.n_alloc[dom]
        n == 0 && continue
        host_av = Array(g.attrs[a])
        for i in 1:n
            v = decode_value(enc, a, host_av[i])
            v !== nothing && set_subpart!(result, i, a, v)
        end
    end

    result
end

function Base.deepcopy(g::GPUACSet)
    GPUACSet(
        g.schema,
        Dict(k => copy(v) for (k,v) in pairs(g.active)),
        Dict(k => copy(v) for (k,v) in pairs(g.homs)),
        Dict(k => copy(v) for (k,v) in pairs(g.attrs)),
        copy(g.n_alloc),
        Dict(k => Ref(v[]) for (k,v) in pairs(g.n_live))
    )
end
