"""
    GPUACSet

GPU-resident representation of an ACSet.  Every column is a `CuArray` so
that rewriting kernels operate without host round-trips.

Layout per object type `o`:
  active[o]  :: CuVector{Bool}   — live element flag (tombstone = false)
  homs[h]    :: CuVector{Int32}  — foreign key for morphism h (1-based, 0=unset)
  attrs[a]   :: CuVector{Int32}  — encoded integer attribute value (0=wildcard)

`n_alloc[o]` is the allocated array capacity; `n_live[o]` is the current
count of non-tombstoned elements.  Both are host-side integers; only the
arrays themselves live on the GPU.
"""
struct GPUACSet
    schema  :: SchemaInfo
    active  :: Dict{Symbol, CuVector{Bool}}    # per object type
    homs    :: Dict{Symbol, CuVector{Int32}}   # per morphism
    attrs   :: Dict{Symbol, CuVector{Int32}}   # per attribute
    n_alloc :: Dict{Symbol, Int}               # capacity per obj type
    n_live  :: Dict{Symbol, Ref{Int}}          # live count per obj type
end

"""
    upload_acset(world, schema, enc) -> GPUACSet

Encode `world` and upload every column to the GPU.
"""
function upload_acset(world, schema::SchemaInfo, enc::AttributeEncoder)::GPUACSet
    active = Dict{Symbol, CuVector{Bool}}()
    homs   = Dict{Symbol, CuVector{Int32}}()
    attrs  = Dict{Symbol, CuVector{Int32}}()
    n_alloc = Dict{Symbol, Int}()
    n_live  = Dict{Symbol, Ref{Int}}()

    for o in schema.obj_types
        n = nparts(world, o)
        n_alloc[o] = n
        n_live[o]  = Ref(n)
        active[o]  = CUDA.ones(Bool, n)    # all elements start live
    end

    for h in schema.homs
        owner = schema.hom_dom[h]
        n     = nparts(world, owner)
        if n == 0
            homs[h] = CuArray(Int32[])
        else
            host_fk = Int32[subpart(world, i, h) for i in 1:n]
            homs[h] = CuArray(host_fk)
        end
    end

    for a in schema.attrs
        owner = schema.attr_dom[a]
        n     = nparts(world, owner)
        if n == 0
            attrs[a] = CuArray(Int32[])
        else
            host_av = Int32[encode_value(enc, a, subpart(world, i, a)) for i in 1:n]
            attrs[a] = CuArray(host_av)
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

    # New IDs after compaction: old_id → new_id (0 = deleted)
    new_id = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        flags  = host_act[o]
        n_live = sum(flags)
        add_parts!(result, o, n_live)
        mapping = zeros(Int, length(flags))
        cursor  = 0
        for (old, alive) in enumerate(flags)
            alive || continue
            cursor += 1
            mapping[old] = cursor
        end
        new_id[o] = mapping
    end

    # Set morphisms (skip deleted sources/targets)
    for h in schema.homs
        owner = schema.hom_dom[h]
        cod   = schema.hom_cod[h]
        fks   = host_hom[h]
        flags = host_act[owner]
        for (old_i, (alive, tgt)) in enumerate(zip(flags, fks))
            alive || continue
            new_i   = new_id[owner][old_i]
            new_tgt = tgt > 0 ? new_id[cod][tgt] : 0
            new_tgt > 0 && set_subpart!(result, new_i, h, new_tgt)
        end
    end

    # Set attributes
    for a in schema.attrs
        owner = schema.attr_dom[a]
        avs   = host_att[a]
        flags = host_act[owner]
        for (old_i, (alive, enc_v)) in enumerate(zip(flags, avs))
            alive || continue
            new_i = new_id[owner][old_i]
            v = decode_value(enc, a, enc_v)
            v !== nothing && set_subpart!(result, new_i, a, v)
        end
    end

    result
end
