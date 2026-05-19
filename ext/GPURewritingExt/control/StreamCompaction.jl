"""
Stream compaction — remove tombstoned elements from a `GPUACSet`.

DPO deletions leave "holes" (active=false) in the GPU arrays, degrading
memory access coalescing over time.  Periodic compaction packs live elements
into contiguous slots, rebuilds foreign-key columns, and reports a new_id
mapping (old element index → new element index, 0 = deleted).
"""

@kernel function mark_live_kernel!(
    active   :: AbstractVector{Bool},
    live_ids :: AbstractVector{Int32}   # output: cumulative count of live elements
)
    i = @index(Global, Linear)
    if i <= length(active)
        live_ids[i] = active[i] ? Int32(1) : Int32(0)
    end
end

@kernel function scatter_kernel!(
    src      :: AbstractVector{Int32},   # source column
    dst      :: AbstractVector{Int32},   # destination column
    new_ids  :: AbstractVector{Int32}    # old_id → new_id (0 = deleted)
)
    i = @index(Global, Linear)
    if i <= length(src)
        new_i = new_ids[i]
        if new_i != 0
            dst[new_i] = src[i]
        end
    end
end

@kernel function remap_fk_kernel!(
    fk_col   :: AbstractVector{Int32},   # foreign key column (in-place update)
    new_ids  :: AbstractVector{Int32}    # old_id → new_id for the target type
)
    i = @index(Global, Linear)
    if i <= length(fk_col)
        old_tgt = fk_col[i]
        if old_tgt != 0
            fk_col[i] = new_ids[old_tgt]
        end
    end
end

"""
    compact_gpu_acset!(g, schema, backend) -> Dict{Symbol, Vector{Int32}}

Compact `g` in-place: remove all tombstoned elements, rebuild FK columns,
and return `new_id_map` (per obj type: old element index → new index, 0 = deleted).

This is a host-orchestrated operation: the prefix-sum and scatter happen via
Julia array operations on host-transferred data, then the compacted arrays
are re-uploaded.  For large worlds a pure-GPU compaction using `accumulate`
or a CUDA CUB call should replace this.
"""
function compact_gpu_acset!(g::GPUACSet, schema::SchemaInfo, backend)
    new_id_map = Dict{Symbol, Vector{Int32}}()

    # ── 1. Compute new_id mapping per object type ──────────────────────────────
    for o in schema.obj_types
        host_active = Array(g.active[o])
        n = length(host_active)
        mapping = zeros(Int32, n)
        cursor  = Int32(0)
        for i in 1:n
            host_active[i] || continue
            cursor += Int32(1)
            mapping[i] = cursor
        end
        new_id_map[o] = mapping
        g.n_live[o][] = Int(cursor)
    end

    # ── 2. Compact each object type's arrays ──────────────────────────────────
    for o in schema.obj_types
        mapping = new_id_map[o]
        n_new   = g.n_live[o][]
        n_new == length(mapping) && continue   # nothing to compact

        host_active = Array(g.active[o])
        new_active  = trues(n_new)
        g.active[o] = CuArray(new_active)
        g.n_alloc[o] = n_new

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            host_fk = Array(g.homs[h])
            new_fk  = zeros(Int32, n_new)
            for (old_i, new_i) in enumerate(mapping)
                new_i == 0 && continue
                new_fk[new_i] = host_fk[old_i]
            end
            g.homs[h] = CuArray(new_fk)
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            host_av = Array(g.attrs[a])
            new_av  = zeros(Int32, n_new)
            for (old_i, new_i) in enumerate(mapping)
                new_i == 0 && continue
                new_av[new_i] = host_av[old_i]
            end
            g.attrs[a] = CuArray(new_av)
        end
    end

    # ── 3. Remap all FK columns whose codomain type was compacted ──────────────
    for h in schema.homs
        cod   = schema.hom_cod[h]
        owner = schema.hom_dom[h]
        mapping = new_id_map[cod]
        all(m -> m == 0 || m == searchsortedfirst(mapping, m), mapping) && continue
        host_fk = Array(g.homs[h])
        for i in eachindex(host_fk)
            old_tgt = host_fk[i]
            old_tgt == 0 && continue
            host_fk[i] = old_tgt <= length(mapping) ? mapping[old_tgt] : Int32(0)
        end
        g.homs[h] = CuArray(host_fk)
    end

    return new_id_map
end
