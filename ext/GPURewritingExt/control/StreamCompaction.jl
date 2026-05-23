"""
Stream compaction — remove tombstoned elements from a `GPUACSet`.

DPO deletions leave "holes" (active=false) in the GPU arrays, degrading
memory access coalescing over time.  Periodic compaction packs live elements
into contiguous slots, rebuilds foreign-key columns, and reports a new_id
mapping (old element index → new element index, 0 = deleted).

Two implementations are provided:
  - GPU-native (CUDA path): uses an on-device prefix-sum + scatter to avoid
    all GPU→CPU transfers except a single scalar per type for n_alloc update.
  - CPU fallback: original host-orchestrated implementation.
"""

# ── Shared helper kernels ─────────────────────────────────────────────────────

@kernel function mark_live_kernel!(
    active   :: AbstractVector{Bool},
    live_ids :: AbstractVector{Int32}
)
    i = @index(Global, Linear)
    if i <= length(active)
        live_ids[i] = active[i] ? Int32(1) : Int32(0)
    end
end

@kernel function scatter_kernel!(
    src      :: AbstractVector{Int32},
    dst      :: AbstractVector{Int32},
    new_ids  :: AbstractVector{Int32}
)
    i = @index(Global, Linear)
    if i <= length(src)
        new_i = new_ids[i]
        if new_i != 0
            dst[new_i] = src[i]
        end
    end
end

@kernel function scatter_bool_kernel!(
    src      :: AbstractVector{Bool},
    dst      :: AbstractVector{Bool},
    new_ids  :: AbstractVector{Int32}
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
    fk_col   :: AbstractVector{Int32},
    new_ids  :: AbstractVector{Int32}
)
    i = @index(Global, Linear)
    if i <= length(fk_col)
        old_tgt = fk_col[i]
        if old_tgt != 0
            fk_col[i] = new_ids[old_tgt]
        end
    end
end

# Fill a Bool array with true for the first n slots, false for the rest.
@kernel function fill_true_prefix_kernel!(
    arr :: AbstractVector{Bool},
    n   :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(arr)
        arr[i] = i <= Int(n)
    end
end

# ── GPU-native compaction ─────────────────────────────────────────────────────

"""
    _compact_gpu_native!(g, schema, backend) -> Dict{Symbol, Vector{Int32}}

GPU-native implementation of stream compaction using on-device prefix-sum
(CUDA cumsum) and scatter kernels.  Only one scalar per type is transferred
CPU←GPU (to update n_alloc).  The new_id mappings stay on-device for FK
remapping, then are discarded.

Returns the new_id mapping (old index → new compact index, 0 = deleted)
as a CPU `Dict{Symbol, Vector{Int32}}` for compatibility with callers that
need to inspect the mapping.
"""
function _compact_gpu_native!(g::GPUACSet, schema::SchemaInfo, backend)
    gpu_new_ids = Dict{Symbol, CuVector{Int32}}()
    old_n       = Dict{Symbol, Int}()

    # Phase 1: compute inclusive prefix-sum new_id maps for each type that needs compaction.
    for o in schema.obj_types
        n      = g.n_alloc[o]
        n_live = g.n_live[o][]
        (n == 0 || n_live == n) && continue

        old_n[o] = n
        live_flags = CuArray{Int32}(undef, n)
        mark_live_kernel!(backend, 256)(live_flags, @view(g.active[o][1:n]); ndrange=n)
        KernelAbstractions.synchronize(backend)
        gpu_new_ids[o] = cumsum(live_flags)   # CUDA.jl inclusive prefix-sum on device
    end

    # Phase 2: compact homs and attrs for each type, using old active flags.
    for o in schema.obj_types
        !haskey(gpu_new_ids, o) && continue
        n       = old_n[o]
        n_live  = g.n_live[o][]
        new_ids = gpu_new_ids[o]
        cap     = length(g.active[o])

        old_active = g.active[o]   # reference to old active array (not yet replaced)

        # Scatter homs and attrs
        for h in schema.homs
            schema.hom_dom[h] == o || continue
            new_fk = CUDA.zeros(Int32, cap)
            scatter_kernel!(backend, 256)(
                @view(g.homs[h][1:n]), new_fk, new_ids; ndrange=n)
            KernelAbstractions.synchronize(backend)
            g.homs[h] = new_fk
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            new_av = CUDA.zeros(Int32, cap)
            scatter_kernel!(backend, 256)(
                @view(g.attrs[a][1:n]), new_av, new_ids; ndrange=n)
            KernelAbstractions.synchronize(backend)
            g.attrs[a] = new_av
        end

        # Compact the active array last (Phase 3 needs old_active for FK remap).
        new_act = CUDA.zeros(Bool, cap)
        fill_true_prefix_kernel!(backend, 256)(new_act, Int32(n_live); ndrange=cap)
        g.active[o] = new_act
        g.n_alloc[o] = n_live
    end

    # Phase 3: remap FK columns whose codomain type was compacted.
    for h in schema.homs
        cod = schema.hom_cod[h]
        !haskey(gpu_new_ids, cod) && continue
        new_ids = gpu_new_ids[cod]
        dom     = schema.hom_dom[h]
        n_dom   = g.n_alloc[dom]
        n_dom == 0 && continue
        remap_fk_kernel!(backend, 256)(
            @view(g.homs[h][1:n_dom]), new_ids; ndrange=n_dom)
    end
    KernelAbstractions.synchronize(backend)

    # Return CPU-side mapping for compatibility (download only what's needed).
    new_id_map = Dict{Symbol, Vector{Int32}}()
    for (o, gids) in gpu_new_ids
        new_id_map[o] = Array(gids)
    end
    # Types that were not compacted get identity-ish mappings (unused by callers)
    for o in schema.obj_types
        !haskey(new_id_map, o) && (new_id_map[o] = collect(Int32(1):Int32(g.n_alloc[o])))
    end
    new_id_map
end

# ── Main entry point ──────────────────────────────────────────────────────────

"""
    compact_gpu_acset!(g, schema, backend) -> Dict{Symbol, Vector{Int32}}

Compact `g` in-place: remove all tombstoned elements, rebuild FK columns,
and return `new_id_map` (per obj type: old element index → new index, 0 = deleted).

On CUDA-functional systems this uses the GPU-native prefix-sum path (no full
GPU→CPU round-trips).  On CPU-only systems it falls back to the host-orchestrated
implementation.
"""
function compact_gpu_acset!(g::GPUACSet, schema::SchemaInfo, backend)
    CUDA.functional() && return _compact_gpu_native!(g, schema, backend)

    # ── CPU fallback (original implementation) ───────────────────────────────
    new_id_map = Dict{Symbol, Vector{Int32}}()

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

    for o in schema.obj_types
        mapping = new_id_map[o]
        n_new   = g.n_live[o][]
        n_new == length(mapping) && continue

        n_new == length(mapping) && continue

        new_active  = trues(n_new)
        g.active[o] = new_active
        g.n_alloc[o] = n_new

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            host_fk = Array(g.homs[h])
            new_fk  = zeros(Int32, n_new)
            for (old_i, new_i) in enumerate(mapping)
                new_i == 0 && continue
                new_fk[new_i] = host_fk[old_i]
            end
            g.homs[h] = new_fk
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            host_av = Array(g.attrs[a])
            new_av  = zeros(Int32, n_new)
            for (old_i, new_i) in enumerate(mapping)
                new_i == 0 && continue
                new_av[new_i] = host_av[old_i]
            end
            g.attrs[a] = new_av
        end
    end

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
        g.homs[h] = host_fk
    end

    return new_id_map
end
