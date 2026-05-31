"""
Zone-partitioned CSP domain building.

A `ZonePartition` maps each (obj_type, slot_id) to a zone index (1..n_zones,
0 = global/unzoned).  For each (zone_idx, obj_type) pair, pre-built GPU
bitmasks restrict CSP domains to zone-local active elements.

This enables "zone-local" rule patterns to be solved with smaller effective
domain sizes.  Combined with the EPS-threshold fix (turbo_block for nc_max=48
with n_vars ≤ 7), zone-local patterns benefit from shared-memory AC-1
propagation even when the global world is large.

The GPUACSet remains the single source of truth; zone masks are thin views
that prune the domain without duplicating world data.

Usage:

    partition = build_zone_partition(g, schema, nc, zone_fn)
    d_gpu = _build_domains_gpu_zoned!(backend, csp, g, schema, scratch,
                                       partition, zone_idx)
    # then use d_gpu in _gpu_turbo_fill_scratch! as usual

After movement rewrites (slot changes zone), call `update_zone_masks!` to
refresh only the affected zones.
"""

struct ZonePartition
    n_zones    :: Int
    # slot_zone[obj_sym][k] = zone index of slot k (0 = global/unzoned)
    slot_zone  :: Dict{Symbol, Vector{Int32}}   # CPU-side
    # zone_masks[(zone_idx, obj_sym)] = nc-length GPU bitmask
    zone_masks :: Dict{Tuple{Int,Symbol}, CuVector{UInt64}}
    nc         :: Int   # number of UInt64 chunks (from world at build time)
end

# ── Build helpers ─────────────────────────────────────────────────────────────

@kernel function _build_zone_mask_kernel!(
    mask      :: AbstractVector{UInt64},
    active    :: AbstractVector{Bool},
    slot_zone :: AbstractVector{Int32},
    target_z  :: Int32,
    nc        :: Int32,
)
    k = @index(Global, Linear)
    if k <= length(active) && active[k] && slot_zone[k] == target_z
        ci, bi = elem_to_chunk(k)
        if ci <= Int(nc)
            Atomix.@atomic mask[ci] |= UInt64(1) << bi
        end
    end
end

"""
    build_zone_partition(g, schema, nc, zone_fn; backend) -> ZonePartition

Build a `ZonePartition` from `g`.

`nc` should be `csp.n_chunks` for the current game (or the maximum across all
CSPs for the schedule).  `zone_fn(obj_sym, slot_id) -> Int` returns the zone
index (1..n_zones) or 0 for global/unzoned objects.
"""
function build_zone_partition(g::GPUACSet,
                               schema::SchemaInfo,
                               nc::Int,
                               zone_fn::Function;
                               backend = CUDA.CUDABackend())::ZonePartition
    # Determine zone membership for every slot (CPU-side, from FK chains)
    slot_zone = Dict{Symbol, Vector{Int32}}()
    n_zones   = 0

    for o in schema.obj_types
        n    = g.n_alloc[o]
        zones = Vector{Int32}(undef, n)
        for k in 1:n
            z = zone_fn(o, k)
            zones[k] = Int32(z)
            n_zones = max(n_zones, z)
        end
        slot_zone[o] = zones
    end

    # Build GPU bitmasks: one per (zone_idx, obj_type)
    zone_masks = Dict{Tuple{Int,Symbol}, CuVector{UInt64}}()
    for o in schema.obj_types
        n = g.n_alloc[o]
        n == 0 && continue
        d_zone = CuVector{Int32}(slot_zone[o])
        for z in 1:n_zones
            mask_gpu = CUDA.zeros(UInt64, nc)
            _build_zone_mask_kernel!(backend, 256)(
                mask_gpu, g.active[o], d_zone, Int32(z), Int32(nc); ndrange = n)
            zone_masks[(z, o)] = mask_gpu
        end
    end
    KernelAbstractions.synchronize(backend)

    ZonePartition(n_zones, slot_zone, zone_masks, nc)
end

"""
    update_zone_masks!(partition, g, schema, affected_types; backend)

Rebuild GPU bitmasks for the affected object types across all zones.
Call after rewrites that change zone membership (e.g., PlatformZone swaps),
passing only the types whose zone assignments changed.
"""
function update_zone_masks!(partition::ZonePartition,
                             g::GPUACSet,
                             schema::SchemaInfo,
                             affected_types::Vector{Symbol},
                             zone_fn::Function;
                             backend = CUDA.CUDABackend())
    nc = partition.nc
    for o in affected_types
        n = g.n_alloc[o]
        n == 0 && continue

        # Recompute slot_zone for this type
        new_zones = Vector{Int32}(undef, n)
        for k in 1:n
            new_zones[k] = Int32(zone_fn(o, k))
        end
        partition.slot_zone[o] = new_zones
        d_zone = CuVector{Int32}(new_zones)

        for z in 1:partition.n_zones
            mask_gpu = CUDA.zeros(UInt64, nc)
            _build_zone_mask_kernel!(backend, 256)(
                mask_gpu, g.active[o], d_zone, Int32(z), Int32(nc); ndrange = n)
            partition.zone_masks[(z, o)] = mask_gpu
        end
    end
    KernelAbstractions.synchronize(backend)
end

# ── Zone-restricted domain building ──────────────────────────────────────────

"""
    _build_domains_gpu_zoned!(backend, csp, g, schema, scratch, partition, zone_idx)

Like `_build_domains_gpu!` but restricts each variable's domain to elements
belonging to `zone_idx`.  Global-typed variables (zone_idx=0 in `slot_zone`)
still get the full active mask.  Returns a view into `scratch.buf_domains`.
"""
function _build_domains_gpu_zoned!(backend,
                                    csp::CSPProblem,
                                    g::GPUACSet,
                                    schema::SchemaInfo,
                                    scratch::GPUScratchBuffers,
                                    partition::ZonePartition,
                                    zone_idx::Int)
    nc = csp.n_chunks
    nv = Int(csp.n_vars)
    d  = @view scratch.buf_domains[1:max(nv * nc, 1)]
    KernelAbstractions.fill!(d, UInt64(0))
    isempty(csp.var_offset) && return d

    type_mask = scratch.buf_type_mask
    type_bases = csp.sorted_type_bases

    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        n         = g.n_alloc[o]
        n == 0 && continue

        KernelAbstractions.fill!(type_mask, UInt64(0))

        # Use zone mask if available; fall back to full active mask for global types
        zone_key = (zone_idx, o)
        if haskey(partition.zone_masks, zone_key)
            zone_mask = partition.zone_masks[zone_key]
            n_mask    = min(length(zone_mask), nc)
            copyto!(type_mask, 1, zone_mask, 1, n_mask)
        else
            # Global-typed object: all active elements are valid
            _build_type_mask_kernel!(backend, 256)(
                type_mask, g.active[o], Int32(nc); ndrange = n)
        end

        for v in base:(next_base - 1)
            off = (v - 1) * nc
            copyto!(d, off + 1, type_mask, 1, nc)
        end
    end
    d
end

# ── Zoned multi-pass match collection ────────────────────────────────────────

"""
    collect_zoned_solutions!(backend, csp, g, schema, enc, scratch, partition;
                              max_solutions) -> Vector{Vector{Int32}}

Run the CSP solver once per zone, restrict domains to zone-local elements,
collect all solutions, and return them with global slot IDs.

Intended for zone-local rule patterns.  Cross-zone rules should use the
standard `gpu_turbo_solve` / `_gpu_turbo_fill_scratch!` path.
"""
function collect_zoned_solutions!(backend,
                                   csp::CSPProblem,
                                   g::GPUACSet,
                                   schema::SchemaInfo,
                                   enc::AttributeEncoder,
                                   scratch::GPUScratchBuffers,
                                   partition::ZonePartition;
                                   max_solutions::Int = 10_000)::Vector{Vector{Int32}}
    all_solutions = Vector{Int32}[]
    hf_flat, hf_offs = _build_hom_fwd_gpu!(backend, g, schema, csp.n_chunks, scratch)

    for z in 1:partition.n_zones
        d_gpu = _build_domains_gpu_zoned!(backend, csp, g, schema, scratch, partition, z)
        _apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc, backend, scratch)
        KernelAbstractions.synchronize(backend)

        sols = gpu_turbo_solve(backend, csp, d_gpu, hf_flat, hf_offs;
                               max_solutions, scratch)
        append!(all_solutions, sols)
    end

    unique(all_solutions)   # deduplicate solutions that appear in multiple zones
end
