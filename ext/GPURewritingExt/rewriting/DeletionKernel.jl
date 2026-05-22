"""
DPO pushout complement — GPU deletion phase.

For a match `m : L → G`, the elements of `L` that are not in the image of
`K → L` must be deleted from `G`.  In DPO mode this is simply a flag flip;
in SPO mode foreign-key chains are additionally cascaded.

`build_to_del_mask`      — builds the flat deletion bitmask on the host.
`gpu_dangling_ok`        — checks the dangling condition entirely on GPU.
`dpo_deletion_kernel!`   — clears active flags in-place (GPU).
`spo_cascade_kernel!`    — cascades deletions through FK chains (GPU).
"""

# ── DPO deletion ──────────────────────────────────────────────────────────────

@kernel function dpo_deletion_kernel!(
    active   :: AbstractVector{Bool},   # element active flags (in-place update)
    to_del   :: AbstractVector{Bool}    # true = delete this G-element
)
    i = @index(Global, Linear)
    if i <= length(active)
        if to_del[i]
            active[i] = false
        end
    end
end

# ── SPO cascade deletion ──────────────────────────────────────────────────────

@kernel function spo_cascade_kernel!(
    active   :: AbstractVector{Bool},
    fk_col   :: AbstractVector{Int32},   # one foreign key column
    changed  :: AbstractVector{Bool}     # signal: any new deletions?
)
    i = @index(Global, Linear)
    if i <= length(fk_col) && active[i]
        tgt = fk_col[i]
        if tgt != 0 && !active[tgt]     # points to deleted element
            active[i]  = false
            changed[1] = true
        end
    end
end

# ── Parallel dangling-condition check ─────────────────────────────────────────

"""
Per-hom GPU kernel: each active, non-deleted source element checks whether
its FK target is being deleted.  Writes 1 into `results[i]` on violation.
No atomics — each thread owns its own output slot.
"""
@kernel function dangling_check_per_hom_kernel!(
    results    :: AbstractVector{Bool},
    active_src :: AbstractVector{Bool},
    fk         :: AbstractVector{Int32},
    to_del_src :: AbstractVector{Bool},
    to_del_tgt :: AbstractVector{Bool},
)
    i = @index(Global, Linear)
    if i <= length(active_src)
        if active_src[i] && !to_del_src[i]
            tgt = Int(fk[i])
            if tgt != 0 && tgt <= length(to_del_tgt) && to_del_tgt[tgt]
                results[i] = true
            end
        end
    end
end

"""
    gpu_dangling_ok(to_del, g, schema, backend) -> Bool

Check the DPO dangling condition entirely on GPU.  For each schema morphism,
launch `dangling_check_per_hom_kernel!` and reduce the result with `sum`.
Returns `true` if the match is safe (no dangling edges).
"""
function gpu_dangling_ok(to_del::CuVector{Bool}, g::GPUACSet,
                          schema::SchemaInfo, backend)::Bool
    g_off = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        g_off[o] = cursor
        cursor += g.n_alloc[o]
    end

    for h in schema.homs
        src_type = schema.hom_dom[h]
        tgt_type = schema.hom_cod[h]
        n_src = g.n_alloc[src_type]
        n_tgt = g.n_alloc[tgt_type]
        (n_src == 0 || n_tgt == 0) && continue

        to_del_src = to_del[g_off[src_type]+1 : g_off[src_type]+n_src]
        to_del_tgt = to_del[g_off[tgt_type]+1 : g_off[tgt_type]+n_tgt]

        results = CUDA.zeros(Bool, n_src)
        dangling_check_per_hom_kernel!(backend, 256)(
            results, g.active[src_type], g.homs[h],
            to_del_src, to_del_tgt; ndrange=n_src)
        KernelAbstractions.synchronize(backend)
        Int(sum(results)) > 0 && return false
    end
    true
end

# ── Host-side helper: build the to_del mask ───────────────────────────────────

"""
    build_to_del_mask(match, cube, schema, g) -> CuVector{Bool}

Build a flat Boolean mask over all G-elements indicating which are deleted
by the rewrite.  The mask is computed on the host from the match and cube
(no GPU download needed), then uploaded.

Use `gpu_dangling_ok` afterwards to check the dangling condition on GPU.
"""
function build_to_del_mask(match::Vector{Int32},
                           cube::AdhesiveCube,
                           schema::SchemaInfo,
                           g::GPUACSet)::CuVector{Bool}
    total = sum(g.n_alloc[o] for o in schema.obj_types; init=0)
    to_del_host = zeros(Bool, total)

    g_offset = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        g_offset[o] = cursor
        cursor += g.n_alloc[o]
    end

    k_img_l = Set{Int}(Int(x) for x in cube.k_to_l)
    for (flat_l, l_type_idx) in enumerate(cube.l_types)
        flat_l ∈ k_img_l && continue
        g_elem = Int(match[flat_l])
        g_elem == 0 && continue
        o = schema.obj_types[l_type_idx]
        to_del_host[g_offset[o] + g_elem] = true
    end

    CuArray(to_del_host)
end
