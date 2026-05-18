"""
DPO pushout complement — GPU deletion phase.

For a match `m : L → G`, the elements of `L` that are not in the image of
`K → L` must be deleted from `G`.  In DPO mode this is simply a flag flip;
in SPO mode foreign-key chains are additionally cascaded.

The `dangling_check_kernel!` must be called (and its output inspected) before
`dpo_deletion_kernel!` to ensure the match satisfies the dangling condition.
"""

# ── Dangling condition check ──────────────────────────────────────────────────

@kernel function dangling_check_kernel!(
    active      :: AbstractVector{Bool},      # current element active flags (all types)
    hom_data    :: AbstractMatrix{Int32},     # [max_n × n_homs] foreign key table
    hom_offsets :: AbstractVector{Int32},     # start offset per morphism in flat layout
    hom_sizes   :: AbstractVector{Int32},     # element count per morphism domain type
    match       :: AbstractVector{Int32},     # flat match: L-variable → G-element
    to_del_mask :: AbstractVector{Bool},      # true if this L-element is being deleted
    n_del       :: Int,
    valid       :: AbstractVector{Bool}       # output: match satisfies dangling cond?
)
    inst = @index(Global, Linear)
    inst > 1 && return   # single-instance check

    ok = true
    # For each morphism h: dom(h) → cod(h), check that no surviving element in
    # dom(h) points to an element being deleted in cod(h).
    for h_idx in 1:length(hom_offsets)
        off  = Int(hom_offsets[h_idx])
        n_h  = Int(hom_sizes[h_idx])
        for src in 1:n_h
            active[off + src] || continue       # source is already deleted
            tgt = hom_data[src, h_idx]
            tgt == 0 && continue
            # Is tgt being deleted?
            # (to_del_mask is indexed over the flat G-element space)
            if active[tgt] && to_del_mask[tgt]
                ok = false
                break
            end
        end
        !ok && break
    end
    valid[1] = ok
end

# ── DPO deletion ──────────────────────────────────────────────────────────────

@kernel function dpo_deletion_kernel!(
    active   :: AbstractVector{Bool},   # element active flags (in-place update)
    to_del   :: AbstractVector{Bool}    # true = delete this G-element
)
    i = @index(Global, Linear)
    i > length(active) && return
    to_del[i] && (active[i] = false)
end

# ── SPO cascade deletion ──────────────────────────────────────────────────────

@kernel function spo_cascade_kernel!(
    active   :: AbstractVector{Bool},
    fk_col   :: AbstractVector{Int32},   # one foreign key column
    changed  :: AbstractVector{Bool}     # signal: any new deletions?
)
    i = @index(Global, Linear)
    i > length(fk_col) && return
    active[i] || return                  # source already deleted
    tgt = fk_col[i]
    tgt == 0 && return
    if !active[tgt]                      # points to deleted element
        active[i]  = false
        changed[1] = true
    end
end

# ── Host-side helper: compute the to_del mask on CPU, upload to GPU ───────────

"""
    build_to_del_mask(match, cube, schema, g) -> (CuVector{Bool}, Bool)

Build a flat Boolean mask over all G-elements indicating which are deleted by
the rewrite.  Also performs the dangling condition check on the host (fast
path for single-match execution).

Returns `(mask, dangling_ok)` where `dangling_ok` is false if the match
violates the dangling condition under DPO.
"""
function build_to_del_mask(match::Vector{Int32},
                           cube::AdhesiveCube,
                           schema::SchemaInfo,
                           g::GPUACSet)::Tuple{CuVector{Bool}, Bool}
    # Compute total G-element count (flat layout matching GPUACSet order)
    total = sum(g.n_alloc[o] for o in schema.obj_types; init=0)
    to_del_host = zeros(Bool, total)

    # Flat offset into G-element space per object type
    g_offset = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        g_offset[o] = cursor
        cursor += g.n_alloc[o]
    end

    # Mark deleted elements: L-elements not in image(K → L)
    k_img_l = Set{Int}(Int(x) for x in cube.k_to_l)
    for (flat_l, l_type_idx) in enumerate(cube.l_types)
        flat_l ∈ k_img_l && continue   # preserved by K
        g_elem = Int(match[flat_l])
        g_elem == 0 && continue
        o = schema.obj_types[l_type_idx]
        to_del_host[g_offset[o] + g_elem] = true
    end

    # Dangling check: for every active source pointing to a to-be-deleted target,
    # the match violates DPO.
    dangling_ok = true
    for h in schema.homs
        owner = schema.hom_dom[h]
        fk    = Array(g.homs[h])
        act   = Array(g.active[owner])
        off_o = g_offset[owner]
        off_c = g_offset[schema.hom_cod[h]]
        for (src_local, (alive, tgt)) in enumerate(zip(act, fk))
            alive || continue
            tgt == 0 && continue
            src_flat = off_o + src_local
            to_del_host[src_flat] && continue  # source is also deleted — ok
            tgt_flat = off_c + Int(tgt)
            if to_del_host[tgt_flat]
                dangling_ok = false
                break
            end
        end
        dangling_ok || break
    end

    CuArray(to_del_host), dangling_ok
end
