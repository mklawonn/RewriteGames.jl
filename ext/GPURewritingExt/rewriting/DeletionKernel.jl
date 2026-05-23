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

# Single-kernel dangling check: one thread per source element, shared violation flag.
# All morphisms are checked in successive launches sharing the same `violation` buffer.
@kernel function dangling_check_all_homs_kernel!(
    violation  :: AbstractVector{Bool},
    active_src :: AbstractVector{Bool},
    fk         :: AbstractVector{Int32},
    to_del_src :: AbstractVector{Bool},
    to_del_tgt :: AbstractVector{Bool},
    src_n      :: Int32,
    tgt_n      :: Int32,
)
    i = @index(Global, Linear)
    if i <= Int(src_n) && active_src[i] && !to_del_src[i]
        tgt = Int(fk[i])
        if tgt != 0 && tgt <= Int(tgt_n) && to_del_tgt[tgt]
            Atomix.@atomic violation[1] |= true
        end
    end
end

"""
    gpu_dangling_ok(to_del, g, schema, backend; buf_violation) -> Bool

Check the DPO dangling condition entirely on GPU.  Launches one kernel per
schema morphism with a shared `violation` flag; a single synchronize and a
single scalar download replace the previous N-per-morphism version.

`buf_violation` is a pre-allocated length-1 `CuVector{Bool}` from the
`GPUScratchBuffers`; pass `nothing` to fall back to a local allocation.
"""
function gpu_dangling_ok(to_del, g::GPUACSet,
                          schema::SchemaInfo, backend;
                          buf_violation = nothing)::Bool
    g_off = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        g_off[o] = cursor
        cursor += g.n_alloc[o]
    end

    viol = buf_violation !== nothing ? buf_violation :
                                      KernelAbstractions.allocate(backend, Bool, 1)
    KernelAbstractions.fill!(viol, false)

    for h in schema.homs
        src_type = schema.hom_dom[h]
        tgt_type = schema.hom_cod[h]
        n_src = g.n_alloc[src_type]
        n_tgt = g.n_alloc[tgt_type]
        (n_src == 0 || n_tgt == 0) && continue

        o_src = g_off[src_type]
        o_tgt = g_off[tgt_type]

        dangling_check_all_homs_kernel!(backend, 256)(
            viol,
            @view(g.active[src_type][1:n_src]),
            @view(g.homs[h][1:n_src]),
            @view(to_del[o_src+1 : o_src+n_src]),
            @view(to_del[o_tgt+1 : o_tgt+n_tgt]),
            Int32(n_src), Int32(n_tgt);
            ndrange = n_src)
    end
    KernelAbstractions.synchronize(backend)
    !Array(viol)[1]
end

# ── Single-sync pipeline kernels ─────────────────────────────────────────────

"""
GPU kernel for building the to_del mask without a CPU round-trip.
`del_l_flats[i]` and `del_l_types[i]` are precomputed in the AdhesiveCube;
`g_type_offs[t]` is the 0-based offset of type t in the flat `buf_to_del`.
Guarded by `buf_fired[1]` so it no-ops when no match was found.
"""
@kernel function build_to_del_kernel!(
    buf_to_del  :: AbstractVector{Bool},
    buf_fired   :: AbstractVector{Int32},
    buf_match   :: AbstractVector{Int32},
    del_l_flats :: AbstractVector{Int32},
    del_l_types :: AbstractVector{Int32},
    g_type_offs :: AbstractVector{Int32},
    n_del       :: Int32,
)
    i = @index(Global, Linear)
    if i <= Int(n_del) && buf_fired[1] != Int32(0)
        l_flat = Int(del_l_flats[i])
        g_elem = Int(buf_match[l_flat])
        if g_elem > 0
            t   = Int(del_l_types[i])
            off = Int(g_type_offs[t])
            buf_to_del[off + g_elem] = true
        end
    end
end

"""
Dangling-condition check that writes its result directly into `buf_fired`
(instead of a separate violation buffer).  When a dangling edge is found,
sets `buf_fired[1] = 0` via atomic AND, marking the rule as non-firing.
"""
@kernel function dangling_check_fired_kernel!(
    buf_fired  :: AbstractVector{Int32},
    active_src :: AbstractVector{Bool},
    fk         :: AbstractVector{Int32},
    to_del_src :: AbstractVector{Bool},
    to_del_tgt :: AbstractVector{Bool},
    src_n      :: Int32,
    tgt_n      :: Int32,
)
    i = @index(Global, Linear)
    if i <= Int(src_n) && buf_fired[1] != Int32(0) && active_src[i] && !to_del_src[i]
        tgt = Int(fk[i])
        if tgt != 0 && tgt <= Int(tgt_n) && to_del_tgt[tgt]
            CUDA.atomic_and!(pointer(buf_fired, 1), Int32(0))
        end
    end
end

"""
Guarded DPO deletion kernel: only clears active flags when `buf_fired[1] != 0`.
"""
@kernel function dpo_deletion_kernel_g!(
    active    :: AbstractVector{Bool},
    to_del    :: AbstractVector{Bool},
    buf_fired :: AbstractVector{Int32},
)
    i = @index(Global, Linear)
    if i <= length(active) && buf_fired[1] != Int32(0) && to_del[i]
        active[i] = false
    end
end

# ── Host-side helper: build the to_del mask ───────────────────────────────────

"""
    build_to_del_mask(match, cube, schema, g; buf_to_del) -> AbstractVector{Bool}

Build a flat Boolean mask over all G-elements indicating which are deleted
by the rewrite.  The mask is computed on the host from the match and cube
(no GPU download needed).

When `buf_to_del` is provided (pre-allocated from `GPUScratchBuffers`),
the mask is written there (via a host-to-device copyto!) rather than
allocating a fresh `CuArray`.  Falls back to `CuArray(to_del_host)` otherwise.
"""
function build_to_del_mask(match::Vector{Int32},
                           cube::AdhesiveCube,
                           schema::SchemaInfo,
                           g::GPUACSet;
                           buf_to_del = nothing)
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

    if buf_to_del !== nothing
        if length(buf_to_del) < total
            # Grow the pre-allocated buffer (rare after initial sizing)
            # Can't update the caller's field here; fall back to fresh allocation
            return CUDA.functional() ? CuArray(to_del_host) : to_del_host
        end
        buf = @view buf_to_del[1:total]
        copyto!(buf, to_del_host)
        return buf
    end
    CUDA.functional() ? CuArray(to_del_host) : to_del_host
end
