"""
    AdhesiveCube

Precomputed span `K ← L → R` structure for incremental match updates after a
DPO rewrite.  Stored as flat Int32 arrays for GPU use.

After a rewrite, the GPU:
1. Marks deleted elements (image of L \\ K in the world).
2. Adds new elements (image of R \\ K in the new world).
3. Uses the cube to efficiently extend surviving partial matches and discover
   new matches via the newly added Δ elements — mirroring `update_cache!` in
   `src/engine/match_cache.jl` but executed on-device.

Fields (all 1-based, 0 = no mapping):
- `n_l_elems`, `n_r_elems`, `n_k_elems`: element counts by type.
- `k_to_l`:  flat array [obj_type, k_idx] → l_idx (K → L map).
- `k_to_r`:  flat array [obj_type, k_idx] → r_idx (K → R map).
- `l_types`:  obj-type index for each L element (matches `SchemaInfo.obj_index`).
- `r_types`:  obj-type index for each R element.
"""
struct AdhesiveCube
    n_l_elems :: Int
    n_r_elems :: Int
    n_k_elems :: Int
    k_to_l    :: Vector{Int32}   # length n_k_elems
    k_to_r    :: Vector{Int32}   # length n_k_elems
    l_types   :: Vector{Int32}   # length n_l_elems: type index of each L elem
    r_types   :: Vector{Int32}   # length n_r_elems: type index of each R elem
    # Per-object block offsets into the flat arrays
    l_offset  :: Dict{Symbol, Int}
    r_offset  :: Dict{Symbol, Int}
    k_offset  :: Dict{Symbol, Int}
    # Precomputed FK / attr data for NEW (non-K) R elements, keyed by object type.
    # new_r_fk[o][h][j]  — for the j-th new element of type o and schema hom h:
    #   > 0  : index into new_r_elems[codom(h)], resolved at rewrite time to r_to_local
    #   < 0  : -(k_flat_index), resolved at rewrite time to match[k_to_l[-val]]
    #   0    : FK unset / null
    # new_r_attr[o][a][j] — encoded Int32 attribute value (0 = AttrVar / unset)
    new_r_fk   :: Dict{Symbol, Dict{Symbol, Vector{Int32}}}
    new_r_attr :: Dict{Symbol, Dict{Symbol, Vector{Int32}}}
    # Precomputed FK / attr updates for PRESERVED (K) elements that differ in R.
    # k_attr_pre[a] = [(k_flat, encoded_val), ...] — skip AttrVars and encoded==0
    # k_fk_pre[h]   = [(k_flat, enc_val), ...] where enc_val encodes the target:
    #   > 0  : index into new_r_elems[codom(h)], resolved to r_to_local at rewrite time
    #   < 0  : -(k_flat_index of target), resolved to match[k_to_l[-enc_val]]
    k_attr_pre :: Dict{Symbol, Vector{Tuple{Int32, Int32}}}
    k_fk_pre   :: Dict{Symbol, Vector{Tuple{Int32, Int32}}}
    # Static counts for n_live bookkeeping (no GPU download needed after rewrite).
    # del_per_type[o] = number of L\K elements of type o (always deleted when rule fires).
    # add_per_type[o] = number of R\K elements of type o (always added when rule fires).
    del_per_type :: Dict{Symbol, Int}
    add_per_type :: Dict{Symbol, Int}
    # Flat L-indices and type indices of deleted elements (precomputed for build_to_del_kernel!).
    del_l_flats :: Vector{Int32}   # flat L indices of L\K elements
    del_l_types :: Vector{Int32}   # schema type index of each deleted element
end

"""
    precompute_adhesive_cube(rule, schema) -> AdhesiveCube

Extract the adhesive cube from a rewrite rule's span `K ← L → R` using the
rule's `left` (I → L) and `right` (I → R) morphisms, where I plays the role
of the interface/gluing object K.
"""

function precompute_adhesive_cube(rule, schema::SchemaInfo;
                                   enc::AttributeEncoder = AttributeEncoder())::AdhesiveCube
    empty_fk     = Dict{Symbol, Dict{Symbol, Vector{Int32}}}()
    empty_attr   = Dict{Symbol, Dict{Symbol, Vector{Int32}}}()
    empty_kattr  = Dict{Symbol, Vector{Tuple{Int32, Int32}}}()
    empty_kfk    = Dict{Symbol, Vector{Tuple{Int32, Int32}}}()

    # Handle the Nothing case (for agent interfaces/queries)
    if rule === nothing
        return AdhesiveCube(0, 0, 0, Int32[], Int32[], Int32[], Int32[],
                            Dict(o => 1 for o in schema.obj_types),
                            Dict(o => 1 for o in schema.obj_types),
                            Dict(o => 1 for o in schema.obj_types),
                            empty_fk, empty_attr, empty_kattr, empty_kfk,
                            Dict{Symbol,Int}(), Dict{Symbol,Int}(),
                            Int32[], Int32[])
    end

    # Handle mock rules with L but no left/right
    if !hasmethod(left, Tuple{typeof(rule)}) && hasproperty(rule, :L)
        L = rule.L
        n_l = sum(nparts(L, o) for o in schema.obj_types)
        l_offset = Dict{Symbol,Int}()
        curr = 1
        l_types = Int32[]
        for o in schema.obj_types
            l_offset[o] = curr
            n = nparts(L, o)
            curr += n
            append!(l_types, fill(Int32(schema.obj_index[o]), n))
        end
        return AdhesiveCube(n_l, 0, 0, Int32[], Int32[], l_types, Int32[],
                            l_offset, Dict(o => 1 for o in schema.obj_types),
                            Dict(o => 1 for o in schema.obj_types),
                            empty_fk, empty_attr, empty_kattr, empty_kfk,
                            Dict{Symbol,Int}(), Dict{Symbol,Int}(),
                            Int32[], Int32[])
    end

    # Extract underlying AlgebraicRewriting rule if we were passed a box
    inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
    if hasproperty(inner_rule, :rule) && hasmethod(left, Tuple{typeof(inner_rule.rule)})
        inner_rule = inner_rule.rule
    end

    L     = codom(left(inner_rule))
    R     = codom(right(inner_rule))
    K     = dom(left(inner_rule))
    l_hom = left(inner_rule)
    r_hom = right(inner_rule)

    # Compute block offsets and total element counts
    l_offset = Dict{Symbol,Int}()
    r_offset = Dict{Symbol,Int}()
    k_offset = Dict{Symbol,Int}()
    n_l = 0; n_r = 0; n_k = 0
    for o in schema.obj_types
        l_offset[o] = n_l + 1
        r_offset[o] = n_r + 1
        k_offset[o] = n_k + 1
        n_l += nparts(L, o)
        n_r += nparts(R, o)
        n_k += nparts(K, o)
    end

    k_to_l = zeros(Int32, n_k)
    k_to_r = zeros(Int32, n_k)
    for o in schema.obj_types
        for k in parts(K, o)
            flat_k = k_offset[o] + (k - 1)
            k_to_l[flat_k] = Int32(l_offset[o] + (l_hom[o](k) - 1))
            k_to_r[flat_k] = Int32(r_offset[o] + (r_hom[o](k) - 1))
        end
    end

    l_types = Int32[]
    r_types = Int32[]
    for o in schema.obj_types
        tidx = Int32(schema.obj_index[o])
        append!(l_types, fill(tidx, nparts(L, o)))
        append!(r_types, fill(tidx, nparts(R, o)))
    end

    # ── Precompute FK and attr data for new (non-K) R elements ───────────────

    # K image in R (flat R indices that are K-preserved)
    k_img_r = Dict{Symbol, Set{Int}}(o => Set{Int}() for o in schema.obj_types)
    for k in 1:n_k
        r_flat = Int(k_to_r[k])
        r_flat == 0 && continue
        o = schema.obj_types[Int(r_types[r_flat])]
        push!(k_img_r[o], r_flat)
    end
    # Reverse map: flat R index → K element index
    r_flat_to_k = Dict{Int, Int}()
    for k in 1:n_k
        r_flat = Int(k_to_r[k])
        r_flat > 0 && (r_flat_to_k[r_flat] = k)
    end

    # Per-type ordered list of new R elements (R-local indices, 1-based)
    new_r_elems = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        nr  = nparts(R, o)
        off = r_offset[o]
        new_r_elems[o] = [i for i in 1:nr if (off + i - 1) ∉ k_img_r[o]]
    end

    new_r_fk   = Dict{Symbol, Dict{Symbol, Vector{Int32}}}()
    new_r_attr = Dict{Symbol, Dict{Symbol, Vector{Int32}}}()

    for o in schema.obj_types
        elems = new_r_elems[o]
        isempty(elems) && continue
        n_new = length(elems)

        fk_o   = Dict{Symbol, Vector{Int32}}()
        attr_o = Dict{Symbol, Vector{Int32}}()

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            tgt_type = schema.hom_cod[h]
            vals = zeros(Int32, n_new)
            for (j, r_local) in enumerate(elems)
                tgt_r = subpart(R, r_local, h)
                tgt_r == 0 && continue
                tgt_r_flat = r_offset[tgt_type] + tgt_r - 1
                if haskey(r_flat_to_k, tgt_r_flat)
                    vals[j] = -Int32(r_flat_to_k[tgt_r_flat])
                else
                    new_idx = findfirst(==(tgt_r), new_r_elems[tgt_type])
                    new_idx !== nothing && (vals[j] = Int32(new_idx))
                end
            end
            fk_o[h] = vals
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            vals = zeros(Int32, n_new)
            for (j, r_local) in enumerate(elems)
                raw = subpart(R, r_local, a)
                raw isa AttrVar && continue
                vals[j] = encode_value(enc, a, raw)
            end
            attr_o[a] = vals
        end

        new_r_fk[o]   = fk_o
        new_r_attr[o] = attr_o
    end

    # ── Precompute attr / FK updates for PRESERVED (K) elements ──────────────

    k_attr_pre = Dict{Symbol, Vector{Tuple{Int32, Int32}}}()
    k_fk_pre   = Dict{Symbol, Vector{Tuple{Int32, Int32}}}()

    for k in 1:n_k
        r_flat = Int(k_to_r[k])
        r_flat == 0 && continue
        o = schema.obj_types[Int(r_types[r_flat])]
        r_local = r_flat - r_offset[o] + 1

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            raw = subpart(R, r_local, a)
            raw isa AttrVar && continue
            encoded = encode_value(enc, a, raw)
            encoded == Int32(0) && continue
            push!(get!(k_attr_pre, a, Tuple{Int32, Int32}[]), (Int32(k), encoded))
        end

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            tgt_type    = schema.hom_cod[h]
            tgt_r_local = subpart(R, r_local, h)
            tgt_r_local == 0 && continue
            tgt_r_flat  = r_offset[tgt_type] + tgt_r_local - 1
            enc_val = if haskey(r_flat_to_k, tgt_r_flat)
                -Int32(r_flat_to_k[tgt_r_flat])
            else
                new_idx = findfirst(==(tgt_r_local), new_r_elems[tgt_type])
                new_idx !== nothing ? Int32(new_idx) : Int32(0)
            end
            enc_val == Int32(0) && continue
            push!(get!(k_fk_pre, h, Tuple{Int32, Int32}[]), (Int32(k), enc_val))
        end
    end

    # ── Precompute deletion metadata for GPU build_to_del_kernel! ────────────
    k_img_l = Set{Int}(Int(i) for i in k_to_l if i != 0)
    del_per_type = Dict{Symbol, Int}()
    del_l_flats  = Int32[]
    del_l_types  = Int32[]
    for o in schema.obj_types
        cnt = 0
        for l in 1:nparts(L, o)
            l_flat = l_offset[o] + l - 1
            if l_flat ∉ k_img_l
                cnt += 1
                push!(del_l_flats, Int32(l_flat))
                push!(del_l_types, Int32(schema.obj_index[o]))
            end
        end
        cnt > 0 && (del_per_type[o] = cnt)
    end

    add_per_type = Dict{Symbol, Int}()
    k_img_r_flat = Set{Int}(keys(r_flat_to_k))
    for o in schema.obj_types
        nr  = nparts(R, o)
        off = r_offset[o]
        cnt = count(r -> (off + r - 1) ∉ k_img_r_flat, 1:nr)
        cnt > 0 && (add_per_type[o] = cnt)
    end

    AdhesiveCube(n_l, n_r, n_k, k_to_l, k_to_r, l_types, r_types,
                 l_offset, r_offset, k_offset, new_r_fk, new_r_attr,
                 k_attr_pre, k_fk_pre,
                 del_per_type, add_per_type, del_l_flats, del_l_types)
end

"""
    GPUAdhesiveCube

GPU-resident copy of the static parts of an `AdhesiveCube`.  Built once at
`compile_schedule` time; reused across every rewrite step.

On non-CUDA systems all fields hold CPU arrays (identical to the CPU path).
"""
struct GPUAdhesiveCube
    k_to_l_gpu      :: Any   # CuVector{Int32}
    new_r_fk_gpu    :: Dict{Symbol, Dict{Symbol, Any}}   # [o][h] → CuVector{Int32}
    new_r_attr_gpu  :: Dict{Symbol, Dict{Symbol, Any}}   # [o][a] → CuVector{Int32}
    # _update_preserved! attr: precomputed l_flat indices and encoded values per attr
    k_attr_l_gpu    :: Dict{Symbol, Any}   # [a] → CuVector{Int32} of l_flat indices
    k_attr_v_gpu    :: Dict{Symbol, Any}   # [a] → CuVector{Int32} of encoded values
    # _update_preserved! FK: precomputed l_flat source indices and enc_vals per hom
    k_fk_l_gpu      :: Dict{Symbol, Any}   # [h] → CuVector{Int32} of l_flat source indices
    k_fk_enc_gpu    :: Dict{Symbol, Any}   # [h] → CuVector{Int32} of encoded target values
    # For GPU build_to_del_kernel!: flat L-indices and type indices of deleted elements
    del_l_flats_gpu :: Any   # CuVector{Int32}
    del_l_types_gpu :: Any   # CuVector{Int32}
    # Static per-type counts for n_live bookkeeping (no GPU download needed)
    del_per_type    :: Dict{Symbol, Int}
    add_per_type    :: Dict{Symbol, Int}
end

"""
    gpu_upload_cube(cube) -> GPUAdhesiveCube

Upload the static parts of `cube` to GPU memory once.  The `k_to_l` lookup
and all precomputed FK / attr arrays become GPU-resident so that rewrite
kernels never touch CPU-side cube data at rewrite time.
"""
function gpu_upload_cube(cube::AdhesiveCube)::GPUAdhesiveCube
    _up(v) = CUDA.functional() ? CuArray(v) : copy(v)

    k_to_l_gpu = _up(cube.k_to_l)

    new_r_fk_gpu = Dict{Symbol, Dict{Symbol, Any}}()
    for (o, fk_o) in cube.new_r_fk
        new_r_fk_gpu[o] = Dict{Symbol, Any}(h => _up(v) for (h, v) in fk_o)
    end

    new_r_attr_gpu = Dict{Symbol, Dict{Symbol, Any}}()
    for (o, attr_o) in cube.new_r_attr
        new_r_attr_gpu[o] = Dict{Symbol, Any}(a => _up(v) for (a, v) in attr_o)
    end

    k_attr_l_gpu = Dict{Symbol, Any}()
    k_attr_v_gpu = Dict{Symbol, Any}()
    for (a, pairs) in cube.k_attr_pre
        l_flats = Int32[Int32(cube.k_to_l[k_flat]) for (k_flat, _) in pairs]
        vals    = Int32[v for (_, v) in pairs]
        k_attr_l_gpu[a] = _up(l_flats)
        k_attr_v_gpu[a] = _up(vals)
    end

    k_fk_l_gpu   = Dict{Symbol, Any}()
    k_fk_enc_gpu = Dict{Symbol, Any}()
    for (h, pairs) in cube.k_fk_pre
        l_flats  = Int32[Int32(cube.k_to_l[k_flat]) for (k_flat, _) in pairs]
        enc_vals = Int32[ev for (_, ev) in pairs]
        k_fk_l_gpu[h]   = _up(l_flats)
        k_fk_enc_gpu[h] = _up(enc_vals)
    end

    del_l_flats_gpu = _up(cube.del_l_flats)
    del_l_types_gpu = _up(cube.del_l_types)

    GPUAdhesiveCube(k_to_l_gpu, new_r_fk_gpu, new_r_attr_gpu,
                    k_attr_l_gpu, k_attr_v_gpu, k_fk_l_gpu, k_fk_enc_gpu,
                    del_l_flats_gpu, del_l_types_gpu,
                    copy(cube.del_per_type), copy(cube.add_per_type))
end

"""
    deleted_l_indices(cube) -> Vector{Int}

Return flat L-element indices that are NOT in the image of K → L, i.e. the
elements that are deleted by the rewrite.
"""
function deleted_l_indices(cube::AdhesiveCube)::Vector{Int}
    img = Set{Int}(Int(i) for i in cube.k_to_l)
    return [i for i in 1:cube.n_l_elems if i ∉ img]
end

"""
    added_r_indices(cube) -> Vector{Int}

Return flat R-element indices that are NOT in the image of K → R, i.e. the
elements freshly introduced by the rewrite.
"""
function added_r_indices(cube::AdhesiveCube)::Vector{Int}
    img = Set{Int}(Int(i) for i in cube.k_to_r)
    return [i for i in 1:cube.n_r_elems if i ∉ img]
end
