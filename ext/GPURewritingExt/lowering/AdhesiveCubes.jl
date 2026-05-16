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
end

"""
    precompute_adhesive_cube(rule, schema) -> AdhesiveCube

Extract the adhesive cube from a rewrite rule's span `K ← L → R` using the
rule's `left` (I → L) and `right` (I → R) morphisms, where I plays the role
of the interface/gluing object K.
"""
function precompute_adhesive_cube(rule, schema::SchemaInfo)::AdhesiveCube
    L    = codom(left(rule))
    R    = codom(right(rule))
    K    = dom(left(rule))      # interface / gluing object
    l_hom = left(rule)          # K → L
    r_hom = right(rule)         # K → R

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

    AdhesiveCube(n_l, n_r, n_k, k_to_l, k_to_r, l_types, r_types,
                 l_offset, r_offset, k_offset)
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
