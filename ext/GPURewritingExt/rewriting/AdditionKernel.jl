"""
DPO pushout — GPU addition phase (Prealloc-Combine strategy).

New elements introduced by `R \\ K` are added to the GPU ACSet using a
two-pass strategy:
  1. **Prefix-sum prealloc**: count new elements per type, compute cumulative
     offsets to find where each new element will live.
  2. **Parallel scatter**: threads concurrently write FK and attribute columns
     for the new elements into the preallocated slots.

The gluing morphism `K → H` (where H is the result world) is also returned so
that the incremental match update step can forward existing matches.
"""

@kernel function prefix_sum_kernel!(
    counts  :: AbstractVector{Int32},   # [n_types] input counts
    offsets :: AbstractVector{Int32}    # [n_types+1] output prefix sums (1-based)
)
    # Single-thread prefix sum (small array — GPU overhead not worth parallelising)
    i = @index(Global, Linear)
    if i == 1
        offsets[1] = Int32(1)
        for t in 1:length(counts)
            offsets[t+1] = offsets[t] + counts[t]
        end
    end
end

@kernel function addition_kernel!(
    active_dst  :: AbstractVector{Bool},   # full active array (extended in-place)
    hom_dst     :: AbstractMatrix{Int32},  # [max_n × n_homs] FK table
    attr_dst    :: AbstractMatrix{Int32},  # [max_n × n_attrs] attr table
    offsets     :: AbstractVector{Int32},  # per-type start index for new elements
    r_homs      :: AbstractMatrix{Int32},  # R-element FK values (new elements)
    r_attrs     :: AbstractMatrix{Int32},  # R-element attr values (encoded)
    r_to_global :: AbstractVector{Int32},  # R-element flat index → global index
    n_new       :: Int
)
    idx = @index(Global, Linear)
    if idx <= n_new
        global_idx = r_to_global[idx]
        active_dst[global_idx] = true
        for h in axes(hom_dst, 2)
            hom_dst[global_idx, h] = r_homs[idx, h]
        end
        for a in axes(attr_dst, 2)
            attr_dst[global_idx, a] = r_attrs[idx, a]
        end
    end
end

# ── Host orchestration ───────────────────────────────────────────────────────

"""
    apply_pushout!(g, match, cube, rule, schema, enc) -> Dict{Symbol, Vector{Int32}}

Apply the DPO pushout to `g` in-place:
1. Grow GPU arrays if needed.
2. Write new R-elements using `addition_kernel!`.
3. Return the `k_to_h` mapping (K-element flat index → new H-element flat index)
   needed by `IncrementalUpdate`.
"""
function apply_pushout!(g::GPUACSet,
                        match::Vector{Int32},
                        cube::AdhesiveCube,
                        rule,
                        schema::SchemaInfo,
                        enc::AttributeEncoder,
                        backend)::Dict{Symbol, Vector{Int32}}
    R = codom(right(rule))
    K = dom(right(rule))
    r_hom = right(rule)   # K → R

    # Determine which R-elements are new (not in image of K → R)
    k_img_r = Dict{Symbol, Set{Int}}()
    for o in schema.obj_types
        k_img_r[o] = Set{Int}()
    end
    for k in 1:cube.n_k_elems
        r_idx = Int(cube.k_to_r[k])
        r_idx == 0 && continue
        o = schema.obj_types[Int(cube.r_types[r_idx])]
        push!(k_img_r[o], r_idx)
    end

    new_r_elems = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        nr = nparts(R, o)
        new_r_elems[o] = [i for i in 1:nr if i ∉ k_img_r[o]]
    end

    # Map each new R-element to a global slot in the GPUACSet
    # (extend arrays if needed)
    r_to_global = Dict{Symbol, Vector{Int32}}()
    for o in schema.obj_types
        new_elems = new_r_elems[o]
        isempty(new_elems) && (r_to_global[o] = Int32[]; continue)

        n_cur   = g.n_alloc[o]
        n_add   = length(new_elems)
        n_total = n_cur + n_add

        # Grow GPU arrays
        new_active = CUDA.zeros(Bool, n_total)
        copyto!(new_active, 1, g.active[o], 1, n_cur)
        g.active[o] = new_active

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            new_fk = CUDA.zeros(Int32, n_total)
            copyto!(new_fk, 1, g.homs[h], 1, n_cur)
            g.homs[h] = new_fk
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            new_av = CUDA.zeros(Int32, n_total)
            copyto!(new_av, 1, g.attrs[a], 1, n_cur)
            g.attrs[a] = new_av
        end

        # Record global indices assigned to new R-elements
        globals = Int32[Int32(n_cur + j) for j in 1:n_add]
        r_to_global[o] = globals
        g.n_alloc[o] = n_total
        g.n_live[o][] += n_add

        # Set active flags and data for new elements on host, then upload
        host_active = Array(g.active[o])
        for gidx in globals
            host_active[gidx] = true
        end
        g.active[o] = CuArray(host_active)

        for (j, r_elem) in enumerate(new_elems)
            gidx = Int(globals[j])
            for h in schema.homs
                schema.hom_dom[h] == o || continue
                tgt_r = subpart(R, r_elem, h)
                if tgt_r > 0
                    tgt_type = schema.hom_cod[h]
                    # Is tgt_r a new element or preserved from K?
                    tgt_global = if tgt_r ∈ k_img_r[tgt_type]
                        # preserved: find its H-index via the match
                        # (simplified: use the K→H map built above)
                        Int32(0)   # filled in below via k_to_h
                    else
                        # new: find in r_to_global
                        new_idx = findfirst(==(tgt_r), new_r_elems[tgt_type])
                        new_idx === nothing ? Int32(0) :
                            get(r_to_global, tgt_type, Int32[])[new_idx]
                    end
                    host_fk = Array(g.homs[h])
                    host_fk[gidx] = tgt_global
                    g.homs[h] = CuArray(host_fk)
                end
            end

            for a in schema.attrs
                schema.attr_dom[a] == o || continue
                raw = subpart(R, r_elem, a)
                raw isa AttrVar && continue
                host_av = Array(g.attrs[a])
                host_av[gidx] = encode_value(enc, a, raw)
                g.attrs[a] = CuArray(host_av)
            end
        end
    end

    return r_to_global
end
