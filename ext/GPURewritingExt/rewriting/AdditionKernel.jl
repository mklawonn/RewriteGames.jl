"""
DPO pushout — GPU addition phase (Prealloc-Combine strategy).

New elements introduced by `R \\ K` are added to the GPU ACSet using a
two-pass strategy:
  1. **Prefix-sum prealloc**: count new elements per type, compute cumulative
     offsets to find where each new element will live.
  2. **Parallel scatter**: threads concurrently write FK and attribute columns
     for the new elements into the preallocated slots.
"""

function apply_pushout!(g::GPUACSet,
                        match::Vector{Int32},
                        cube::AdhesiveCube,
                        rule,
                        schema::SchemaInfo,
                        enc::AttributeEncoder,
                        backend)::Dict{Symbol, Vector{Int32}}
    if rule === nothing
        return Dict(o => Int32[] for o in schema.obj_types)
    end

    R = codom(right(rule))
    K = dom(right(rule))
    r_hom = right(rule)

    # 1. Identify which elements are new
    k_img_r = Dict{Symbol, Set{Int}}()
    for o in schema.obj_types; k_img_r[o] = Set{Int}(); end
    for k in 1:cube.n_k_elems
        r_idx = Int(cube.k_to_r[k])
        r_idx == 0 && continue
        o = schema.obj_types[Int(cube.r_types[r_idx])]
        push!(k_img_r[o], r_idx)
    end

    new_r_elems = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        nr = nparts(R, o)
        off = cube.r_offset[o]
        new_r_elems[o] = [i for i in 1:nr if Int(off + i - 1) ∉ k_img_r[o]]
    end

    # 2. Assign slots and grow arrays
    r_to_local = Dict{Symbol, Vector{Int32}}()
    
    for o in schema.obj_types
        new_elems = new_r_elems[o]
        if isempty(new_elems)
            r_to_local[o] = Int32[]
            continue
        end

        n_cur   = g.n_alloc[o]
        n_add   = length(new_elems)
        n_total = n_cur + n_add
        globals = Int32[Int32(n_cur + j) for j in 1:n_add]
        r_to_local[o] = globals

        # Grow arrays
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
        
        g.n_alloc[o] = n_total
        g.n_live[o][] += n_add
    end

    # 3. Populate new elements (Host-side for simplicity, then upload)
    g_offset = _global_offset(g, schema)
    for o in schema.obj_types
        new_elems = new_r_elems[o]
        isempty(new_elems) && continue
        
        globals = r_to_local[o]
        
        # Update active flags
        host_active = Array(g.active[o])
        for gidx in globals; host_active[gidx] = true; end
        g.active[o] = CuArray(host_active)
        
        # Update FKs
        for h in schema.homs
            schema.hom_dom[h] == o || continue
            host_fk = Array(g.homs[h])
            tgt_type = schema.hom_cod[h]
            off_r_tgt = cube.r_offset[tgt_type]
            
            for (j, r_elem) in enumerate(new_elems)
                gidx = Int(globals[j])
                tgt_r = subpart(R, r_elem, h)
                tgt_r == 0 && continue
                
                tgt_r_flat = Int(off_r_tgt + tgt_r - 1)
                
                if tgt_r_flat ∈ k_img_r[tgt_type]
                    # Preserved element: find in world
                    found_k = 0
                    for k in 1:cube.n_k_elems
                        if Int(cube.k_to_r[k]) == tgt_r_flat
                            found_k = k; break
                        end
                    end
                    if found_k != 0
                        flat_g = Int(match[Int(cube.k_to_l[found_k])])
                        host_fk[gidx] = Int32(flat_g - g_offset[tgt_type])
                    end
                else
                    # New element: find in r_to_local
                    new_idx = findfirst(==(tgt_r), new_r_elems[tgt_type])
                    if new_idx !== nothing
                        host_fk[gidx] = r_to_local[tgt_type][new_idx]
                    end
                end
            end
            g.homs[h] = CuArray(host_fk)
        end
        
        # Update attributes
        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            host_av = Array(g.attrs[a])
            for (j, r_elem) in enumerate(new_elems)
                gidx = Int(globals[j])
                raw = subpart(R, r_elem, a)
                raw isa AttrVar && continue
                host_av[gidx] = encode_value(enc, a, raw)
            end
            g.attrs[a] = CuArray(host_av)
        end
    end

    return r_to_local
end
