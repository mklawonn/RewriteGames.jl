function _build_device_registry(rules_list, csps, cubes, schema, enc)
    n_rules = length(csps)
    n_obj   = length(schema.obj_types)
    n_hom   = length(schema.homs)
    n_attr  = length(schema.attrs)

    # ── 1. Flatten CSP Bytecodes ─────────────────────────────────────────────
    all_bytecodes = TCNBytecode[]
    csp_offsets   = zeros(Int32, n_rules)
    csp_lens      = zeros(Int32, n_rules)
    csp_n_vars    = zeros(Int32, n_rules)

    for i in 1:n_rules
        csp_offsets[i] = length(all_bytecodes)
        append!(all_bytecodes, csps[i].bytecodes)
        csp_lens[i]    = length(csps[i].bytecodes)
        csp_n_vars[i]  = csps[i].n_vars
    end

    # ── 2. Flatten RHS / Addition Data ───────────────────────────────────────
    rhs_n_add = zeros(Int32, n_obj, n_rules)
    all_hom_data = Int32[]
    rhs_hom_offsets = zeros(Int32, n_rules)
    
    all_attr_data = Int32[]
    rhs_attr_offsets = zeros(Int32, n_rules)

    for i in 1:n_rules
        rule = rules_list[i]
        cube = cubes[i]
        rhs_hom_offsets[i] = length(all_hom_data)
        rhs_attr_offsets[i] = length(all_attr_data)
        
        if rule === nothing; continue; end
        
        inner_rule = hasproperty(rule, :rule) ? rule.rule : rule
        if hasproperty(inner_rule, :rule) && hasmethod(left, Tuple{typeof(inner_rule.rule)})
            inner_rule = inner_rule.rule
        end
        R = codom(right(inner_rule))
        
        new_indices = added_r_indices(cube)
        for idx in new_indices
            tidx = cube.r_types[idx]
            rhs_n_add[tidx, i] += 1
        end
        
        r_flat_to_new = zeros(Int32, cube.n_r_elems)
        counts = zeros(Int32, n_obj)
        for idx in new_indices
            tidx = cube.r_types[idx]
            counts[tidx] += 1
            r_flat_to_new[idx] = counts[tidx]
        end
        
        r_flat_to_k = zeros(Int32, cube.n_r_elems)
        for k_idx in 1:cube.n_k_elems
            r_flat_to_k[cube.k_to_r[k_idx]] = k_idx
        end

        for idx in new_indices
            tidx = cube.r_types[idx]
            o = schema.obj_types[tidx]
            r_local = idx - cube.r_offset[o] + 1
            
            for h in schema.homs
                schema.hom_dom[h] == o || continue
                tgt_r_local = subpart(R, r_local, h)
                if tgt_r_local == 0
                    push!(all_hom_data, 0)
                    continue
                end
                
                tgt_type = schema.hom_cod[h]
                tgt_r_flat = cube.r_offset[tgt_type] + tgt_r_local - 1
                
                if r_flat_to_new[tgt_r_flat] > 0
                    push!(all_hom_data, r_flat_to_new[tgt_r_flat])
                elseif r_flat_to_k[tgt_r_flat] > 0
                    k_idx = r_flat_to_k[tgt_r_flat]
                    push!(all_hom_data, -Int32(cube.k_to_l[k_idx]))
                else
                    push!(all_hom_data, 0)
                end
            end
            
            for a in schema.attrs
                schema.attr_dom[a] == o || continue
                val = subpart(R, r_local, a)
                if val isa AttrVar
                    push!(all_attr_data, 0)
                else
                    push!(all_attr_data, encode_value(enc, a, val))
                end
            end
        end
    end

    DeviceRuleRegistry(
        CuArray(all_bytecodes),
        CuArray(csp_offsets),
        CuArray(csp_lens),
        CuArray(csp_n_vars),
        CuArray(vec(rhs_n_add)),
        CuArray(all_hom_data),
        CuArray(rhs_hom_offsets),
        CuArray(all_attr_data),
        CuArray(rhs_attr_offsets)
    )
end
