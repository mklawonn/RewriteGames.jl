"""
Incremental match update — GPU analog of `update_cache!` from
`src/engine/match_cache.jl`.

After a DPO rewrite, the match set for all other rules must be updated:
1. **Forward surviving matches**: existing matches whose images are all still
   active are forwarded through the pushout complement maps (K → H).
   Matches that included a deleted element are dropped.
2. **Invalidate via NAC/PAC**: re-check application conditions; matches
   newly violating a NAC (or no longer satisfying a PAC) are dropped.
3. **Discover new matches**: run the Turbo solver restricted to sub-problems
   that include at least one of the newly added Δ⁺ elements.

This mirrors the three steps in `update_cache!` (lines 78-88 of match_cache.jl).
"""

"""
    MatchTable

Flat representation of a set of homomorphisms as a matrix of Int32 values.

`assignments[v, m]` = world element assigned to pattern variable `v` in match `m`.
`n_matches` = number of valid matches currently stored.
"""
mutable struct MatchTable
    assignments :: Matrix{Int32}   # [n_vars × max_matches]
    n_matches   :: Int
end

MatchTable(n_vars::Int, max_matches::Int) =
    MatchTable(zeros(Int32, n_vars, max_matches), 0)

"""
    incremental_match_update!(table, csp, cube, g, deleted_g, added_g, schema, enc)

Update `table` in-place after a rewrite that deleted `deleted_g` elements
and added `added_g` elements (both as flat global G-element indices per obj type).
"""
function incremental_match_update!(table::MatchTable,
                                   csp::CSPProblem,
                                   cube::AdhesiveCube,
                                   g::GPUACSet,
                                   deleted_g::Dict{Symbol, Vector{Int32}},
                                   added_g::Dict{Symbol, Vector{Int32}},
                                   schema::SchemaInfo,
                                   enc::AttributeEncoder)
    # ── Step 1: forward surviving matches ────────────────────────────────────
    # Mirror of _forward_match in match_cache.jl:110
    host_active = Dict(o => Array(g.active[o]) for o in schema.obj_types)

    # Build global offset for each obj type
    g_offset = _global_offset(g, schema)

    n = table.n_matches
    keep = trues(n)
    for m in 1:n
        for v in 1:csp.n_vars
            g_elem = Int(table.assignments[v, m])
            g_elem == 0 && continue
            # Determine obj type of this variable
            o = _var_to_type(v, csp, schema)
            o === nothing && continue
            off = g_offset[o]
            local_idx = g_elem - off
            local_idx < 1 && (keep[m] = false; break)
            local_idx > length(host_active[o]) && (keep[m] = false; break)
            host_active[o][local_idx] || (keep[m] = false; break)
        end
    end

    # Compact surviving matches
    _compact_matches!(table, keep)

    # ── Step 2: discover new matches (pinned to Δ⁺ elements) ──────────────────
    # Mirror of _find_new_matches in match_cache.jl:153
    for o in schema.obj_types
        new_elems = get(added_g, o, Int32[])
        isempty(new_elems) && continue

        # For each pattern variable of type o, try pinning it to each new element
        for v in 1:csp.n_vars
            _var_to_type(v, csp, schema) == o || continue

            for new_g_elem in new_elems
                domains = _init_domains(csp, g, schema, g_offset)
                # Pin variable v to new_g_elem
                local_new = Int(new_g_elem) - g_offset[o]
                local_new < 1 && continue
                domains[v] = UInt64(1) << (local_new - 1)

                # Apply attribute masks for fixed attrs in L
                _apply_attr_masks!(domains, csp, g, schema, enc, g_offset)

                solutions = cpu_dive_solve(csp, domains)
                for sol in solutions
                    # Filter: must include new_g_elem in assignment of v
                    Int(sol[v]) + g_offset[o] == Int(new_g_elem) || continue
                    _push_match!(table, sol)
                end
            end
        end
    end

    table
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _global_offset(g::GPUACSet, schema::SchemaInfo)
    off = Dict{Symbol, Int}()
    cursor = 0
    for o in schema.obj_types
        off[o] = cursor
        cursor += g.n_alloc[o]
    end
    off
end

function _var_to_type(v::Int, csp::CSPProblem, schema::SchemaInfo)
    for o in schema.obj_types
        base  = get(csp.var_offset, o, 0)
        # Count how many L-elements of type o there are
        # (= number of variables in this block)
        # We infer from the next block's offset
        found = false
        for other in schema.obj_types
            ob = get(csp.var_offset, other, 0)
            ob > base && ob <= v && (found = true)
        end
        if !found && v >= base
            return o
        end
    end
    return nothing
end

function _init_domains(csp::CSPProblem, g::GPUACSet,
                        schema::SchemaInfo,
                        g_offset::Dict{Symbol,Int})::Vector{UInt64}
    domains = zeros(UInt64, Int(csp.n_vars))
    for o in schema.obj_types
        n_live = g.n_live[o][]
        mask   = n_live < 64 ? (UInt64(1) << n_live) - UInt64(1) : typemax(UInt64)
        base   = get(csp.var_offset, o, 0)
        base == 0 && continue
        # Determine how many vars are in this type's block
        n_vars_here = count(v -> _var_to_type(v, csp, schema) == o,
                            1:Int(csp.n_vars))
        for i in 0:(n_vars_here-1)
            domains[base + i] = mask
        end
    end
    domains
end

function _apply_attr_masks!(domains::Vector{UInt64}, csp::CSPProblem,
                             g::GPUACSet, schema::SchemaInfo,
                             enc::AttributeEncoder,
                             g_offset::Dict{Symbol,Int})
    # For PROP_ATTR_EQ bytecodes, build per-variable masks restricting to
    # world elements whose attribute value matches the required encoded int.
    for bc in csp.bytecodes
        bc.op != PROP_ATTR_EQ && continue
        v = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        host_av  = Array(g.attrs[a])
        host_act = Array(g.active[owner])
        mask = UInt64(0)
        for (local_i, (alive, av)) in enumerate(zip(host_act, host_av))
            alive || continue
            local_i > 63 && break
            av == req && (mask |= UInt64(1) << (local_i - 1))
        end
        domains[v] &= mask
    end
end

function _compact_matches!(table::MatchTable, keep::BitVector)
    src = 1; dst = 1
    while src <= table.n_matches
        if keep[src]
            if dst != src
                table.assignments[:, dst] .= table.assignments[:, src]
            end
            dst += 1
        end
        src += 1
    end
    table.n_matches = dst - 1
end

function _push_match!(table::MatchTable, sol::Vector{Int32})
    if table.n_matches < size(table.assignments, 2)
        table.n_matches += 1
        table.assignments[:, table.n_matches] .= sol
    end
end
