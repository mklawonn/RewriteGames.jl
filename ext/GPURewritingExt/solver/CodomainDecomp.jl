"""
Compact codomain decomposition for pinned-agent hom-search solves.

The dominant cost in agent-dense schedules is the per-agent pinned solve, which is
handed the WHOLE world as its codomain (global `nc` chunks).  For a pinned agent the
match can only involve elements FK-reachable from the agent through the rule's own
morphisms, so the real codomain is tiny.  This module restricts each solve to that
local neighborhood, remapped to a compact `1..k` index space (small `nc_local`), so the
shared-memory `turbo_block` solver runs O(nc_local) propagation in a small `@localmem`
(many blocks/SM).  See plan: please-write-a-plan-cozy-wall.md.

Correctness is by construction: the compact domains are taken by RESTRICTING the
already-built world initial domain (`base_d`, which already carries type + attribute
masks) to the neighborhood and remapping; the compact `hom_forward` is the world FK
restricted+remapped.  The bytecodes (which reference variable/hom indices, never world
slots) are reused unchanged.  Solutions are translated back to world indices.

The gate `RG_DECOMP_DIAG` asserts the back-translated decomposed solution SET equals the
full-world `gpu_turbo_solve` set per solve (see `_exec_agent_loop_batched!`).
"""

# Map each CSP variable index (1..n_vars) to its object type, from sorted_type_bases.
function _decomp_var_types(csp::CSPProblem)
    vt = Vector{Symbol}(undef, Int(csp.n_vars))
    tb = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(tb)
        next_base = idx < length(tb) ? tb[idx + 1][1] : Int(csp.n_vars) + 1
        for v in base:(next_base - 1)
            vt[v] = o
        end
    end
    vt
end

# Schema homs that are constraints in this CSP: both endpoints are rule-variable types.
function _decomp_relevant_homs(schema::SchemaInfo, csp::CSPProblem)
    [h for h in schema.homs
       if haskey(csp.var_offset, schema.hom_dom[h]) &&
          haskey(csp.var_offset, schema.hom_cod[h])]
end

# Number of L-elements (CSP variables) of each object type, from sorted_type_bases.
function _decomp_nvars_per_type(csp::CSPProblem)
    nv = Dict{Symbol,Int}()
    tb = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(tb)
        next_base = idx < length(tb) ? tb[idx + 1][1] : Int(csp.n_vars) + 1
        nv[o] = next_base - base
    end
    nv
end

"""
    _decomp_gather(schema, csp, fk_cols, n_alloc, seed_obj, seed_slots) -> Dict{Symbol,Vector{Int}}

Fixpoint over the rule's morphism edges (both directions), seeded with one or more slots
of the seed type (a pinned agent, or an anchored-decomposition cell), returning the
sorted world slots reachable per rule-variable type.  `fk_cols[h]` is the host FK column
of hom `h` (length n_alloc[hom_dom[h]]).  Over-approximates the exact match support
(safe: any valid match element is reachable), bounded by the local connectivity.
"""
_decomp_gather(schema::SchemaInfo, csp::CSPProblem,
               fk_cols::Dict{Symbol,Vector{Int32}}, n_alloc::Dict{Symbol,Int},
               agent_obj::Symbol, agent_slot::Int) =
    _decomp_gather(schema, csp, fk_cols, n_alloc, agent_obj, Int[agent_slot])
function _decomp_gather(schema::SchemaInfo, csp::CSPProblem,
                        fk_cols::Dict{Symbol,Vector{Int32}},
                        n_alloc::Dict{Symbol,Int},
                        agent_obj::Symbol, seed_slots::Vector{Int})
    rel = _decomp_relevant_homs(schema, csp)
    sets = Dict{Symbol,Set{Int}}(o => Set{Int}() for o in keys(csp.var_offset))
    union!(sets[agent_obj], seed_slots)
    # ── Connectivity closure (bounds the universe).  FREEZE the agent type: never grow
    #    its set, so reverse-FK neighbors (e.g. this platform's FuelTokens) stay local to
    #    THIS agent instead of cascading to siblings via shared hubs (squadron/zone). ──
    changed = true
    while changed
        changed = false
        for h in rel
            A = schema.hom_dom[h]; B = schema.hom_cod[h]; col = fk_cols[h]
            if B != agent_obj                              # forward: targets of sources
                for w in sets[A]
                    (1 <= w <= length(col)) || continue
                    t = Int(col[w])
                    if t > 0 && !(t in sets[B]); push!(sets[B], t); changed = true; end
                end
            end
            if A != agent_obj && !isempty(sets[B])         # reverse: sources into B
                for w in 1:length(col)
                    t = Int(col[w])
                    if t > 0 && (t in sets[B]) && !(w in sets[A]); push!(sets[A], w); changed = true; end
                end
            end
        end
    end
    # ── AC-1 narrowing of SINGLE-VARIABLE types (sound: a 1-var type's set IS that
    #    variable's domain).  For functional constraint h:A→B, prune A to elements mapping
    #    into B, and B to the image of A.  Multi-var types (which would conflate distinct
    #    L-elements, e.g. ZoneA vs ZoneB) are left at the connectivity set; the solver
    #    separates them.  Over-approx inputs ⇒ monotone-safe; the set-equiv gate confirms. ──
    nvar = _decomp_nvars_per_type(csp)
    is_single(o) = o != agent_obj && get(nvar, o, 0) == 1
    changed = true
    while changed
        changed = false
        for h in rel
            A = schema.hom_dom[h]; B = schema.hom_cod[h]; col = fk_cols[h]
            if is_single(B)
                img = Set{Int}()
                for w in sets[A]
                    (1 <= w <= length(col)) || continue
                    t = Int(col[w]); t > 0 && push!(img, t)
                end
                ni = intersect(sets[B], img)
                if length(ni) != length(sets[B]); sets[B] = ni; changed = true; end
            end
            if is_single(A)
                keep = Set{Int}(w for w in sets[A] if 1 <= w <= length(col) && Int(col[w]) in sets[B])
                if length(keep) != length(sets[A]); sets[A] = keep; changed = true; end
            end
        end
    end
    # Types NOT reachable from the seed through the rule's own homs (type-level,
    # undirected) form unanchored components and range over the full active set:
    # isolated types (in no relevant hom — the original fallback) and whole
    # connected-but-unseeded components (e.g. a TLAM-style rule whose slot/squadron
    # side has no hom path to the anchor).  A REACHABLE type that ends empty must
    # stay empty — that is the correct "no candidates" (e.g. a NAC element with
    # none linked to this seed).
    reach = Set{Symbol}((agent_obj,))
    changed = true
    while changed
        changed = false
        for h in rel
            A = schema.hom_dom[h]; B = schema.hom_cod[h]
            if (A in reach) != (B in reach)
                push!(reach, A); push!(reach, B); changed = true
            end
        end
    end
    for (o, _) in csp.var_offset
        if !(o in reach) && isempty(sets[o])
            for w in 1:get(n_alloc, o, 0); push!(sets[o], w); end
        end
    end
    Dict{Symbol,Vector{Int}}(o => sort!(collect(s)) for (o, s) in sets)
end

"""
    decomposed_pinned_solve(backend, csp, schema, agent_obj, agent_slot,
                            base_d_host, fk_cols, n_alloc) -> Vector{Vector{Int32}}

Solve the pinned-agent CSP over the compact local codomain and return solutions in WORLD
indices (length n_vars each).  `base_d_host` = host copy of the world initial domain
(`_build_domains_gpu!` + attr masks, UNPINNED), `fk_cols` = host FK columns of relevant
homs, both built once per box.  Pure host setup + one small upload + one solve.
"""
function decomposed_pinned_solve(backend, csp::CSPProblem, schema::SchemaInfo,
                                  agent_obj::Symbol, agent_slot::Int,
                                  base_d_host::Vector{UInt64},
                                  fk_cols::Dict{Symbol,Vector{Int32}},
                                  n_alloc::Dict{Symbol,Int})
    nbhd = _decomp_gather(schema, csp, fk_cols, n_alloc, agent_obj, agent_slot)
    _decomp_compact_solve(backend, csp, schema, nbhd, base_d_host, fk_cols)
end

"""
    _decomp_compact_solve(backend, csp, schema, nbhd, base_d_host, fk_cols)
        -> Vector{Vector{Int32}}

Solve the CSP over the compact codomain `nbhd` (world slots per rule-variable
type, from `_decomp_gather`) and return solutions in WORLD indices.  Shared by
the pinned-agent and anchored-cell decompositions.
"""
function _decomp_compact_solve(backend, csp::CSPProblem, schema::SchemaInfo,
                                nbhd::Dict{Symbol,Vector{Int}},
                                base_d_host::Vector{UInt64},
                                fk_cols::Dict{Symbol,Vector{Int32}})
    nv      = Int(csp.n_vars)
    nc_w    = csp.n_chunks
    vt      = _decomp_var_types(csp)

    # Per-type local index maps.
    l2w = Dict{Symbol,Vector{Int}}()        # local idx -> world slot
    w2l = Dict{Symbol,Dict{Int,Int}}()      # world slot -> local idx
    for (o, ws) in nbhd
        l2w[o] = ws
        w2l[o] = Dict{Int,Int}(w => i for (i, w) in enumerate(ws))
    end
    k_max  = maximum((length(ws) for ws in values(l2w)); init = 1)
    nc_l   = max(cld(k_max, 64), 1)

    # Compact initial domains: restrict base_d's bits to the neighborhood, remap to local.
    dl = zeros(UInt64, nv * nc_l)
    for v in 1:nv
        o   = vt[v]
        ws  = l2w[o]
        offw = (v - 1) * nc_w
        offl = (v - 1) * nc_l
        for (li, w) in enumerate(ws)
            ciw, biw = elem_to_chunk(w)
            (ciw <= nc_w && (base_d_host[offw + ciw] & (UInt64(1) << biw)) != 0) || continue
            cil, bil = elem_to_chunk(li)
            dl[offl + cil] |= (UInt64(1) << bil)
        end
    end

    # Compact hom_forward aligned to schema.homs indexing (only relevant homs filled).
    rel = Set(_decomp_relevant_homs(schema, csp))
    hom_offs = Int32[Int32(0)]; total = 0
    src_cnt  = Int[]
    for h in schema.homs
        domo = schema.hom_dom[h]
        n_local = haskey(l2w, domo) ? max(length(l2w[domo]), 1) : 1
        push!(src_cnt, n_local)
        total += n_local * nc_l
        push!(hom_offs, Int32(total))
    end
    hf = zeros(UInt64, max(total, 1))
    for (hidx, h) in enumerate(schema.homs)
        (h in rel) || continue
        A = schema.hom_dom[h]; B = schema.hom_cod[h]
        col = fk_cols[h]; off = Int(hom_offs[hidx])
        for (ls, w) in enumerate(l2w[A])
            (1 <= w <= length(col)) || continue
            t = Int(col[w]); t > 0 || continue
            lt = get(w2l[B], t, 0); lt > 0 || continue
            cil, bil = elem_to_chunk(lt)
            hf[off + (ls - 1) * nc_l + cil] = (UInt64(1) << bil)
        end
    end

    # Compact CSP: same vars/bytecodes, small nc; hom_forward field unused by the solver.
    domain_sizes = Int32[Int32(length(l2w[vt[v]])) for v in 1:nv]
    csp_l = CSPProblem(csp.n_vars, csp.var_offset, domain_sizes, csp.bytecodes,
                       csp.nac_groups, csp.pac_groups, csp.agent_var_map,
                       Vector{Vector{UInt64}}(), nc_l, csp.sorted_type_bases)

    d_gpu   = CuArray(dl)
    hf_gpu  = CuArray(hf)
    ho_gpu  = CuArray(hom_offs)
    sols_l  = gpu_turbo_solve(backend, csp_l, d_gpu, hf_gpu, ho_gpu; scratch = nothing)

    # Back-translate local solution indices to world slots (rows 1..nv are meaningful).
    out = Vector{Vector{Int32}}()
    for s in sols_l
        ws = Vector{Int32}(undef, nv)
        ok = true
        for v in 1:nv
            li = v <= length(s) ? Int(s[v]) : 0
            tab = l2w[vt[v]]
            if 1 <= li <= length(tab)
                ws[v] = Int32(tab[li])
            else
                ok = false; break
            end
        end
        ok && push!(out, ws)
    end
    out
end
