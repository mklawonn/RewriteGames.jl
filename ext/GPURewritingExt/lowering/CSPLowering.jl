"""
    CSPProblem

GPU-ready constraint satisfaction problem for one rewrite rule's pattern L.

Fields:
- `n_vars`:      Total number of pattern variables (one per element of L).
- `var_offset`:  Per-object-type start index in the flat variable array.
- `domain_sizes`: Maximum domain size for each variable (= nparts(world, ob)).
- `bytecodes`:   Constraint packets in execution order.
- `nac_groups`:  Number of distinct NAC groups (0 if no NACs).
- `pac_groups`:  Number of distinct PAC groups (0 if no PACs).
- `n_chunks`:    Number of UInt64 words per domain variable (= ceil(max_world_size/64)).
- `hom_forward`: `hom_forward[h][(w-1)*n_chunks + c]` = chunk c of h(w)'s bitmask.
"""
struct CSPProblem
    n_vars            :: Int32
    var_offset        :: Dict{Symbol, Int}   # obj type → first variable index (1-based)
    domain_sizes      :: Vector{Int32}       # one per variable
    bytecodes         :: Vector{TCNBytecode}
    nac_groups        :: Int32
    pac_groups        :: Int32
    agent_var_map     :: Vector{Int32}       # mapping from agent interface elements to L variables
    hom_forward       :: Vector{Vector{UInt64}}
    n_chunks          :: Int                 # number of UInt64 words per domain variable
    sorted_type_bases :: Vector{Tuple{Int, Symbol}}  # [(base, obj_type)] sorted by base, pre-computed
end

"""
    lower_rule_to_csp(rule, world, schema, enc) -> CSPProblem

Translate a rewrite rule's pattern (its left-hand side `L`) and any
NAC/PAC application conditions into a `CSPProblem`.

Variable assignment: for each object type `o` in `schema.obj_types`, the
L-elements of type `o` are assigned consecutive variable indices starting at
`var_offset[o]`.  Variables run from 1 to `n_vars` (0 = unused sentinel).

Bytecodes emitted (in order):
1. `DOMAIN_SIZE`  — one per variable, encoding the world part count for that type.
2. `PROP_FUNC`    — one per morphism edge in L (structural propagation).
3. `PROP_ATTR_EQ` — one per concrete attribute value in L (fixed attribute match).
4. `PROP_NEQ`     — monic pairs (all pairs of same-type variables when rule is monic).
5. `NAC_REIF`     — one per constraint in each negative application condition.
6. `PAC_REIF`     — one per constraint in each positive application condition.
"""
function lower_rule_to_csp(rule, world, schema::SchemaInfo,
                            enc::AttributeEncoder;
                            n_chunks::Int = cld(max(1, isempty(schema.obj_types) ? 1 :
                                maximum(nparts(world, o) for o in schema.obj_types)), 64))::CSPProblem
    
    
    # Extract underlying AlgebraicRewriting rule if we were passed a box
    inner_rule = if hasproperty(rule, :rule)
        rule.rule
    else
        rule
    end

    L = if hasmethod(left, Tuple{typeof(inner_rule)})
        codom(left(inner_rule))
    elseif hasproperty(inner_rule, :_left) && inner_rule._left !== nothing
        codom(inner_rule._left)
    elseif hasproperty(inner_rule, :L)
        inner_rule.L
    elseif hasproperty(inner_rule, :rule) && hasmethod(left, Tuple{typeof(inner_rule.rule)})
        codom(left(inner_rule.rule))
    else
        error("lower_rule_to_csp: could not extract L from rule")
    end


    S = acset_schema(L)

    # ── 1. Assign variable indices ─────────────────────────────────────────────
    var_offset = Dict{Symbol,Int}()
    cursor = 1
    for o in schema.obj_types
        n = nparts(L, o)
        if n > 0
            var_offset[o] = cursor
            cursor += n
        end
    end
    n_vars = cursor - 1

    # ── 2. Domain size for each variable ──────────────────────────────────────
    domain_sizes = Int32[]
    for o in schema.obj_types
        n = nparts(L, o)
        if n > 0
            sz = Int32(nparts(world, o))
            for _ in 1:nparts(L, o)
                push!(domain_sizes, sz)
            end
        end
    end

    bytecodes = TCNBytecode[]

    # ── 3. DOMAIN_SIZE bytecodes ───────────────────────────────────────────────
    for (v, sz) in enumerate(domain_sizes)
        # push!(bytecodes, tcn(DOMAIN_SIZE; var1=v, param1=domain_sizes[v]))
    end

    # ── 4. PROP_FUNC: structural morphism constraints ─────────────────────────
    for (h_idx, h) in enumerate(schema.homs)
        hom_ob = schema.hom_dom[h]
        nparts(L, hom_ob) == 0 && continue
        for i in parts(L, hom_ob)
            j = subpart(L, i, h)           # j is the target element in L
            j == 0 && continue
            v_i = get(var_offset, hom_ob, 0) + (i - 1)
            v_j = get(var_offset, schema.hom_cod[h], 0) + (j - 1)
            # If either variable is 0 (type not in L), this constraint cannot be satisfied if L is well-formed.
            # But here we just skip if not in L.
            (get(var_offset, hom_ob, 0) == 0 || get(var_offset, schema.hom_cod[h], 0) == 0) && continue
            push!(bytecodes, tcn(PROP_FUNC; var1=v_i, var2=v_j, param1=h_idx))
        end
    end

    # ── 5. PROP_ATTR_EQ: fixed attribute value constraints ────────────────────
    for (a_idx, a) in enumerate(schema.attrs)
        owner = schema.attr_dom[a]
        nparts(L, owner) == 0 && continue
        for i in parts(L, owner)
            raw = subpart(L, i, a)
            raw isa AttrVar && continue          # free attribute variable — skip
            encoded = encode_value(enc, a, raw)
            encoded == Int32(0) && continue      # unknown value — skip
            haskey(var_offset, owner) || continue
            v_i = var_offset[owner] + (i - 1)
            push!(bytecodes, tcn(PROP_ATTR_EQ; var1=v_i,
                                 param1=a_idx, param2=encoded))
        end
    end

    # ── 6. PROP_NEQ: monic constraints (all distinct within each type) ─────────
    # Check both the outer wrapper (PlayerRuleApp) and the inner AlgebraicRewriting rule
    _monic_src = hasproperty(rule, :monic) ? rule :
                 hasproperty(inner_rule, :monic) ? inner_rule : nothing
    _monic = _monic_src !== nothing ? _monic_src.monic : false
    if _monic === true
        for o in schema.obj_types
            cnt = nparts(L, o)
            cnt == 0 && continue
            base = var_offset[o]
            for i in 1:cnt, j in (i+1):cnt
                push!(bytecodes,
                      tcn(PROP_NEQ; var1=base+(i-1), var2=base+(j-1)))
            end
        end
    elseif _monic isa AbstractVector
        for o in _monic
            haskey(var_offset, o) || continue
            cnt = nparts(L, o)
            base = var_offset[o]
            for i in 1:cnt, j in (i+1):cnt
                push!(bytecodes,
                      tcn(PROP_NEQ; var1=base+(i-1), var2=base+(j-1)))
            end
        end
    end

    # ── 7. NAC_REIF: negative application conditions ──────────────────────────
    nac_count = Int32(0)
    if hasproperty(rule, :conditions)
        for cond in rule.conditions
            _lower_ac!(bytecodes, cond, schema, var_offset, enc,
                       NAC_REIF, nac_count += Int32(1))
        end
    end

    # ── 8. PAC_REIF: positive application conditions ──────────────────────────
    pac_count = Int32(0)
    if hasproperty(rule, :pos_conditions)
        for cond in rule.pos_conditions
            _lower_ac!(bytecodes, cond, schema, var_offset, enc,
                       PAC_REIF, pac_count += Int32(1))
        end
    end

    # ── 9. agent_var_map: mapping from agent interface to L variables ────────
    agent_var_map = Int32[]
    if hasproperty(rule, :in_hom) && rule.in_hom isa Catlab.CategoricalAlgebra.ACSetTransformation
        # PlayerRuleApp case
        h = rule.in_hom
        dom_L = dom(h)
        for o in schema.obj_types
            haskey(var_offset, o) || continue
            for i in parts(dom_L, o)
                tgt = h[o](i)
                v_idx = var_offset[o] + (tgt - 1)
                push!(agent_var_map, Int32(v_idx))
            end
        end
    elseif hasproperty(rule, :in_agent) && rule.in_agent isa Catlab.CategoricalAlgebra.ACSetTransformation
        # RuleApp case
        h = rule.in_agent
        dom_L = dom(h)
        for o in schema.obj_types
            haskey(var_offset, o) || continue
            for i in parts(dom_L, o)
                tgt = h[o](i)
                v_idx = var_offset[o] + (tgt - 1)
                push!(agent_var_map, Int32(v_idx))
            end
        end
    end

    # ── Precompute world hom forward bitmasks for PROP_FUNC propagation ─────
    # hom_forward[h][(w-1)*n_chunks + c] = chunk c of bitmask for element w.
    nc = n_chunks
    hom_forward = Vector{UInt64}[]
    for h in schema.homs
        nA = nparts(world, schema.hom_dom[h])
        fwd = zeros(UInt64, max(nA, 1) * nc)
        for w in 1:nA
            target = subpart(world, w, h)
            if target > 0
                ci, bi = elem_to_chunk(target)
                ci <= nc && (fwd[(w-1)*nc + ci] |= UInt64(1) << bi)
            end
        end
        push!(hom_forward, fwd)
    end

    sorted_bases = sort([(base, o) for (o, base) in pairs(var_offset)], by=first)

    CSPProblem(Int32(n_vars), var_offset, domain_sizes, bytecodes,
               nac_count, pac_count, agent_var_map, hom_forward, nc, sorted_bases)
end

function _lower_ac!(bytecodes, cond, schema, var_offset, enc, op_code, group_id)
    # Application conditions carry their own pattern graph; we emit PROP_FUNC-
    # style constraints tagged with the group id via the NAC_REIF/PAC_REIF op.
    # The GPU kernel interprets reification bytecodes as "if all these match
    # simultaneously the auxiliary var3 is forced to 1".
    ac_L = try; codom(left(cond)); catch; return; end
    S    = acset_schema(ac_L)
    for h in schema.homs
        dom_ob = schema.hom_dom[h]
        nparts(ac_L, dom_ob) == 0 && continue
        for i in parts(ac_L, dom_ob)
            j = subpart(ac_L, i, h)
            j == 0 && continue
            # Reuse the base L variable offsets when the AC shares them
            v_i = get(var_offset, dom_ob, 0) + (i - 1)
            v_j = get(var_offset, schema.hom_cod[h], 0) + (j - 1)
            v_i == 0 || v_j == 0 && continue
            push!(bytecodes,
                  tcn(op_code; var1=v_i, var2=v_j, var3=0, param1=group_id))
        end
    end
end
