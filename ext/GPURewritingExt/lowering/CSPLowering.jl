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
"""
struct CSPProblem
    n_vars       :: Int32
    var_offset   :: Dict{Symbol, Int}   # obj type → first variable index (1-based)
    domain_sizes :: Vector{Int32}       # one per variable
    bytecodes    :: Vector{TCNBytecode}
    nac_groups   :: Int32
    pac_groups   :: Int32
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
                            enc::AttributeEncoder)::CSPProblem
    L = codom(left(rule))   # pattern graph (LHS of the span)
    S = acset_schema(L)

    # ── 1. Assign variable indices ─────────────────────────────────────────────
    var_offset = Dict{Symbol,Int}()
    cursor = 1
    for o in schema.obj_types
        var_offset[o] = cursor
        cursor += nparts(L, o)
    end
    n_vars = cursor - 1

    # ── 2. Domain size for each variable ──────────────────────────────────────
    domain_sizes = Int32[]
    for o in schema.obj_types
        sz = Int32(nparts(world, o))
        for _ in 1:nparts(L, o)
            push!(domain_sizes, sz)
        end
    end

    bytecodes = TCNBytecode[]

    # ── 3. DOMAIN_SIZE bytecodes ───────────────────────────────────────────────
    for (v, sz) in enumerate(domain_sizes)
        push!(bytecodes, tcn(DOMAIN_SIZE; var1=v, param1=sz))
    end

    # ── 4. PROP_FUNC: structural morphism constraints ─────────────────────────
    for (h_idx, h) in enumerate(schema.homs)
        hom_ob = schema.hom_dom[h]
        nparts(L, hom_ob) == 0 && continue
        for i in parts(L, hom_ob)
            j = subpart(L, i, h)           # j is the target element in L
            j == 0 && continue
            v_i = var_offset[hom_ob] + (i - 1)
            v_j = var_offset[schema.hom_cod[h]] + (j - 1)
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
            v_i = var_offset[owner] + (i - 1)
            push!(bytecodes, tcn(PROP_ATTR_EQ; var1=v_i,
                                 param1=a_idx, param2=encoded))
        end
    end

    # ── 6. PROP_NEQ: monic constraints (all distinct within each type) ─────────
    if hasproperty(rule, :monic) && rule.monic === true
        for o in schema.obj_types
            cnt = nparts(L, o)
            base = var_offset[o]
            for i in 1:cnt, j in (i+1):cnt
                push!(bytecodes,
                      tcn(PROP_NEQ; var1=base+(i-1), var2=base+(j-1)))
            end
        end
    elseif hasproperty(rule, :monic) && rule.monic isa AbstractVector
        for o in rule.monic
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

    CSPProblem(Int32(n_vars), var_offset, domain_sizes, bytecodes,
               nac_count, pac_count)
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
