"""
    fire_auto_rules!(state::GameState, auto_rules::Vector{AutoRule})
        -> Vector{NamedTuple}

Fire each `AutoRule` in order against the current world state.  Returns one
named tuple `(rule_name, rule, match)` per match that was actually applied,
preserving firing order.
"""
function fire_auto_rules!(state::GameState, auto_rules::Vector{AutoRule})
    results = NamedTuple[]
    for ar in auto_rules
        # Snapshot matches before applying any rewrites; mid-loop world mutation
        # would invalidate an open iterator but leave already-visited matches stale.
        snapshot = collect(get_matches(ar.rule, state.world))
        for m in snapshot
            should_fire = if ar.prob_attr === nothing
                true
            else
                prob = _get_prob_attr(state.world, m, ar.prob_attr)
                rand() < prob
            end
            if should_fire
                state.world = rewrite_match(ar.rule, m)
                push!(results, (rule_name=ar.name, rule=ar.rule, match=m))
            end
        end
    end
    return results
end

# ─── helpers ──────────────────────────────────────────────────────────────────

"""
    _get_prob_attr(W, match, attr::Symbol) -> Float64

Attempt to read the probability attribute `attr` from the first matched part
in the world ACSet.  Returns 1.0 if the attribute or part cannot be found.
"""
function _get_prob_attr(W, match, attr::Symbol)
    try
        comp = first(values(components(match)))
        part = first(collect(comp))
        return Float64(subpart(W, part, attr))
    catch _
        return 1.0
    end
end
