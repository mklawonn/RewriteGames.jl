"""
    fire_auto_rules!(state::GameState, auto_rules::Vector{AutoRule})
        -> Vector{NamedTuple}

Fire each `AutoRule` in order against the current world state.  Returns a
vector of named tuples recording which rules fired and how many matches were
applied, for inclusion in the `Experience.info` field.
"""
function fire_auto_rules!(state::GameState, auto_rules::Vector{AutoRule})
    results = NamedTuple[]
    for ar in auto_rules
        matches = get_matches(ar.rule, state.world)
        fired   = 0
        for m in matches
            should_fire = if ar.prob_attr === nothing
                true
            else
                # Roll against the probability attribute on the first component.
                prob = _get_prob_attr(state.world, m, ar.prob_attr)
                rand() < prob
            end
            if should_fire
                state.world = rewrite_match(ar.rule, m)
                fired += 1
                # Re-enumerate matches against updated world
                matches = get_matches(ar.rule, state.world)
            end
        end
        push!(results, (rule=ar.name, fired=fired))
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
    catch
        return 1.0
    end
end
