"""
    enumerate_all_matches(entry::RuleEntry, world) -> Vector{Action}

Return every legal `Action` for the given `RuleEntry` in the current world,
applying the optional post-filter.
"""
function enumerate_all_matches(entry::RuleEntry, world)
    matches = get_matches(entry.rule, world)

    if entry.post_filter !== nothing
        matches = filter(m -> entry.post_filter(world, m), matches)
    end

    return [Action(entry, m) for m in matches]
end

"""
    enumerate_legal_actions(lib::RuleLibrary, world) -> Vector{Action}

Return all legal actions for a player given the current world state.
"""
function enumerate_legal_actions(lib::RuleLibrary, world)
    actions = Action[]
    for entry in lib.entries
        append!(actions, enumerate_all_matches(entry, world))
    end
    return actions
end

"""
    apply_rule!(world, action::Action) -> new_world

Apply the chosen action and return the new world.
"""
function apply_rule!(world, action::Action)
    return rewrite_match(action.entry.rule, action.match)
end

"""
    rule_index(lib::RuleLibrary, entry::RuleEntry) -> Int

Return the 1-based index of `entry` in `lib`, or 0 if not found.
"""
function rule_index(lib::RuleLibrary, entry::RuleEntry)
    for (i, e) in enumerate(lib.entries)
        if e === entry
            return i
        end
    end
    return 0
end
