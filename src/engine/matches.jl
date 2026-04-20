"""
    enumerate_all_matches(entry::RuleEntry, state::GameState, player::Symbol, idx::Int)
        -> Vector{Action}

Return every legal `Action` for the given `RuleEntry` in the current world,
subject to budget limits and the optional post-filter.
"""
function enumerate_all_matches(entry::RuleEntry, state::GameState,
                                player::Symbol, idx::Int)
    # Check budget
    key = (player, idx)
    if haskey(state.counters, key) && state.counters[key] <= 0
        return Action[]
    end

    W = state.world
    matches = get_matches(entry.rule, W)

    # Apply post-filter if provided
    if entry.post_filter !== nothing
        matches = filter(m -> entry.post_filter(W, m), matches)
    end

    return [Action(entry, m) for m in matches]
end

"""
    enumerate_legal_actions(lib::RuleLibrary, state::GameState, player::Symbol)
        -> Vector{Action}

Return all legal actions for a player given the current game state.
"""
function enumerate_legal_actions(lib::RuleLibrary, state::GameState, player::Symbol)
    actions = Action[]
    for (idx, entry) in enumerate(lib.entries)
        append!(actions, enumerate_all_matches(entry, state, player, idx))
    end
    return actions
end

"""
    apply_rule!(state::GameState, action::Action, player::Symbol, rule_idx::Int)

Apply the chosen action to the game state in-place (mutates `state.world` and
decrements the rule's budget counter).
"""
function apply_rule!(state::GameState, action::Action, player::Symbol, rule_idx::Int)
    new_world = rewrite_match(action.entry.rule, action.match)
    state.world = new_world

    key = (player, rule_idx)
    if haskey(state.counters, key)
        state.counters[key] -= 1
    end
    return state
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
