"""
    AbstractAgent

Supertype for all game-playing agents.  Concrete subtypes must implement:

    select_action(agent, state::EncodedState, legal_actions::Vector{Action}) -> Action
"""
abstract type AbstractAgent end

"""
    select_action(agent::AbstractAgent, state::EncodedState,
                  legal_actions::Vector{Action}) -> Action

Choose one action from `legal_actions` given the current encoded state.
Concrete agent types override this method.
"""
function select_action end

# ─── Action ───────────────────────────────────────────────────────────────────

"""
    Action

A concrete game action: a reference to a `RuleEntry` and the specific ACSet
transformation (match morphism) that was found in the current world.
"""
struct Action
    entry :: RuleEntry
    match :: Any        # ACSetTransformation from AlgebraicRewriting
end
