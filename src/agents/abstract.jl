"""
    AbstractAgent

Supertype for all game-playing agents.  Concrete subtypes must implement:

    select_action(agent, state::GameState, legal_actions::Vector{Action}) -> Action
"""
abstract type AbstractAgent end

"""
    select_action(agent::AbstractAgent, state::GameState,
                  legal_actions::Vector{Action}) -> Action

Choose one action from `legal_actions` given the current game state.
Concrete agent types override this method.  The raw ACSet world is available
as `state.world`; call `elements_graph(state)` for the category-of-elements
representation.
"""
function select_action end

# ─── Action ───────────────────────────────────────────────────────────────────

"""
    Action

A concrete game action: a reference to a box (e.g. `PlayerRuleApp`) and the
specific ACSet transformation (match morphism) found in the current world.
"""
struct Action
    entry :: Any    # PlayerRuleApp or RuleEntry
    match :: Any    # ACSetTransformation from AlgebraicRewriting
end

Base.:(==)(a::Action, b::Action) = a.entry === b.entry && a.match === b.match
Base.hash(a::Action, h::UInt)    = hash(objectid(a.entry), hash(objectid(a.match), h))

Base.show(io::IO, a::Action) = print(io, "Action(:$(a.entry.name))")
