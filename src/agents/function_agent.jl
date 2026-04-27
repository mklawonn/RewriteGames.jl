"""
    FunctionAgent(f)

An agent backed by any Julia callable `f` with signature:

    f(state::GameState, legal_actions::Vector{Action}) -> Action

This covers random policies, rule-based heuristics, trained Flux models, MCTS
trees, or any other strategy expressible as a Julia function.  The raw ACSet
world is available as `state.world`; use `elements_graph(state)` for the
category-of-elements representation.

# Example
```julia
random_agent = FunctionAgent((state, actions) -> rand(actions))
```
"""
struct FunctionAgent <: AbstractAgent
    f :: Function
end

function select_action(agent::FunctionAgent, state::GameState,
                       legal_actions::Vector{Action})
    return agent.f(state, legal_actions)
end
