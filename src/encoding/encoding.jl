"""
    elements_graph(state::GameState)

Return the category of elements of the game world ACSet, using Catlab's
`elements()`.  This provides a homogeneous node/edge representation of the
world that is useful as input to graph neural networks.

The raw ACSet is always accessible directly via `state.world`.
"""
elements_graph(state::GameState) = elements(state.world)
