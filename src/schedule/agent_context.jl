"""
    AgentContext

Identifies the specific ACSet part that a `ForEachStep` is currently
iterating over.  Threaded as an optional argument through `execute_step!`;
`nothing` means no restriction — actions are enumerated globally.

# Fields
- `ob`:    The object type being iterated (e.g. `:Wolf`).
- `id`:    1-based part index of the current instance in the world.
- `stack`: Previous `(ob, id)` pairs from enclosing `ForEachStep` nodes
           (outermost first).  Empty for the outermost context.
"""
struct AgentContext
    ob    :: Symbol
    id    :: Int
    stack :: Vector{Tuple{Symbol,Int}}
end

AgentContext(ob::Symbol, id::Int) = AgentContext(ob, id, Tuple{Symbol,Int}[])

"""
    push_context(ctx::AgentContext, ob::Symbol, id::Int) -> AgentContext

Return a new `AgentContext` for a nested `ForEachStep`, preserving the outer
context chain in `stack`.
"""
function push_context(ctx::AgentContext, ob::Symbol, id::Int)
    AgentContext(ob, id, vcat(ctx.stack, [(ctx.ob, ctx.id)]))
end

Base.show(io::IO, ctx::AgentContext) =
    print(io, "AgentContext(:$(ctx.ob), id=$(ctx.id))")

# ─── Context-aware action enumeration ────────────────────────────────────────

"""
    enumerate_legal_actions_in_context(lib, state, player, context)
        -> Vector{Action}

Like `enumerate_legal_actions`, but when `context` is non-`nothing`, retains
only those actions whose match morphism maps at least one domain element of
type `context.ob` to `context.id` in the world.

This restricts a `PlayerStep` inside `ForEachStep(:Wolf, ...)` to actions that
directly involve the specific wolf instance being iterated.

## Implementation note

`components(match)[ob]` is a `FinFunction` whose `collect` gives the vector of
world-part ids that the left-hand pattern's `ob` parts map to.  If the rule's
left-hand pattern has no parts of type `context.ob`, the component maps 0
elements and the image is empty — the match is excluded, as intended.
"""
function enumerate_legal_actions_in_context(
    lib     :: RuleLibrary,
    state   :: GameState,
    player  :: Symbol,
    context :: Union{AgentContext, Nothing},
)
    actions = enumerate_legal_actions(lib, state, player)
    context === nothing && return actions
    ob = context.ob
    id = context.id
    return filter(actions) do action
        comps = components(action.match)
        haskey(comps, ob) || return false
        id ∈ collect(comps[ob])
    end
end
