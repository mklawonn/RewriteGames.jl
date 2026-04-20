"""
    RuleEntry

A `Rule` together with an optional usage budget and an optional post-filter.

# Fields
- `rule`: The AlgebraicRewriting rewrite rule.
- `name`: Optional symbolic name for display/serialization.
- `budget`: Maximum number of times this rule may fire per game (nothing = unlimited).
- `post_filter`: An optional function `(W, match) -> Bool` applied after pattern
  matching to discard matches that satisfy a threshold condition not expressible as
  exact attribute equality in the ACSet pattern.
"""
struct RuleEntry
    rule        :: Any          # AlgebraicRewriting.Rule
    name        :: Symbol
    budget      :: Union{Int, Nothing}
    post_filter :: Union{Function, Nothing}
end

function RuleEntry(rule; name::Symbol=:unnamed, budget=nothing, post_filter=nothing)
    RuleEntry(rule, name, budget, post_filter)
end

"""
    RuleLibrary

An ordered collection of `RuleEntry` values for one player.
"""
struct RuleLibrary
    entries :: Vector{RuleEntry}
end

RuleLibrary(entries::Vector) = RuleLibrary(convert(Vector{RuleEntry}, entries))

Base.length(lib::RuleLibrary)            = length(lib.entries)
Base.iterate(lib::RuleLibrary, args...)  = iterate(lib.entries, args...)
Base.getindex(lib::RuleLibrary, i)       = lib.entries[i]
Base.eachindex(lib::RuleLibrary)         = eachindex(lib.entries)
