"""
    AutoRule

A rule that fires automatically between player turns.

# Fields
- `rule`: The AlgebraicRewriting rewrite rule.
- `name`: Optional symbolic name.
- `prob_attr`: If provided, the name of a Float64 attribute in the ACSet whose
  value is used as the probability that the rule fires for each match.
  When `nothing` the rule fires deterministically for every match found.
"""
struct AutoRule
    rule      :: Any          # AlgebraicRewriting.Rule
    name      :: Symbol
    prob_attr :: Union{Symbol, Nothing}
end

function AutoRule(rule; name::Symbol=:auto, prob_attr=nothing)
    AutoRule(rule, name, prob_attr)
end
