using Statistics: mean

"""
    win_rate(experiences::Vector{Experience}, player::Symbol) -> Float64

Return the fraction of completed episodes in `experiences` where `player` was
the winner.  Only experiences with `done == true` are counted as episode
boundaries; episodes without a winner (draw / timeout) do not count for any
player.
"""
function win_rate(experiences::Vector{Experience}, player::Symbol)
    terminals = filter(e -> e.done, experiences)
    isempty(terminals) && return 0.0
    wins = count(e -> e.winner === player, terminals)
    return wins / length(terminals)
end

"""
    episode_length(experiences::Vector{Experience}) -> Int

Return the number of steps in `experiences`.  Assumes all experiences belong to
a single episode.
"""
episode_length(experiences::Vector{Experience}) = length(experiences)

"""
    action_counts(experiences::Vector{Experience}) -> Dict{Symbol,Int}

Count how many times each named rule was chosen across all experiences.
Passes (action === nothing) are counted under the key `:pass`.
"""
function action_counts(experiences::Vector{Experience})
    counts = Dict{Symbol, Int}()
    for exp in experiences
        key = exp.action === nothing ? :pass : exp.action.entry.name
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end
