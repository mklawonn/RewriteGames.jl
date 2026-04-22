using Statistics: mean

"""
    winner(hist::GameHistory) -> Union{Symbol, Nothing}

Return the winner of the episode, or `nothing` if the game timed out without
a winner.  Looks at the terminal narrative entry for the last recorded turn.
"""
function winner(hist::GameHistory)
    isempty(hist._step_turns) && return nothing
    get_terminal(hist, last(hist._step_turns))
end

export winner

"""
    win_rate(hists::Vector{GameHistory}, player::Symbol) -> Float64

Return the fraction of episodes in `hists` where `player` was the winner.
Episodes without any winner (draw / timeout) do not contribute to any player's
win count.
"""
function win_rate(hists::Vector{GameHistory}, player::Symbol)
    isempty(hists) && return 0.0
    count(h -> winner(h) === player, hists) / length(hists)
end

"""
    episode_length(hist::GameHistory) -> Int

Return the number of player-action turns in `hist` (excludes the initial world
snapshot at t = 0).
"""
episode_length(hist::GameHistory) = history_length(hist)

"""
    action_counts(hist::GameHistory) -> Dict{Symbol, Int}

Count how many times each named rule was chosen across all turns.
Passes (no chosen action) are counted under the key `:pass`.
"""
function action_counts(hist::GameHistory)
    counts = Dict{Symbol, Int}()
    for t in hist._step_turns
        ch  = get_chosen(hist, t)
        key = ch === nothing ? :pass : ch.rule_name
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end
