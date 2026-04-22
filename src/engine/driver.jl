# ─── run_game ─────────────────────────────────────────────────────────────────

"""
    run_game(game::Game, agents::Dict{Symbol,<:AbstractAgent}; T_max=1000)
        -> GameHistory

Run a single complete episode and return the full game history as seven
parallel temporal narratives over `DiscreteTime`.

Delegates to `ScheduledGameDriver` (defined in `engine/scheduled_driver.jl`),
which executes `game.schedule` once per round until the terminal predicate fires
or `T_max` steps are reached.
"""
function run_game(game::Game, agents::Dict{Symbol, <:AbstractAgent}; T_max::Int=1000)
    # _run_scheduled_game is defined in engine/scheduled_driver.jl, loaded after
    # this file; the forward reference is resolved at call-time, not load-time.
    return _run_scheduled_game(game, agents; T_max=T_max)
end
