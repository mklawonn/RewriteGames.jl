"""
    Experience

A single step of interaction between a player and the game environment.

# Fields
- `player`:        Symbol naming the active player.
- `state`:         `EncodedState` capturing the world *before* the action.
- `legal_actions`: All `Action`s available to the player this turn.
- `action`:        The chosen `Action`, or `nothing` if the player passed.
- `next_state`:    `EncodedState` capturing the world *after* the action
                   and after auto-rules have fired.
- `done`:          Whether the game terminated after this step.
- `winner`:        Winning player symbol, or `nothing`.
- `info`:          Metadata dict (auto-rule results, budget snapshot, …).
"""
struct Experience
    player        :: Symbol
    state         :: EncodedState
    legal_actions :: Vector{Action}
    action        :: Union{Action, Nothing}
    next_state    :: EncodedState
    done          :: Bool
    winner        :: Union{Symbol, Nothing}
    info          :: Dict{Symbol, Any}
end

# ─── GameDriver ───────────────────────────────────────────────────────────────

"""
    GameDriver

Mutable driver that advances a game step by step.

# Fields
- `game`:    The `Game` definition.
- `agents`:  Dict mapping player Symbols to `AbstractAgent` instances.
- `state`:   Current `GameState`.
- `T_max`:   Maximum turns before the game is forced to end.
- `_done`:   Internal flag set once the terminal predicate is satisfied.
- `_winner`: Internal winner value once done.
"""
mutable struct GameDriver
    game    :: Game
    agents  :: Dict{Symbol, AbstractAgent}
    state   :: GameState
    T_max   :: Int
    _done   :: Bool
    _winner :: Union{Symbol, Nothing}
end

function GameDriver(game::Game, agents::Dict{Symbol, <:AbstractAgent}; T_max::Int=1000)
    world = game.initial()
    state = GameState(world, game)
    GameDriver(game, Dict{Symbol, AbstractAgent}(agents), state, T_max, false, nothing)
end

# ─── step! ────────────────────────────────────────────────────────────────────

"""
    step!(driver::GameDriver) -> Experience

Advance the game by one player turn and return the resulting `Experience`.

1. Determine the active player (round-robin by turn number).
2. Encode the current state.
3. Enumerate legal actions.
4. Ask the agent to select an action (or pass if no actions are available).
5. Apply the chosen action.
6. Fire all auto-rules.
7. Check the terminal predicate.
8. Return the `Experience`.
"""
function step!(driver::GameDriver)
    game  = driver.game
    state = driver.state

    # Active player
    n_players  = length(game.players)
    player_idx = mod1(state.turn, n_players)
    player     = game.players[player_idx]
    agent      = driver.agents[player]

    # Encode state before action
    enc_before = encode_state(state.world, state.counters, state.turn, driver.T_max)

    # Legal actions
    lib           = game.rules[player]
    legal_actions = enumerate_legal_actions(lib, state, player)

    # Agent selects action
    chosen_action = if isempty(legal_actions)
        nothing
    else
        select_action(agent, enc_before, legal_actions)
    end

    # Apply chosen action
    if chosen_action !== nothing
        idx = rule_index(lib, chosen_action.entry)
        apply_rule!(state, chosen_action, player, idx)
    end

    # Fire auto-rules
    auto_results = fire_auto_rules!(state, game.auto)

    # Check terminal predicate
    done, winner = game.terminal(state.world)

    # Encode state after action + auto-rules
    enc_after = encode_state(state.world, state.counters, state.turn, driver.T_max)

    # Advance turn counter
    state.turn += 1

    # Force done if T_max reached
    if state.turn > driver.T_max
        done   = true
        winner = winner  # preserve any winner set by terminal predicate
    end

    driver._done   = done
    driver._winner = winner

    info = Dict{Symbol, Any}(
        :auto_results   => auto_results,
        :budget_snapshot => copy(state.counters),
    )

    return Experience(player, enc_before, legal_actions, chosen_action,
                      enc_after, done, winner, info)
end

# ─── Iteration interface ──────────────────────────────────────────────────────

function Base.iterate(driver::GameDriver)
    driver._done && return nothing
    exp = step!(driver)
    return (exp, nothing)
end

function Base.iterate(driver::GameDriver, ::Nothing)
    driver._done && return nothing
    exp = step!(driver)
    return (exp, nothing)
end

# ─── run_game ─────────────────────────────────────────────────────────────────

"""
    run_game(game::Game, agents::Dict{Symbol,<:AbstractAgent}; T_max=1000)
        -> Vector{Experience}

Run a single complete episode and return all experiences.
"""
function run_game(game::Game, agents::Dict{Symbol, <:AbstractAgent}; T_max::Int=1000)
    driver = GameDriver(game, agents; T_max=T_max)
    experiences = Experience[]
    for exp in driver
        push!(experiences, exp)
        exp.done && break
    end
    return experiences
end
