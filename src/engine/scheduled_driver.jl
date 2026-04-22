using Random: shuffle!

# ─── StepResult ───────────────────────────────────────────────────────────────

"""
    StepResult

Lightweight return value from `execute_step!` that signals whether the game
ended during that subtree so composite nodes can short-circuit.
"""
struct StepResult
    done   :: Bool
    winner :: Union{Symbol, Nothing}
end

# ─── ScheduledGameDriver ──────────────────────────────────────────────────────

"""
    ScheduledGameDriver

Drives a game whose `game.schedule` field is a `GameStep` tree.  Each call to
`run_schedule!` executes one full pass of the schedule tree.  The outer loop
(until terminal or T_max) is handled by `_run_scheduled_game`.

Constructed automatically by `run_game` when `game.schedule !== nothing`.
"""
mutable struct ScheduledGameDriver
    game    :: Game
    agents  :: Dict{Symbol, AbstractAgent}
    state   :: GameState
    T_max   :: Int
    _done   :: Bool
    _winner :: Union{Symbol, Nothing}
    history :: GameHistory
end

Base.show(io::IO, d::ScheduledGameDriver) =
    print(io, "ScheduledGameDriver(turn=$(d.state.turn)/$(d.T_max), done=$(d._done))")

function ScheduledGameDriver(game::Game, agents::Dict{Symbol, <:AbstractAgent};
                              T_max::Int = 1000)
    world   = game.initial()
    state   = GameState(world, game)
    history = GameHistory(world)   # records initial world at t = 0
    ScheduledGameDriver(game, Dict{Symbol,AbstractAgent}(agents),
                        state, T_max, false, nothing, history)
end

# ─── execute_step! dispatch ───────────────────────────────────────────────────

"""
    execute_step!(driver, step, path, context) -> StepResult

Recursively execute `step` against `driver.state`.  Narrative data is
accumulated in `driver.history` by `PlayerStep` leaves.

- `path`:    Accumulated schedule path (`Vector{Symbol}`); each node pushes
             its `name` before recursing.
- `context`: `nothing` or an `AgentContext` from a surrounding `ForEachStep`.
"""
function execute_step! end

# ── PlayerStep ────────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: PlayerStep,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    game    = driver.game
    state   = driver.state
    player  = step.player
    agent   = driver.agents[player]
    my_path = push!(copy(path), step.name)

    # t_left is the left endpoint of the transition interval for this step.
    # state.turn starts at 1; the first player step spans [0, 1].
    t_left = state.turn - 1

    enc_before    = encode_state(state.world, state.counters, state.turn, driver.T_max)
    lib           = game.rules[player]
    legal_actions = enumerate_legal_actions_in_context(lib, state, player, context)

    chosen = isempty(legal_actions) ? nothing :
             select_action(agent, enc_before, legal_actions)

    if chosen !== nothing
        apply_rule!(state, chosen, player, rule_index(lib, chosen.entry))
    end

    done, winner = game.terminal(state.world)
    state.turn  += 1
    done = done || state.turn > driver.T_max

    # Record action narratives (transition t_left → t_left+1)
    record_step!(driver.history;
        chosen_action = chosen,
        legal_actions = legal_actions,
        player        = player,
        path          = my_path,
        winner        = done ? winner : nothing,
        t             = t_left)

    # Record world state after the action (and any auto rules that fired earlier
    # via AutoStep siblings in the same Seq node).
    record_world!(driver.history, state.world, t_left + 1)

    driver._done   = done
    driver._winner = winner
    return StepResult(done, winner)
end

# ── AutoStep ──────────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: AutoStep,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    rules = step.rules === nothing ? driver.game.auto : step.rules
    fire_auto_rules!(driver.state, rules)

    done, winner = driver.game.terminal(driver.state.world)
    done = done || driver.state.turn > driver.T_max
    driver._done   = done
    driver._winner = winner
    return StepResult(done, winner)
end

# ── Seq ───────────────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: Seq,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    my_path = push!(copy(path), step.name)
    for child in step.steps
        result = execute_step!(driver, child, my_path, context)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ── Cond ──────────────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: Cond,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    my_path = push!(copy(path), step.name)
    idx = step.pred(driver.state.world)
    1 <= idx <= length(step.branches) ||
        error("Cond '$(step.name)': predicate returned branch index $idx " *
              "but only $(length(step.branches)) branch(es) defined")
    return execute_step!(driver, step.branches[idx], my_path, context)
end

# ── WhileStep ─────────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: WhileStep,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    my_path = push!(copy(path), step.name)
    iter = 0
    while step.cond(driver.state.world) && !driver._done
        iter += 1
        iter > step.max_iter &&
            error("WhileStep '$(step.name)': max_iter=$(step.max_iter) exceeded; " *
                  "ensure the condition eventually becomes false")
        result = execute_step!(driver, step.body, my_path, context)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ── ForEachStep ───────────────────────────────────────────────────────────────

function execute_step!(
    driver  :: ScheduledGameDriver,
    step    :: ForEachStep,
    path    :: Vector{Symbol},
    context :: Union{AgentContext, Nothing},
) :: StepResult

    my_path = push!(copy(path), step.name)

    # Snapshot part ids at iteration start; apply ordering before any rewrites.
    ids = collect(Int, parts(driver.state.world, step.ob))
    step.order === :random   && shuffle!(ids)
    step.order === :reversed && reverse!(ids)

    for id in ids
        # Skip if this instance was deleted by an earlier rewrite in this loop.
        id <= nparts(driver.state.world, step.ob) || continue

        new_ctx = context === nothing ? AgentContext(step.ob, id) :
                                        push_context(context, step.ob, id)

        result = execute_step!(driver, step.body, my_path, new_ctx)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ─── run_schedule! ────────────────────────────────────────────────────────────

"""
    run_schedule!(driver::ScheduledGameDriver) -> Bool

Execute one full pass of `game.schedule`.  Returns `true` if the game ended
during this pass.  Narrative data accumulates in `driver.history`.
"""
function run_schedule!(driver::ScheduledGameDriver)
    execute_step!(driver, driver.game.schedule, Symbol[], nothing)
    return driver._done
end

# ─── Dispatched run_game implementation ──────────────────────────────────────

"""
    _run_scheduled_game(game, agents; T_max) -> GameHistory

Internal implementation for `run_game`.  Loops over full schedule passes until
the terminal predicate fires or T_max is reached, returning the accumulated
`GameHistory`.
"""
function _run_scheduled_game(game::Game, agents::Dict{Symbol, <:AbstractAgent};
                              T_max::Int = 1000)
    driver = ScheduledGameDriver(game, agents; T_max = T_max)
    while !driver._done && driver.state.turn <= driver.T_max
        steps_before = length(driver.history._step_turns)
        run_schedule!(driver)
        # Stop if the schedule fired no PlayerSteps (degenerate schedule guard)
        length(driver.history._step_turns) == steps_before && break
        driver._done && break
    end
    return driver.history
end
