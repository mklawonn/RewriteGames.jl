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
`run_schedule!` executes one full pass of the schedule tree and returns all
`Experience` records emitted during that pass.  The outer loop (until terminal
or T_max) is handled by `run_game`.

Constructed automatically by `run_game` when `game.schedule !== nothing`.
"""
mutable struct ScheduledGameDriver
    game    :: Game
    agents  :: Dict{Symbol, AbstractAgent}
    state   :: GameState
    T_max   :: Int
    _done   :: Bool
    _winner :: Union{Symbol, Nothing}
end

Base.show(io::IO, d::ScheduledGameDriver) =
    print(io, "ScheduledGameDriver(turn=$(d.state.turn)/$(d.T_max), done=$(d._done))")

function ScheduledGameDriver(game::Game, agents::Dict{Symbol, <:AbstractAgent};
                              T_max::Int = 1000)
    game.schedule === nothing && error(
        "ScheduledGameDriver requires game.schedule to be a GameStep; " *
        "use GameDriver for schedule=nothing (round-robin) games.")
    world = game.initial()
    state = GameState(world, game)
    ScheduledGameDriver(game, Dict{Symbol,AbstractAgent}(agents),
                        state, T_max, false, nothing)
end

# ─── execute_step! dispatch ───────────────────────────────────────────────────

"""
    execute_step!(driver, step, path, context, out_exps) -> StepResult

Recursively execute `step` against `driver.state`.

- `path`:     Accumulated schedule path (`Vector{Symbol}`); each node pushes
              its `name` before recursing.
- `context`:  `nothing` or an `AgentContext` from a surrounding `ForEachStep`.
- `out_exps`: `Vector{Experience}` that leaf `PlayerStep` nodes append to.
"""
function execute_step! end

# ── PlayerStep ────────────────────────────────────────────────────────────────

function execute_step!(
    driver   :: ScheduledGameDriver,
    step     :: PlayerStep,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
) :: StepResult

    game    = driver.game
    state   = driver.state
    player  = step.player
    agent   = driver.agents[player]
    my_path = push!(copy(path), step.name)

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

    enc_after = encode_state(state.world, state.counters, state.turn, driver.T_max)

    push!(out_exps, Experience(
        player, enc_before, legal_actions, chosen,
        enc_after, done, winner,
        Dict{Symbol,Any}(:budget_snapshot => copy(state.counters),
                         :context         => context),
        my_path,
    ))

    driver._done   = done
    driver._winner = winner
    return StepResult(done, winner)
end

# ── AutoStep ──────────────────────────────────────────────────────────────────

function execute_step!(
    driver   :: ScheduledGameDriver,
    step     :: AutoStep,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
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
    driver   :: ScheduledGameDriver,
    step     :: Seq,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
) :: StepResult

    my_path = push!(copy(path), step.name)
    for child in step.steps
        result = execute_step!(driver, child, my_path, context, out_exps)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ── Cond ──────────────────────────────────────────────────────────────────────

function execute_step!(
    driver   :: ScheduledGameDriver,
    step     :: Cond,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
) :: StepResult

    my_path = push!(copy(path), step.name)
    idx = step.pred(driver.state.world)
    1 <= idx <= length(step.branches) ||
        error("Cond '$(step.name)': predicate returned branch index $idx " *
              "but only $(length(step.branches)) branch(es) defined")
    return execute_step!(driver, step.branches[idx], my_path, context, out_exps)
end

# ── WhileStep ─────────────────────────────────────────────────────────────────

function execute_step!(
    driver   :: ScheduledGameDriver,
    step     :: WhileStep,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
) :: StepResult

    my_path = push!(copy(path), step.name)
    iter = 0
    while step.cond(driver.state.world) && !driver._done
        iter += 1
        iter > step.max_iter &&
            error("WhileStep '$(step.name)': max_iter=$(step.max_iter) exceeded; " *
                  "ensure the condition eventually becomes false")
        result = execute_step!(driver, step.body, my_path, context, out_exps)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ── ForEachStep ───────────────────────────────────────────────────────────────

function execute_step!(
    driver   :: ScheduledGameDriver,
    step     :: ForEachStep,
    path     :: Vector{Symbol},
    context  :: Union{AgentContext, Nothing},
    out_exps :: Vector{Experience},
) :: StepResult

    my_path = push!(copy(path), step.name)

    # Snapshot part ids at iteration start; apply ordering before any rewrites.
    ids = collect(Int, parts(driver.state.world, step.ob))
    step.order === :random   && shuffle!(ids)
    step.order === :reversed && reverse!(ids)

    for id in ids
        # Skip if this instance was deleted by an earlier rewrite in this loop.
        # parts() returns 1:nparts, so id > nparts means the part is gone.
        id <= nparts(driver.state.world, step.ob) || continue

        new_ctx = context === nothing ? AgentContext(step.ob, id) :
                                        push_context(context, step.ob, id)

        result = execute_step!(driver, step.body, my_path, new_ctx, out_exps)
        result.done && return result
    end
    return StepResult(driver._done, driver._winner)
end

# ─── run_schedule! ────────────────────────────────────────────────────────────

"""
    run_schedule!(driver::ScheduledGameDriver) -> Vector{Experience}

Execute one full pass of `game.schedule` and return all `Experience` records
emitted during that pass.
"""
function run_schedule!(driver::ScheduledGameDriver)
    out = Experience[]
    execute_step!(driver, driver.game.schedule, Symbol[], nothing, out)
    return out
end

# ─── Dispatched run_game implementation ──────────────────────────────────────

"""
    _run_scheduled_game(game, agents; T_max) -> Vector{Experience}

Internal implementation for `run_game` when `game.schedule !== nothing`.
Loops over full schedule passes until the terminal predicate fires or T_max
is reached.
"""
function _run_scheduled_game(game::Game, agents::Dict{Symbol, <:AbstractAgent};
                              T_max::Int = 1000)
    driver   = ScheduledGameDriver(game, agents; T_max = T_max)
    all_exps = Experience[]
    while !driver._done && driver.state.turn <= driver.T_max
        exps = run_schedule!(driver)
        append!(all_exps, exps)
        # Stop if the schedule emitted nothing (empty schedule guard) or game ended
        isempty(exps) && break
        all_exps[end].done && break
    end
    return all_exps
end
