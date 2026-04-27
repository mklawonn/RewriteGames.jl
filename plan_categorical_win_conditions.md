# Plan: Categorical Effect Handlers for Win Conditions

## Objective
Refactor the RewriteGames engine and the Tic-Tac-Toe tutorial to use pure categorical effect handlers (wiring diagrams and `Query` boxes) for evaluating win conditions. This eliminates the need for an external, global `terminal(world)` Julia callback and fully grounds the game loop in the semantics of traced monoidal categories.

## 1. Tutorial Revisions (`tutorials/TicTacToeGameplay.qmd`)

The tutorial will be updated to demonstrate the "Proper Categorical Handler" approach.

### What to Remove:
*   **The `ttt_terminal` function:** Completely remove this block and its usage. We will no longer rely on a global Julia function to halt the game.
*   **The dedicated win-check schedules:** Remove `x_won_check_gs` (and its migrated `O` counterpart). We don't need a separate sub-schedule just to evaluate wins if we are handling them directly in the main trace loop.

### What to Add / Modify:
*   **Define `Query` Boxes:** Introduce AlgebraicRewriting `Query` boxes for the winning structural patterns.
    ```julia
    x_win_query = Query(x_win_pattern, :check_x_win)
    o_win_query = Query(o_win_pattern, :check_o_win)
    ```
*   **Rewrite the Main Schedule (`game_sched`):** Update the `mk_game_sched` block to act as our `try/catch` handler. The schedule will interleave player turns with the `Query` boxes. If a query matches, it immediately routes to a dedicated exit wire (`x_won` or `o_won`).
    ```julia
    game_sched = mk_game_sched(
        (trace_arg=:I,), (init=:I,), N,
        (X=X_sched_gs, O=O_sched_gs, check_x=x_win_query, check_o=o_win_query, mw=merge_wires(I)),
        quote
            x_moved, x_tie = X([init, trace_arg])
            x_won, x_cont  = check_x(x_moved)
            
            o_moved, o_tie = O(x_cont)
            o_won, o_cont  = check_o(o_moved)
            
            tie = mw(x_tie, o_tie)
            return o_cont, x_won, o_won, tie
        end
    )
    ```
*   **Update Execution:** Change the `run_game_sched!` call to no longer pass a `terminal` argument. Show how the engine deduces the winner strictly from the active exit wire.

## 2. Core Engine Source Alterations (`src/`)

To support this generally, the `Game` definition and the execution engine must be decoupled from the mandatory `terminal` function.

### A. `src/core/game.jl` & `src/dsl.jl`
*   **Remove `terminal`:** Remove the `terminal` field from the `Game` struct and the `@game` macro.
*   **Add `win_conditions` (Optional):** Introduce an optional `win_conditions` parameter (e.g., a `Dict{Symbol, Any}` mapping player names to their winning `Query` or `RuleApp`).
*   **DSL Update:** Allow the `@game` macro to accept `win_conditions: ...` instead of `terminal: ...`.

### B. Schedule Injection Logic
*   **Dynamic Handler Injection:** If a user defines a sequence of turns but provides `win_conditions` separately, the engine should automatically compile the "handler" wiring diagram. It will inject the win-check queries after each respective player's turn and route successful matches to player-specific exit wires.
*   **Manual Override:** If `win_conditions` is *not* provided, the engine assumes the user has manually wired the exit conditions into their provided schedule (as we will do manually in the tutorial).

### C. `src/engine/sched_runner.jl`
*   **Remove `terminal` Evaluation:** `run_game_sched!` and `_exec_player!` will no longer invoke a `terminal` function to determine if the game is over.
*   **Determine Winners via Control Flow:** The engine will evaluate the final state of the diagram's wires when the schedule halts. If the schedule exits on a wire designated for a specific player's win (e.g., the `x_won` wire), the engine retroactively records that player as the winner for the final `Experience` records.
*   **Graceful Timeouts:** Ensure that if `T_max` is reached without an exit wire activating, the engine correctly handles it as a timeout/draw.