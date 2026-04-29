"""
Game-episode throughput benchmarks.

Measures wall time for:
  single_episode   – one game with random agents, three match strategies
  batch_200        – 200 games (the Part-5 statistics run in the tutorial)
"""

include(joinpath(@__DIR__, "ttt_setup.jl"))

const BENCH_EPISODES = BenchmarkGroup()

# Pre-build the three schedule variants
const SCHED_BASELINE     = build_game_sched(use_cache=false, use_fast=false)
const SCHED_FAST         = build_game_sched(use_cache=false, use_fast=true)
const SCHED_CACHE        = build_game_sched(use_cache=true,  use_fast=false)

# ── Single episode ────────────────────────────────────────────────────────────

BENCH_EPISODES["single_episode"] = BenchmarkGroup()

BENCH_EPISODES["single_episode"]["baseline"] = @benchmarkable begin
    Random.seed!(42)
    run_game_sched!($SCHED_BASELINE, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
end

BENCH_EPISODES["single_episode"]["fast_path"] = @benchmarkable begin
    Random.seed!(42)
    run_game_sched!($SCHED_FAST, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
end

BENCH_EPISODES["single_episode"]["incremental_cache"] = @benchmarkable begin
    Random.seed!(42)
    run_game_sched!($SCHED_CACHE, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
end

# ── Batch of 200 episodes ─────────────────────────────────────────────────────

BENCH_EPISODES["batch_200"] = BenchmarkGroup()

BENCH_EPISODES["batch_200"]["baseline"] = @benchmarkable begin
    Random.seed!(42)
    [run_game_sched!($SCHED_BASELINE, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
     for _ in 1:200]
end seconds=60

BENCH_EPISODES["batch_200"]["fast_path"] = @benchmarkable begin
    Random.seed!(42)
    [run_game_sched!($SCHED_FAST, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
     for _ in 1:200]
end seconds=60

BENCH_EPISODES["batch_200"]["incremental_cache"] = @benchmarkable begin
    Random.seed!(42)
    [run_game_sched!($SCHED_CACHE, $TTT_GAME, $RANDOM_AGENTS; T_max=20)
     for _ in 1:200]
end seconds=60
