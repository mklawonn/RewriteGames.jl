"""
RewriteGames benchmark suite.

Entry point for BenchmarkTools.  Loads three sub-suites:

  match_enumeration  – micro-benchmarks of the homomorphism search step
  game_episodes      – full-episode throughput (random agents)
  rl_training        – REINFORCE self-play training (10 updates × 25 episodes)

Run from the repo root:

    julia --project=benchmark benchmark/benchmarks.jl

or via BenchmarkTools tune/run workflow:

    using BenchmarkTools, Pkg
    Pkg.activate("benchmark")
    include("benchmark/benchmarks.jl")
    tune!(SUITE)
    results = run(SUITE; verbose=true)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using BenchmarkTools
using Random

const SUITE = BenchmarkGroup()

# Load sub-suites
include(joinpath(@__DIR__, "suite", "match_enumeration.jl"))
include(joinpath(@__DIR__, "suite", "game_episodes.jl"))
include(joinpath(@__DIR__, "suite", "rl_training.jl"))

SUITE["match_enumeration"] = BENCH_MATCH
SUITE["game_episodes"]     = BENCH_EPISODES
SUITE["rl_training"]       = BENCH_RL

# ── Driver (when executed as a script) ────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Tuning benchmarks ===")
    tune!(SUITE)

    println("\n=== Running benchmarks ===")
    results = run(SUITE; verbose=true, seconds=30)

    println("\n=== Results ===")
    show(stdout, MIME("text/plain"), results)

    # Write findings
    findings_path = joinpath(@__DIR__, "results", "findings.md")
    include(joinpath(@__DIR__, "suite", "write_findings.jl"))
    write_findings(results, findings_path)
    println("\nFindings written to $findings_path")
end
