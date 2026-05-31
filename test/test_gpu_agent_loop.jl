"""
GPU agent-loop (`BOX_AGENT_LOOP`) execution tests.

Regression coverage for three properties the GPU runner must guarantee:

  1. An `agent(pra; n=:X)` box actually *executes* on the GPU (it used to be
     compiled but silently skipped — the move phase never ran).
  2. The loop iterates once per live agent instance, read dynamically from the
     world, so worlds with **more than 256** instances are fully covered rather
     than truncated by the old `MAX_CHUNKS=4` (256-element) cap.
  3. The firing count equals the number of agent instances for every world
     size, i.e. the CPU spec `length(homomorphisms(single-vertex, world)) == nv`.

(Per RewriteGames.jl/AGENT.md we do not compare CPU vs GPU experience counts
directly — the CPU and GPU runners enumerate in different orders and the bare
top-level agent sched uses different turn/T_max accounting. We instead assert
the GPU count equals the instance count, which is the property that matters.)

The rule preserves the matched vertex (K = L = one vertex) and adds a self-loop,
so the agent interface is a single vertex and the loop ranges over every vertex.
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

@testset "GPU agent loop (BOX_AGENT_LOOP)" begin
    K_v    = Graph(1)
    L_v    = Graph(1)
    R_loop = Graph(1); add_edge!(R_loop, 1, 1)
    rule_loop = Rule(ACSetTransformation(K_v, L_v, V=[1]),
                     ACSetTransformation(K_v, R_loop, V=[1]))

    pra_loop = PlayerRuleApp(:add_loop, rule_loop, K_v, :alice)
    gs       = agent(pra_loop; n=:V)
    agents   = Dict(:alice => FunctionAgent((s, acts) -> first(acts)))
    never    = (W) -> (false, nothing)

    @testset "executes once per vertex (small world)" begin
        exps = gpu_run_game_sched!(gs, Graph(5), agents; T_max=3)
        @test count(e -> e.player == :alice, exps) == 5
    end

    @testset "no 256-element cap — large world ($(310) vertices)" begin
        # 310 > 256: under the old MAX_CHUNKS=4 clamp the loop would have stopped
        # short; the dynamic n_live-based count must cover all of them.
        exps = gpu_run_game_sched!(gs, Graph(310), agents; T_max=3)
        @test count(e -> e.player == :alice, exps) == 310
    end

    @testset "firing count == instance count across sizes (incl. >256)" begin
        for N in (1, 7, 64, 257, 300)
            exps = gpu_run_game_sched!(gs, Graph(N), agents; T_max=3)
            @test count(e -> e.player == :alice, exps) == N
        end
    end
end
