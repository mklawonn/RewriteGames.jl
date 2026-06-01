"""
GPU NAC/PAC enforcement tests.

The GPU CSP solver does not reify NACs that introduce new elements (e.g. "a
vertex already has a self-loop"); those are enforced by the host-side
`_filter_nac_solutions` post-filter.  Two GPU code paths used to bypass that
filter and so fired rules whose NAC should have blocked them:

  * the single-sync **native pipeline** (`_gpu_native_pipeline!`), and
  * the **GPU-player fast path** in `_gpu_solve_inplace!`.

Both now route rules carrying NAC/PAC conditions to the standard, NAC-filtered
solve path.  These tests pin a player rule against a world where the NAC blocks
every match — the rule must NOT fire.
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

@testset "GPU NAC enforcement (player fast path)" begin
    # Rule: add a self-loop to a vertex that does not already have one.
    K      = Graph(1)
    L      = Graph(1)
    R      = Graph(1); add_edge!(R, 1, 1)
    L_loop = Graph(1); add_edge!(L_loop, 1, 1)
    rule   = Rule(homomorphism(K, L; monic=true),
                  homomorphism(K, R; monic=true);
                  ac = [NAC(homomorphism(L, L_loop; monic=true))])

    pra    = PlayerRuleApp(:add_loop_if_none, rule, K, :alice)
    gs     = mk_game_sched((;), (init=:I,), Names(Dict("I" => K)), (r=pra,),
                           quote
                               s, f = r(init)
                               return s, f
                           end)
    # GPUFunctionPlayer is an AbstractGPUPlayer → exercises the fast path.
    agents = Dict{Symbol, AbstractAgent}(:alice => GPUFunctionPlayer((_, _c, _n, _t) -> 1))

    @testset "NAC blocks every match → rule does not fire" begin
        # Both vertices already have a self-loop, so the NAC blocks both matches.
        G = Graph(2); add_edge!(G, 1, 1); add_edge!(G, 2, 2)
        exps = gpu_run_game_sched!(gs, G, agents; T_max = 3)
        # No firing: every candidate is NAC-violating and must be filtered out.
        @test count(e -> e.player == :alice, exps) == 0
    end

    @testset "NAC satisfied → rule fires" begin
        # No self-loops yet, so the rule may add one (positive control).
        G = Graph(2)
        exps = gpu_run_game_sched!(gs, G, agents; T_max = 1)
        @test count(e -> e.player == :alice, exps) >= 1
    end
end
