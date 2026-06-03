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

    @testset "NAC filters selectively (mixed candidates)" begin
        # v1, v2 already have self-loops; v3 does not.  The NAC must block the two
        # looped vertices but leave v3 a valid match — the filter has to select,
        # not act all-or-nothing.  This exercises the GPU-native NAC path
        # (single new element = the self-loop edge, FKs to the pinned vertex).
        G = Graph(3); add_edge!(G, 1, 1); add_edge!(G, 2, 2)
        exps = gpu_run_game_sched!(gs, G, agents; T_max = 1)
        @test count(e -> e.player == :alice, exps) == 1   # fires once, on v3
        wf = exps[end].next_state.world
        has_loop(v) = any(i -> wf[i, :src] == v && wf[i, :tgt] == v, parts(wf, :E))
        @test all(has_loop, parts(wf, :V))   # v3 received its self-loop
        @test nparts(wf, :E) == 3            # and no duplicate loop on v1/v2
    end
end

@testset "GPU NAC enforcement (general path: 2-new-element NAC)" begin
    # Rule: add a self-loop to a vertex that has NO outgoing edge.  The NAC adds
    # a NEW vertex w AND a NEW edge v→w, so its forbidden structure has TWO new
    # elements and the edge's target is itself new.  That disqualifies the
    # single-element NacSpec fast path and routes to the general GPU-native
    # condition solver (`_gpu_filter_conditions`, which lowers the condition
    # pattern and runs `gpu_dive_solve` pinned to the match).
    K     = Graph(1)
    L     = Graph(1)
    R     = Graph(1); add_edge!(R, 1, 1)
    L_out = Graph(2); add_edge!(L_out, 1, 2)         # v has an out-edge to a new vertex
    # Build the NAC morphism explicitly (maps the matched vertex to vertex 1, the
    # source of the new edge); `homomorphism(L, L_out; monic=true)` errors here
    # because two monic homs exist (v→1, v→2) and it expects a unique one.
    nac_mor = ACSetTransformation(L, L_out; V=[1])
    rule  = Rule(homomorphism(K, L; monic=true),
                 homomorphism(K, R; monic=true);
                 ac = [NAC(nac_mor)])
    pra   = PlayerRuleApp(:loop_if_no_out, rule, K, :alice)
    gs    = mk_game_sched((;), (init=:I,), Names(Dict("I" => K)), (r=pra,),
                          quote
                              s, f = r(init)
                              return s, f
                          end)
    agents = Dict{Symbol, AbstractAgent}(:alice => GPUFunctionPlayer((_, _c, _n, _t) -> 1))

    @testset "NAC blocks every match (all vertices have an out-edge)" begin
        G = Graph(2); add_edge!(G, 1, 2); add_edge!(G, 2, 1)   # 2-cycle
        exps = gpu_run_game_sched!(gs, G, agents; T_max = 3)
        @test count(e -> e.player == :alice, exps) == 0
    end

    @testset "NAC satisfied → rule fires (no out-edges)" begin
        G = Graph(2)                                            # no edges
        exps = gpu_run_game_sched!(gs, G, agents; T_max = 1)
        @test count(e -> e.player == :alice, exps) >= 1
    end

    @testset "general filter == CPU homsearch filter (mixed world)" begin
        # Only v1 has an out-edge (1→2); v2, v3 do not.  The general NAC path must
        # block v1 and leave v2/v3.  RG_NAC_DIAG makes the engine run BOTH the GPU
        # and CPU filters per solve and count kept-set mismatches.
        ext = Base.get_extension(RewriteGames, :GPURewritingExt)
        ext._NAC_DIAG_CHECKS[] = 0; ext._NAC_DIAG_MISM[] = 0
        ENV["RG_NAC_DIAG"] = "1"
        try
            G = Graph(3); add_edge!(G, 1, 2)
            exps = gpu_run_game_sched!(gs, G, agents; T_max = 1)
            @test count(e -> e.player == :alice, exps) >= 1   # fires on a valid (out-edge-free) vertex
        finally
            delete!(ENV, "RG_NAC_DIAG")
        end
        @test ext._NAC_DIAG_CHECKS[] > 0     # the general path was actually exercised
        @test ext._NAC_DIAG_MISM[]   == 0    # GPU-native filter kept the identical set to CPU
    end
end
