"""
Tests for count-weighted random-descent "take N" sampling (Feature 1).

The bar (per design): every sampled match must be a *valid* solution (a member of
the full enumeration), the count semantics must hold, and the sampler must be able
to reach every solution.  Uniformity is best-effort and only loosely checked.

Compares against `turbo_homomorphisms(L, G)` (full enumeration) on both the CPU
reference (`cpu_sample_solve`) and the GPU kernel (`gpu_turbo_sample`), reached via
the `take`/`seed` keywords on `turbo_homomorphisms`.
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

# Flat assignment tuple of a homomorphism, for set membership / comparison.
_assign(m, L) = (S = acset_schema(L); Tuple(m[o](i) for o in ob(S) for i in parts(L, o)))
_assign_set(homs, L) = Set(_assign(m, L) for m in homs)

# (name, L, G) cases spanning a range of solution-space sizes.
const SAMPLING_CASES = [
    ("path3→path4",      path_graph(Graph, 3), path_graph(Graph, 4)),
    ("path2→path5",      path_graph(Graph, 2), path_graph(Graph, 5)),
    ("edge→cycle4",      path_graph(Graph, 2), cycle_graph(Graph, 4)),
    ("path3→cycle5",     path_graph(Graph, 3), cycle_graph(Graph, 5)),
]

@testset "take sampling — $(name)" for (name, L, G) in SAMPLING_CASES
    full     = RewriteGames.turbo_homomorphisms(L, G; backend = nothing)
    full_set = _assign_set(full, L)
    S        = length(full_set)
    @test S > 0

    for (label, backend) in (
            ("CPU", nothing),
            ("GPU", CUDA.functional() ? CUDA.CUDABackend() : nothing))
        # Skip the GPU row when CUDA is unavailable (backend falls back to CPU,
        # already covered by the CPU row).
        label == "GPU" && backend === nothing && continue

        @testset "$label" begin
            # Validity + count semantics across several seeds and take sizes.
            for take in (1, 2, max(1, S ÷ 2), S, S + 5)
                for seed in 0:3
                    sampled = RewriteGames.turbo_homomorphisms(
                        L, G; backend = backend, take = take, seed = seed)
                    sset = _assign_set(sampled, L)
                    @test length(sampled) == length(sset)          # distinct
                    @test issubset(sset, full_set)                  # all valid
                    @test length(sampled) <= take                  # respects cap
                    @test length(sampled) <= S
                    if take >= S
                        @test sset == full_set                      # take ≥ |sols| ⇒ all
                    end
                end
            end

            # Coverage: across many seeds, every solution is reachable.
            if S <= 40
                seen = Set{NTuple}()
                for seed in 0:200
                    for m in RewriteGames.turbo_homomorphisms(
                            L, G; backend = backend, take = 2, seed = seed)
                        push!(seen, _assign(m, L))
                    end
                    length(seen) == S && break
                end
                @test seen == full_set
            end
        end
    end
end

# Soft, non-flaky uniformity check on a symmetric case: edge → 4-cycle has 8
# automorphism-equivalent matches; over many single draws no solution should be
# wildly over/under-represented.
@testset "take sampling — soft uniformity" begin
    L, G = path_graph(Graph, 2), cycle_graph(Graph, 4)
    backend = CUDA.functional() ? CUDA.CUDABackend() : nothing
    full_set = _assign_set(RewriteGames.turbo_homomorphisms(L, G; backend = nothing), L)
    S = length(full_set)

    counts = Dict{NTuple, Int}()
    ndraws = 0
    for seed in 0:1199
        for m in RewriteGames.turbo_homomorphisms(L, G; backend = backend, take = 1, seed = seed)
            counts[_assign(m, L)] = get(counts, _assign(m, L), 0) + 1
            ndraws += 1
        end
    end
    @test length(counts) == S                       # every solution sampled
    expected = ndraws / S
    # Generous band: no solution off by more than 3× from uniform expectation.
    for (_, c) in counts
        @test expected / 3 <= c <= expected * 3
    end
end

# End-to-end through the GPU player fast path in `_gpu_solve_inplace!`: a
# GPUFunctionPlayer (AbstractGPUPlayer) triggers the scratch-based sampler.
@testset "take sampling — scheduler fast path" begin
    if !CUDA.functional()
        @test_skip "CUDA required for the GPU player fast path"
    else
        # Monic "add edge between two distinct vertices": n_vars=2, no NAC, so the
        # fast path (and thus `_gpu_turbo_sample_scratch!`) is exercised.
        L2 = Graph(2); R2 = Graph(2); add_edge!(R2, 1, 2)
        rule_e = Rule(ACSetTransformation(L2, L2, V = [1, 2]),
                      ACSetTransformation(L2, R2, V = [1, 2]); monic = true)
        pra = PlayerRuleApp(:add_e, rule_e, Graph(), :alice)
        gs  = mk_game_sched((;), (init = :I,), Names(Dict("I" => Graph())), (add_e = pra,),
                            quote
                                s, f = add_e(init)
                                return s, f
                            end)
        agents = Dict{Symbol, AbstractAgent}(:alice => GPUFunctionPlayer((_, _c, _n, _t) -> 1))

        # Runs and applies a valid rewrite with sampling on (take) and off (nothing).
        for take in (nothing, 1, 3)
            exps = gpu_run_game_sched!(gs, Graph(5), agents; T_max = 1, take = take)
            @test count(e -> e.player == :alice, exps) >= 1
            w = exps[end].next_state.world
            @test ne(w) == 1                          # exactly one edge added
            @test src(w, 1) != tgt(w, 1)              # endpoints distinct (monic)
        end

        # Sampling actually varies which match is presented: vary the seed and
        # collect the chosen edge.  With ~20 monic matches we expect >1 distinct.
        chosen = Set{Tuple{Int,Int}}()
        for sd in 0:19
            exps = gpu_run_game_sched!(gs, Graph(5), agents; T_max = 1, take = 1, sample_seed = sd)
            w = exps[end].next_state.world
            ne(w) >= 1 && push!(chosen, (src(w, 1), tgt(w, 1)))
        end
        @test length(chosen) >= 2
    end
end
