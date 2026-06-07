# Ordinal attribute threshold constraints (PROP_ATTR_LEQ / PROP_ATTR_GEQ).
#
# Verifies that `turbo_homomorphisms(L, G; thresholds=...)` restricts matches by
# an ordinal attribute threshold and agrees with the CPU ground truth, on both
# the CPU-turbo path and (when available) the GPU path.

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

@present SchFuelG(FreeSchema) begin
    V::Ob
    F::AttrType
    fuel::Attr(V, F)
end
@acset_type FuelG_(SchFuelG, index = [])
const FuelG = FuelG_{Int}

# Matched fuel values (sorted) for a set of homs L→G, keyed by L's single vertex.
matched_fuels(homs, G) = sort([subpart(G, h[:V](1), :fuel) for h in homs])

@testset "Ordinal attribute thresholds" begin
    G = @acset FuelG begin V = 8; fuel = [0, 1, 2, 3, 4, 5, 6, 7] end
    L = @acset FuelG begin V = 1; F = 1; fuel = [AttrVar(1)] end

    cases = [
        (:leq, 3, [0, 1, 2, 3]),
        (:lt,  3, [0, 1, 2]),
        (:geq, 5, [5, 6, 7]),
        (:gt,  5, [6, 7]),
    ]

    backends = Any[nothing]
    CUDA.functional() && push!(backends, CUDA.CUDABackend())

    for (op, val, expected) in cases
        ths = [(:V, 1, :fuel, op, val)]
        for backend in backends
            homs = RewriteGames.turbo_homomorphisms(L, G; backend = backend, thresholds = ths)
            @test matched_fuels(homs, G) == expected
        end
    end

    # No thresholds → all 8 vertices match (regression: registry default empty).
    for backend in backends
        homs = RewriteGames.turbo_homomorphisms(L, G; backend = backend)
        @test length(homs) == 8
    end

    # CPU match-predicate parity helper.
    @testset "attr_threshold_pred parity" begin
        pred = RewriteGames.attr_threshold_pred(
            [RewriteGames.AttrThreshold(:V, 1, :fuel, :leq, 3)])
        all_homs = homomorphisms(L, G)
        kept = filter(pred, all_homs)
        @test sort([subpart(G, h[:V](1), :fuel) for h in kept]) == [0, 1, 2, 3]
    end
end

# ── Phase 2: affine attribute mutation (delta) ───────────────────────────────
@testset "Affine attribute deltas (mutation)" begin
    if !CUDA.functional()
        @test_skip "no GPU"
    else
        @present SchFuelM(FreeSchema) begin V::Ob; F::AttrType; fuel::Attr(V, F) end
        @acset_type FuelM_(SchFuelM, part_type = BitSetParts)
        FuelM = FuelM_{Int}
        cat = ACSetCategory(MADVarACSetCat(FuelM()))

        # Rule: match a V with fuel ≥ 1, structural identity, post-rewrite fuel -= 1.
        Lp = @acset FuelM begin V = 1; F = 1; fuel = [AttrVar(1)] end
        rule = Rule(homomorphism(Lp, Lp; cat = cat), homomorphism(Lp, Lp; cat = cat); cat = cat)
        set_attr_thresholds!(rule, [(:V, 1, :fuel, :geq, 1)])
        set_attr_deltas!(rule, [(:V, 1, :fuel, -1)])

        I  = FuelM()
        mw = merge_wires(I)
        app = PlayerRuleApp(:decr, rule, I, :p; cat = cat)
        N = Names(Dict("I" => I))
        sched = mk_game_sched((tr = :I,), (init = :I,), N, (r = app, mw = mw),
            quote
                p, f = r(mw(init, tr))
                cont = mw(p, f)
                return cont
            end; cat = cat)

        # Identity-encode fuel (encoded == value + 1) so a delta is a pure int add.
        discr = Dict{Symbol, Pair{Function, Function}}(
            :fuel => (v -> Int32(Int(v) + 1)) => (i -> Int(i) - 1))
        agents = Dict{Symbol, AbstractAgent}(:p => GPUFunctionPlayer((_, _c, _n, _t) -> 1))

        W = @acset FuelM begin V = 1; fuel = [3] end
        exps = gpu_run_game_sched!(sched, W, agents; T_max = 5, discretizers = discr)
        wf = exps[end].next_state.world
        # fuel 3→2→1→0 over three firings; turns 4–5 don't match (fuel ≥ 1 fails).
        @test subpart(wf, 1, :fuel) == 0
    end
end
