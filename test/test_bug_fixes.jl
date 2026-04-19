using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Bug-fix regression tests" begin

    @testset "BUG-1: encode_state does not alias live counters" begin
        W = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
        counters = Dict{Tuple{Symbol,Int},Int}((:p, 1) => 5)

        enc = encode_state(W, counters, 1, 50)

        # Mutate the original counters dict
        counters[(:p, 1)] = 0

        # enc.raw.counters must still hold the snapshot value (5), not 0
        @test enc.raw.counters[(:p, 1)] == 5
    end

    @testset "BUG-2: fire_auto_rules! applies all matched morphisms" begin
        # Use a deterministic auto-rule: add one vertex.
        # Start with Graph(2); after firing, every match that was snapshotted
        # should be applied once.  The key check is that `fired` reflects the
        # actual number of applications, not zero (which the old dead code path
        # would have produced when reassignment was silently ignored).
        I_empty = Graph()
        R_one_v = Graph(1)
        rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                          ACSetTransformation(I_empty, R_one_v))
        ar = AutoRule(rule_add_v; name=:grow)

        W = Graph(0)          # empty graph
        state = GameState(W, Dict{Tuple{Symbol,Int},Int}(), 1)

        results = fire_auto_rules!(state, [ar])

        # There is exactly 1 match of the empty-LHS rule in any graph,
        # so exactly 1 application should have been made.
        @test results[1].fired == 1
        @test nparts(state.world, :V) == 1
    end

end
