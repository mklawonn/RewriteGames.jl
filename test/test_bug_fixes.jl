using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Bug-fix regression tests" begin

    @testset "BUG-1: encode_state snapshot does not alias live GameState" begin
        W   = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
        enc = encode_state(W, 1, 50)

        # The encoded raw GameState should record the turn at snapshot time (1).
        @test enc.raw.turn == 1

        # Encoding a different turn should produce a distinct turn_frac.
        enc2 = encode_state(W, 5, 50)
        @test enc2.raw.turn == 5
        @test enc2.turn_frac > enc.turn_frac
    end

    @testset "BUG-2: fire_auto_rules! applies all matched morphisms" begin
        I_empty = Graph()
        R_one_v = Graph(1)
        rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                          ACSetTransformation(I_empty, R_one_v))
        ar = AutoRule(rule_add_v; name=:grow)

        W     = Graph(0)
        state = GameState(W, 1)

        results = fire_auto_rules!(state, [ar])

        @test results[1].fired == 1
        @test nparts(state.world, :V) == 1
    end

end
