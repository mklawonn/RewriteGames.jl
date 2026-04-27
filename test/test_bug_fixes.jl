using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Bug-fix regression tests" begin

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
