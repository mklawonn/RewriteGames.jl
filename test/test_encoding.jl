using Test
using RewriteGames
using Catlab

@testset "elements_graph utility" begin
    W     = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
    state = GameState(W, 1)

    eg = elements_graph(state)

    @testset "returns a non-nothing value" begin
        @test eg !== nothing
    end

    @testset "raw ACSet accessible directly" begin
        @test state.world === W
        @test state.turn  == 1
    end
end
