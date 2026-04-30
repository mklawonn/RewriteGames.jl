using Test
using Catlab
using RewriteGames

@testset "Core types" begin
    @testset "Game constructor" begin
        game = Game(
            nothing;
            players  = [:red, :blue],
            terminal = (W) -> (false, nothing),
            initial  = () -> nothing,
        )
        @test game.players == [:red, :blue]
        @test game.schema  === nothing
    end

    @testset "GameState" begin
        empty_graph = Graph()
        gs = GameState(empty_graph, 1)
        @test gs.world === empty_graph
        @test gs.turn  == 1

        gs2 = copy(gs)
        @test gs2.turn == gs.turn
    end

    @testset "nplayers" begin
        game = Game(nothing; players=[:a, :b, :c],
                    terminal=(W)->(false,nothing), initial=()->nothing)
        @test nplayers(game) == 3
        @test length(game) == 3
    end
end
