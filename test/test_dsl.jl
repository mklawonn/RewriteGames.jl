using Test
using RewriteGames

@testset "DSL @game macro" begin
    @testset "@game basic construction" begin
        g = @game nothing begin
            players:  alice, bob
            terminal: (W) -> (false, nothing)
            initial:  () -> nothing
        end

        @test g isa Game
        @test g.players == [:alice, :bob]
    end

    @testset "@game single player" begin
        g = @game nothing begin
            players:  solo
            terminal: (W) -> (false, nothing)
            initial:  () -> nothing
        end
        @test g.players == [:solo]
    end

    @testset "@game schema is stored" begin
        g = @game :my_schema begin
            players:  p
            terminal: (W) -> (false, nothing)
            initial:  () -> nothing
        end
        @test g.schema === :my_schema
    end

    @testset "@game missing players clause errors" begin
        @test_throws ErrorException @macroexpand @game nothing begin
            terminal: (W) -> (false, nothing)
            initial:  () -> nothing
        end
    end

    @testset "@game unrecognised clause errors" begin
        @test_throws ErrorException @macroexpand @game nothing begin
            players: p
            rules: Dict()
        end
    end
end
