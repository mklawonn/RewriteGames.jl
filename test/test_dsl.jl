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

    @testset "@game terminal defaults to nothing" begin
        g = @game nothing begin
            players: solo
            initial: () -> nothing
        end
        @test g isa Game
        @test g.terminal === nothing
        @test g.win_conditions === nothing
    end

    @testset "@game win_conditions clause" begin
        g = @game nothing begin
            players:        alice, bob
            initial:        () -> nothing
            win_conditions: Dict{Symbol,Any}(:alice_won => :alice, :tie => nothing)
        end
        @test g isa Game
        @test g.terminal === nothing
        @test g.win_conditions isa Dict
        @test g.win_conditions[:alice_won] === :alice
        @test g.win_conditions[:tie]       === nothing
    end

    @testset "@game terminal and win_conditions can coexist" begin
        g = @game nothing begin
            players:        solo
            terminal:       (W) -> (false, nothing)
            initial:        () -> nothing
            win_conditions: Dict{Symbol,Any}(:won => :solo)
        end
        @test g.terminal !== nothing
        @test g.win_conditions[:won] === :solo
    end
end
