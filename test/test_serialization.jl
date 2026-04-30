using Test
using RewriteGames
using Catlab

@testset "Serialization tests" begin
    W     = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    state = GameState(W, 1)

    exp = Experience(
        :alice,
        state,
        Action[],
        nothing,
        state,
        true,
        :alice,
        Dict{Symbol, Any}(:auto_results => []),
    )

    path = tempname() * ".arrow"

    @testset "write and read round-trip" begin
        write_experiences(path, [exp])
        @test isfile(path)

        rows = read_experiences(path)
        @test length(rows) == 1

        row = rows[1]
        @test row.player == "alice"
        @test row.turn   == Int32(1)
        @test row.done   == true
        @test row.winner == "alice"
        @test row.action_rule_name == "nothing"
    end

    @testset "multiple experiences" begin
        exps  = [exp, exp, exp]
        path2 = tempname() * ".arrow"
        write_experiences(path2, exps)
        rows2 = read_experiences(path2)
        @test length(rows2) == 3
    end
end
