using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Serialization tests" begin
    W   = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    enc = encode_state(W, 1, 50)

    I_empty = Graph()
    R_one_v = Graph(1)
    rule  = Rule(ACSetTransformation(I_empty, I_empty),
                 ACSetTransformation(I_empty, R_one_v))
    entry = RuleEntry(rule; name=:add_vertex)

    exp = Experience(
        :alice,
        enc,
        Action[],
        nothing,
        enc,
        true,
        :alice,
        Dict{Symbol, Any}(:auto_results => []),
        Symbol[],
    )

    path = tempname() * ".arrow"

    @testset "write and read round-trip" begin
        write_experiences(path, [exp])
        @test isfile(path)

        rows = read_experiences(path)
        @test length(rows) == 1

        row = rows[1]
        @test row.player == "alice"
        @test row.done   == true
        @test row.winner == "alice"
        @test row.action_rule_name == "nothing"
        @test row.turn_frac ≈ enc.turn_frac
    end

    @testset "multiple experiences" begin
        exps  = [exp, exp, exp]
        path2 = tempname() * ".arrow"
        write_experiences(path2, exps)
        rows2 = read_experiences(path2)
        @test length(rows2) == 3
    end
end
