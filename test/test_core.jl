using Test
using RewriteGames

@testset "Core types" begin
    @testset "RuleEntry" begin
        fake_rule = "rule_stub"
        e1 = RuleEntry(fake_rule)
        @test e1.name   == :unnamed
        @test e1.budget === nothing
        @test e1.post_filter === nothing

        e2 = RuleEntry(fake_rule; name=:attack, budget=3)
        @test e2.name   == :attack
        @test e2.budget == 3

        filter_fn = (W, m) -> true
        e3 = RuleEntry(fake_rule; post_filter=filter_fn)
        @test e3.post_filter === filter_fn
    end

    @testset "RuleLibrary" begin
        entries = [RuleEntry("r1"; name=:a), RuleEntry("r2"; name=:b)]
        lib = RuleLibrary(entries)
        @test length(lib) == 2
        @test lib[1].name == :a
        @test lib[2].name == :b
        @test collect(lib) == entries
    end

    @testset "AutoRule" begin
        ar1 = AutoRule("rule_stub")
        @test ar1.name      == :auto
        @test ar1.prob_attr === nothing

        ar2 = AutoRule("rule_stub"; name=:env, prob_attr=:efficacy)
        @test ar2.name      == :env
        @test ar2.prob_attr == :efficacy
    end

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
        gs = GameState(nothing, 1)
        @test gs.world === nothing
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
