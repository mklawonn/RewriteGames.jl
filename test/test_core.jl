using Test
using RewriteGames

@testset "Core types" begin
    @testset "RuleEntry" begin
        # Minimal rule stub (not an actual rewrite rule — avoids AlgebraicRewriting
        # dependency in the core tests)
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
            rules    = Dict(
                :red  => [RuleEntry("r1"; name=:move)],
                :blue => [RuleEntry("r2"; name=:build, budget=5)],
            ),
            auto     = AutoRule[],
            terminal = (W) -> (false, nothing),
            initial  = () -> nothing,
        )
        @test game.players == [:red, :blue]
        @test haskey(game.rules, :red)
        @test haskey(game.rules, :blue)
        @test game.rules[:blue][1].budget == 5
        @test isempty(game.auto)
    end

    @testset "GameState counters" begin
        e_limited   = RuleEntry("r"; name=:lim,   budget=4)
        e_unlimited = RuleEntry("r"; name=:unlim)
        lib = RuleLibrary([e_limited, e_unlimited])
        game = Game(
            nothing;
            players  = [:p],
            rules    = Dict(:p => lib),
            terminal = (W) -> (false, nothing),
            initial  = () -> nothing,
        )
        gs = GameState(nothing, game)
        @test gs.counters[(:p, 1)] == 4
        @test !haskey(gs.counters, (:p, 2))
        @test gs.turn == 1
    end
end
