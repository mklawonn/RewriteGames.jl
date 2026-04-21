using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Analysis utilities" begin
    # Build a small set of experiences to analyse
    W = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    enc = encode_state(W, Dict{Tuple{Symbol,Int},Int}(), 1, 50)

    rule_stub = RuleEntry("r"; name=:move)
    act = Action(rule_stub, nothing)

    exp_win = Experience(:alice, enc, [act], act, enc, true, :alice,
                         Dict{Symbol,Any}())
    exp_draw = Experience(:bob, enc, [act], nothing, enc, true, nothing,
                          Dict{Symbol,Any}())
    exp_mid  = Experience(:alice, enc, [act], act, enc, false, nothing,
                          Dict{Symbol,Any}())

    exps = [exp_mid, exp_win, exp_draw]

    @testset "win_rate" begin
        @test win_rate(exps, :alice) ≈ 0.5   # 1 win out of 2 terminals
        @test win_rate(exps, :bob)   ≈ 0.0   # 0 wins
        @test win_rate(Experience[], :alice) == 0.0
    end

    @testset "episode_length" begin
        @test episode_length(exps) == 3
        @test episode_length(Experience[]) == 0
    end

    @testset "action_counts" begin
        counts = action_counts(exps)
        @test counts[:move] == 2   # exp_mid and exp_win each have action :move
        @test counts[:pass] == 1   # exp_draw passed
    end
end
