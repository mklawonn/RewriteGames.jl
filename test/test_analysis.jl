using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Analysis utilities" begin
    W   = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    enc = encode_state(W, 1, 50)

    rule_stub = RuleEntry("r"; name=:move)
    act = Action(rule_stub, nothing)

    exp_win  = Experience(:alice, enc, [act], act,     enc, true,  :alice,  Dict{Symbol,Any}(), Symbol[])
    exp_draw = Experience(:bob,   enc, [act], nothing, enc, true,  nothing, Dict{Symbol,Any}(), Symbol[])
    exp_mid  = Experience(:alice, enc, [act], act,     enc, false, nothing, Dict{Symbol,Any}(), Symbol[])

    exps = [exp_mid, exp_win, exp_draw]

    @testset "win_rate" begin
        @test win_rate(exps, :alice) ≈ 0.5
        @test win_rate(exps, :bob)   ≈ 0.0
        @test win_rate(Experience[], :alice) == 0.0
    end

    @testset "episode_length" begin
        @test episode_length(exps) == 3
        @test episode_length(Experience[]) == 0
    end

    @testset "action_counts" begin
        counts = action_counts(exps)
        @test counts[:move] == 2
        @test counts[:pass] == 1
    end
end
