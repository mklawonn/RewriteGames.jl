using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Analysis utilities" begin
    W = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end

    # Minimal rule for creating Action objects (add a vertex; L = empty graph)
    I_empty = Graph()
    R_one_v = Graph(1)
    rule_stub = Rule(ACSetTransformation(I_empty, I_empty),
                     ACSetTransformation(I_empty, R_one_v))
    entry = RuleEntry(rule_stub; name=:move)
    act   = Action(entry, ACSetTransformation(I_empty, W))

    # Episode 1: alice wins
    hist_win = GameHistory(W)
    record_step!(hist_win;
        chosen_action = act, legal_actions = [act],
        player = :alice, path = Symbol[], winner = :alice, t = 0)
    record_world!(hist_win, W, 1)

    # Episode 2: no winner (draw / timeout)
    hist_draw = GameHistory(W)
    record_step!(hist_draw;
        chosen_action = nothing, legal_actions = [act],
        player = :bob, path = Symbol[], winner = nothing, t = 0)
    record_world!(hist_draw, W, 1)

    # Episode 3: three steps — alice acts twice, bob passes once; alice wins last
    hist_multi = GameHistory(W)
    record_step!(hist_multi;
        chosen_action = act, legal_actions = [act],
        player = :alice, path = Symbol[], winner = nothing, t = 0)
    record_world!(hist_multi, W, 1)
    record_step!(hist_multi;
        chosen_action = nothing, legal_actions = [act],
        player = :bob, path = Symbol[], winner = nothing, t = 1)
    record_world!(hist_multi, W, 2)
    record_step!(hist_multi;
        chosen_action = act, legal_actions = [act],
        player = :alice, path = Symbol[], winner = :alice, t = 2)
    record_world!(hist_multi, W, 3)

    @testset "winner" begin
        @test winner(hist_win)   === :alice
        @test winner(hist_draw)  === nothing
        @test winner(hist_multi) === :alice
        @test winner(GameHistory(W)) === nothing   # no steps recorded
    end

    @testset "win_rate" begin
        hists = [hist_win, hist_draw]
        @test win_rate(hists, :alice) ≈ 0.5   # 1 win out of 2 episodes
        @test win_rate(hists, :bob)   ≈ 0.0   # 0 wins
        @test win_rate(GameHistory[], :alice) == 0.0
    end

    @testset "episode_length" begin
        @test episode_length(hist_win)         == 1
        @test episode_length(hist_multi)       == 3
        @test episode_length(GameHistory(W))   == 0   # only initial world, no steps
    end

    @testset "action_counts" begin
        counts = action_counts(hist_multi)
        @test counts[:move] == 2   # alice's two :move actions
        @test counts[:pass] == 1   # bob's pass
    end
end
