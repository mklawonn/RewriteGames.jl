using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

# ─── Fixtures ────────────────────────────────────────────────────────────────

I_empty = Graph()
R_one_v = Graph(1)
rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                  ACSetTransformation(I_empty, R_one_v))

@testset "JSON Game/GameSched serialization" begin

    # ─── acset_to_dict / dict_to_acset ───────────────────────────────────────

    @testset "acset round-trip: empty graph" begin
        W = Graph()
        d = RewriteGames.acset_to_dict(W)
        W2 = RewriteGames.dict_to_acset(Graph, d)
        @test nparts(W2, :V) == 0
        @test nparts(W2, :E) == 0
    end

    @testset "acset round-trip: graph with vertices only" begin
        W = Graph(3)
        d = RewriteGames.acset_to_dict(W)
        W2 = RewriteGames.dict_to_acset(Graph, d)
        @test nparts(W2, :V) == 3
        @test nparts(W2, :E) == 0
    end

    @testset "acset round-trip: graph with vertices and edges" begin
        W = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
        d = RewriteGames.acset_to_dict(W)
        W2 = RewriteGames.dict_to_acset(Graph, d)
        @test nparts(W2, :V) == 3
        @test nparts(W2, :E) == 2
        @test collect(subpart(W2, :src)) == [1, 2]
        @test collect(subpart(W2, :tgt)) == [2, 3]
    end

    # ─── rule_to_dict / dict_to_rule ─────────────────────────────────────────

    @testset "rule round-trip: add-vertex rule" begin
        d = RewriteGames.rule_to_dict(rule_add_v)
        r2 = RewriteGames.dict_to_rule(d, Graph)
        @test r2 isa Rule
        # LHS domain and codomain sizes match original
        @test nparts(dom(r2.L), :V) == 0
        @test nparts(codom(r2.L), :V) == 0
        @test nparts(codom(r2.R), :V) == 1
    end

    # ─── write_game / read_game: no win_conditions ───────────────────────────

    @testset "write_game/read_game round-trip (terminal, no win_conditions)" begin
        game = Game(nothing;
                    players   = [:alice],
                    terminal  = (W) -> (nparts(W, :V) >= 3, nothing),
                    initial   = () -> Graph(0))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)

        path = tempname() * ".json"
        write_game(path, game, gs)
        @test isfile(path)

        game2, gs2 = read_game(path, Graph, N)
        @test game2 isa Game
        @test gs2 isa GameSched
        @test Set(game2.players) == Set([:alice])
        # Initial state is correctly reconstructed
        W0 = game2.initial()
        @test nparts(W0, :V) == 0
    end

    @testset "reconstructed GameSched runs correctly" begin
        game = Game(nothing;
                    players   = [:alice],
                    terminal  = (W) -> (nparts(W, :V) >= 3, nothing),
                    initial   = () -> Graph(0))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)

        path = tempname() * ".json"
        write_game(path, game, gs)
        game2, gs2 = read_game(path, Graph, N)

        agents = Dict(:alice => FunctionAgent((s, a) -> rand(a)))
        done_at_3 = (W) -> (nparts(W, :V) >= 3, nothing)
        exps = run_game_sched!(gs2, Graph(0), agents;
                               T_max=10, terminal=done_at_3)
        @test !isempty(exps)
    end

    # ─── write_game / read_game: with win_conditions ─────────────────────────

    @testset "win_conditions with Symbol values round-trips correctly" begin
        game = Game(nothing;
                    players        = [:alice, :bob],
                    initial        = () -> Graph(0),
                    win_conditions = Dict{Symbol, Any}(:alice_won => :alice,
                                                       :bob_won   => :bob))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N, (box=pra,),
            quote won, tie = box(init); return won, tie end)

        path = tempname() * ".json"
        write_game(path, game, gs)
        game2, _ = read_game(path, Graph, N)
        @test game2.win_conditions !== nothing
        @test game2.win_conditions[:alice_won] == :alice
        @test game2.win_conditions[:bob_won]   == :bob
    end

    @testset "win_conditions=nothing serializes and reads back as nothing" begin
        game = Game(nothing;
                    players        = [:alice],
                    initial        = () -> Graph(0),
                    win_conditions = nothing)
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N, (box=pra,),
            quote won, tie = box(init); return won, tie end)

        path = tempname() * ".json"
        write_game(path, game, gs)
        game2, _ = read_game(path, Graph, N)
        @test game2.win_conditions === nothing
    end

    # ─── box_to_dict / dict_to_box ───────────────────────────────────────────

    @testset "PlayerRuleApp box round-trips with use_cache and match_limit" begin
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice;
                            use_cache=true, match_limit=5)
        d = RewriteGames.box_to_dict(pra)
        @test d["type"] == "PlayerRuleApp"
        @test d["name"] == "add_v"
        @test d["player"] == "alice"
        @test d["use_cache"] == true
        @test d["match_limit"] == 5

        pra2 = RewriteGames.dict_to_box(d, Graph, I_empty)
        @test pra2 isa PlayerRuleApp
        @test pra2.name == :add_v
        @test pra2.player == :alice
        @test pra2.use_cache == true
        @test pra2.match_limit == 5
    end

    @testset "MergeWires box serializes correctly" begin
        mw = merge_wires(I_empty)
        d = RewriteGames.box_to_dict(mw)
        @test d["type"] == "MergeWires"
    end

    # ─── multiple players ─────────────────────────────────────────────────────

    @testset "multi-player game preserves player list order" begin
        players = [:alice, :bob, :carol]
        game = Game(nothing;
                    players   = players,
                    terminal  = (W) -> (false, nothing),
                    initial   = () -> Graph(0))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N, (box=pra,),
            quote won, tie = box(init); return won, tie end)

        path = tempname() * ".json"
        write_game(path, game, gs)
        game2, _ = read_game(path, Graph, N)
        @test game2.players == players
    end

end
