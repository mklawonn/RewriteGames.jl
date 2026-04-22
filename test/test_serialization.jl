using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Serialization tests" begin
    W0 = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    W1 = @acset Graph begin V=3; E=1; src=[1]; tgt=[2] end

    # Build a minimal GameHistory with one step
    hist = GameHistory(W0)
    record_step!(hist;
        chosen_action = nothing,
        legal_actions = Action[],
        player        = :alice,
        path          = [:sched, :alice],
        winner        = :alice,
        t             = 0)
    record_world!(hist, W1, 1)

    dir = mktempdir()

    @testset "write_history creates expected files" begin
        write_history(dir, hist)
        @test isfile(joinpath(dir, "world",  "0.json"))
        @test isfile(joinpath(dir, "world",  "1.json"))
        @test isfile(joinpath(dir, "scalars.json"))
        # No chosen action was recorded, so chosen/0.json should not exist
        @test !isfile(joinpath(dir, "chosen", "0.json"))
    end

    @testset "read_history round-trips scalar narratives" begin
        hist2 = read_history(dir, Graph)
        @test history_length(hist2) == 1
        @test get_player(hist2, 0)   == :alice
        @test get_path(hist2, 0)     == [:sched, :alice]
        @test get_terminal(hist2, 0) == :alice
    end

    @testset "read_history round-trips world narrative" begin
        hist2  = read_history(dir, Graph)
        w0_rt  = get_world(hist2, 0)
        w1_rt  = get_world(hist2, 1)
        @test w0_rt isa Graph
        @test nparts(w0_rt, :V) == 2
        @test nparts(w0_rt, :E) == 1
        @test nparts(w1_rt, :V) == 3
    end

    @testset "write_history with chosen span" begin
        I_empty = Graph()
        R_one_v = Graph(1)
        rule  = Rule(ACSetTransformation(I_empty, I_empty),
                     ACSetTransformation(I_empty, R_one_v))
        entry = RuleEntry(rule; name=:add_vertex)
        act   = Action(entry, ACSetTransformation(I_empty, W0))

        hist3 = GameHistory(W0)
        record_step!(hist3;
            chosen_action = act, legal_actions = [act],
            player = :alice, path = Symbol[], winner = nothing, t = 0)
        record_world!(hist3, W1, 1)

        dir3 = mktempdir()
        write_history(dir3, hist3)
        @test isfile(joinpath(dir3, "chosen",    "0.json"))
        @test isfile(joinpath(dir3, "available", "0.json"))

        hist4 = read_history(dir3, Graph)
        ch = get_chosen(hist4, 0)
        @test ch !== nothing
        @test ch.rule_name == :add_vertex
        @test ch.L isa Graph
        @test ch.K isa Graph
        @test ch.R isa Graph
    end
end
