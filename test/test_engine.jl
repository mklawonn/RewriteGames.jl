using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Engine tests" begin

    # ── Minimal graph schema ──────────────────────────────────────────────────
    # Objects: V, E; morphisms: src::Hom(E,V), tgt::Hom(E,V)
    # We use Catlab's built-in Graph type which has exactly this schema.

    # ── Rule 1: add a vertex ──────────────────────────────────────────────────
    I_empty = Graph()
    L_empty = Graph()
    R_one_v = Graph(1)

    l_addv = ACSetTransformation(I_empty, L_empty)
    r_addv = ACSetTransformation(I_empty, R_one_v)
    rule_add_vertex = Rule(l_addv, r_addv)

    # ── Rule 2: add an edge between two existing vertices ────────────────────
    I_two_v = Graph(2)
    L_two_v = Graph(2)
    R_two_v_one_e = Graph(2)
    add_edges!(R_two_v_one_e, [1], [2])

    l_adde = ACSetTransformation(I_two_v, L_two_v;  V=[1,2])
    r_adde = ACSetTransformation(I_two_v, R_two_v_one_e; V=[1,2])
    rule_add_edge = Rule(l_adde, r_adde)

    # ── Game definition ───────────────────────────────────────────────────────
    T_MAX = 50

    game = Game(
        nothing;
        players = [:alice, :bob],
        rules   = Dict(
            :alice => [RuleEntry(rule_add_vertex; name=:add_vertex)],
            :bob   => [RuleEntry(rule_add_edge;   name=:add_edge)],
        ),
        auto     = AutoRule[],
        terminal = (W) -> begin
            done   = nparts(W, :V) >= 8 || nparts(W, :E) >= 6
            winner = nparts(W, :V) >= 8 ? :alice :
                     nparts(W, :E) >= 6 ? :bob   : nothing
            (done, winner)
        end,
        initial  = () -> Graph(2),
    )

    agents = Dict{Symbol, AbstractAgent}(
        :alice => FunctionAgent((state, actions) -> rand(actions)),
        :bob   => FunctionAgent((state, actions) -> rand(actions)),
    )

    # ── Episode-level invariants (checked across 10 random episodes) ──────────
    @testset "run_game terminates within T_max" begin
        for _ in 1:10
            hist = run_game(game, agents; T_max=T_MAX)
            @test history_length(hist) > 0
            @test history_length(hist) <= T_MAX
        end
    end

    # ── GameHistory API contract (checked once on a single episode) ───────────
    @testset "GameHistory fields have correct types" begin
        hist = run_game(game, agents; T_max=T_MAX)
        t0   = first(hist._step_turns)

        @test get_player(hist, t0) isa Symbol
        @test get_player(hist, t0) in game.players
        @test get_path(hist, t0)   isa Vector{Symbol}
        @test get_world(hist, t0)  isa Graph
        @test get_world(hist, 0)   isa Graph   # initial world

        # chosen span is (rule_name, L, K, R) or nothing
        ch = get_chosen(hist, t0)
        if ch !== nothing
            @test ch.rule_name isa Symbol
            @test ch.L isa Graph
            @test ch.K isa Graph
            @test ch.R isa Graph
        end
    end

    # ── Terminal semantics ────────────────────────────────────────────────────
    @testset "Game produces a complete history" begin
        hist = run_game(game, agents; T_max=T_MAX)
        # turns() includes initial world (t=0) plus one world per player step
        @test length(turns(hist)) == history_length(hist) + 1
        # last world snapshot corresponds to the final game state
        @test get_world(hist, last(turns(hist))) isa Graph
    end

    # ── Agent behaviour ───────────────────────────────────────────────────────
    @testset "FunctionAgent with pass-through function" begin
        first_agent = FunctionAgent((state, actions) -> first(actions))
        agents2 = Dict{Symbol, AbstractAgent}(
            :alice => first_agent,
            :bob   => FunctionAgent((s, a) -> rand(a)),
        )
        hist = run_game(game, agents2; T_max=T_MAX)
        @test history_length(hist) > 0
    end

    # ── Budget limiting ───────────────────────────────────────────────────────
    @testset "Budget limiting works" begin
        game_budgeted = Game(
            nothing;
            players = [:p],
            rules   = Dict(
                :p => [RuleEntry(rule_add_vertex; name=:add_vertex, budget=3)],
            ),
            terminal = (W) -> (nparts(W, :V) >= 10, nothing),
            initial  = () -> Graph(1),
        )
        agents3 = Dict{Symbol, AbstractAgent}(
            :p => FunctionAgent((s, a) -> isempty(a) ? error("no actions") : rand(a)),
        )
        # With budget=3 and terminal at 10 vertices, the budget is exhausted after
        # 3 moves; the player passes on all subsequent turns until T_max.
        hist = run_game(game_budgeted, agents3; T_max=20)
        @test history_length(hist) > 0
        @test get(action_counts(hist), :add_vertex, 0) <= 3
    end
end
