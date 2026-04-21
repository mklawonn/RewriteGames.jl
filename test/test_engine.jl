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
            exps = run_game(game, agents; T_max=T_MAX)
            @test !isempty(exps)
            @test length(exps) <= T_MAX
            @test exps[end].done == true
        end
    end

    # ── Experience struct contract (checked once on a single episode) ─────────
    @testset "Experience fields have correct types" begin
        exps = run_game(game, agents; T_max=T_MAX)
        exp  = first(exps)

        @test exp.player in game.players
        @test exp.state     isa EncodedState
        @test exp.next_state isa EncodedState
        @test exp.state.node_features  isa Matrix{Float32}
        @test exp.next_state.node_features isa Matrix{Float32}
        @test size(exp.state.edge_index,    1) == 2   # COO: 2 rows
        @test size(exp.next_state.edge_index, 1) == 2
        @test size(exp.state.node_features, 2) ==
              size(exp.next_state.node_features, 2)   # consistent feature width
        @test exp.legal_actions isa Vector{Action}
        @test exp.action isa Union{Action, Nothing}
        @test 0.0f0 <= exp.state.turn_frac     <= 1.0f0
        @test 0.0f0 <= exp.next_state.turn_frac <= 1.0f0
        @test exp.info isa Dict{Symbol, Any}
        @test exp.schedule_path isa Vector{Symbol}
    end

    # ── Terminal semantics ────────────────────────────────────────────────────
    @testset "Exactly one experience per episode has done=true" begin
        exps = run_game(game, agents; T_max=T_MAX)
        @test count(e -> e.done, exps) == 1
        @test exps[end].done
    end

    # ── Agent behaviour ───────────────────────────────────────────────────────
    @testset "FunctionAgent with pass-through function" begin
        first_agent = FunctionAgent((state, actions) -> first(actions))
        agents2 = Dict{Symbol, AbstractAgent}(
            :alice => first_agent,
            :bob   => FunctionAgent((s, a) -> rand(a)),
        )
        exps = run_game(game, agents2; T_max=T_MAX)
        @test !isempty(exps)
        @test exps[end].done
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
        exps = run_game(game_budgeted, agents3; T_max=20)
        @test !isempty(exps)
        @test length(filter(e -> e.action !== nothing, exps)) <= 3
    end
end
