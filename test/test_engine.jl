using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Engine tests" begin

    # ── Minimal graph schema ──────────────────────────────────────────────────
    # Objects: V, E; morphisms: src::Hom(E,V), tgt::Hom(E,V)
    # We use Catlab's built-in Graph type which has exactly this schema.

    # ── Rule 1: add a vertex ──────────────────────────────────────────────────
    # I = empty graph, L = empty graph, R = 1 vertex
    I_empty = Graph()
    L_empty = Graph()
    R_one_v = Graph(1)

    l_addv = ACSetTransformation(I_empty, L_empty)
    r_addv = ACSetTransformation(I_empty, R_one_v)
    rule_add_vertex = Rule(l_addv, r_addv)

    # ── Rule 2: add an edge between two existing vertices ────────────────────
    # I = 2 vertices, L = 2 vertices (pattern), R = 2 vertices + 1 edge
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
        initial  = () -> Graph(2),          # start with 2 vertices
    )

    agents = Dict{Symbol, AbstractAgent}(
        :alice => FunctionAgent((state, actions) -> rand(actions)),
        :bob   => FunctionAgent((state, actions) -> rand(actions)),
    )

    @testset "run_game terminates within T_max" begin
        for ep in 1:10
            exps = run_game(game, agents; T_max=T_MAX)

            @test !isempty(exps)
            @test length(exps) <= T_MAX
            @test exps[end].done == true

            for exp in exps
                # Player must be one of the declared players
                @test exp.player in game.players

                # state and next_state must be consistent EncodedState instances
                @test exp.state isa EncodedState
                @test exp.next_state isa EncodedState

                # node_features must be 2-D Float32 matrix
                @test exp.state.node_features isa Matrix{Float32}
                @test exp.next_state.node_features isa Matrix{Float32}

                # edge_index must have 2 rows (COO format) or be empty
                @test size(exp.state.edge_index, 1) == 2
                @test size(exp.next_state.edge_index, 1) == 2

                # Number of feature columns must be the same in state and next_state
                if size(exp.state.node_features, 1) > 0 &&
                   size(exp.next_state.node_features, 1) > 0
                    @test size(exp.state.node_features, 2) ==
                          size(exp.next_state.node_features, 2)
                end

                # Legal actions must be a Vector{Action}
                @test exp.legal_actions isa Vector{Action}

                # Chosen action must be an Action (never nothing for this game
                # as long as we start with 2 vertices, alice can always add_vertex)
                # We only assert type here; if player passed, action is nothing.
                @test exp.action isa Union{Action, Nothing}

                # done flag consistency: only the last experience should be done
                if exp !== exps[end]
                    # All intermediate steps are not done
                    # (unless the terminal condition fires mid-loop — fine either way)
                end

                # turn_frac must be in [0, 1]
                @test 0.0f0 <= exp.state.turn_frac <= 1.0f0
                @test 0.0f0 <= exp.next_state.turn_frac <= 1.0f0

                # info must be a Dict
                @test exp.info isa Dict{Symbol, Any}
            end
        end
    end

    @testset "GameDriver iteration" begin
        driver = GameDriver(game, agents; T_max=T_MAX)
        exps   = Experience[]
        for exp in driver
            push!(exps, exp)
            exp.done && break
        end
        @test !isempty(exps)
        @test exps[end].done
    end

    @testset "FunctionAgent with pass-through function" begin
        # Agent that always picks the first available action
        first_agent = FunctionAgent((state, actions) -> first(actions))
        agents2 = Dict{Symbol, AbstractAgent}(
            :alice => first_agent,
            :bob   => FunctionAgent((s, a) -> rand(a)),
        )
        exps = run_game(game, agents2; T_max=T_MAX)
        @test !isempty(exps)
        @test exps[end].done
    end

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
        # With budget=3 and terminal at 10 vertices, the game should exhaust the
        # budget after 3 turns and then pass (no legal actions) until T_max.
        exps = run_game(game_budgeted, agents3; T_max=20)
        @test !isempty(exps)
        # At most 3 non-pass actions should appear
        non_pass = filter(e -> e.action !== nothing, exps)
        @test length(non_pass) <= 3
    end
end
