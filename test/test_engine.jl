using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "Engine tests" begin

    # ── Minimal graph schema ──────────────────────────────────────────────────
    I_empty = Graph()
    R_one_v = Graph(1)
    l_addv  = ACSetTransformation(I_empty, I_empty)
    r_addv  = ACSetTransformation(I_empty, R_one_v)
    rule_add_vertex = Rule(l_addv, r_addv)

    I_two_v      = Graph(2)
    R_two_v_one_e = Graph(2)
    add_edges!(R_two_v_one_e, [1], [2])
    l_adde = ACSetTransformation(I_two_v, Graph(2); V=[1,2])
    r_adde = ACSetTransformation(I_two_v, R_two_v_one_e; V=[1,2])
    rule_add_edge = Rule(l_adde, r_adde)

    N = Names(Dict("" => I_empty, "I" => I_empty))

    done_at_8v_or_6e = (W) -> begin
        done   = nparts(W, :V) >= 8 || nparts(W, :E) >= 6
        winner = nparts(W, :V) >= 8 ? :alice :
                 nparts(W, :E) >= 6 ? :bob   : nothing
        (done, winner)
    end

    # Build a two-player schedule: alice adds a vertex, bob adds an edge
    pra_alice = PlayerRuleApp(:add_vertex, rule_add_vertex, I_empty, :alice)
    pra_bob   = PlayerRuleApp(:add_edge,   rule_add_edge,   I_two_v, :bob)

    game_sched = mk_game_sched(
        (trace_arg=:I,), (init=:I,), N,
        (a=pra_alice, b=pra_bob, mw=merge_wires(I_empty)),
        quote
            a_moved, a_pass = a([init, trace_arg])
            b_moved, b_pass = b(a_moved)
            cont = mw(mw(a_pass, b_moved), b_pass)
            return cont
        end)

    game = Game(
        nothing;
        players = [:alice, :bob],
        terminal = done_at_8v_or_6e,
        initial  = () -> Graph(2),
    )

    T_MAX = 50

    agents = Dict{Symbol, AbstractAgent}(
        :alice => FunctionAgent((state, actions) -> rand(actions)),
        :bob   => FunctionAgent((state, actions) -> rand(actions)),
    )

    # ── Episode-level invariants ──────────────────────────────────────────────
    @testset "run_game_sched! terminates within T_max" begin
        for _ in 1:5
            exps = run_game_sched!(game_sched, game, agents; T_max=T_MAX)
            @test !isempty(exps)
        end
    end

    # ── Experience struct contract ────────────────────────────────────────────
    @testset "Experience fields have correct types" begin
        exps = run_game_sched!(game_sched, game, agents; T_max=T_MAX)
        @test !isempty(exps)
        exp = first(exps)

        @test exp.player in game.players
        @test exp.state      isa GameState
        @test exp.next_state isa GameState
        @test exp.state.world      !== nothing
        @test exp.next_state.world !== nothing
        @test exp.state.turn >= 1
        @test exp.next_state.turn >= exp.state.turn
        @test exp.legal_actions isa Vector{Action}
        @test exp.action isa Union{Action, Nothing}
        @test exp.info isa Dict{Symbol, Any}
        @test exp.schedule_path isa Vector{Symbol}
    end

    # ── Agent behaviour ───────────────────────────────────────────────────────
    @testset "FunctionAgent with deterministic first-pick" begin
        first_agent = FunctionAgent((state, actions) -> first(actions))
        agents2 = Dict{Symbol, AbstractAgent}(
            :alice => first_agent,
            :bob   => FunctionAgent((s, a) -> rand(a)),
        )
        exps = run_game_sched!(game_sched, game, agents2; T_max=T_MAX)
        @test !isempty(exps)
    end

end
