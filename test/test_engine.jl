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

    N = Names(Dict("" => I_empty, "I" => I_empty, "I_two_v" => I_two_v))

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

    # ── match_limit ───────────────────────────────────────────────────────────

    @testset "match_limit caps legal_actions per turn" begin
        # Start with 4 vertices so rule_add_edge has multiple matches.
        # With match_limit=1, each Experience should have at most 1 legal action.
        pra_limited = PlayerRuleApp(:add_edge, rule_add_edge, I_two_v, :bob;
                                    match_limit=1)
        sched_limited = mk_game_sched(
            (;), (init=:I_two_v,), N,
            (b=pra_limited,),
            quote
                moved, tie = b(init)
                return moved, tie
            end)
        exps = run_game_sched!(sched_limited, Graph(4), agents; T_max=5)
        @test !isempty(exps)
        @test all(e -> length(e.legal_actions) <= 1, exps)
    end

    @testset "match_limit with use_cache caps legal_actions per turn" begin
        pra_cached = PlayerRuleApp(:add_edge, rule_add_edge, I_two_v, :bob;
                                   match_limit=1, use_cache=true)
        sched_cached = mk_game_sched(
            (;), (init=:I_two_v,), N,
            (b=pra_cached,),
            quote
                moved, tie = b(init)
                return moved, tie
            end)
        exps = run_game_sched!(sched_cached, Graph(4), agents; T_max=5)
        @test !isempty(exps)
        @test all(e -> length(e.legal_actions) <= 1, exps)
    end

    # ── Exit-wire winner detection ────────────────────────────────────────────

    # One-shot schedule: alice adds a vertex then exits.
    # Port 1 (alice_won) fires on success; port 2 (tie) fires when no moves.
    alice_wins_sched = mk_game_sched(
        (;), (init=:I,), N,
        (a=pra_alice,),
        quote
            alice_won, tie = a(init)
            return alice_won, tie
        end)

    @testset "winner_wires sets winner on final Experience" begin
        ww = Dict{Symbol, Union{Symbol,Nothing}}(:alice_won => :alice, :tie => nothing)
        exps = run_game_sched!(alice_wins_sched, Graph(2), agents;
                               T_max=10, winner_wires=ww)
        @test !isempty(exps)
        @test exps[end].done   === true
        @test exps[end].winner === :alice
    end

    @testset "Game.win_conditions used automatically in Game overload" begin
        game_wc = Game(
            nothing;
            players        = [:alice, :bob],
            initial        = () -> Graph(2),
            win_conditions = Dict{Symbol, Any}(:alice_won => :alice, :tie => nothing),
        )
        exps = run_game_sched!(alice_wins_sched, game_wc, agents; T_max=10)
        @test !isempty(exps)
        @test exps[end].done   === true
        @test exps[end].winner === :alice
    end

    @testset "terminal kwarg backward compatibility" begin
        always_alice = (W) -> (true, :alice)
        exps = run_game_sched!(alice_wins_sched, Graph(2), agents;
                               T_max=10, terminal=always_alice)
        @test !isempty(exps)
        @test exps[end].done   === true
        @test exps[end].winner === :alice
    end

    # One-shot schedule where bob tries to add an edge to an empty graph;
    # rule_add_edge requires 2 existing vertices, so bob always has no matches
    # and the tie exit wire fires.
    bob_tries_sched = mk_game_sched(
        (;), (init=:I,), N,
        (b=pra_bob,),
        quote
            moved, tie = b(init)
            return moved, tie
        end)

    @testset "winner_wires: tie exit wire produces winner=nothing" begin
        ww = Dict{Symbol, Union{Symbol,Nothing}}(:moved => :bob, :tie => nothing)
        # Graph(0) has no vertices, so rule_add_edge has no matches → tie fires.
        exps = run_game_sched!(bob_tries_sched, Graph(0), agents;
                               T_max=10, winner_wires=ww)
        @test !isempty(exps)
        @test exps[end].done   === true
        @test exps[end].winner === nothing
    end

end
