using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

# ─── Shared fixtures ──────────────────────────────────────────────────────────

I_empty = Graph()
R_one_v = Graph(1)
rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                  ACSetTransformation(I_empty, R_one_v))

never_terminal = (W) -> (false, nothing)
done_at_3      = (W) -> (nparts(W, :V) >= 3, nothing)

rand_agent = FunctionAgent((s, a) -> rand(a))

@testset "Wiring-diagram scheduling layer" begin

    # ─── PlayerRuleApp construction ───────────────────────────────────────────

    @testset "PlayerRuleApp construction" begin
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        @test pra.name   == :add_v
        @test pra.player == :alice
        @test pra.rule   === rule_add_v
        @test pra.init   === I_empty
        @test pra._inner isa RuleApp
    end

    @testset "PlayerRuleApp with cat keyword" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :bob; cat=𝒞)
        @test pra.player == :bob
        @test pra.cat    === 𝒞
        @test pra._inner isa RuleApp
    end

    # ─── Body parser ─────────────────────────────────────────────────────────

    @testset "_parse_body: simple two-output box" begin
        body = quote
            success, failure = box(init)
            return success, failure
        end
        steps, rets = RewriteGames._parse_body(body)
        @test length(steps) == 1
        @test steps[1].box     == :box
        @test steps[1].inputs  == [:init]
        @test steps[1].outputs == [:success, :failure]
        @test rets == [:success, :failure]
    end

    @testset "_parse_body: merge wire list input" begin
        body = quote
            out1, out2 = box([w1, w2])
            return out1, out2
        end
        steps, rets = RewriteGames._parse_body(body)
        @test steps[1].inputs == [:w1, :w2]
    end

    @testset "_parse_body: nested merge calls are flattened" begin
        body = quote
            result = mw(mw(a, b), c)
            return result
        end
        steps, rets = RewriteGames._parse_body(body)
        # mw(a,b) → _gstmp_1; mw(_gstmp_1, c) → result
        @test length(steps) == 2
        @test steps[1].outputs[1] == :_gstmp_1
        @test steps[2].inputs     == [:_gstmp_1, :c]
        @test steps[2].outputs    == [:result]
        @test rets == [:result]
    end

    # ─── mk_game_sched ───────────────────────────────────────────────────────

    @testset "mk_game_sched produces a GameSched" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box(init)
                return moved, tie
            end)
        @test gs isa GameSched
        @test haskey(gs._player_map, :box)
        @test gs._player_map[:box] === pra
        @test gs._init_names  == [:init]
        @test gs._trace_names == Symbol[]
        @test gs._ret_names   == [:moved, :tie]
    end

    # ─── run_game_sched! ─────────────────────────────────────────────────────

    @testset "run_game_sched! terminates and returns experiences" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        agents = Dict(:p => rand_agent)
        exps = run_game_sched!(gs, Graph(0), agents;
                               T_max=10, terminal=done_at_3)
        @test !isempty(exps)
        @test exps[end].done == true
    end

    @testset "run_game_sched! player picks from legal actions" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p)
        # Agent always picks the first action
        first_agent = FunctionAgent((s, a) -> first(a))
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        agents = Dict(:p => first_agent)
        exps = run_game_sched!(gs, Graph(0), agents;
                               T_max=10, terminal=done_at_3)
        # Every step should have an action (the rule always fires on Graph)
        active = filter(e -> e.player == :p, exps)
        @test all(e -> e.action !== nothing, active)
        @test exps[end].done == true
    end

    @testset "run_game_sched! with Game convenience overload" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        game = Game(nothing;
                    players  = [:p],
                    terminal = done_at_3,
                    initial  = () -> Graph(0))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        exps = run_game_sched!(gs, game, Dict(:p => rand_agent); T_max=10)
        @test !isempty(exps)
        @test exps[end].done == true
    end

    # ─── view_sched delegation ────────────────────────────────────────────────

    @testset "view_sched(GameSched) delegates to inner schedule" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p)
        gs = mk_game_sched(
            (;), (init=:I,), N,
            (box=pra,),
            quote
                moved, tie = box(init)
                return moved, tie
            end)
        # view_sched should not error — it returns a Graphviz object
        result = view_sched(gs; names=N)
        @test result !== nothing
    end

end
