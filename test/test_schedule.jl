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
        @test pra.in_hom === I_empty
        @test pra.out_hom === I_empty
        @test pra._inner isa RuleApp
    end

    @testset "PlayerRuleApp with cat keyword" begin
        𝒞 = ACSetCategory(CSetCat(Graph()))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :bob; cat=𝒞)
        @test pra.player == :bob
        @test pra.cat    === 𝒞
        @test pra._inner isa RuleApp
    end

    @testset "PlayerRuleApp with match_limit" begin
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice; match_limit=3)
        @test pra.match_limit == 3
        # default is nothing
        pra2 = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        @test pra2.match_limit === nothing
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

    # ─── tryrule ────────────────────────────────────────────────────────────────

    @testset "tryrule(PlayerRuleApp) produces a GameSched" begin
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = tryrule(pra)
        @test gs isa GameSched
        @test gs._init_names == [:init]
        @test length(gs._ret_names) == 1
        @test haskey(gs._player_map, :add_v)
    end

    @testset "tryrule result can be run via run_game_sched!" begin
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = tryrule(pra)
        agents = Dict(:alice => rand_agent)
        exps = run_game_sched!(gs, Graph(0), agents; T_max=5, terminal=done_at_3)
        @test !isempty(exps)
    end

    # ─── PlayerRuleApp ⋅ composition ────────────────────────────────────────────

    @testset "pra1 ⋅ pra2 produces a GameSched with both players" begin
        using Catlab.Theories: ⋅
        pra1 = PlayerRuleApp(:add_v1, rule_add_v, I_empty, :alice)
        pra2 = PlayerRuleApp(:add_v2, rule_add_v, I_empty, :bob)
        gs = pra1 ⋅ pra2
        @test gs isa GameSched
        @test haskey(gs._player_map, :add_v1)
        @test haskey(gs._player_map, :add_v2)
        @test gs._init_names == [:init]
    end

    @testset "pra1 ⋅ pra2 schedule runs both players" begin
        using Catlab.Theories: ⋅
        pra1 = PlayerRuleApp(:av1, rule_add_v, I_empty, :alice)
        pra2 = PlayerRuleApp(:av2, rule_add_v, I_empty, :bob)
        gs = pra1 ⋅ pra2
        agents = Dict(:alice => rand_agent, :bob => rand_agent)
        exps = run_game_sched!(gs, Graph(0), agents; T_max=5, terminal=done_at_3)
        @test !isempty(exps)
        players = [e.player for e in exps]
        @test :alice in players
        @test :bob in players
    end

    # ─── PlayerRuleApp ⊗ tensor ─────────────────────────────────────────────────

    @testset "pra1 ⊗ pra2 produces a GameSched with two init wires" begin
        using Catlab.Theories: ⊗
        pra1 = PlayerRuleApp(:av1, rule_add_v, I_empty, :alice)
        pra2 = PlayerRuleApp(:av2, rule_add_v, I_empty, :bob)
        gs = pra1 ⊗ pra2
        @test gs isa GameSched
        @test length(gs._init_names) == 2
        @test haskey(gs._player_map, :av1)
        @test haskey(gs._player_map, :av2)
    end

    # ─── GameSched ⋅ composition ────────────────────────────────────────────────

    @testset "gs1 ⋅ gs2 chains two schedules" begin
        using Catlab.Theories: ⋅
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra1 = PlayerRuleApp(:av1, rule_add_v, I_empty, :alice)
        pra2 = PlayerRuleApp(:av2, rule_add_v, I_empty, :bob)
        # Use tryrule to get single-output schedules (⋅ for GameSched requires 1-output ports)
        gs1 = tryrule(pra1)
        gs2 = tryrule(pra2)
        gs = gs1 ⋅ gs2
        @test gs isa GameSched
        agents = Dict(:alice => rand_agent, :bob => rand_agent)
        exps = run_game_sched!(gs, Graph(0), agents; T_max=5, terminal=done_at_3)
        @test !isempty(exps)
    end

    # ─── _collect_player_apps: nested GameSched ──────────────────────────────────

    @testset "_collect_player_apps recurses into nested GameSched" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra_inner = PlayerRuleApp(:inner_av, rule_add_v, I_empty, :alice)
        # tryrule gives a single-output schedule, safe to use as a nested box
        gs_inner = tryrule(pra_inner)
        pra_outer = PlayerRuleApp(:outer_av, rule_add_v, I_empty, :bob)
        gs_outer = mk_game_sched((;), (init=:I,), N,
                                  (sub=gs_inner, b2=pra_outer, mw=merge_wires(I_empty)),
                                  quote
                                      mid = sub(init)
                                      moved, tie = b2(mid)
                                      out = mw(moved, tie)
                                      return out
                                  end)
        apps = RewriteGames._collect_player_apps(gs_outer)
        # inner_av is the NamedTuple key tryrule used (same as pra.name in this case)
        @test haskey(apps, :inner_av)
        # b2 is the NamedTuple key used for pra_outer in gs_outer
        @test haskey(apps, :b2)
    end

    # ─── use_cache regression ────────────────────────────────────────────────────

    @testset "use_cache=true does not crash with run_game_sched!" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p; use_cache=true)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        exps = run_game_sched!(gs, Graph(0), Dict(:p => rand_agent);
                               T_max=5, terminal=done_at_3)
        @test !isempty(exps)
        @test exps[end].done == true
    end

    # ─── view_fn (fog-of-war) ────────────────────────────────────────────────────

    @testset "view_fn restricts matches to subworld" begin
        # view_fn returning the full world as subworld with identity inclusion
        identity_view = (player, world) -> begin
            nv = nparts(world, :V)
            ne = nparts(world, :E)
            id = ACSetTransformation(world, world; V=collect(1:nv), E=collect(1:ne))
            (world, id)
        end
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p;
                            view_fn=identity_view)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        exps = run_game_sched!(gs, Graph(0), Dict(:p => rand_agent);
                               T_max=5, terminal=done_at_3)
        @test !isempty(exps)
        # All experiences come from player :p
        @test all(e -> e.player == :p, exps)
    end

    # ─── winner_wires with exit wires ────────────────────────────────────────────

    @testset "winner_wires: success wire tags winner correctly" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N, (box=pra,),
            quote
                won, tie = box(init)
                return won, tie
            end)
        ww = Dict{Symbol, Union{Symbol,Nothing}}(:won => :alice, :tie => nothing)
        exps = run_game_sched!(gs, Graph(0), Dict(:alice => rand_agent);
                               T_max=3, winner_wires=ww)
        @test !isempty(exps)
        @test exps[end].done == true
        @test exps[end].winner === :alice
    end

    @testset "winner_wires: no-match exit wire gives winner=nothing" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        # rule_add_v always matches (empty interface), so 'tie' never fires
        # but if we start with a world that already satisfies terminal, done fires.
        # Use a rule that can only fail: require 5 vertices to match.
        R5 = Graph(5)
        l5 = ACSetTransformation(R5, R5; V=collect(1:5), E=Int[])
        r5 = ACSetTransformation(R5, R5; V=collect(1:5), E=Int[])
        rule_noop5 = Rule(l5, r5)   # only matches if world has ≥ 5 vertices
        pra5 = PlayerRuleApp(:noop5, rule_noop5, R5, :alice)
        gs5 = mk_game_sched(
            (;), (init=:I,), Names(Dict("" => R5, "I" => R5)), (box=pra5,),
            quote
                won, tie = box(init)
                return won, tie
            end)
        ww = Dict{Symbol, Union{Symbol,Nothing}}(:won => :alice, :tie => nothing)
        # Graph(0) has no vertices → no match → tie fires
        exps = run_game_sched!(gs5, Graph(0), Dict(:alice => rand_agent);
                               T_max=3, winner_wires=ww)
        @test !isempty(exps)
        @test exps[end].done == true
        @test exps[end].winner === nothing
    end

    # ─── T_max enforcement ───────────────────────────────────────────────────────

    @testset "T_max=0 causes immediate termination" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :p)
        gs = mk_game_sched(
            (trace_arg=:I,), (init=:I,), N,
            (box=pra, mw=merge_wires(I_empty)),
            quote
                moved, tie = box([init, trace_arg])
                return moved, tie
            end)
        exps = run_game_sched!(gs, Graph(0), Dict(:p => rand_agent); T_max=0)
        @test all(e -> e.done, exps)
    end

    # ─── player_migrate with name_map ────────────────────────────────────────────

    @testset "player_migrate renames PlayerRuleApp boxes via name_map" begin
        N = Names(Dict("" => I_empty, "I" => I_empty))
        pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
        gs = mk_game_sched(
            (;), (init=:I,), N, (add_v=pra,),
            quote
                moved, tie = add_v(init)
                return moved, tie
            end)
        F = identity  # trivial functor — schema unchanged
        player_map = Dict{Symbol, Symbol}(:alice => :alice)
        name_map   = Dict{Symbol, Symbol}(:add_v => :renamed_add_v)
        gs2 = player_migrate(F, gs, player_map; name_map=name_map)
        @test gs2 isa GameSched
        # _player_map is keyed by NamedTuple key (:add_v), but the PRA.name is renamed
        @test haskey(gs2._player_map, :add_v)
        @test gs2._player_map[:add_v].name == :renamed_add_v
    end

end
