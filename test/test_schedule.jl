using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using Random

# ─── Shared fixtures ──────────────────────────────────────────────────────────
# We use the Catlab Graph schema throughout: Ob={V,E}, Hom={src,tgt}.

I_empty  = Graph()
R_one_v  = Graph(1)
rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                  ACSetTransformation(I_empty, R_one_v))
entry_add_v = RuleEntry(rule_add_v; name=:add_vertex)

# Terminal: never (unless overridden in specific tests)
never_terminal = (W) -> (false, nothing)
done_at_3 = (W) -> (nparts(W, :V) >= 3, nothing)

# Helper: random-selecting FunctionAgent
rand_agent = FunctionAgent((s, a) -> rand(a))

@testset "GameStep schedule layer" begin

    # ─── Layer 1: Type structure ───────────────────────────────────────────────

    @testset "Type hierarchy" begin
        @test PlayerStep(:alice) isa GameStep
        @test PlayerStep(:alice) isa PlayerStep
        @test PlayerStep(:alice).player == :alice
        @test PlayerStep(:alice).name == :alice          # default name = player
        @test PlayerStep(:alice; name=:a).name == :a

        @test AutoStep() isa GameStep
        @test AutoStep() isa AutoStep
        @test AutoStep().rules === nothing
        ar = AutoRule(rule_add_v; name=:grow)
        @test AutoStep([ar]).rules == [ar]
        @test Auto() isa AutoStep                        # alias works

        @test Seq(PlayerStep(:a), PlayerStep(:b)) isa Seq
        @test Seq(PlayerStep(:a), PlayerStep(:b)) isa GameStep
        @test length(Seq(PlayerStep(:a), PlayerStep(:b)).steps) == 2

        @test Cond(W -> 1, PlayerStep(:a)) isa Cond
        @test Cond(W -> 1, PlayerStep(:a)) isa GameStep
        @test length(Cond(W -> 1, PlayerStep(:a), PlayerStep(:b)).branches) == 2

        @test WhileStep(W -> false, AutoStep()) isa WhileStep
        @test WhileStep(W -> false, AutoStep()) isa GameStep
        @test WhileStep(W -> false, AutoStep()).max_iter == 1000
        @test WhileStep(W -> false, AutoStep(); max_iter=5).max_iter == 5

        @test ForEachStep(:V, PlayerStep(:a)) isa ForEachStep
        @test ForEachStep(:V, PlayerStep(:a)) isa GameStep
        @test ForEachStep(:V, PlayerStep(:a)).order == :natural
        @test ForEachStep(:V, PlayerStep(:a); order=:random).order == :random
    end

    # ─── Layer 2: AgentContext ─────────────────────────────────────────────────

    @testset "AgentContext construction and nesting" begin
        ctx = AgentContext(:Wolf, 2)
        @test ctx.ob == :Wolf
        @test ctx.id == 2
        @test isempty(ctx.stack)

        ctx2 = push_context(ctx, :Sheep, 5)
        @test ctx2.ob == :Sheep
        @test ctx2.id == 5
        @test length(ctx2.stack) == 1
        @test ctx2.stack[1] == (:Wolf, 2)

        ctx3 = push_context(ctx2, :Grass, 1)
        @test ctx3.ob == :Grass
        @test ctx3.id == 1
        @test length(ctx3.stack) == 2
    end

    # ─── Layer 3: Individual node semantics ───────────────────────────────────

    @testset "PlayerStep emits one Experience per evaluation" begin
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(0),
            schedule = PlayerStep(:p),
        )
        agents = Dict(:p => rand_agent)
        driver = ScheduledGameDriver(game, agents; T_max=20)
        exps = run_schedule!(driver)
        @test length(exps) == 1
        @test exps[1].player == :p
        @test exps[1].schedule_path == [:p]
        @test nparts(driver.state.world, :V) == 1
    end

    @testset "AutoStep uses game.auto when rules=nothing" begin
        ar = AutoRule(rule_add_v; name=:grow)
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            auto     = [ar],
            terminal = never_terminal,
            initial  = () -> Graph(0),
            schedule = AutoStep(),      # should use game.auto = [ar]
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        run_schedule!(driver)
        @test nparts(driver.state.world, :V) == 1   # auto-rule added one vertex
    end

    @testset "AutoStep with explicit rules overrides game.auto" begin
        ar_default = AutoRule(rule_add_v; name=:default_grow)
        # AutoStep with an explicit empty list fires nothing, even though game.auto
        # contains ar_default (which would add a vertex if it ran).
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            auto     = [ar_default],
            terminal = never_terminal,
            initial  = () -> Graph(2),
            schedule = AutoStep(AutoRule[]),   # explicit empty list
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        run_schedule!(driver)
        # game.auto would have added a vertex → 3. Explicit empty list → still 2.
        @test nparts(driver.state.world, :V) == 2
    end

    @testset "Seq short-circuits on terminal" begin
        # After player :a acts, world has 3 vertices → done_at_3 fires.
        # Player :b should NOT get an experience.
        game = Game(nothing;
            players  = [:a, :b],
            rules    = Dict(:a => [entry_add_v], :b => [entry_add_v]),
            terminal = done_at_3,
            initial  = () -> Graph(2),        # start at 2 → one more → done
            schedule = Seq(PlayerStep(:a), PlayerStep(:b)),
        )
        agents = Dict(:a => rand_agent, :b => rand_agent)
        driver = ScheduledGameDriver(game, agents; T_max=20)
        exps = run_schedule!(driver)
        @test length(exps) == 1
        @test exps[1].player == :a
        @test exps[1].done == true
    end

    @testset "Cond routes to correct branch" begin
        # Branch 1 when V >= 3, branch 2 otherwise.
        # Large world (3 vertices) → branch 1 → player :big
        # Small world (1 vertex)   → branch 2 → player :small
        for (start_v, expected_player) in [(3, :big), (1, :small)]
            game = Game(nothing;
                players  = [:big, :small],
                rules    = Dict(:big => [entry_add_v], :small => [entry_add_v]),
                terminal = never_terminal,
                initial  = () -> Graph(start_v),
                schedule = Cond(
                    W -> nparts(W, :V) >= 3 ? 1 : 2,
                    PlayerStep(:big),
                    PlayerStep(:small),
                ),
            )
            agents = Dict(:big => rand_agent, :small => rand_agent)
            driver = ScheduledGameDriver(game, agents; T_max=20)
            exps = run_schedule!(driver)
            @test length(exps) == 1
            @test exps[1].player == expected_player
        end
    end

    @testset "WhileStep iterates until condition false" begin
        # Condition: V < 5.  Body = PlayerStep(:p) which adds a vertex.
        # Start at 2 vertices → fires 3 times → 5 vertices.
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(2),
            schedule = WhileStep(W -> nparts(W, :V) < 5, PlayerStep(:p)),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        exps = run_schedule!(driver)
        @test length(exps) == 3
        @test nparts(driver.state.world, :V) == 5
    end

    @testset "WhileStep max_iter safety cap" begin
        # Condition always true, body = AutoStep with no-op auto-rules list.
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => RuleEntry[]),
            terminal = never_terminal,
            initial  = () -> Graph(0),
            schedule = WhileStep(W -> true, AutoStep(AutoRule[]); max_iter=5),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        @test_throws ErrorException run_schedule!(driver)
    end

    @testset "ForEachStep runs body once per part" begin
        # World has 3 vertices; ForEachStep iterates over :V (vertex ids 1,2,3).
        # Each iteration: PlayerStep(:p) with add_vertex rule → 3 experiences.
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(3),
            schedule = ForEachStep(:V, PlayerStep(:p)),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        exps = run_schedule!(driver)
        @test length(exps) == 3
    end

    @testset "ForEachStep context stored in experience info" begin
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(2),
            schedule = ForEachStep(:V, PlayerStep(:p)),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        exps = run_schedule!(driver)
        @test length(exps) == 2
        ctx1 = exps[1].info[:context]
        ctx2 = exps[2].info[:context]
        @test ctx1 isa AgentContext
        @test ctx1.ob == :V
        @test ctx1.id == 1
        @test ctx2.id == 2
    end

    @testset "ForEachStep schedule_path includes foreach name" begin
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(1),
            schedule = ForEachStep(:V, PlayerStep(:p); name=:vloop),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        exps = run_schedule!(driver)
        @test :vloop ∈ exps[1].schedule_path
        @test :p     ∈ exps[1].schedule_path
    end

    # ─── Layer 4: ScheduledGameDriver integration ──────────────────────────────

    @testset "run_game dispatches to ScheduledGameDriver when schedule set" begin
        game = Game(nothing;
            players  = [:a, :b],
            rules    = Dict(:a => [entry_add_v], :b => [entry_add_v]),
            terminal = done_at_3,
            initial  = () -> Graph(0),
            schedule = Seq(PlayerStep(:a), PlayerStep(:b)),
        )
        agents = Dict(:a => rand_agent, :b => rand_agent)
        exps = run_game(game, agents; T_max=20)
        @test !isempty(exps)
        @test exps[end].done == true
        @test exps[end].schedule_path != Symbol[]   # path is non-empty
    end

    @testset "run_game auto-generates round-robin schedule when none provided" begin
        # Omitting schedule: triggers auto-generation: Seq(PlayerStep(:p), AutoStep())
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = done_at_3,
            initial  = () -> Graph(0),
        )
        @test game.schedule isa Seq
        exps = run_game(game, Dict(:p => rand_agent); T_max=20)
        @test !isempty(exps)
        @test exps[end].done == true
        # All experiences come from the ScheduledGameDriver; schedule_path is non-empty
        @test all(e -> !isempty(e.schedule_path), exps)
    end

    @testset "T_max enforced when terminal never fires" begin
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(0),
            schedule = PlayerStep(:p),
        )
        exps = run_game(game, Dict(:p => rand_agent); T_max=4)
        # Each PlayerStep increments turn; we stop when turn > T_max
        @test !isempty(exps)
        @test exps[end].done == true
    end

    @testset "schedule_path correctly identifies nested position" begin
        # Seq(name=:round, Seq(name=:inner, PlayerStep(:p; name=:step)))
        # Expected path: [:seq, :inner, :step]  (outer Seq uses default :seq name)
        game = Game(nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_add_v]),
            terminal = never_terminal,
            initial  = () -> Graph(0),
            schedule = Seq(
                Seq(PlayerStep(:p; name=:step); name=:inner);
                name=:round,
            ),
        )
        driver = ScheduledGameDriver(game, Dict(:p => rand_agent); T_max=20)
        exps = run_schedule!(driver)
        @test exps[1].schedule_path == [:round, :inner, :step]
    end

    # ─── Layer 5: DSL ─────────────────────────────────────────────────────────

    @testset "@game schedule: clause sets game.schedule" begin
        sched = Seq(PlayerStep(:p))
        g = @game nothing begin
            players: p
            p: [entry_add_v]
            terminal: (W) -> (false, nothing)
            initial: () -> Graph(0)
            schedule: sched
        end
        @test g.schedule === sched
    end

    @testset "@game without schedule: clause auto-generates round-robin" begin
        g = @game nothing begin
            players: p
            p: RuleEntry[]
            terminal: (W) -> (false, nothing)
            initial: () -> nothing
        end
        # Auto-generated schedule is a Seq node containing a PlayerStep + AutoStep
        @test g.schedule isa Seq
        steps = g.schedule.steps
        @test length(steps) == 2
        @test steps[1] isa PlayerStep && steps[1].player == :p
        @test steps[2] isa AutoStep
    end

end
