"""
GPU schedule execution tests.

Tests that `gpu_run_game_sched!` correctly executes non-trivial schedules
and produces results equivalent to the CPU `run_game_sched!`.

Uses simple Graph rules so the tests run without GPU hardware (the Turbo
solver falls back to CPU when CUDA is not functional, but the host-driven
schedule loop always runs on the host).
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

# ── Shared fixtures ───────────────────────────────────────────────────────────

I_empty = Graph()
R_one_v = Graph(1)

# Rule: add one isolated vertex (empty L = empty K)
rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                  ACSetTransformation(I_empty, R_one_v))

# Rule: delete one vertex (L = {v}, K = {}, R = {})
L_one_v = Graph(1)
rule_del_v = Rule(ACSetTransformation(I_empty, L_one_v),
                  ACSetTransformation(I_empty, I_empty))

# Rule: add edge between two DISTINCT vertices (monic, L = {v1, v2}, K = {v1, v2}, R = {v1, v2, e})
L_two_v = Graph(2)
R_edge   = Graph(2); add_edge!(R_edge, 1, 2)
rule_add_e = Rule(ACSetTransformation(L_two_v, L_two_v, V=[1,2]),
                  ACSetTransformation(L_two_v, R_edge, V=[1,2]);
                  monic=true)

N_empty = Names(Dict("I" => I_empty))
first_agent = FunctionAgent((s, acts) -> first(acts))
never_terminal = (W) -> (false, nothing)

# ── 1. Empty-pattern rule (trivial match) ─────────────────────────────────────

@testset "GPU schedule — add vertex (empty pattern)" begin
    pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (add_v=pra,),
                        quote
                            success, fail = add_v(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)
    exps = gpu_run_game_sched!(gs, Graph(0), agents; T_max=1)
    @test length(exps) == 1
    @test exps[1].player == :alice
end



# ── 2. Non-trivial pattern — vertex deletion ──────────────────────────────────

@testset "GPU schedule — delete vertex (n_vars=1 pattern)" begin
    pra = PlayerRuleApp(:del_v, rule_del_v, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (del_v=pra,),
                        quote
                            success, fail = del_v(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)

    # 3 vertices → should fire once, delete vertex 1, exit via success
    exps = gpu_run_game_sched!(gs, Graph(3), agents; T_max=5)
    @test length(exps) == 1
    @test exps[1].player == :alice

    # 0 vertices → no match, exit via fail
    exps_empty = gpu_run_game_sched!(gs, Graph(0), agents; T_max=5)
    @test length(exps_empty) == 0
end

# ── 3. CPU / GPU equivalence — single application ─────────────────────────────

@testset "GPU schedule — CPU/GPU equivalence (del_v)" begin
    pra = PlayerRuleApp(:del_v, rule_del_v, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (del_v=pra,),
                        quote
                            success, fail = del_v(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)
    G = Graph(4)

    exps_cpu = run_game_sched!(gs, G, agents; T_max=1, terminal=never_terminal)
    exps_gpu = gpu_run_game_sched!(gs, G, agents; T_max=1)
    @test length(exps_cpu) == length(exps_gpu)
end

# ── 4. Edge-addition rule (structural FKs in new elements) ───────────────────

@testset "GPU schedule — add edge (FK in new R element)" begin
    pra = PlayerRuleApp(:add_e, rule_add_e, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (add_e=pra,),
                        quote
                            success, fail = add_e(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)

    # 2 vertices → matches (v1→v2); fires once, adds an edge
    exps = gpu_run_game_sched!(gs, Graph(2), agents; T_max=2)
    @test length(exps) >= 1

    # 1 vertex → no pair to match
    exps_one = gpu_run_game_sched!(gs, Graph(1), agents; T_max=2)
    @test length(exps_one) == 0
end

# ── 5. Monic rule — GPU honours PROP_NEQ constraint ─────────────────────────

@testset "GPU schedule — monic add_e (no self-loop on single vertex)" begin
    # monic=true: src ≠ tgt, so Graph(1) has no valid match
    pra = PlayerRuleApp(:add_e, rule_add_e, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (add_e=pra,),
                        quote
                            success, fail = add_e(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)

    exps_one = gpu_run_game_sched!(gs, Graph(1), agents; T_max=2)
    @test length(exps_one) == 0   # monic: no match on single vertex

    exps_two = gpu_run_game_sched!(gs, Graph(2), agents; T_max=2)
    @test length(exps_two) == 1   # matches (1→2) or (2→1)
end

# ── 6. CPU/GPU equivalence — edge addition ────────────────────────────────────

@testset "GPU schedule — CPU/GPU equivalence (add_e)" begin
    pra = PlayerRuleApp(:add_e, rule_add_e, I_empty, :alice)
    gs  = mk_game_sched((;), (init=:I,), N_empty, (add_e=pra,),
                        quote
                            success, fail = add_e(init)
                            return success, fail
                        end)
    agents = Dict(:alice => first_agent)
    G = Graph(3)

    exps_cpu = run_game_sched!(gs, G, agents; T_max=1, terminal=never_terminal)
    exps_gpu = gpu_run_game_sched!(gs, G, agents; T_max=1)
    @test length(exps_cpu) == length(exps_gpu)
end

# ── 7. BOX_NATIVE_RULE — add vertex ──────────────────────────────────────────
#
# RuleApp (no player) compiles to BOX_NATIVE_RULE and routes through
# _gpu_native_pipeline!.  We pair a native rule with a PlayerRuleApp so the
# player's experience count tells us whether the native rule fired.
#
# Wire routing: every output wire is explicitly returned (no discard) because
# the wiring-diagram theory does not support `delete` on world-state wires.

@testset "GPU schedule — native add_v (BOX_NATIVE_RULE)" begin
    rapp = RuleApp(:native_add_v, rule_add_v, I_empty)
    pra  = PlayerRuleApp(:del_v, rule_del_v, I_empty, :alice)
    # native add_v fires (empty pattern always matches) → world gains a vertex
    # player del_v then fires on that vertex → 1 experience
    gs = mk_game_sched((;), (init=:I,), N_empty, (native_add_v=rapp, del_v=pra),
                        quote
                            added, fail_n = native_add_v(init)
                            success, fail_p = del_v(added)
                            return success, fail_p, fail_n
                        end)
    agents = Dict(:alice => first_agent)

    exps = gpu_run_game_sched!(gs, Graph(0), agents; T_max=2)
    @test length(exps) == 1
    @test exps[1].player == :alice
end

# ── 8. BOX_NATIVE_RULE — delete vertex ───────────────────────────────────────

@testset "GPU schedule — native del_v (BOX_NATIVE_RULE)" begin
    rapp = RuleApp(:native_del_v, rule_del_v, I_empty)
    pra  = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
    # native del_v fires if there are vertices; player add_v then runs
    gs = mk_game_sched((;), (init=:I,), N_empty, (native_del_v=rapp, add_v=pra),
                        quote
                            deleted, fail_n = native_del_v(init)
                            success, fail_p = add_v(deleted)
                            return success, fail_p, fail_n
                        end)
    agents = Dict(:alice => first_agent)

    # 3 vertices → native del fires → player add fires → 1 experience
    exps = gpu_run_game_sched!(gs, Graph(3), agents; T_max=2)
    @test length(exps) == 1

    # 0 vertices → native del fails → schedule exits via fail_n wire → 0 experiences
    exps_empty = gpu_run_game_sched!(gs, Graph(0), agents; T_max=2)
    @test length(exps_empty) == 0
end

# ── 9. BOX_NATIVE_RULE — add edge (FK writes in new R elements) ───────────────

@testset "GPU schedule — native add_e (BOX_NATIVE_RULE)" begin
    rapp = RuleApp(:native_add_e, rule_add_e, I_empty)
    pra  = PlayerRuleApp(:del_v, rule_del_v, I_empty, :alice)
    gs = mk_game_sched((;), (init=:I,), N_empty, (native_add_e=rapp, del_v=pra),
                        quote
                            added, fail_n = native_add_e(init)
                            success, fail_p = del_v(added)
                            return success, fail_p, fail_n
                        end)
    agents = Dict(:alice => first_agent)

    # 2 vertices → native add_e fires (monic match finds distinct pair)
    exps = gpu_run_game_sched!(gs, Graph(2), agents; T_max=2)
    @test length(exps) == 1

    # 1 vertex → no distinct pair → native add_e fails → 0 experiences
    exps_one = gpu_run_game_sched!(gs, Graph(1), agents; T_max=2)
    @test length(exps_one) == 0
end

# ── 10. CPU/GPU equivalence — native add_v ────────────────────────────────────

@testset "GPU schedule — CPU/GPU equivalence (native add_v)" begin
    rapp = RuleApp(:native_add_v, rule_add_v, I_empty)
    pra  = PlayerRuleApp(:del_v, rule_del_v, I_empty, :alice)
    gs = mk_game_sched((;), (init=:I,), N_empty, (native_add_v=rapp, del_v=pra),
                        quote
                            added, fail_n = native_add_v(init)
                            success, fail_p = del_v(added)
                            return success, fail_p, fail_n
                        end)
    agents = Dict(:alice => first_agent)

    for G in [Graph(0), Graph(2)]
        exps_cpu = run_game_sched!(gs, G, agents; T_max=2, terminal=never_terminal)
        exps_gpu = gpu_run_game_sched!(gs, G, agents; T_max=2)
        @test length(exps_cpu) == length(exps_gpu)
    end
end

# ── 11. CPU/GPU equivalence — native del_v ────────────────────────────────────

@testset "GPU schedule — CPU/GPU equivalence (native del_v)" begin
    rapp = RuleApp(:native_del_v, rule_del_v, I_empty)
    pra  = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
    gs = mk_game_sched((;), (init=:I,), N_empty, (native_del_v=rapp, add_v=pra),
                        quote
                            deleted, fail_n = native_del_v(init)
                            success, fail_p = add_v(deleted)
                            return success, fail_p, fail_n
                        end)
    agents = Dict(:alice => first_agent)

    for n in [0, 1, 3]
        G = Graph(n)
        exps_cpu = run_game_sched!(gs, G, agents; T_max=2, terminal=never_terminal)
        exps_gpu = gpu_run_game_sched!(gs, G, agents; T_max=2)
        @test length(exps_cpu) == length(exps_gpu)
    end
end

# ── Traced schedule loops multiple turns ──────────────────────────────────────
#
# A schedule with a trace wire (`tr=:I`) and a first box written with both the
# init and trace inputs (`add_v([init, tr])`) must loop for T_max turns, firing
# once per turn — not exit after turn 1.  Regression for two GPU-runner bugs:
#   * the trace RETURN wire was never fed back into the trace INPUT wire, and
#   * a box's extra input wires (the `tr` in `[init, tr]`) were dropped, so the
#     body could only ever fire on turn 1 (when `init` was active).

@testset "GPU schedule — traced schedule loops T_max turns" begin
    pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
    gs  = mk_game_sched((tr=:I,), (init=:I,), N_empty,
                        (add_v=pra, mw=merge_wires(I_empty)),
                        quote
                            s, f = add_v([init, tr])
                            out  = mw(s, f)
                            return out
                        end)
    agents = Dict(:alice => first_agent)
    for T in (1, 4)
        exps = gpu_run_game_sched!(gs, Graph(0), agents; T_max=T)
        @test count(e -> e.player == :alice, exps) == T
    end
end

# ── Per-turn world snapshots (track_turn_worlds) ──────────────────────────────
#
# Same traced add_v schedule: one vertex per turn, so with `track_turn_worlds`
# the t-th experience must carry pre/post worlds with t−1 / t vertices (end-of-
# turn snapshots) plus the fired box name in `info[:rule]`.  Without the flag
# the legacy decode (pre = episode start, post = final world, empty info) must
# be unchanged.

@testset "GPU schedule — track_turn_worlds per-turn snapshots" begin
    pra = PlayerRuleApp(:add_v, rule_add_v, I_empty, :alice)
    gs  = mk_game_sched((tr=:I,), (init=:I,), N_empty,
                        (add_v=pra, mw=merge_wires(I_empty)),
                        quote
                            s, f = add_v([init, tr])
                            out  = mw(s, f)
                            return out
                        end)
    agents = Dict(:alice => first_agent)
    T = 4

    exps = gpu_run_game_sched!(gs, Graph(0), agents; T_max=T,
                               track_turn_worlds=true)
    @test length(exps) == T
    for (t, e) in enumerate(exps)
        @test e.state.turn == t
        @test nparts(e.state.world, :V)      == t - 1
        @test nparts(e.next_state.world, :V) == t
        @test e.info[:rule] == :add_v
    end

    # Flag off: legacy decode unchanged (start/final worlds, no :rule info)
    legacy = gpu_run_game_sched!(gs, Graph(0), agents; T_max=T)
    @test length(legacy) == T
    for e in legacy
        @test nparts(e.state.world, :V)      == 0
        @test nparts(e.next_state.world, :V) == T
        @test !haskey(e.info, :rule)
    end
end
