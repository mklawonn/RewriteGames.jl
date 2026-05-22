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
