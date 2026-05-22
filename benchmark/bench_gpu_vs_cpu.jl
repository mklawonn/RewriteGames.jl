"""
GPU vs CPU performance benchmark.

Compares `gpu_run_game_sched!` (Turbo CSP solver + scatter-write rewriting)
against `run_game_sched!` (Catlab homomorphism search + AlgebraicRewriting).

Without CUDA hardware both paths run on CPU but with entirely different engines:
  CPU path  — Catlab backtracking homomorphism search + AlgebraicRewriting DPO
  GPU path  — Turbo chunked-bitmask CSP solver + KernelAbstractions CPU kernels

Run:
    julia --project=test benchmark/bench_gpu_vs_cpu.jl
"""

using Catlab, AlgebraicRewriting
using RewriteGames
using CUDA
using Statistics: median
using Dates

println("=== GPU vs CPU Performance Benchmark ===")
println("Julia $(VERSION)  |  CUDA functional: $(CUDA.functional())")
println("Date: $(Dates.now())\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function bench(f, n)
    f()  # one warmup to absorb residual JIT
    [(@elapsed f()) for _ in 1:n]
end

function report(label, cpu_t, gpu_t)
    c = 1000 * median(cpu_t)
    g = 1000 * median(gpu_t)
    ratio  = c / g
    winner = ratio >= 1.0 ? "GPU" : "CPU"
    println("  $(rpad(label, 44))  CPU $(lpad(round(c; digits=2), 8)) ms  " *
            "GPU $(lpad(round(g; digits=2), 8)) ms  " *
            "$(winner) $(round(max(ratio, 1/ratio); digits=2))×")
end

# ─────────────────────────────────────────────────────────────────────────────
# Rules and schedules
# ─────────────────────────────────────────────────────────────────────────────

I_empty = Graph()
N_empty = Names(Dict("I" => I_empty))
first_agent = FunctionAgent((s, acts) -> first(acts))
agents = Dict{Symbol, AbstractAgent}(:agent => first_agent)

# Rule A: delete one vertex
L_v = Graph(1)
rule_del_v = Rule(ACSetTransformation(I_empty, L_v),
                  ACSetTransformation(I_empty, I_empty))

# Rule B: match one edge and delete it (preserves src and tgt vertices)
L_e = Graph(2); add_edge!(L_e, 1, 2)
K_e = Graph(2)
rule_del_e = Rule(ACSetTransformation(K_e, L_e, V=[1,2]),
                  ACSetTransformation(K_e, K_e, V=[1,2]))

# Rule C: identity on one edge — pure pattern match, no structural change
rule_id_e = Rule(ACSetTransformation(L_e, L_e, V=[1,2], E=[1]),
                 ACSetTransformation(L_e, L_e, V=[1,2], E=[1]);
                 monic=true)

function one_shot_sched(rule, name)
    pra = PlayerRuleApp(name, rule, I_empty, :agent)
    mk_game_sched((;), (init=:I,), N_empty, NamedTuple{(name,)}((pra,)),
                  Meta.parse("begin ok, fail = $name(init); return ok, fail end"))
end

gs_delv  = one_shot_sched(rule_del_v, :del_v)
gs_dele  = one_shot_sched(rule_del_e, :del_e)
gs_id_e  = one_shot_sched(rule_id_e,  :id_e)

function make_ring(n)
    G = Graph(n)
    for i in 1:n; add_edge!(G, i, mod1(i+1, n)); end
    G
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1 — compile / setup overhead (paid once per rule+world pair)
# ─────────────────────────────────────────────────────────────────────────────

println("─── 1. One-time compile overhead ───────────────────────────────────────")
println("  CPU = build MatchCache;  GPU = compile_schedule + AdhesiveCubes\n")

for n in [10, 50, 100, 200]
    G = make_ring(n)

    t_cpu = @elapsed MatchCache(rule_del_e, nothing, G; match_limit=typemax(Int))

    # compile_schedule is the expensive part; run with T_max=1 on a tiny world
    # to force it, then we measure just the first gpu call on the real world
    t_gpu = @elapsed gpu_run_game_sched!(gs_dele, G, agents; T_max=1)

    println("  n=$(lpad(n,3)) vertices:  " *
            "CPU MatchCache $(lpad(round(1000t_cpu; digits=1), 7)) ms  |  " *
            "GPU compile+run $(lpad(round(1000t_gpu; digits=1), 7)) ms")
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2 — single-step: del_v (1-variable pattern)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 2. Single step: del_v  (1-variable pattern, T_max=1) ───────────────")
println("  Median over 30 calls (compile excluded)\n")

for n in [10, 50, 100, 200]
    G = Graph(n)
    cpu_t = bench(30) do; run_game_sched!(gs_delv, G, agents; T_max=1); end
    gpu_t = bench(30) do; gpu_run_game_sched!(gs_delv, G, agents; T_max=1); end
    report("n=$(lpad(n,3)) vertices", cpu_t, gpu_t)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3 — single-step: del_e (2-variable + FK constraint)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 3. Single step: del_e  (2-variable + FK, T_max=1) ──────────────────")
println("  Ring graph: n vertices, n edges  (median over 30 calls)\n")

for n in [10, 50, 100, 200]
    G = make_ring(n)
    cpu_t = bench(30) do; run_game_sched!(gs_dele, G, agents; T_max=1); end
    gpu_t = bench(30) do; gpu_run_game_sched!(gs_dele, G, agents; T_max=1); end
    report("n=$(lpad(n,3)) vertices, $(lpad(n,3)) edges", cpu_t, gpu_t)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4 — multi-step loop: drain all edges (T_max = n)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 4. Multi-step loop: delete all edges  (T_max = N) ──────────────────")
println("  Exercises the scheduler loop + repeated match+rewrite  (10 runs)\n")

for n in [10, 30, 50]
    G = make_ring(n)
    cpu_t = bench(10) do; run_game_sched!(gs_dele, G, agents; T_max=n); end
    gpu_t = bench(10) do; gpu_run_game_sched!(gs_dele, G, agents; T_max=n); end
    report("N=$(lpad(n,2)) edges, $(lpad(n,2)) steps", cpu_t, gpu_t)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 5 — pure matching: identity rule on one edge
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 5. Pure matching: identity edge rule  (no rewrite, T_max=1) ────────")
println("  Isolates CSP solve vs homomorphism search  (median over 30 calls)\n")

for n in [10, 50, 100, 200]
    G = make_ring(n)
    cpu_t = bench(30) do; run_game_sched!(gs_id_e, G, agents; T_max=1); end
    gpu_t = bench(30) do; gpu_run_game_sched!(gs_id_e, G, agents; T_max=1); end
    report("n=$(lpad(n,3)) vertices, $(lpad(n,3)) edges", cpu_t, gpu_t)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 6 — amortized per-episode cost (100 episodes, compile paid once)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 6. Amortized per-episode cost  (100 episodes, T_max=1) ─────────────")
println("  compile_schedule paid once before the loop; shows steady-state cost\n")

for (label, gs, n) in [
        ("del_v n=50",  gs_delv, 50),
        ("del_e n=50",  gs_dele, 50),
        ("del_v n=200", gs_delv, 200),
        ("del_e n=200", gs_dele, 200),
    ]
    G = label[end-2:end] == "del_v" || startswith(label, "del_v") ? Graph(n) : make_ring(n)
    # pick world type based on rule
    G = contains(label, "del_v") ? Graph(n) : make_ring(n)

    # CPU: MatchCache built once, then 100 episodes
    _cache = MatchCache(contains(label, "del_v") ? rule_del_v : rule_del_e,
                        nothing, G; match_limit=typemax(Int))
    t_cpu = @elapsed for _ in 1:100
        run_game_sched!(gs, G, agents; T_max=1)
    end

    # GPU: compile on first call, then 100 more
    gpu_run_game_sched!(gs, G, agents; T_max=1)
    t_gpu = @elapsed for _ in 1:100
        gpu_run_game_sched!(gs, G, agents; T_max=1)
    end

    c_per = 1000 * t_cpu / 100
    g_per = 1000 * t_gpu / 100
    ratio = c_per / g_per
    winner = ratio >= 1.0 ? "GPU" : "CPU"
    println("  $(rpad(label, 14))  " *
            "CPU $(lpad(round(c_per; digits=3), 8)) ms/ep  " *
            "GPU $(lpad(round(g_per; digits=3), 8)) ms/ep  " *
            "$(winner) $(round(max(ratio, 1/ratio); digits=2))×")
end

println("\nDone.")
