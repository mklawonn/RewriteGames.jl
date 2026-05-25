"""
Flux GPU-player vs CPU random-player benchmark.

Compares three strategies on the same game schedule:
  (A) CPU schedule  + FunctionAgent(rand)        — baseline random player
  (B) GPU schedule  + FunctionAgent(rand)         — GPU solver, same random choice
  (C) GPU schedule  + GPUFunctionPlayer(Flux MLP) — GPU solver, network scores matches

Without CUDA hardware all three run on CPU but via different engines:
  CPU schedule — Catlab backtracking homomorphism search + AlgebraicRewriting DPO
  GPU schedule — Turbo chunked-bitmask CSP solver + KernelAbstractions CPU kernels

The Flux MLP receives the raw candidate matrix (AbstractArray{Int32,2}) directly from
the solver scratch buffer — no world download occurs on the GPU path.

Run from the repo root (use the main project env, which has CUDA configured):
    julia --project=. benchmark/bench_flux_gpu_player.jl
"""

using Catlab, AlgebraicRewriting
using RewriteGames
using Flux
using CUDA
using Statistics: median
using Dates

println("=== Flux GPU-player vs CPU random-player benchmark ===")
println("Julia $(VERSION)  |  CUDA functional: $(CUDA.functional())")
println("Flux $(pkgversion(Flux))")
println("Date: $(Dates.now())\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function bench(f, n_timed; n_warmup=2)
    for _ in 1:n_warmup; f(); end
    [(@elapsed f()) for _ in 1:n_timed]
end

function ms(t); round(1000 * t; digits=2); end

function report3(label, t_a, t_b, t_c)
    ma = ms(median(t_a)); mb = ms(median(t_b)); mc = ms(median(t_c))
    println("  $(rpad(label, 22))  rand/CPU $(lpad(ma,7)) ms  " *
            "rand/GPU $(lpad(mb,7)) ms  " *
            "Flux/GPU $(lpad(mc,7)) ms  " *
            "(GPU-Flux vs CPU-rand: $(round(median(t_a)/median(t_c); digits=2))×)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Rules and schedules (del_v: 1-variable pattern)
# ─────────────────────────────────────────────────────────────────────────────

I_empty = Graph()
N_empty = Names(Dict("I" => I_empty))
L_v     = Graph(1)
rule_del_v = Rule(ACSetTransformation(I_empty, L_v),
                  ACSetTransformation(I_empty, I_empty))

pra    = PlayerRuleApp(:del_v, rule_del_v, I_empty, :agent)
gs_delv = mk_game_sched((;), (init=:I,), N_empty, (del_v=pra,),
    Meta.parse("begin ok, fail = del_v(init); return ok, fail end"))

# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

# (A) CPU baseline: random choice among legal actions
cpu_rand = FunctionAgent((state, acts) -> rand(acts))

# (B) GPU schedule with the same random choice (FunctionAgent still works in
#     the GPU scheduler; it downloads the world for the agent's view)
gpu_rand = FunctionAgent((state, acts) -> rand(acts))

# (C) Tiny untrained Flux MLP as a GPUFunctionPlayer.
#     Input:  candidates [n_vars × n_sols] = [1 × n_sols] — one vertex index per match
#     Output: scalar score per match → argmax → chosen column index
#
#     The model stays on CPU when CUDA is absent.  With CUDA, move via Flux.gpu().
#     On the GPU fast path (CUDA + scratch available), candidates is a CuArray subview
#     and the model runs entirely on device.  Without CUDA, candidates is a plain
#     Matrix{Int32} and the forward pass is CPU-only.
n_vars = 1   # del_v matches exactly one vertex
flux_model = Flux.Chain(
    Flux.Dense(n_vars => 32, Flux.relu),
    Flux.Dense(32 => 16, Flux.relu),
    Flux.Dense(16 => 1),
)
if CUDA.functional()
    flux_model = Flux.gpu(flux_model)
    println("Flux model moved to GPU.\n")
else
    println("CUDA not functional — Flux model runs on CPU (still exercises the player API).\n")
end

flux_player = GPUFunctionPlayer((g, cands, n_sols, turn) -> begin
    # cands: AbstractArray{Int32,2} — GPU subview when CUDA available, else CPU Matrix
    x = Float32.(Array(cands))   # bring to CPU (no-op when already CPU)
    scores = vec(flux_model(x))
    Int(argmax(scores))
end)

never_terminal = (W) -> (false, nothing)

agents_cpu  = Dict{Symbol, AbstractAgent}(:agent => cpu_rand)
agents_grnd = Dict{Symbol, AbstractAgent}(:agent => gpu_rand)
agents_flux = Dict{Symbol, AbstractAgent}(:agent => flux_player)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1 — first-call compile overhead
# ─────────────────────────────────────────────────────────────────────────────

println("─── 1. First-call compile overhead ─────────────────────────────────────")
for n in [10, 100, 500]
    G = Graph(n)
    t_a = @elapsed run_game_sched!(gs_delv, G, agents_cpu;  T_max=1, terminal=never_terminal)
    t_b = @elapsed gpu_run_game_sched!(gs_delv, G, agents_grnd; T_max=1)
    t_c = @elapsed gpu_run_game_sched!(gs_delv, G, agents_flux; T_max=1)
    println("  n=$(lpad(n,4)) vertices:  " *
            "rand/CPU $(lpad(ms(t_a),7)) ms  " *
            "rand/GPU $(lpad(ms(t_b),7)) ms  " *
            "Flux/GPU $(lpad(ms(t_c),7)) ms")
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2 — steady-state single step (compile excluded)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 2. Steady-state single step (T_max=1, median of 50 calls) ──────────")

for n in [10, 50, 100, 200, 500, 1000]
    G = Graph(n)
    t_a = bench(50) do
        run_game_sched!(gs_delv, G, agents_cpu; T_max=1, terminal=never_terminal)
    end
    t_b = bench(50) do
        gpu_run_game_sched!(gs_delv, G, agents_grnd; T_max=1)
    end
    t_c = bench(50) do
        gpu_run_game_sched!(gs_delv, G, agents_flux; T_max=1)
    end
    report3("n=$(lpad(n,4)) vertices", t_a, t_b, t_c)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3 — Flux forward-pass overhead (isolated)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 3. Isolated Flux forward-pass overhead ──────────────────────────────")
println("  One forward pass of the MLP on [n_vars × n_sols] candidate matrix\n")

for n_sols in [1, 10, 50, 100, 500]
    cands = rand(Int32(1):Int32(100), n_vars, n_sols)
    t = bench(20) do
        x = Float32.(cands)
        vec(flux_model(x))
    end
    println("  n_sols=$(lpad(n_sols,4)) candidates:  $(lpad(ms(median(t)),7)) ms")
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4 — multi-step loop: amortized per-step cost (T_max = n)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 4. Multi-step loop: drain all vertices (T_max=N, 10 runs) ──────────")

for n in [10, 30, 50, 100]
    G = Graph(n)
    t_a = bench(3) do
        run_game_sched!(gs_delv, G, agents_cpu; T_max=n, terminal=never_terminal)
    end
    t_b = bench(3) do
        gpu_run_game_sched!(gs_delv, G, agents_grnd; T_max=n)
    end
    t_c = bench(3) do
        gpu_run_game_sched!(gs_delv, G, agents_flux; T_max=n)
    end
    report3("N=$(lpad(n,3)), $(lpad(n,3)) steps", t_a, t_b, t_c)
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("""
\n─── Summary ──────────────────────────────────────────────────────────────────
  Columns: (A) CPU schedule + random FunctionAgent  [baseline]
           (B) GPU schedule + random FunctionAgent  [same policy, different engine]
           (C) GPU schedule + Flux MLP GPUFunctionPlayer  [network policy, GPU engine]

  Speedup column: median(A) / median(C) for n=200, T_max=1
""")

G200 = Graph(200)
ta = bench(() -> run_game_sched!(gs_delv, G200, agents_cpu; T_max=1, terminal=never_terminal), 30)
tb = bench(() -> gpu_run_game_sched!(gs_delv, G200, agents_grnd; T_max=1), 30)
tc = bench(() -> gpu_run_game_sched!(gs_delv, G200, agents_flux; T_max=1), 30)
println("  n=200, T_max=1:")
println("    (A) rand/CPU  median $(lpad(ms(median(ta)),7)) ms")
println("    (B) rand/GPU  median $(lpad(ms(median(tb)),7)) ms")
println("    (C) Flux/GPU  median $(lpad(ms(median(tc)),7)) ms")
println("    (A)vs(B): $(round(median(ta)/median(tb); digits=2))×   " *
        "(A)vs(C): $(round(median(ta)/median(tc); digits=2))×   " *
        "(B)vs(C): $(round(median(tb)/median(tc); digits=2))× (Flux overhead relative to rand/GPU)")

println("\nDone.")
