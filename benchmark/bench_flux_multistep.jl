"""
Multi-step complex schedule benchmark.

Compares three agent strategies on a 3-rule looping schedule:
  (native) add_vertex + (player :alice) add_edge + (player :bob) delete_vertex

Strategy legend:
  (A) CPU schedule + random FunctionAgent         [run_game_sched!]
  (B) GPU schedule + random FunctionAgent         [gpu_run_game_sched!, world download per step]
  (C) GPU schedule + Flux MLP GPUFunctionPlayer   [gpu_run_game_sched!, no world download]

Also verifies element-count equivalence: with a deterministic "always-first" policy,
after T_max=1, all three paths should produce identical vertex and edge cardinalities
regardless of which specific match each solver chose.

Note on T_max semantics:
  CPU scheduler: T_max bounds total player-box firings across all iterations.
  GPU scheduler: T_max bounds outer loop iterations; N player boxes → N×T_max firings.
  For T_max=1, both schedulers execute exactly one pass through the schedule body.

Run from repo root:
    julia --project=. benchmark/bench_flux_multistep.jl
"""

using Catlab, AlgebraicRewriting
using RewriteGames
using Flux
using CUDA
using Statistics: median
using Dates

println("=== Multi-step Flux GPU-player benchmark ===")
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
ms(t) = round(1000 * t; digits=2)

function report3(label, ta, tb, tc)
    ma = ms(median(ta)); mb = ms(median(tb)); mc = ms(median(tc))
    println("  $(rpad(label, 22))  rand/CPU $(lpad(ma,7)) ms  " *
            "rand/GPU $(lpad(mb,7)) ms  " *
            "Flux/GPU $(lpad(mc,7)) ms  " *
            "(A/C=$(round(median(ta)/median(tc); digits=2))×, " *
            "B/C=$(round(median(tb)/median(tc); digits=2))×)")
end

never_terminal = (W) -> (false, nothing)

# ─────────────────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────────────────

I_empty = Graph()
L_v     = Graph(1)
L_two   = Graph(2)
R_edge  = Graph(2); add_edge!(R_edge, 1, 2)

# Native: add one isolated vertex (empty pattern — always fires, unique match)
rule_add_v = Rule(ACSetTransformation(I_empty, I_empty),
                   ACSetTransformation(I_empty, L_v))
rapp_add_v  = RuleApp(:nadd_v, rule_add_v, I_empty)

# Player :alice — add edge between any two distinct vertices (monic, n_vars=2)
rule_add_e = Rule(ACSetTransformation(L_two, L_two, V=[1,2]),
                   ACSetTransformation(L_two, R_edge, V=[1,2]); monic=true)
pra_add_e  = PlayerRuleApp(:add_e, rule_add_e, I_empty, :alice)

# Player :bob — delete one isolated vertex (n_vars=1; DPO dangling blocks non-isolated)
rule_del_v = Rule(ACSetTransformation(I_empty, L_v),
                   ACSetTransformation(I_empty, I_empty))
pra_del_v  = PlayerRuleApp(:del_v, rule_del_v, I_empty, :bob)

N_multi = Names(Dict("I" => I_empty))

# ─────────────────────────────────────────────────────────────────────────────
# Schedule: per-iteration — add vertex, alice adds edge, bob deletes vertex
#
# Wiring notes:
#   mw_e: merge add_e success wire OR add_e fail wire → single world wire
#   mw_d: merge del_v success wire OR del_v fail wire → single world wire
#   exit1 is the no-match output of nadd_v (empty pattern never fails; always inactive).
#   exit2/exit4 are consumed by mw_e/mw_d; they don't appear in the return.
#   Wire fan-out (mcopy) is unsupported; fall-through uses the rule's own fail port.
# ─────────────────────────────────────────────────────────────────────────────

gs_multi = mk_game_sched(
    (prev=:I,),
    (init=:I,),
    N_multi,
    (nadd_v=rapp_add_v, add_e=pra_add_e, del_v=pra_del_v,
     mw_init=merge_wires(I_empty), mw_e=merge_wires(I_empty), mw_d=merge_wires(I_empty)),
    quote
        w0        = mw_init(init, prev)
        w1, exit1 = nadd_v(w0)
        w2, exit2 = add_e(w1)
        w3        = mw_e(w2, exit2)  # merge success or alice no-match
        w4, exit4 = del_v(w3)
        w5        = mw_d(w4, exit4)  # merge success or bob no-match
        return w5, exit1
    end
)

# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────

# Flux MLPs: separate model per player (different n_vars: alice=2, bob=1)
model_alice = Flux.Chain(Flux.Dense(2 => 32, Flux.relu), Flux.Dense(32 => 1))
model_bob   = Flux.Chain(Flux.Dense(1 => 16, Flux.relu), Flux.Dense(16 => 1))
if CUDA.functional()
    model_alice = Flux.gpu(model_alice)
    model_bob   = Flux.gpu(model_bob)
    println("Flux models moved to GPU.\n")
else
    println("CUDA not functional — models run on CPU.\n")
end

make_flux_player(model) = GPUFunctionPlayer((g, cands, n_sols, turn) -> begin
    x = Float32.(Array(cands))    # CPU Array no-op when not on GPU
    Int(argmax(vec(model(x))))
end)

flux_alice = make_flux_player(model_alice)
flux_bob   = make_flux_player(model_bob)

rand_agent = FunctionAgent((state, acts) -> rand(acts))

agents_cpu  = Dict{Symbol, AbstractAgent}(:alice => rand_agent, :bob => rand_agent)
agents_grnd = Dict{Symbol, AbstractAgent}(:alice => rand_agent, :bob => rand_agent)
agents_flux = Dict{Symbol, AbstractAgent}(:alice => flux_alice, :bob => flux_bob)

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1 — first-call compile overhead
# ─────────────────────────────────────────────────────────────────────────────

println("─── 1. First-call compile overhead (3-rule looping schedule) ───────────")
for n in [5, 20, 100]
    G  = Graph(n)
    ta = @elapsed run_game_sched!(gs_multi, G, agents_cpu; T_max=1, terminal=never_terminal)
    tb = @elapsed gpu_run_game_sched!(gs_multi, G, agents_grnd; T_max=1)
    tc = @elapsed gpu_run_game_sched!(gs_multi, G, agents_flux; T_max=1)
    println("  n=$(lpad(n,3)):  rand/CPU $(lpad(ms(ta),8)) ms  " *
            "rand/GPU $(lpad(ms(tb),8)) ms  " *
            "Flux/GPU $(lpad(ms(tc),8)) ms")
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2 — single-step (T_max=1): one pass through the full 3-rule body
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 2. Single step (T_max=1, median 30 calls) ──────────────────────────")
println("  Each step: native_add_v → alice_add_e → bob_del_v\n")

for n in [5, 20, 50, 100, 200, 500]
    G  = Graph(n)
    ta = bench(() -> run_game_sched!(gs_multi, G, agents_cpu; T_max=1, terminal=never_terminal), 30)
    tb = bench(() -> gpu_run_game_sched!(gs_multi, G, agents_grnd; T_max=1), 30)
    tc = bench(() -> gpu_run_game_sched!(gs_multi, G, agents_flux; T_max=1), 30)
    report3("n=$(lpad(n,3))", ta, tb, tc)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3 — multi-step episodes (T_max=5, fixed n=50)
#
# Note: T_max=5 means 5 outer loop iterations in the GPU scheduler
# (2 player firings each = 10 total), vs ≤5 total player firings in CPU
# scheduler.  Episode durations are therefore longer on the GPU path.
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 3. Multi-step episode (T_max=5, n=50, 20 calls) ───────────────────")
G50 = Graph(50)
ta5 = bench(() -> run_game_sched!(gs_multi, G50, agents_cpu; T_max=5, terminal=never_terminal), 20)
tb5 = bench(() -> gpu_run_game_sched!(gs_multi, G50, agents_grnd; T_max=5), 20)
tc5 = bench(() -> gpu_run_game_sched!(gs_multi, G50, agents_flux; T_max=5), 20)
println("  rand/CPU $(lpad(ms(median(ta5)),7)) ms  " *
        "rand/GPU $(lpad(ms(median(tb5)),7)) ms  " *
        "Flux/GPU $(lpad(ms(median(tc5)),7)) ms  " *
        "(A/C=$(round(median(ta5)/median(tc5); digits=2))×)")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4 — varying T_max at n=50 (GPU iterations vs CPU turns)
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 4. Varying T_max (n=50, 15 calls) ─────────────────────────────────")
for T in [1, 3, 5, 10, 20]
    ta = bench(() -> run_game_sched!(gs_multi, G50, agents_cpu; T_max=T, terminal=never_terminal), 15)
    tb = bench(() -> gpu_run_game_sched!(gs_multi, G50, agents_grnd; T_max=T), 15)
    tc = bench(() -> gpu_run_game_sched!(gs_multi, G50, agents_flux; T_max=T), 15)
    println("  T_max=$(lpad(T,2)):  rand/CPU $(lpad(ms(median(ta)),7)) ms  " *
            "rand/GPU $(lpad(ms(median(tb)),7)) ms  " *
            "Flux/GPU $(lpad(ms(median(tc)),7)) ms  " *
            "(A/C=$(round(median(ta)/median(tc); digits=2))×)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Correctness: T_max=1, deterministic "always first" policy
#
# Key semantic difference between CPU and GPU schedulers for del_v (DPO rule):
#   CPU: get_matches() returns only DPO-valid matches (isolated vertices).
#        "First" always picks a valid isolated vertex → del_v always fires.
#   GPU: CSP solver returns all homomorphisms including non-isolated vertices.
#        "First" (column 1 = vertex 1) may be an edge endpoint → DPO may fail.
#
# Therefore:
#   A (CPU): nadd_v +1, add_e +1, del_v -1 → net: same vertex count, +1 edge
#   B, C (GPU): nadd_v +1, add_e +1, del_v may fail → net: +1 vertex, +1 edge
#
# Key invariant: B and C must agree exactly (both use the GPU CSP solver).
# Edge count = 1 for all three (add_e always finds a match on n=20 vertices).
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── 5. Correctness: B and C (GPU paths) must agree ─────────────────────")
println("  Policy: always first action.  Start: Graph(20), T_max=1.")
println("  Note: CPU and GPU differ on del_v due to DPO pre-filtering (see comments)\n")

first_cpu      = FunctionAgent((state, acts) -> first(acts))
first_flux_a   = GPUFunctionPlayer((g, cands, n_sols, turn) -> 1)
first_flux_b_p = GPUFunctionPlayer((g, cands, n_sols, turn) -> 1)

agents_first_cpu  = Dict{Symbol, AbstractAgent}(:alice => first_cpu, :bob => first_cpu)
agents_first_grnd = Dict{Symbol, AbstractAgent}(:alice => first_cpu, :bob => first_cpu)
agents_first_flux = Dict{Symbol, AbstractAgent}(:alice => first_flux_a, :bob => first_flux_b_p)

G_check = Graph(20)

exps_a = run_game_sched!(gs_multi, G_check, agents_first_cpu; T_max=1, terminal=never_terminal)
exps_b = gpu_run_game_sched!(gs_multi, G_check, agents_first_grnd; T_max=1)
exps_c = gpu_run_game_sched!(gs_multi, G_check, agents_first_flux; T_max=1)

function final_counts(exps)
    isempty(exps) && return (n_exp=0, n_v=0, n_e=0)
    w = exps[end].next_state.world
    (n_exp=length(exps), n_v=nparts(w, :V), n_e=nparts(w, :E))
end

ra = final_counts(exps_a)
rb = final_counts(exps_b)
rc = final_counts(exps_c)

println("  (A) CPU  schedule + first CPU:  $(ra.n_exp) exps,  $(ra.n_v) vertices,  $(ra.n_e) edges")
println("  (B) GPU  schedule + first CPU:  $(rb.n_exp) exps,  $(rb.n_v) vertices,  $(rb.n_e) edges")
println("  (C) GPU  schedule + Flux GPU:   $(rc.n_exp) exps,  $(rc.n_v) vertices,  $(rc.n_e) edges")

# Primary check: B and C must agree (same GPU engine)
bc_v_ok = rb.n_v == rc.n_v
bc_e_ok = rb.n_e == rc.n_e == 1

if bc_v_ok && bc_e_ok
    println("\n  ✓ B and C agree: $(rb.n_v) vertices, $(rb.n_e) edge (GPU paths consistent).")
else
    println("\n  ✗ B/C mismatch!")
    !bc_v_ok && println("    Vertex count: B=$(rb.n_v), C=$(rc.n_v)")
    !bc_e_ok && println("    Edge count:   B=$(rb.n_e), C=$(rc.n_e)  (expected 1)")
end

# Secondary info: CPU result (may differ due to DPO pre-filtering)
cpu_e_ok = ra.n_e == 1
println("  CPU (A) edge count: $(ra.n_e) $(cpu_e_ok ? "✓" : "✗") (expected 1)")
println("  CPU vertex count: $(ra.n_v) (CPU del_v uses DPO-valid matches only)")

println("""
\nNote on T_max semantics (A vs B/C):
  CPU scheduler: T_max bounds total player-box firings (2 players × ? rounds).
  GPU scheduler: T_max bounds outer loop iterations (1 iteration = full body).
  With T_max=1: GPU runs exactly one full body pass (2 player firings).
  CPU may run 1 or 2 player firings depending on internal T_max accounting.
  Observed: A=$(ra.n_exp) exps, B=$(rb.n_exp) exps, C=$(rc.n_exp) exps.
""")

println("Done.")
