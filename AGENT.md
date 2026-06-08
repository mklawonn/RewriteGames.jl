# RewriteGames.jl — Agent Notes

## Environment and Testing

### Single Environment Strategy
We use a **single environment** for both development and testing. Test-only dependencies (like `Test.jl` and `Revise.jl`) are managed in the root `Project.toml` under the `[extras]` and `[targets]` sections.

**Mandate:** Do NOT create a `test/Project.toml` or `test/Manifest.toml`. This ensures that CUDA artifacts and version-sensitive dependencies remain synchronized across the entire project.

### Running Tests
Use the standard Julia command:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```
This is now the verified way to run tests. It correctly inherits the root project's CUDA configuration.

## Ambient Category and `rewrite_match`

### Background

AlgebraicRewriting's `rewrite_match(rule, m)` calls `infer_acset_cat(m)` when no `cat=`
argument is supplied. That function returns `MADVarACSetCat` only when
`hasvar(dom(m)) || hasvar(codom(m))` — i.e., when the match morphism's domain (the
rule's `L`) or codomain (the world) contains AttrType parts (AttrVars). The world state
never contains AttrVars. So if a rule's `L` also has no AttrVars, `infer_acset_cat`
falls back to `MADACSetCat`, which is incompatible with matches produced by
`get_matches(...; cat=MADVarACSetCat)`.

### Engine Fix (`sched_runner.jl`)

`_exec_native_rule!` computes `_cat` at line 278 and passes it to `get_matches`, but
the `rewrite_match` call at line 292 originally omitted `cat=`. **Fixed:** that call now
passes `cat=_cat` explicitly so the same category is used for both match search and
rewrite application.

The `PlayerRuleApp` path (`_exec_player!`) is not affected: it uses
`rewrite_match_maps(box.rule, chosen.match; cat=_cat)` and derives the new world from
`codom(maps[:rh])`, never calling bare `rewrite_match`.

### What `Rule(l, r; cat=...)` Does NOT Do

`Rule` uses `cat` during construction for span validation but **does not store it as a
field**. Passing `cat=FALCON_CAT` to `Rule` has no effect on subsequent `rewrite_match`
calls. Always propagate the category explicitly through `get_matches` → `rewrite_match`.

### Rules for Engine Contributors

1. **Always pass `cat=` to `rewrite_match`.** Derive it from `box.cat`, the game
   schedule's `cat`, or a known ACSetCategory constant. Never rely on `infer_acset_cat`.

2. **`_cat` is already computed in both execution paths.** In `_exec_player!` it is
   `box.cat` (or `infer_acset_cat(world)` as fallback). In `_exec_native_rule!` it is
   `box.cat` (or `infer_acset_cat(L)` as fallback). Use it.

3. **The `infer_acset_cat(world)` and `infer_acset_cat(L)` fallbacks are fragile.**
   The world state has no AttrVars, so both return `MADACSetCat` if the caller did not
   supply `box.cat`. Games that use rules with AttrVars must set `cat=` on every
   `PlayerRuleApp`. See the game-specific AGENT.md (e.g., `FalconRewriteGame/AGENT.md`)
   for guidance on when a rule's L has or lacks AttrVars.

4. **Free-floating AttrVars are rejected by Catlab.** An AttrType part (Sym, Boo, Num)
   that is allocated but not referenced by any attribute in the same ACSet is
   "free-floating". `backtracking_search` will error with
   `Cannot search for morphisms with free-floating variables`. Adding a dummy AttrType
   part just to make `hasvar(L)` true is never the correct fix — pass `cat=` explicitly
   instead.

### `match_cache.jl`

`MatchCache` stores and propagates `cat` through all internal `homomorphism`,
`get_matches`, and `can_match` calls. No bare category inference occurs there.
The cache must be constructed with the same `cat` that the game uses for match search
(`MatchCache(rule, FALCON_CAT, world)`).

---

## Preferred Style for Rule Construction (follow the LV example)

The canonical reference is `lotka_volterra_example.jl` in this repository.

### Convenience aliases

Define at the top of every rule file:

```julia
const AV  = AttrVar
const AV1 = AttrVar(1)
```

### `@acset` over `add_parts!`

Construct K, L, R pattern ACSets declaratively:

```julia
# Good
K = @acset MyACSet begin
    Sym = 2; Boo = 1; Sheep = 1; V = 1
    sheep_loc = [1]; sheep_eng = [AV1]; sheep_dir = [AV(2)]
end

# Avoid
K = MyACSet()
add_parts!(K, :Sym, 2); add_part!(K, :Boo)
add_part!(K, :Sheep; sheep_loc=1, sheep_eng=AttrVar(1), sheep_dir=AttrVar(2))
add_part!(K, :V)
```

### `homomorphism` over `ACSetTransformation` + identity maps

Compute span morphisms and NAC inclusions with `homomorphism`:

```julia
# Good — AttrVars present, category inferred
l = homomorphism(K, L; monic=true)
r = homomorphism(K, R; monic=true)

# Good — no AttrVars in K/L, pass cat= explicitly
l = homomorphism(K, L; cat=MY_CAT, monic=true)

# Good — NAC as monic inclusion
L_nac = copy(L)
add_part!(L_nac, :SomeToken; some_fk=1)
nac = NAC(homomorphism(L, L_nac; monic=true))

# Avoid
comps = _id_comps(K)
l = ACSetTransformation(K, L; cat=MY_CAT, comps...)
nac = NAC(ACSetTransformation(L, L_nac; cat=MY_CAT, _id_comps(L)...))
```

### Representable presheaves (yoneda cache)

For minimal single-object patterns, use the yoneda cache:

```julia
gSheep, gWolf, gV, … = ob_generators(FinCat(SchLV))
yLV = yoneda_cache(LV; clear=true)
S = ob_map(yLV, gSheep)   # generic sheep — minimal ACSet with one Sheep + all attrs
W = ob_map(yLV, gWolf)
G = ob_map(yLV, gV)
```

Representables are the natural interfaces for agent morphisms and for patterns where
you want the most general match against a single entity.

### `expr` for attribute-modifying rules

When a rule changes an attribute value without adding or removing objects, use `expr`:

```julia
# Good: turn a sheep left without deleting and re-creating it
Rule(l, r;
     expr = (Dir = [((d,),) -> left(d)],),
     cat  = MY_CAT)
```

`expr` is a `NamedTuple` keyed by AttrType name. Each entry is a vector of functions
(one per R AttrVar) receiving inherited value tuples and returning the new value.
This avoids the spurious object deletion/recreation that would otherwise be needed
to change an attribute.

---

## Known Issue: EPS Threshold Too Conservative for nc_max=48

**Context (commit a0d7082):** To fix a shared-memory overflow when `@localmem UInt64
(nc_max*128,)` allocated exactly 48 KB with no room for other kernel arrays, the
`use_eps` condition was changed to `nc_max * 1024 + 256 > 49152` (i.e., nc_max ≥ 48
always uses EPS). This over-corrects: for n_vars ≤ 7 with nc_max=48 the workspace is
only 43 KB and turbo_block would fit fine.

**Impact:** The Falcon Tornado scenario (nc_max=48) routes all CSP calls to the EPS
pipeline (global-memory workspace) instead of turbo_block (shared memory, ~100× faster
per access). EPS is GPU-parallel and cpu_propagate! fast-fails infeasible cases, so
absolute throughput may still be acceptable, but turbo_block would be faster.

**Proper fix:** Change `nvnm16 = nc_max * 128` to `nvnm16 = 7 * nc_max * 16` (allocate
for 7 variables instead of 8), and update `use_eps` to:
```julia
use_eps = nc == 1 || n_vars > 7 || n_vars * nc_max * 128 + 256 > 49152
```
This restores turbo_block for n_vars ≤ 7 with nc_max=48 (43 KB + overhead < 48 KB),
routes n_vars=8 and nc_max ≥ 64 to EPS. nc_max=64 (>4096 elements/type) must always
use EPS regardless — the 48 KB shared-memory limit is a hard physical constraint.

---

## Known Issue: `done` flag in `gpu_run_game_sched!`

`run_game_sched!` (CPU) sets `experience.done = true` on the final experience
whenever the terminal predicate fires or `turn > T_max`.

`gpu_run_game_sched!` can exit for a third reason: a schedule exit wire fires
(no active trace wire to continue). In that case the terminal predicate is not
consulted and the last experience is pushed with `done = false`.

**Impact:** Code that inspects `last(exps).done` to detect episode termination
will behave differently depending on which engine ran the episode.

**Possible fix:** In the GPU runner's exit-wire path, call the terminal predicate
(or unconditionally set `done = true`) on the final experience before returning,
mirroring the CPU runner's behaviour in `_exec_player!`.

## GPU Match Ordering Diverges from CPU

The GPU CSP solver and Catlab's `backtracking_search` enumerate matches in
**different orders**.  Deterministic agents such as "always pick first action"
will therefore take different trajectories on `run_game_sched!` vs
`gpu_run_game_sched!`, producing different episode lengths and action sequences
even from the same initial world.

**Consequences for tests:**
- Do not assert `length(exps_cpu) == length(exps_gpu)` in equivalence tests.
- Do not compare action sequences or world states step-by-step.
- Safe invariants: both produce non-empty experiences; all experiences have
  valid `player`/`state` fields; both terminate within `T_max`.

**The RNG-consumption corollary (cost a lot of debugging time — read this).** Even
holding the engine *fixed*, two code paths that make *identical* match/filter
decisions can produce **different trajectories** if they consume the global RNG
differently. Concretely: the CPU NAC filter runs Catlab `homomorphisms` (which
touches the RNG), the GPU-native NAC filter does not — so a stochastic-policy
episode under `RG_FORCE_CPU_NAC` diverges from the GPU path *even though every
kept solution set is identical*. **Do not** "equivalence-test" a filter/solver
change by comparing episode trajectories on a fixed seed. Compare the **per-solve
kept solution SET** instead (see `_nac_diag` / env `RG_NAC_DIAG`, which runs both
filters on the same world and counts set mismatches). The pipeline is otherwise
deterministic per process (same seed + same model ⇒ identical episode), so a
trajectory diff is a real signal *only* when RNG consumption is held equal.

## GPU Kernel Compilation (JIT Tax) — Val{NM} and Val{NVNM16}

### Background

`turbo_block_kernel!` and `turbo_eps_kernel!` use `Val{NM}` (= nc_max) and
`Val{NVNM16}` as compile-time type parameters to enable `@localmem` static
allocation and `MVector{NM,...}` register arrays. Julia/CUDA compiles a separate
LLVM specialization for each distinct `(NM, NVNM16)` pair encountered at runtime.

**Original code** used `nvnm16 = n_vars * nc_max * 16`, creating one specialization
per (nc_max, n_vars) pair — up to 8 distinct `Val{NVNM16}` values per nc_max. With
nc_max=48, each specialization requires ~15-30 min of LLVM compilation → 2-4 hours
before the first training episode runs.

**Fix (commit 002b481):** Changed to `nvnm16 = nc_max * 128` (= 8 × nc_max × 16,
the maximum possible n_vars × nc_max × 16 when `use_eps` enforces n_vars ≤ 8).
All n_vars values sharing the same nc_max now compile to one `Val{NVNM16}`
specialization → compilation time reduced to ~15-30 min total.

The kernel's `@localmem UInt64 (NVNM16,)` over-allocates slightly for n_vars < 8,
but all actual indexing uses `n_vars_nm = n_vars * NM`, so no out-of-bounds access.

### Precompile Note

A bare `using RewriteGames` does NOT trigger GPU kernel compilation because the
CUDA extension only loads when both `RewriteGames` and `CUDA` are in scope. To
warm the kernel cache before launching multiple training processes, run a
precompile command that imports CUDA and exercises the solver once:
```julia
using RewriteGames, CUDA
# trigger one small solve to compile kernels
```
Alternatively, accept the per-process JIT penalty on first episode (~15-30 min
with the nvnm16 fix, vs. 2-4 hours before).

---

## `GPUFunctionPlayer` for Deterministic GPU Testing

`GPUFunctionPlayer(f)` wraps a plain Julia function as a GPU-compatible agent
without requiring a neural network.  Useful for writing deterministic tests:

```julia
# always pick the first match
gpu_agents = Dict{Symbol, AbstractAgent}(
    :blue => GPUFunctionPlayer((g, cands, n_sols, t) -> 1),
    :red  => GPUFunctionPlayer((g, cands, n_sols, t) -> 1),
)
exps = gpu_run_game_sched!(gs, world, gpu_agents; T_max=10)
```

`f` receives: the GPU-resident `GPUACSet` `g`, a `CuArray{Int32,2}` of shape
`[n_vars × n_sols]`, the solution count `n_sols`, and the turn number.
It must return an `Int` index into the solution columns (1-based).

---

## GPU Agent Loops (`BOX_AGENT_LOOP`) Now Execute on the GPU Runner

**Background.** `compile_schedule` lowered `agent(pra; n=:X)` boxes to a
`BOX_AGENT_LOOP` opcode with a registered agent interface and a body
sub-schedule, but the host-orchestrated runner (`run_gpu_schedule!`) had **no
handler** for that opcode — only `WEAKEN`, `COIN`, `NATIVE_RULE`, and
`PLAYER_RULE`.  An agent box therefore fell through every branch: its input wire
stayed set, no output wire activated, and the move phase silently never ran
(platforms never moved).  Agent loops only worked on the CPU runner
(`_exec_subsched!`).

**Fixes (branch `fix/gpu-agent-loop-residency`):**

1. **Runtime execution.** The per-box dispatch was factored into
   `_dispatch_gpu_box!(box, b_idx, sched, …)` (shared by the main per-turn loop
   and the agent-loop body runner).  A new `BOX_AGENT_LOOP` branch calls
   `_exec_agent_loop!`, which runs the body sub-schedule once per **live agent
   instance** — `k = g.n_live[agent_obj][]` (e.g. `:Platform`).  The world `g`
   mutates in place, so it threads through iterations automatically.  This
   mirrors the CPU `for am in homomorphisms(agent_interface, world)` semantics.

2. **No fixed cap.** `k` is read from `g.n_live` each time the box runs, so a
   larger world iterates over *more* instances instead of dropping the overflow.
   A compile-time unroll to a fixed `N` would have re-introduced exactly the
   silent-skip cap that lifting `MAX_CHUNKS` removed.  Reading `k` is a single
   host-side scalar — no GPU→CPU *data* round-trip.

3. **Compiler double-wrap bug.** In `_process_steps!`, an inline agent-loop box
   compiled its body with `compile_schedule(box, …)` while `box._agent_name`
   was still set, which re-entered the top-level agent-loop wrapping and
   produced a *nested* `BOX_AGENT_LOOP`.  The runtime would then have iterated
   the body `k²` times.  Fixed by stripping `_agent_name` (building `inner`)
   before compiling the body, matching the top-level wrapping path.

4. **Experience attribution.** Agent-loop body firings push `GpuRewriteEvent`s
   tagged with the *parent* agent-loop box index (`event_box_idx`), and Phase 5
   of `gpu_run_game_sched!` resolves the acting player for `BOX_AGENT_LOOP`
   boxes via `_agent_loop_body_player` (the body's player, e.g. `:blue`) rather
   than the parent box's `box_players` entry (which holds the agent *object*,
   e.g. `:Platform`).  `GpuRewriteEvent` stays `isbits` (no `Symbol` field) so
   the on-device master kernel's `CuArray{GpuRewriteEvent}` is unaffected.

Covered by `test/test_gpu_agent_loop.jl` (firing count == instance count for
worlds up to 310 instances).

## GPU Residency: CPU AC Fast-Fail Removed

`gpu_turbo_solve` used to `Array(d_gpu)` + `Array(hf_flat_gpu)` and run
`cpu_propagate!` before launching the GPU kernel, to skip infeasible CSPs.  That
was a GPU→CPU **data** round-trip on the hot path (once per solve), violating the
"everything stays on the GPU" invariant flagged in the TODOs.  Removed: the
turbo/EPS kernels run AC-1 propagation themselves and return zero solutions for
infeasible CSPs, so correctness is unchanged.  If infeasible-slot throughput
ever needs a pre-filter, add a GPU-resident feasibility kernel (no host copy).

## DONE: EPS Threshold Fix (formerly TODO #1)

Fixed in `ext/GPURewritingExt/solver/DiveSolveKernel.jl` on branch `feature/gpu-shared-mem-graph`:

- `nvnm16 = 7 * nc_max * 16` (was `nc_max * 128`)
- `use_eps = nc == 1 || n_vars > 7 || n_vars * nc_max * 128 + 256 > 49152`

For nc_max=48, n_vars ≤ 7: 43,264 bytes < 49,152 → `turbo_block` restored.

---

## DONE: GPU Graph Representation of El(world) (formerly TODO #2)

Implemented in `ext/GPURewritingExt/rewriting/GPUGraphData.jl` on branch `feature/gpu-shared-mem-graph`:

- **`GPUGraphData`** struct: COO edge list + node features + tombstone mask, all GPU-resident. Node indices = `obj_node_offset[type] + slot_id` (stable across deletions).
- **`build_gpu_graph(g, schema, enc)`**: builds from GPUACSet via thin CPU-side download of active/homs/attrs, then uploads COO + features.
- **`update_graph_deletions!` / `update_graph_additions!`**: GPU-kernel incremental updates after rewrites.
- **`live_coo(graph)`**: extracts live-edge COO for GNN input construction.
- **`GPUSchedulerState.graph_data`**: maintained by scheduler; built lazily on first GNN player call, refreshed after each rewrite (`graph_dirty` flag).
- **`AbstractGNNPlayer`**: new abstract type (subtype of `AbstractGPUPlayer`); scheduler builds graph and passes it as `graph_data` keyword to `select_action_gpu`.

---

## DONE: Coproduct Zone Decomposition (formerly Stretch TODO)

Implemented in `ext/GPURewritingExt/control/ZonePartition.jl` on branch `feature/gpu-shared-mem-graph`:

- **`ZonePartition`**: pre-built per-zone domain bitmasks for each (zone_idx, obj_type) pair. GPUACSet remains single source of truth; masks restrict CSP domain without data duplication.
- **`build_zone_partition(g, schema, nc, zone_fn)`**: user supplies `zone_fn(obj_sym, slot) → zone_idx`.
- **`_build_domains_gpu_zoned!`**: like `_build_domains_gpu!` but uses zone masks; global types get full active mask.
- **`collect_zoned_solutions!`**: runs solver once per zone, collects and deduplicates solutions with global slot IDs.
- **`update_zone_masks!`**: selective rebuild after movement rewrites.
- Pass via `gpu_run_game_sched!(...; zone_partition=partition)`.

FalconRewriteGame provides **`build_falcon_zone_partition(g, schema)`** and **`update_falcon_zone_partition!`** in `src/zone_partition.jl` with the full FK-chain derivation for all 28 FalconACSet object types.

### Design Note: Domain Masking vs True Sharding

The current implementation uses **domain masking** (sparse domains in the global nc-chunk space).  After the EPS threshold fix, `turbo_block` now runs for nc_max=48 with n_vars ≤ 7, so zone-local patterns with sparse domains benefit from shared-memory AC-1. For a further win, true sharding (separate GPUACSet per zone with zone-local nc_max ≈ 3–4) would be the next step, but requires infrastructure for shard-local upload/rewrite/sync not yet implemented.

---

## FalconGNNPlayer (New Agent Type)

Implemented in `FalconRewriteGame/src/gnn_agent.jl` on branch `feature/gpu-shared-mem-graph`:

- **`FalconGNNPlayer`**: `AbstractGNNPlayer` with two-layer `GraphConv` GNN + MLP scorer.
- `select_action_gpu` receives `graph_data::GPUGraphData`, constructs a `GNNGraph` from `live_coo`, runs GNN forward entirely on device, gathers embeddings for match variables via CUDA fancy indexing, scores solutions, returns argmax (1 Int32 to CPU).
- **`make_falcon_gnn_agents`**: convenience constructor for Blue/Red agents.
- Add `GraphNeuralNetworks` v1.1.0 as FalconRewriteGame dependency.

---

## Performance: the engine is HOST-TRANSFER bound, not compute bound (2026-06)

> **Game-name + staleness note (2026-06-08):** FalconRewriteGame now has a SINGLE
> game, `falcon_tornado_A`. Older perf lessons below name the former `game_E` (the
> agent-dense multimove token game) and `game_full` (the full attribute scenario =
> `falcon_tornado_A` at `fraction=1.0`). The "host-bound / no single dominant cost"
> conclusions here predate codomain-decomposition + intra-bytecode propagation — see
> "GPU CSP propagation — intra-bytecode is the DEFAULT" below: at full `T_MAX` on the
> full scenario the turbo_block propagation IS the dominant compute cost (~2.75× win).

A profiling pass on an agent-dense episode (the former game_E) found the GPU CSP solve
kernels have **~0 self-time** — the cost is on the host. **Profile self-time and
optimize what the profile shows; do not assume the GPU kernels are the
bottleneck.** Two host costs dominated, both now fixed:

### DONE: `download_acset` vectorized (bulk column writes)

`download_acset` (`rewriting/GPUACSet.jl`) rebuilt the host ACSet with a
per-element `set_subpart!` loop per live part per hom/attr — the **#1 self-time**
(~27% of an episode), and it runs ~2×/turn (terminal check + any GNN-agent graph
rebuild). Replaced with vectorized compaction (`cumsum` of the active-flag
vector + `findall`) and **bulk** `set_subpart!(result, :, name, col)` writes, with
a per-element fallback only when a hom target is missing/deleted (`0`) or an
attribute decodes to `nothing` (preserves the original "skip" semantics exactly).
Equivalence: `test/test_gpu_download.jl`. Result: **~2.6–3× faster agent-game
episode**; the per-call download win grows with schema richness (Falcon has many
homs/attrs). Lesson: building a Catlab ACSet element-by-element is expensive —
always write whole columns.

### DONE: GPU-native general NAC/PAC (no host homsearch, no world download)

After the download fix, the new #1 cost was the CPU application-condition filter
(`_filter_nac_solutions`): it downloads the world and runs a host-side Catlab
`homomorphisms` **per candidate solution**. The single-new-element `NacSpec` fast
path couldn't cover conditions whose forbidden structure has ≥2 new elements
(e.g. Falcon's red `sam_aim` NAC3: a fresh `ThreatSystem` + a `ShotAt` referencing
it), so those fell to the CPU. **Fix (the principled one): check a condition with
the SAME GPU solver a rule uses.** `lower_pattern_to_csp` (`lowering/CSPLowering.jl`)
lowers the condition's extended pattern `ac_L` into its own `CSPProblem` — a
variable for *every* element including the NAC's new ones, `PROP_FUNC` +
concrete-attr `PROP_ATTR_EQ`, free `AttrVar`s left unconstrained, non-monic
(mirrors `homomorphisms(ac_L, world; no_bind=true)`). `_gpu_filter_conditions`
(`control/Scheduler.jl`) pins the shared-`L` variables to the candidate match
(`_pin_csp_var!`) and runs `gpu_dive_solve`: a NAC fires on ≥1 solution, a PAC
needs one. Wired as **tier 2** in `_gpu_solve_inplace!` between the `NacSpec` fast
path (tier 1) and the CPU homsearch (tier 3, now only a fallback for patterns that
fail to lower). Reuses `_build_domains_gpu`/`_build_hom_fwd_gpu` (which read the
live `g`), so **no world download**. Validated by per-solve kept-set equivalence
(`_nac_diag`, env `RG_NAC_DIAG`) — see the RNG-consumption note above for *why*
set-equivalence (not trajectory) is the correct gate.

### Catlab gotcha: `homomorphism(...; monic=true)` errors when non-unique

`homomorphism(X, Y; monic=true)` throws `Exceeded 1: [...]` when **more than one**
monic homomorphism exists (e.g. `Graph(1) → Graph(2)`), rather than returning the
first. When you need a specific morphism (e.g. a NAC inclusion) and it isn't
unique, build it explicitly: `ACSetTransformation(L, L_out; V=[...], ...)`.

---

## Batched-PINNED agent loop + batched scoring (`perf/incremental-batched-agent`)

`BOX_AGENT_LOOP` has a fast path `_exec_agent_loop_batched!` (`control/Scheduler.jl`)
for the common "each agent picks one move" body (a single `PLAYER_RULE` box +
WEAKEN plumbing; agent is a CSP variable; a GPU player).  It builds the whole-world
`hom_forward` + base domains **once per box**, then does a **PINNED solve per
agent** (copy base domains → `_pin_agent_var!` → `gpu_turbo_solve`), so each
agent's matches are byte-identical to the sequential loop's pinned solve.  This
fixes the reverted `perf/batch-agent-loop`, which solved once UNPINNED and grouped
by the agent variable: the solver returns one witness per corner, so
interchangeable-resource matches (a Platform with several FuelTokens) diverged from
the per-agent enumeration.  Falls back to the per-instance `_exec_agent_loop!` for
any body it can't handle; `RG_NO_BATCH_AGENT` forces the fallback.  Gate:
`RG_AGENT_DIAG` (each agent's turbo solve == reference `gpu_dive_solve`, compared on
the first `n_vars` rows — the scratch turbo path returns buffer-width vectors padded
past `n_vars`; only rows `1..n_vars` are meaningful and ever indexed downstream).

**Semantics:** the batched loop solves all agents against the BOX-ENTRY world (a
snapshot) and applies their moves with no compaction in between (deletes tombstone,
adds extend the high-water mark, so box-entry slots stay valid).  This is
**simultaneous-move** semantics; the sequential loop is order-dependent (an earlier
move can enable a later one).  They are NOT required to match — simultaneous is the
intended behaviour.

**Batched scoring** — `select_action_gpu_batched` (declared in `src/RewriteGames.jl`)
scores ALL agents in ONE player call: the agent loop collects every agent's
candidates, then calls it once.  The default loops `select_action_gpu` (so non-GNN
GPU players are unchanged); a GNN player overrides it with a single batched (masked)
transformer forward.

**Perf lesson (measured on the full scenario, A40, T_MAX=2):** building `hom_forward` ONCE
instead of 315×/turn gave **no wall-time win (1.00×)** — the per-agent rebuilds are
~free (GPU-parallel), confirming the engine is host/sync-bound, not compute-bound.
After the merged sync work (`perf/fewer-syncs`, `perf/batch-nac`), a flat profile
shows **no single dominant cost**: stream compaction, GNN scoring, rewrite apply,
tier-1 NAC, per-episode `compile_schedule`, and solve-result download are each
~5–13%.  Remaining single-episode levers: batched GNN scoring (done) and a
cross-episode `compile_schedule` cache (deferred — a training-throughput win whose
cache must guard `enc`/world consistency).  NB: low-variance timing needs a
deterministic red + seeded model; a random red roughly doubles full-scenario episode
time (more SAM activity → more downstream work).

---

## GPU CSP propagation — intra-bytecode is the DEFAULT (2026-06-08, ~2.75× on full game)

`turbo_block_kernel!`'s per-block AC-1 propagation fixpoint now has **three modes**,
chosen at compile time via `Val{PROP_MODE}` from `_prop_mode()` (env):

- **mode 2 — intra-bytecode (DEFAULT).** Bytecodes processed serially, but each
  `PROP_FUNC` is parallelized over its `nc` domain chunks across the 32 lanes — lane
  owns chunk `c`, so narrowing is **race-free (no atomics)**. Three sync-free phase
  fns `_intra_forward!` (lane → shared `new_d1_sh[c]`), `_intra_backward!` (lane owns
  chunk c, narrows v1 = new_d1_sh[c] and v2 &= OR-over-v1-bits of hom column c),
  `_intra_simple!` (PROP_NEQ/EQ on tid==1), driven by a small inline skeleton at the
  dive + solve sites (the bytecode loop + 2 barriers/bytecode stay in the kernel).
- **mode 0 — serial (`RG_SERIAL_PROP`).** The single-thread (`tid==1`) reference /
  kill-switch — what the kernel did before. `_propagate_serial!`.
- **mode 1 — block-parallel over bytecodes (`RG_BLOCK_PROP`).** `_propagate_block!`;
  a **~6-8% regression** on the full scenario, kept only for hypothetical many-bytecode work.

**Why (profile via env `RG_SOLVE_DIAG`, logged in `_launch_turbo_block!`):** every
full-scenario turbo_block solve has **nc=45** (huge domains, ~2880 elems/var) but
**n_bc=2-9** (few bytecodes). Spreading bytecodes across lanes (mode 1) has ~no
width; parallelizing each `PROP_FUNC`'s ~2880-element scan over the lanes (mode 2)
cuts its critical path ~32×.

**Measured (A40, `build_scenario(1.0)` ~2838-target world, `build_planning_exec_sched_attr`,
GNN blue vs fixed red, isolated, n=5):** intra **9.3 s/turn** vs serial **25.5 →
~2.75×**. This **revises the "host/sync-bound, no single dominant cost" lesson above**:
that was T_MAX=2 and *predates codomain decomposition*. Once codomain decomp removed
the move-solve cost, the per-turn time concentrated in the non-move turbo_block solves
(SEAD/engage, nc=45), which are **propagation-dominated** — hence intra's ~2.75×. The
remaining ~9.3 s/turn is now the dive/solve DFS + featurization.

**Validation:** solver set-equivalence intra == serial == Catlab for PROP_FUNC
(nc=2..41) + PROP_NEQ (nc=2,8); full suite 1025/1025 with intra default AND with
`RG_SERIAL_PROP`. Local solver A/B harness: build a CSP via the ext internals, call
`gpu_turbo_solve` twice toggling the env mode, `issetequal` + compare to Catlab
`homomorphisms` (force nc≥2, i.e. >64 target parts, or it dispatches to EPS).

**KernelAbstractions / GPU-Julia gotchas (hard-won; apply to any new device code in
this kernel):**
- **Device fns called from a KA `@kernel` MUST use UNTYPED args.** Typed args with
  free type params (`dom::DOM … where DOM`) compile standalone and under raw `@cuda`
  but throw `InvalidIRError: jl_f_throw_methoderror` *only* inside the KA `@kernel`.
  `_propagate_*!` / `_intra_*!` are all untyped.
- **`@synchronize` cannot live inside a called device fn (or a macro that expands to
  one).** Keep the barrier-bearing loop inline in the kernel; split per-pass work
  into sync-free phase fns.
- **`Atomix.@atomic` has no method on `@localmem` (shared) arrays** → use
  `CUDA.atomic_and!(pointer(dom,i), v)`, or design for race-freedom (intra's
  chunk-ownership needs no atomics).
- Allocate `MVector` / `@localmem` at function/kernel top, never inside a dynamic loop.

**Timescale history (full-scenario turn):** intra-bytecode is the latest of a stack of
GPU-solver wins — codomain decomposition (~6×), batched Tier-2 NAC filter (~1.36×), GPU
attribute thresholds, now intra propagation. A session carrying the old ~100-130 s/turn
mental model should expect **~9 s/turn** on the full scenario today.

## Attribute support on the GPU engine (on `main`, commit 9687827)

The GPU solver matches and modifies attributes on-device (no CPU round-trip):
- **Ordinal threshold matching** — `PROP_ATTR_LEQ` / `PROP_ATTR_GEQ` bytecodes
  (`TCNBytecode.jl`) pre-filter a variable's domain by a `≤`/`≥` bound; set per-rule
  via `set_attr_thresholds!(rule, thresholds)` (`src/schedule/attr_thresholds.jl`,
  exported).
- **Affine attribute deltas** — `set_attr_deltas!(rule, deltas)` applies affine
  updates to attribute values on rewrite, GPU-resident.

NB: FalconRewriteGame's AGENT.md still carries an older "GPU engine has no attribute
arithmetic — use tokens, not Counter+`expr`" caveat. Affine deltas partially supersede
that (affine updates work on the GPU now); a general `expr` is still unsupported.
