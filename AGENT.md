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

## TODO: Fix EPS Threshold (Too Conservative for nc_max=48)

See **Known Issue: EPS Threshold Too Conservative for nc_max=48** above for full diagnosis and the correct fix recipe. Summary:

- Change `nvnm16 = nc_max * 128` → `nvnm16 = 7 * nc_max * 16`
- Update `use_eps` to `nc == 1 || n_vars > 7 || n_vars * nc_max * 128 + 256 > 49152`
- This restores `turbo_block` for n_vars ≤ 7 with nc_max=48 (43 KB < 48 KB limit)
- nc_max=64 must always use EPS regardless (57 KB > 48 KB hard limit)

File: `ext/GPURewritingExt/solver/DiveSolveKernel.jl` — both the `gpu_turbo_solve` and `_gpu_turbo_fill_scratch!` call sites.

---

## TODO: Reduce CPU/GPU Round-trips in GNNAgent

### Current Bottleneck

Every time a rewrite rule fires and marks the world dirty, `select_action_gpu` in the training script does:

1. **Full world download**: `ext.download_acset(g_world, enc, FalconACSet)` — transfers the entire `GPUACSet` to CPU.
2. **CPU graph construction**: `world_to_graph_data(world_cpu)` — builds edge lists, type-id arrays, and node-attr tensors on CPU.
3. **GNN forward + embedding download**: `p.model.gnn(gnn_g, x_init)` runs on GPU, but the input graph was built on CPU and uploaded; output embeddings are then downloaded back.

Every agent call (even with a cached world) also incurs:

4. **Match download**: `Array(cands)` — downloads all candidate match columns from device.
5. **Match fuser round-trip**: `CUDA.cu(all_match_inputs)` upload + `p.model.match_fuser(...)` + `Array(Flux.cpu(...))` download.
6. **Logit download**: `Array(logits)` — downloads transformer output to select an action.

The `world_dirty` flag caches steps 1–3 across multiple calls within the same schedule step, but steps 4–6 happen on every agent invocation.

### Goal: GPU-Side Graph Construction After Rewrites

The ideal architecture keeps the world on the GPU throughout an episode:

- Maintain a `GPUGraphData` structure (edge index, type IDs, node-attr tensor) resident on device.
- After a rewrite fires, **incrementally update** `GPUGraphData` on the GPU rather than re-downloading the whole ACSet:
  - Insertions: append new node rows and update affected edge lists in place.
  - Deletions: mark nodes/edges as inactive (masked attention) rather than reallocating.
- The GNN input graph is already on device → no CPU upload needed.
- GNN embeddings stay on device; the match fuser and transformer operate entirely on device.
- Only the final scalar action index is returned to CPU (1 Int32 per call).

### Immediate (Lower-Effort) Wins

1. **GPU match encoding**: Build `all_match_inputs` on the GPU from already-resident embeddings and the `cands` array — eliminate `Array(cands)` + CPU loop + `CUDA.cu(all_match_inputs)`.
2. **Fused GNN + match_fuser**: Run GNN and match scoring without intermediate host transfers.
3. **GPU-side argmax**: Compute the action index on device; download only 1 Int32 instead of the full logit vector.

### Full Solution: Incremental GPU Graph Updates

Requires a `GPUGraphData` type in `GPURewritingExt` that the GPU runner updates after each rewrite application (the new world already lives on device as `codom(maps[:rh])`). Needs:

- A device-side CSR or COO graph representation.
- A mapping from ACSet part IDs to node indices.
- Kernel or scatter/gather ops for incremental node/edge insertion and deletion.

This eliminates the dominant per-step latency and is the highest-impact optimization for episode throughput.

---

## Stretch TODO: Decompose CSP for Shared Memory Regardless of Problem Size

> **Note for future sessions:** If the user asks to "tackle all TODOs," confirm whether they also mean this stretch TODO — it is a significant research/engineering effort distinct from the items above.

### Motivation

`turbo_block` (shared memory) is ~100× faster per access than `turbo_eps` (global memory). The current design routes to EPS whenever the domain bitset exceeds 48 KB (nc_max ≥ 48 on A40 GPUs). For the full Falcon Tornado scenario (nc_max=48), every CSP call uses EPS.

### Approach: User-Supplied ACSet Partition

The ACSet can be partitioned by a user-supplied decomposition — e.g., a geographic zone partition of `Target` and `Platform` objects. A CSP for a pattern that matches only within one zone has nc_max proportional to that zone's object count, not the total world size.

For Falcon Tornado with 4 zones averaging ~710 targets per zone: nc_max per zone ≈ ⌈710/64⌉ = 12, well within the nc_max=16 turbo_block range.

### Interface Sketch

```julia
# User supplies a partition function mapping (object type, part ID) → shard index
partition = ZonePartition(world)   # inspects PlatformZone / TargetZone FKs

# GPU runner shards the world per partition, solves each shard with turbo_block,
# and re-indexes matches to global IDs before returning to the agent
exps = gpu_run_game_sched!(sched, world, agents; partition=partition, T_max=T_max)
```

### Caveats

- Rules whose patterns span multiple partitions (e.g., "platform in zone A engages target in zone A") are the common case — the partition must respect each rule's variable typing so all variables for a given match fall within one shard.
- Cross-partition rules must fall back to EPS or a multi-shard join.
- Infrastructure for sharding `GPUACSet` and merging match results across shards does not yet exist.
