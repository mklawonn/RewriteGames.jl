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
