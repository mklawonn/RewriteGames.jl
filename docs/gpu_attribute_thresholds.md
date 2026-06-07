# GPU attribute support: discretized ordinal attributes + threshold constraints

Status: **design / plan**. Motivated by the `game_F` `aar` bottleneck (a fuel cap
expressed as 8 interchangeable `FuelToken` parts → depth‑8 NAC dive per candidate
= ~100% of per‑turn time). The token-cloud pattern exists only because the GPU
pipeline historically couldn't match/mutate attributes; this plan closes that gap.

## TL;DR

Most of the machinery the user asked for **already exists**. The encoder already
discretizes ordinal attributes into ranks where `<`/`≤` are meaningful, the device
already stores encoded `Int32` attribute columns, and the Turbo bytecode already
**defines** a `PROP_ATTR_LEQ` op. What's missing is (1) a rule-level way to
*state* a threshold, (2) **emitting** `PROP_ATTR_LEQ` in lowering, and (3)
*consuming* it at domain-init (4 small sites). That's **Phase 1** and it's the
whole win for `aar`'s cap check, `rtb_bingo`, and `kill_depleted`. **Phase 2**
(affine attribute mutation on GPU) is a larger lift needed to fully replace
counter tokens (`move` −1, `aar` +1, `sam_damage` −dmg).

---

## 1. What already exists (verified)

### 1a. Discretization — `ext/GPURewritingExt/lowering/AttributeEncoder.jl`
`AttributeEncoder` maps Julia attribute values ↔ GPU `Int32` IDs:
- **Nominal** (`Symbol`/`Bool`/discrete): each distinct value → unique Int32
  (encounter order). Used for `==` matching.
- **Ordinal** (`Real` subtypes): values **sorted and assigned ranks 1..n so that
  `<` and `≤` hold on the integer rank.** ← exactly the requested discretization.
- **Custom**: `discretizers::Dict{Symbol, Pair{encode_fn, decode_fn}}` — a
  per-attribute **binning hook**, already plumbed through
  `gpu_run_game_sched!(...; discretizers=...)` (`GPURewritingExt.jl:164,173`).
- `Int32(0)` is reserved as "unset / wildcard".

So **ordered integer attributes and custom binning are done.**

### 1b. Device storage — `rewriting/GPUACSet.jl`
`g.attrs[a] :: CuVector{Int32}` — encoded attribute value per element (0 = wildcard).
Comparisons against these columns are trivial integer ops in a kernel.

### 1c. Equality matching end-to-end (the template to copy)
- **Lowering** (`lowering/CSPLowering.jl:122`): each concrete attribute value in
  `L` → `tcn(PROP_ATTR_EQ; var1, param1=attr_idx, param2=encode_value(...))`.
  `AttrVar`s are skipped (free wildcard).
- **Consumed at domain-init** (NOT in the dive loop): `PROP_ATTR_EQ` builds a
  bitmask `{w : attrs[a][w] == req}` and ANDs it into the variable's domain.
  Four consumer sites:
  - `control/Scheduler.jl:344` `_apply_attr_masks_gpu_device!` (on-device,
    `_attr_mask_fill_kernel!` two-pass, no download) — the hot path.
  - `control/Scheduler.jl:417` `_apply_attr_masks_gpu!` (CPU fallback).
  - `rewriting/IncrementalUpdate.jl:301` (incremental domain refresh).
  - `GPURewritingExt.jl:344` `_apply_attr_masks_world!` (host world path).
- **Dive/propagation kernels do NOT see attributes** — `DiveSolveKernel.jl:803`:
  "Attribute / domain-size constraints are pre-baked into the initial domains, so
  only PROP_FUNC / PROP_EQ / PROP_NEQ are handled here." This is the key
  architectural fact: **attribute constraints are domain pre-filters, not search
  steps.** Thresholds inherit this for free.

### 1d. The threshold op is already DEFINED — `solver/TCNBytecode.jl:60`
```
PROP_ATTR_LEQ = 0x0005  # var1's attribute column param1 must be ≤ param2 (ordinal only)
```
…but it is **never emitted** by lowering and **never consumed** at domain-init.

---

## 2. What's missing

| Capability | Status | Needed for |
|---|---|---|
| Ordinal discretization (ranks) | ✅ exists | thresholds |
| Custom binning (`discretizers`) | ✅ exists | continuous attrs (lat/lon/range) |
| Device `Int32` attr columns | ✅ exists | all |
| `PROP_ATTR_EQ` match | ✅ exists | `==` matches (e.g. fuel==0) |
| `PROP_ATTR_LEQ` op defined | ✅ exists | thresholds |
| **Rule-level threshold spec** | ❌ build | thresholds |
| **Emit `PROP_ATTR_LEQ` in lowering** | ❌ build | thresholds |
| **Consume `PROP_ATTR_LEQ` at domain-init** | ❌ build | thresholds |
| **Affine attribute mutation (`expr`) on GPU** | ❌ build | counter rules (move/aar/sam_damage) |

---

## 3. Phase 1 — threshold matching (`val ≤ k`, `val ≥ k`, `val < k`, `val > k`)

Small, localized, **no dive/propagation kernel changes** (pre-filter only).

### 3a. Rule-level threshold spec (the only new public surface)
AlgebraicRewriting's `Rule`/`RuleApp` can't store this (and the ACSet pattern
can't encode `≤`). Add a RewriteGames-side side-channel. Two options:

- **(preferred) wrapper helper** `with_attr_thresholds(app, thresholds)` where
  `app` is a `PlayerRuleApp`/`RuleApp` and `thresholds` is
  `Vector{NamedTuple{(:ob,:idx,:attr,:op,:val)}}` (e.g.
  `(ob=:Platform, idx=2, attr=:plat_fuel, op=:leq, val=7)` = "L's Platform #2 has
  fuel ≤ 7"). Stored in a field on `PlayerRuleApp` (it's a RewriteGames struct —
  add `attr_thresholds::Vector{...}=[]`) or, for native `RuleApp`, in a
  side registry keyed by `objectid(rule)`.
- mirrors AlgebraicRewriting's own `predicates=` concept (homsearch supports
  per-attribute predicate fns); we restrict to the GPU-expressible subset
  (`≤ ≥ < >` against a constant on an ordinal/nominal-ordered attr).

The CPU `get_matches` path can enforce the same thresholds via its `predicates=`
kwarg so CPU and GPU agree (needed for A/B determinism tests).

### 3b. Lowering — `lower_rule_to_csp` (CSPLowering.jl)
After the `PROP_ATTR_EQ` loop, add:
```julia
for t in attr_thresholds(rule)              # (ob, idx, attr, op, val)
    haskey(var_offset, t.ob) || continue
    v  = var_offset[t.ob] + (t.idx - 1)
    a  = attr_index(schema, t.attr)
    k  = encode_value(enc, t.attr, t.val)   # ordinal rank of the threshold
    k == 0 && continue                       # value not in encoder's range → handle (see 3d)
    op, kk = t.op == :leq ? (PROP_ATTR_LEQ, k) :
             t.op == :lt  ? (PROP_ATTR_LEQ, k-1) :          # < k  ==  ≤ k-1 on ranks
             t.op == :geq ? (PROP_ATTR_GEQ, k) :            # new op, or complement
             t.op == :gt  ? (PROP_ATTR_GEQ, k+1) : error()
    push!(bytecodes, tcn(op; var1=v, param1=a, param2=kk))
end
```
Add a `PROP_ATTR_GEQ = 0x0006` op (symmetric to LEQ) so `≥`/`>` are first-class
rather than via domain complement.

### 3c. Domain-init consumption (the 4 sites in §1c)
Mirror the `PROP_ATTR_EQ` branch with a comparison-mode. Device kernel
`_attr_mask_fill_kernel!` (`Scheduler.jl:309`) gains a `cmp::Int32` arg:
```
cmp == 0:  attrs[i] == req        # EQ (existing)
cmp == 1:  1 <= attrs[i] <= req   # LEQ  (exclude 0 = wildcard/unset)
cmp == 2:  attrs[i] >= req        # GEQ
```
Each consumer loops `bc.op in (PROP_ATTR_EQ, PROP_ATTR_LEQ, PROP_ATTR_GEQ)` and
passes the matching `cmp`. ~10 lines per site, 4 sites + 1 kernel.

### 3d. Encoder edge case — threshold value not present in the world
`encode_value` returns 0 for values it hasn't seen (e.g. cap=7 when no element
currently has exactly 7). For thresholds we need the **rank position**, not exact
membership. Add `encode_threshold(enc, attr, v, op)` that returns
`searchsortedfirst/​last` rank for ordinal attrs (the boundary rank even if `v`
isn't an existing value), independent of exact presence. Trivial extension of the
existing ordinal `searchsorted` logic.

**Phase 1 deliverable:** rules can pre-filter candidates by `attr {≤,<,≥,>} const`
at domain-init cost (one bitmask AND), with zero dive-kernel overhead.

---

## 4. Phase 2 — affine attribute mutation on GPU (`attr := matched_attr ± c`)

Needed to replace **counter** tokens (fuel/health) entirely. Today the addition
path (`rewriting/AdditionKernel.jl`) writes only **static** encoded values
(`new_r_attr[o][a][j]`, precomputed from `R`) or inherits matched values;
`_update_preserved!` patches static K-side changes. No value is a *function* of
the match, so AlgebraicRewriting `expr=(Eng=[vs->vs-1])` has no GPU analogue.

### Design: compile-time affine `expr` for ordinal/counter attributes
Restrict to the GPU-expressible subset: `new_val = source_val + c` where `c` is a
compile-time integer and the attribute is **identity-encoded** (value == rank;
true for small-range integer counters like fuel 0..8, health 0..n). Then `+c` in
real space == `+c` in rank space — a pure on-device integer add, no decode.

- Detect this case when lowering the rule's `exprs` (AlgebraicRewriting `Rule`
  *does* carry `exprs`): if `expr` for AttrType `T` is affine in a single matched
  AttrVar, record `(target_elem, attr, source_elem, delta)`.
- Extend `_update_preserved!` / the addition kernel to, for such entries, read
  `g.attrs[a][match[source_elem]]`, add `delta`, clamp to `[1, n_ranks]`, write to
  the target element's column. One new kernel mirroring `write_attr_kernel!`.
- **Out of scope:** non-affine expr, or affine expr on *binned* attrs (where
  rank arithmetic ≠ value arithmetic) — those stay CPU-only or are rejected at
  compile time with a clear error.

This is the bigger lift (~1 new kernel + lowering of `exprs` + clamp semantics)
but is well-bounded and covers every counter in game_F.

---

## 5. `aar` fix using the new functionality

Represent fuel as an **ordinal integer attribute** instead of a `FuelToken` cloud:
`plat_fuel::Attr(Platform, Num)` (identity-encoded 0..FUEL_CAP).

- **Match** (Phase 1): receiver `plat_fuel ≤ FUEL_CAP-1` via `PROP_ATTR_LEQ`.
  The 8-`FuelToken` cap NAC — the entire bottleneck — **disappears**; the cap is
  one domain-mask AND.
- **Set** (Phase 2): receiver `plat_fuel := plat_fuel + 1` via affine expr.
- Candidate count is still O(tankers × receivers in zone), but each candidate is
  now O(1) (no NAC dive), so `aar` drops from ~235 s/turn to the ~0.01 s/turn the
  other boxes already run at.

**Interim fix (no engine change, available now):** gate `aar` on a `NeedsFuel`
token (added by `move`/bingo when fuel is low) and consume it — cuts candidates to
tankers × low-fuel receivers and removes the cap NAC. Use this to unblock game_F
while Phases 1–2 land.

## 6. Other game_F rules that benefit (token → integer attribute)

Migrating `FuelToken`→`plat_fuel` and `HealthToken`→`plat_health` (two ordinal
attrs) **deletes two Ob tables and all per-token matching**:

| Rule | Today (tokens) | With attributes | Needs |
|---|---|---|---|
| `aar` | 8-token cap NAC (O(P²)×dive) | `fuel ≤ cap-1` + `+1` | P1 + P2 |
| `rtb_bingo` | NAC: no `FuelToken` | `fuel == 0` (PROP_ATTR_EQ) | **works today** once fuel is an attr |
| `kill_depleted` | NAC: no `HealthToken` | `health == 0` | **works today** once health is an attr |
| `move_platform` | consume 1 `FuelToken` | `fuel ≥ 1` + `−1` | P1 + P2 |
| `sam_damage` | match+delete `dmg` `HealthToken`s (the `any=true`/"Exceeded 1" hack) | `health ≥ dmg` + `−dmg` | P1 + P2 (also removes the interchangeable-token blowup) |
| `mk_deploy_*` | seed `n_fuel`+`n_health` tokens | set `fuel=n_fuel, health=n_health` (static R) | **works today** (static attr set already supported) |

`establish_link`/`isr`/`sead`/`dead`/`engage` are structural — no change.

Net: `aar` fixed, `sam_damage`'s interchangeable-token blowup eliminated, two Ob
tables removed, and the schema moves off the "tokens everywhere" workaround for
all counter quantities — which is the general capability the user asked for.

## 7. Effort / sequencing

1. **Phase 1** (threshold matching): ~1–2 days. New op `PROP_ATTR_GEQ`, lowering
   emit, 4 domain-init consumers + 1 kernel `cmp` arg, `encode_threshold`,
   `with_attr_thresholds` API, CPU `predicates` parity, A/B determinism test.
   Unblocks `rtb_bingo`/`kill_depleted` immediately and `aar`'s cap check.
2. **Phase 2** (affine expr mutation): ~2–4 days. Lower affine `exprs`, new
   source-relative write kernel, clamp, identity-encoding guard + compile error
   for unsupported expr/binned cases. Unblocks `aar`/`move`/`sam_damage` sets.
3. **game_F migration**: swap `FuelToken`/`HealthToken` for `plat_fuel`/
   `plat_health`; rewrite the 6 rules above; re-verify on GPU.

## 8. Risks / notes
- **Determinism:** GPU thresholds must match the CPU `predicates` path exactly
  (A/B test, like the batched-NAC work). The shared ordinal encoder makes this
  natural.
- **Binned attrs + mutation don't mix:** affine expr is only valid for
  identity-encoded ordinals; reject binned-attr expr at compile time.
- **Encoder growth:** ordinal ranks shift as new values appear
  (`extend_encoder!`); thresholds are re-encoded per solve from the live encoder,
  so this is consistent, but worth a test when the value set grows mid-episode.
- **Scope:** Phase 1 alone removes the `aar` bottleneck's NAC and is low-risk;
  Phase 2 is the general counter-attribute capability.
```
