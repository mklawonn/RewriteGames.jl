# GPU Performance Plan: Eliminating Remaining CPU Bottlenecks

This document catalogues every remaining CPU bottleneck in `gpu_run_game_sched!` and
provides a concrete implementation plan for each one.  The bottlenecks are organized
by the phase of execution in which they appear, with difficulty and expected impact
annotated for prioritization.

**Current state (after recent session)**:  
The pipeline runs a full DPO rewrite episode without reading from the Catlab ACSet R
at rewrite time, without downloading active/FK arrays to build domains, and without
the terminal-predicate false-download.  The kernel bounds crash (nc > MAX_CHUNKS) is
fixed.  Benchmarks show 1–4× GPU speedup over CPU for graph rewriting workloads,
growing with world size.

**Critical clarification on "Turbo"**:  
The name "Turbo" in `turbo_homomorphisms` and throughout the codebase refers to the
AAAI-26 paper "A GPU-based Constraint Programming Solver" (Talbot, 2026) and its open-
source implementation at https://github.com/ptal/turbo/tree/aaai2026.  Our TCN
bytecode format (`TCNBytecode`, 16 bytes, matching Turbo's struct) was correctly
adopted from that work.  However, **the parallel multi-block GPU solver that is the
paper's central contribution has not been implemented**.  Our `dive_solve_kernel!`
runs with `ndrange=1` — a single GPU thread executing sequential DFS — while Turbo's
algorithm distributes the search tree across hundreds of CUDA blocks executing in
parallel.  The benchmark speedups observed so far come entirely from the GPU-resident
data structure (GPUACSet), GPU rewriting kernels, and eliminating CPU round-trips;
the matching step itself, which dominates cost, still runs on one thread.

---

## Executive Summary of Remaining Bottlenecks

| # | Bottleneck | Phase | Impact | Difficulty |
|---|-----------|-------|--------|------------|
| 1 | Per-solve GPU buffer allocations | solve | High | Low |
| 2 | `_apply_attr_masks_gpu_device!` CPU download | solve | Medium | Low |
| 3 | `build_to_del_mask` CPU + upload | rewrite | Medium | Medium |
| 4 | `gpu_dangling_ok` per-morphism alloc + CPU reduction | rewrite | Medium | Low |
| 5 | `apply_pushout!` small CuArray allocations | rewrite | Medium | Low |
| 6 | `_update_preserved!` CuArray allocations | rewrite | Low | Low |
| 7 | Stream compaction — full CPU round-trip | compaction | High | Medium |
| 8 | `_choose_gpu_match` + Catlab `homomorphisms` call | player | Very High | Medium |
| 9 | Missing Turbo multi-block parallel solver (core of Turbo) | solver | **Critical** | High |
| 10 | Multiple `KernelAbstractions.synchronize` per solve | solve | Medium | Low |
| 11 | `MAX_CHUNKS = 4` hard limit (256 elements/type) | solver | High | Medium |
| 12 | `IncrementalUpdate.jl` entirely CPU-side | cache | High | High |
| 13 | Experience pre-state always `initial_world` | decoding | Medium | Medium |
| 14 | `download_acset` CPU-side compaction | decoding | Medium | Medium |
| 15 | `hom_fwd_offs` recomputed and re-uploaded every solve | solve | Low | Low |
| 16 | `sort()` in `_build_domains_gpu` | solve | Negligible | Trivial |
| 17 | `propagation_kernel!` unused in main path | solver | Medium | High |

---

## Bottleneck 1 — Per-Solve GPU Buffer Allocations

**File:** `ext/GPURewritingExt/solver/DiveSolveKernel.jl`,
         `ext/GPURewritingExt/control/Scheduler.jl`

**Diagnosis:**  
Every call to `_gpu_solve_inplace!` allocates and discards multiple GPU buffers:
- `gpu_dive_solve`: `b_gpu` (bytecodes), `sol_gpu` (solutions matrix), `cnt_gpu`
  (count), `work_gpu` (16-level DFS workspace of size `n_vars * MAX_CHUNKS × 16`)
- `_build_hom_fwd_gpu`: `hf_flat` (all FK bitmasks), `hf_offs` (offsets array)
- `_build_domains_gpu`: `d` (domain array of length `n_vars * nc`), and a
  temporary `type_mask` (length `nc`) per object type

Each `KernelAbstractions.allocate` touches the CUDA allocator, which serializes on a
global lock and forces a synchronize when the memory manager needs to trim.  For small
worlds (the common case in RL training) these allocations dominate measured latency.

**Fix:**  
Add persistent pre-allocated buffers to `GPUSchedulerState`.  Size them at
`compile_schedule` time based on the largest CSP in the schedule.

```julia
mutable struct GPUSchedulerState
    # ... existing fields ...
    # Persistent GPU scratch buffers, sized at construction:
    buf_domains     :: CuVector{UInt64}        # n_vars_max * nc
    buf_hf_flat     :: CuVector{UInt64}        # max total hom-fwd words
    buf_hf_offs     :: CuVector{Int32}         # n_homs + 1
    buf_bytecodes   :: CuVector{TCNBytecode}   # max n_bytecodes
    buf_solutions   :: CuMatrix{Int32}         # n_vars_max × max_solutions
    buf_sol_count   :: CuVector{Int32}         # [1]
    buf_workspace   :: CuMatrix{UInt64}        # n_vars_max * MAX_CHUNKS × 16
    buf_type_mask   :: CuVector{UInt64}        # nc (shared across types)
    buf_to_del      :: CuVector{Bool}          # sum(n_alloc) across all types
    buf_dang_result :: CuVector{Bool}          # max n_alloc across all types
end
```

`_gpu_solve_inplace!` calls `KernelAbstractions.fill!(state.buf_domains, 0)` and
`KernelAbstractions.fill!(state.buf_sol_count, 0)` before each solve instead of
allocating.  `gpu_dive_solve` gains a new signature accepting pre-allocated buffers.

`buf_hf_flat` and `buf_hf_offs` can be reused directly because `_build_hom_fwd_gpu`
writes them entirely before the solver reads them.

`buf_to_del` and `buf_dang_result` are shared across all per-hom dangling checks
(see Bottleneck 4).

**Verification:** `@allocated gpu_run_game_sched!(...)` should return 0 in steady
state (after the first call).  Benchmark should show 10–30% latency reduction for
small worlds where allocation dominates.

---

## Bottleneck 2 — `_apply_attr_masks_gpu_device!` CPU Download

**File:** `ext/GPURewritingExt/control/Scheduler.jl`

**Diagnosis:**  
For each `PROP_ATTR_EQ` bytecode, `_apply_attr_masks_gpu_device!` downloads
`Array(g.attrs[a])`, builds a bitmask on CPU, then re-uploads a slice.  Rules with
concrete attribute values in their LHS (e.g., "match a vertex with label = red") hit
this on every solve.

**Fix:**  
Add a GPU kernel that builds the attr mask on-device and ANDs it into the domain
array:

```julia
@kernel function _attr_mask_and_kernel!(
    domains  :: AbstractVector{UInt64},   # [n_vars * nc], modified in place
    attrs    :: AbstractVector{Int32},    # g.attrs[a]
    active   :: AbstractVector{Bool},     # g.active[owner]
    var_off  :: Int32,                    # 0-based word offset for this variable
    req      :: Int32,
    nc       :: Int32,
)
    i = @index(Global, Linear)           # element index (1-based)
    i <= length(active) || return
    active[i] || return
    attrs[i] != req && return
    ci, bi = elem_to_chunk(i)
    ci <= Int(nc) || return
    # Build per-element contribution; atomic OR into a staging mask,
    # then AND-reduce into domains[(var_off + ci)].
    # Because we need AND (not OR), a two-pass approach is required:
    # Pass 1: fill!(mask, 0); atomic-OR bits for matching elements into mask.
    # Pass 2: atomic-AND mask into domains.
    # Alternatively: pass 1 fills a shared mask[], pass 2 ANDs it in.
    Atomix.@atomic mask[ci] |= UInt64(1) << bi
end
```

Two-pass approach within `_apply_attr_masks_gpu_device!`:
1. `fill!(state.buf_attr_mask, UInt64(0))` (pre-allocated `nc`-length buffer)
2. Launch `_attr_mask_fill_kernel!` (OR bits for matching elements)
3. Launch `_attr_mask_and_kernel!` (AND the mask into `domains[var_off..var_off+nc]`)

No `Array(g.attrs[a])` call needed.  The `active` flag is already on GPU.

**Verification:** Benchmark a rule that matches a concrete attribute value.  The
`Array(g.attrs[a])` line should disappear from the profile.

---

## Bottleneck 3 — `build_to_del_mask` CPU Computation + Upload

**File:** `ext/GPURewritingExt/rewriting/DeletionKernel.jl`

**Diagnosis:**  
`build_to_del_mask` builds a `zeros(Bool, total)` on CPU by iterating `cube.l_types`
and `cube.k_to_l`, then uploads via `CuArray(to_del_host)`.  The work is O(n_l_elems)
CPU-side plus a PCIe transfer of `sum(g.n_alloc)` bytes.

**Fix:**  
Pre-upload the per-rule `AdhesiveCube` data needed for the deletion mask as persistent
GPU arrays in `CompiledGPUSched` at `compile_schedule` time.

New fields on `CompiledGPUSched` (or in a companion struct per rule):
```julia
# Flat L-element indices that are NOT in image(K→L) — i.e., deleted indices
del_l_flat :: CuVector{Int32}   # one per rule
# Offsets of each object type within the flat G-element layout
g_type_offsets :: CuVector{Int32}  # length n_obj_types, computed at solve time
```

New kernel:
```julia
@kernel function build_to_del_kernel!(
    to_del      :: AbstractVector{Bool},   # pre-zeroed, length = sum(n_alloc)
    match       :: AbstractVector{Int32},  # solution from solver
    del_l_flat  :: AbstractVector{Int32},  # pre-uploaded cube data: L flat indices to delete
    l_types     :: AbstractVector{Int32},  # pre-uploaded cube.l_types
    g_off       :: AbstractVector{Int32},  # per-type G-layout offsets
)
    i = @index(Global, Linear)            # index into del_l_flat
    i <= length(del_l_flat) || return
    flat_l  = Int(del_l_flat[i])
    g_elem  = Int(match[flat_l])
    g_elem == 0 && return
    type_idx = Int(l_types[flat_l])
    slot = Int(g_off[type_idx]) + g_elem
    to_del[slot] = true
end
```

`match` is the solution vector from `gpu_dive_solve`, which is already on CPU after
`Array(sol_gpu)`.  To avoid re-uploading it, `match` could be kept on GPU and passed
directly (requires the kernel to receive a GPU pointer).

The `g_off` array changes every rewrite (as `n_alloc` grows).  It can be rebuilt on
CPU (fast: n_obj_types additions) and uploaded in one small transfer before launch.

**Verification:** `build_to_del_mask` should no longer appear in CPU profiles.  The
deletion mask should be built and resident on GPU before the dangling check.

---

## Bottleneck 4 — `gpu_dangling_ok` Per-Morphism Alloc + CPU Reduction

**File:** `ext/GPURewritingExt/rewriting/DeletionKernel.jl`

**Diagnosis:**  
`gpu_dangling_ok` launches one kernel per schema morphism, allocates a fresh
`CUDA.zeros(Bool, n_src)` result buffer each time, synchronizes after each morphism,
and reduces with `Int(sum(results)) > 0` which downloads the result to CPU.  For a
schema with 5 morphisms this means 5 allocations, 5 syncs, and 5 CPU reductions.

**Fix:**  
Combine all morphisms into a single kernel launch using a pre-allocated persistent
flag buffer:

```julia
@kernel function dangling_check_all_homs_kernel!(
    violation   :: AbstractVector{Bool},   # [1], write true if any dangling edge
    active_src  :: AbstractVector{Bool},
    fk          :: AbstractVector{Int32},
    to_del_src  :: AbstractVector{Bool},
    to_del_tgt  :: AbstractVector{Bool},
    src_n       :: Int32,
    tgt_n       :: Int32,
)
    i = @index(Global, Linear)
    i <= Int(src_n) || return
    active_src[i] || return
    to_del_src[i] && return
    tgt = Int(fk[i])
    tgt == 0 && return
    tgt > Int(tgt_n) && return
    to_del_tgt[tgt] || return
    violation[1] = true   # racy write but idempotent (true-only)
end
```

The `violation` buffer is `state.buf_violation :: CuVector{Bool}` (length 1).
Reset it to `false` once before the loop.  Launch each morphism's kernel with
`violation` shared.  A single synchronize + single `Array(state.buf_violation)[1]`
at the end replaces N syncs + N CPU reductions.

**Note:** The racy write `violation[1] = true` is safe because: (a) only true is ever
written, and (b) a missed write merely means the dangling check runs one more iteration
— but since the violation never clears once set, the final `Array()` call will see it.
Use `Atomix.@atomic violation[1] |= true` if correctness over-conservatism is a
concern.

**Verification:** Confirm the dangling check detects violations for rules where the
match would create dangling edges.  The fix should collapse N syncs to 1 and eliminate
N per-morphism allocations.

---

## Bottleneck 5 — `apply_pushout!` Small CuArray Allocations

**File:** `ext/GPURewritingExt/rewriting/AdditionKernel.jl`

**Diagnosis:**  
For each new R element of each type, `apply_pushout!` allocates:
- `d_globals = CuArray(globals)` — slot indices of new elements (length = n_add per type)
- `CuArray(fk_vals)` — per-hom FK values (length = n_add, one CuArray per morphism per type)
- `CuArray(attr_vals)` — per-attr values (one CuArray per attribute per type)

For rules that add many elements these can be many small PCIe transfers and allocations.

**Fix:**  
Add a pre-allocated staging buffer to `GPUSchedulerState`:
```julia
buf_pushout_slots  :: CuVector{Int32}   # max n_add across all rules × all types
buf_pushout_vals   :: CuVector{Int32}   # max n_add × max(n_homs, n_attrs)
```

`apply_pushout!` receives the state's staging buffers and writes into `buf_pushout_slots[1:n_add]` (on CPU — this is a pinned-memory copyto!, not a full CuArray construction). The write is then triggered with a `copyto!` to the pre-allocated `CuVector`.

Alternatively: use CUDA pinned (page-locked) host memory for the staging arrays so
that PCIe transfers are DMA-capable and overlap with GPU computation.

The reallocation path (when `n_next > cap`) already does a full CuArray replacement
and is inherently expensive — flag it for future attention but accept it as amortized
rare.

**Verification:** Rules with addition of elements should show reduced allocation
count per call.

---

## Bottleneck 6 — `_update_preserved!` CuArray Allocations

**File:** `ext/GPURewritingExt/rewriting/AdditionKernel.jl`

**Diagnosis:**  
`_update_preserved!` allocates `CuArray(slots)` and `CuArray(vals)` for each
attr/hom that has changed K-elements.  These are typically very small (the number of
K-elements whose attrs/FKs change is usually zero or a handful).

**Fix:**  
Same staging-buffer approach as Bottleneck 5.  The existing `buf_pushout_vals` from
Bottleneck 5 can serve double duty, or a dedicated `buf_update_slots :: CuVector{Int32}`
and `buf_update_vals :: CuVector{Int32}` of modest capacity (e.g., 256 elements) can
be used.  When the number of preserved updates exceeds the buffer capacity, fall back
to a direct `CuArray(...)` allocation (this is a degenerate case).

**Verification:** Rules with preserved-element updates should show zero additional GPU
allocations in steady state.

---

## Bottleneck 7 — Stream Compaction Full CPU Round-Trip

**File:** `ext/GPURewritingExt/control/StreamCompaction.jl`

**Diagnosis:**  
`compact_gpu_acset!` is currently implemented as a full CPU round-trip:
1. `Array(g.active[o])` for all types → compute new_id mapping on CPU
2. `Array(g.homs[h])` for all morphisms → scatter new values on CPU
3. `Array(g.attrs[a])` for all attrs → scatter on CPU
4. Re-upload everything as new `CuArray`s

For a world with 200 vertices and 200 edges this means ~4× GPU→CPU transfers plus
~4× CPU→GPU re-uploads for each compaction event.  The compaction is triggered every
`compact_every` rewrites (default 100), so it appears infrequently but is expensive.

**Fix:**  
Full GPU compaction using a prefix-sum approach (already partially scaffolded in the
existing `mark_live_kernel!` and `scatter_kernel!`):

**Step 1 — Compute new_id (parallel prefix-sum of active flags):**
```julia
# For each type o:
live_flags = Int32.(g.active[o])            # 0 or 1 per element, still on GPU
new_ids = CUDA.accumulate(+, live_flags)    # exclusive prefix-sum → new IDs
# Or use CUB.DeviceScan.ExclusiveSum via CUDA.jl's low-level API
```

`CUDA.accumulate(+, ...)` is an inclusive prefix-sum; subtract the original flag to
get the exclusive (0-based new ID).  Elements where `active[i] = false` get the same
ID as the next live element — they are ignored because they won't appear as sources in
the scatter.

**Step 2 — Scatter active columns (GPU scatter kernels):**
```julia
@kernel function compact_scatter_bool_kernel!(dst, src, active, new_ids, n)
    i = @index(Global, Linear)
    i <= n || return
    active[i] || return
    dst[new_ids[i]] = src[i]
end

@kernel function compact_scatter_int32_kernel!(dst, src, active, new_ids, n)
    i = @index(Global, Linear)
    i <= n || return
    active[i] || return
    dst[new_ids[i]] = src[i]
end
```

**Step 3 — Remap FK columns (GPU remap kernel, already exists as `remap_fk_kernel!`):**
The existing `remap_fk_kernel!` in `StreamCompaction.jl` is correct; it just needs to
be used in the GPU-native path with the `new_ids` arrays staying on device.

**Step 4 — Update `n_alloc` and `n_live`:**
Retrieve only the total count per type: `n_new = Int(Array(new_ids)[end])`.  This is a
single 4-byte scalar transfer per type (vs downloading the entire active array).

**Allocation strategy:** Pre-allocate destination buffers for compacted arrays at
`GPUSchedulerState` construction time (same capacity as source arrays).  Swap source
and destination pointers in `GPUACSet` after compaction.

**Verification:** `compact_gpu_acset!` should produce zero `Array()` calls except for
the scalar `n_new` downloads.  Run a test episode with `compact_every=1` (compact after
every rewrite) and verify correctness against the CPU-compaction baseline.

---

## Bottleneck 8 — `_choose_gpu_match` + Catlab `homomorphisms` Call

**File:** `ext/GPURewritingExt/control/Scheduler.jl`

**Diagnosis:**  
This is the single largest bottleneck for PLAYER_RULE episodes.

For non-GPU players (`AbstractAgent`), `_choose_gpu_match` does:
1. `download_acset(g, enc, world_type)` — full GPU→CPU transfer
2. For each solution: `_sol_gpu_to_compact(sol, ...)` + `_assignment_to_hom(...)`,
   which calls `homomorphisms(L, G; initial=comps)` — a full Catlab backtracking
   search on the CPU!

The Catlab call is triggered once per solution candidate, so for a rule with 200 edge
matches (common in ring graphs) this is 200 Catlab homomorphism searches per turn.

**Fix A — Eliminate `_assignment_to_hom` (correct and easy):**  
The CSP solution vector IS already a valid homomorphism assignment — the solver
guarantees it satisfies all FK and attribute constraints.  The Catlab call was added to
handle `AttrVar` binding (where the match assigns concrete values to attribute
variables).  For attribute-free rules this binding is a no-op.

Replace `_assignment_to_hom` with a direct constructor:

```julia
function _sol_to_action(sol::Vector{Int32}, L, world_host, csp, schema, gpu_to_compact)
    compact_sol = _sol_gpu_to_compact(sol, csp, schema, gpu_to_compact)
    comps = Dict{Symbol, Vector{Int}}()
    S = acset_schema(L)
    for o in ob(S)
        base = get(csp.var_offset, o, 0)
        base == 0 && continue
        n = nparts(L, o)
        comps[o] = [Int(compact_sol[base + i - 1]) for i in 1:n]
    end
    # Build ACSetTransformation directly — no homomorphism search
    try
        ACSetTransformation(comps, L, world_host)
    catch
        nothing
    end
end
```

`ACSetTransformation(comps, L, world_host)` constructs the morphism directly from the
assignment dictionary without search.  AttrVar values are left unbound (wildcard);
downstream code that needs concrete attr values must use the `enc` to decode from
`g.attrs[a]`.

**Fix B — GPU Player interface (preferred long term):**  
Require all agents used with `gpu_run_game_sched!` to implement `AbstractGPUPlayer`,
which receives the raw GPU solution matrix and returns an index.  The `download_acset`
and Catlab call are both eliminated.  A thin adapter `CPUAgentAdapter` can wrap any
`AbstractAgent` with Fix A's direct-constructor approach.

**Fix C — Lazy world download (partial improvement):**  
Download `world_host` once per turn (not once per solution candidate) outside the
solution loop.  This is already done correctly; the main cost is the per-candidate
Catlab search.

**Verification:** Benchmark a PLAYER_RULE episode with a non-GPU agent.  CPU profile
should no longer show `homomorphisms` calls inside `_choose_gpu_match`.  Confirm that
the action chosen matches the agent's preference (by running against a deterministic
first-match agent and comparing solution indices).

---

## Bottleneck 9 — Missing Turbo Multi-Block Parallel Solver

**File:** `ext/GPURewritingExt/solver/DiveSolveKernel.jl`

**Diagnosis:**  
`dive_solve_kernel!` is launched with `ndrange=1`.  One GPU thread executes the entire
DFS — AC-1 propagation, branching, and solution recording — sequentially.  An RTX 2070
has 2304 CUDA cores idle while that one thread runs.  For a ring graph with 200 edges,
the identity-edge rule has 200 valid homomorphisms; the single thread finds them one at
a time while 99.9% of compute is unused.

This is not a minor omission.  The parallel multi-block search algorithm described in
the Turbo paper (Talbot 2026, AAAI-26) and implemented at
`github.com/ptal/turbo/tree/aaai2026` is the entire reason our solver is called
"Turbo" — and it has not been built.  What we have is a CPU DFS ported to a single GPU
thread.

### The Turbo Algorithm (from paper + source)

Turbo decomposes the search tree into 2^D independent **subproblems** using a binary
depth-D path encoding.  Each subproblem is a leaf of a complete binary tree of depth D;
the path from root to leaf tells you which branch to take at each of the D split
decisions.  Blocks claim subproblems on-demand from a shared atomic counter
`next_subproblem` and explore each subproblem's subtree independently.  The algorithm
has two phases per subproblem:

**Diving phase** (`dive_blk`, Algorithm 3 in the paper):  
A block descends from the root to its assigned subproblem by following the binary path.
At each of the D levels, it:
1. Runs parallel propagation (`propagate_blk`) across all threads in the block.
2. If a leaf node or failure is detected, uses the skip operation to jump to the next
   unvisited subproblem and updates `next_subproblem` atomically.
3. Otherwise, extracts the branch bit from the subproblem index and pins the split
   variable to that branch.

The skip operation is pure bit arithmetic.  If the block is at depth `rd` and the path
so far is `b₁…bᵢ`, then skipping all subproblems that share this prefix is done by
incrementing the integer `(target >> rd)` and left-shifting back by `rd`.  No
synchronization between blocks is needed during diving.

**Solving phase** (`solve_blk`, Algorithm 4 — sequential backtracking within a block):  
After diving, the block is at a depth-D node.  It runs a full backtracking search from
that point using parallel propagation.  Solutions found are written atomically.  When
the subtree is exhausted, the block calls `next()` to atomically increment
`next_subproblem` and claims the next unsolved subproblem.

**Within-block parallel propagation** (`propagate_blk`):  
All threads in a block cooperate to propagate the TCN bytecodes.  Threads are assigned
groups of 32 contiguous bytecodes (one warp per group).  Within a warp, each thread
propagates its bytecode; the warp repeats until fixpoint.  This is the "warp-centric"
scheduling from Talbot, Pinel, and Bouvry (2022).  93% of Turbo's GPU time is spent
here.

**Memory layout**:  
Turbo stores variable domains in **shared memory** when they fit (227KB per block on an
H100, 48KB on the RTX 2070).  When shared memory is insufficient, it falls back to
global memory.  The variables' domains are the hot data — they are read and written on
every bytecode propagation.

### Our Domain Representation vs Turbo's

Turbo uses integer interval domains `[lb, ub]`.  We use bitset domains
`UInt64[nc]` per variable.  Our representation is strictly more expressive (we can
represent arbitrary finite sets, not just intervals), and our propagation logic maps
onto the same bytecode format.  The `PROP_FUNC` constraint (our FK morphism
propagator) has no direct analog in Turbo, which is a general-purpose solver; it is
our addition.

The bitset representation makes our propagation slightly heavier than Turbo's interval
arithmetic (one UInt64 AND vs one integer comparison), but far simpler than
Turbo's view-based propagators.

### What Needs To Be Built

**Phase 1 — Block-parallel propagation kernel (prerequisite):**  
Replace the inline single-thread AC-1 loop in `dive_solve_kernel!` with a
block-parallel version where each thread handles one bytecode per propagation round.
All threads in a block cooperate via `__syncthreads()`.  The shared variable domains
must be accessible to all threads in the block.

```julia
# Conceptual block-parallel propagation (one call per fixpoint iteration)
@kernel function propagate_block_kernel!(
    domains   :: AbstractMatrix{UInt64},  # [n_vars * nc × n_blocks], one column per block
    hom_fwd   :: AbstractVector{UInt64},  # read-only FK tables
    hom_offs  :: AbstractVector{Int32},
    bytecodes :: AbstractVector{TCNBytecode},
    changed   :: AbstractVector{Bool},    # [n_blocks], true if any domain shrank
)
    tid = @index(Local, Linear)           # thread within block
    bid = @index(Group, Linear)           # block index
    bc  = bytecodes[tid]                  # each thread owns one bytecode
    # ... propagate bc, write result back to domains[:, bid], OR into changed[bid]
    @synchronize()
end
```

Domain storage: `n_vars * nc` UInt64 words per block.  For n_vars=5, nc=4, this is
160 bytes per block — trivially fits in shared memory.  For n_vars=10, nc=4, still 320
bytes.  Shared memory is viable for all patterns we expect to encounter.

**Phase 2 — Multi-block EPS dive-and-solve kernel:**  
Launch `B` blocks, each running the dive-and-solve loop.  One shared atomic counter
`nextsub` (stored in global memory) assigns subproblems to blocks on-demand.

```julia
@kernel function turbo_solve_kernel!(
    domains_root   :: AbstractVector{UInt64},  # initial domains (read-only)
    bytecodes      :: AbstractVector{TCNBytecode},
    n_bc           :: Int,
    n_vars         :: Int,
    nc             :: Int,
    D              :: Int,                     # subproblem depth (log₂ of num subproblems)
    nextsub        :: AbstractVector{Int32},   # [1], atomic subproblem counter
    solutions      :: AbstractMatrix{Int32},   # [n_vars × max_solutions]
    sol_count      :: AbstractVector{Int32},   # [1], atomic solution counter
    max_solutions  :: Int,
    hom_fwd_flat   :: AbstractVector{UInt64},
    hom_fwd_offs   :: AbstractVector{Int32},
)
    tid   = @index(Local, Linear)   # thread within block (1..blocksize)
    bid   = @index(Group, Linear)   # block index (1..B)

    # --- Shared memory layout (per block) ---
    # domains_blk[1..n_vars*nc]: current domain state for this block
    # Each thread handles ceil(n_vars*nc / blocksize) domain words during copy

    # --- Main loop ---
    # while (mysub = atomic_inc(nextsub)) < 2^D:
    #   1. Restore domains_blk ← domains_root
    #   2. Dive phase: for each of D levels, propagate_block + branch on bit
    #   3. if reached subproblem without failure:
    #      Solve phase: backtrack within block (thread 0 drives stack,
    #                   all threads participate in propagation)
    #   4. else: skip (nextsub atomic_max with skip target)
end
```

Choosing D: Turbo sets `D = ⌈log₂(blockDim.x × 300)⌉`.  For 256 threads/block and
300× factor, D ≈ 16.  For our smaller patterns (1–10 variables, few solutions) a
smaller D (8–12) is appropriate, giving 256–4096 subproblems.

**Phase 3 — Backtracking within a block (thread-0-driven, all-threads-propagate):**  
During the solving phase, thread 0 manages the DFS stack (which branch to expand
next).  All threads participate in each `propagate_block` call.  Thread 0 checks for
solution or failure after propagation and updates the stack.

This is simpler than the diving phase because no cross-block coordination is needed.
The stack depth is at most D levels (since we already dove D levels before the solve
phase begins).

### Implementation Notes

- **Shared memory in KernelAbstractions**: Use `@localmem UInt64 (n_vars_max * nc_max,)`
  with a compile-time size, or pass a preallocated shared-memory pointer via
  `@index(Group, Linear)` offset arithmetic.  KernelAbstractions supports shared memory
  via `@localmem` for statically-known sizes.

- **Atomic nextsub**: Use `Atomix.@atomic nextsub[1] += 1` for the claim operation and
  `Atomix.@atomic nextsub[1] = max(nextsub[1], skip_target)` for the skip operation.

- **Number of blocks B**: Turbo selects B using `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  to maximize occupancy given the shared memory footprint.  For our RTX 2070 (48KB
  shared/block), with 320 bytes of domain data + ~100 bytes for fixpoint state, B ≈ 48
  blocks per SM × 18 SMs = 864 blocks.  Each block gets 256 threads → 221,184 threads
  total vs current 1.

- **Solution deduplication**: For "find all solutions" queries, different blocks
  exploring different subtrees will never find the same solution (each block explores a
  disjoint subtree).  Solutions discovered in the solving phase (within-block DFS) may
  be found in different orders by different blocks but are deduplicated by the
  uniqueness of the subproblem assignment.  No deduplication needed.

- **PROP_FUNC within parallel propagation**: Our FK-based propagation touches two
  variables (not just one) and reads from `hom_fwd_flat`.  For block-parallel
  propagation, the hom_fwd table should be loaded into shared memory when it fits
  (typically yes — for a ring graph with 200 edges and nc=4, the hom_fwd for the :src
  or :tgt morphism is 200 × 4 × 8 = 6400 bytes).  If not, global memory accesses are
  still coalesced across warps since consecutive elements share the same FK table.

- **KernelAbstractions limitations**: `@localmem` requires a compile-time size.  Use a
  `Val{N}` parameter for `n_vars * nc`, compiled per-rule at `compile_schedule` time.
  This is the same approach needed for Bottleneck 11 (parameterized MAX_CHUNKS).

### Expected Speedup

The paper reports ~103,000 nodes/second on an H100 vs 14,623 for view-based propagation
(7× faster propagation).  On the RTX 2070 (about 4× less compute than H100), expect
~25,000 nodes/second.  For a ring-200 edge-match query with 200 solutions, the single-
thread solver currently explores the full search tree sequentially; with 864 blocks,
the 200 solutions can be found across many simultaneous subtrees.  Expected end-to-end
speedup for the matching step: **10–100× over the current single-thread kernel**.

**Verification:** `turbo_homomorphisms` equivalence tests (230/230) must continue to
pass.  Add a benchmark comparing solution-finding time for the identity-edge rule on
Ring(200) between old single-thread kernel and new multi-block kernel.  The new kernel
should find all 200 solutions faster than the old kernel finds the first one.

---

## Bottleneck 10 — Multiple `KernelAbstractions.synchronize` Per Solve

**File:** `ext/GPURewritingExt/control/Scheduler.jl`,
         `ext/GPURewritingExt/rewriting/DeletionKernel.jl`

**Diagnosis:**  
In a single solve-and-rewrite step, the following synchronize calls appear:
1. After `_build_type_mask_kernel!` (inside `_build_domains_gpu`, once per type)
2. After `_build_hom_fwd_kernel!` (end of `_build_hom_fwd_gpu` — currently absent)
3. In `_gpu_solve_inplace!` before `gpu_dive_solve`
4. After `KernelAbstractions.synchronize(backend)` in `gpu_dive_solve`
5. After each morphism in `gpu_dangling_ok` (N syncs for N morphisms)
6. After `KernelAbstractions.synchronize(backend)` in `_gpu_apply_inplace!`

Each synchronize drains the CUDA command queue, preventing pipelining of independent
GPU operations.  Items 1 and 5 are the most impactful.

**Fix:**  
1. Remove the `KernelAbstractions.synchronize(backend)` inside `_build_domains_gpu`
   (after `_build_type_mask_kernel!`).  The subsequent `copyto!` to the domain buffer
   is a GPU-to-GPU copy that is already ordered by CUDA's implicit stream dependencies.
2. Collapse N per-morphism syncs in `gpu_dangling_ok` to 1 (see Bottleneck 4).
3. Use CUDA streams to pipeline: launch `_build_hom_fwd_kernel!` and
   `_build_type_mask_kernel!` on separate streams so they overlap.

In general: the only mandatory syncs are (a) before any `Array()` call that reads
results back to CPU, and (b) at episode end.  All intermediate GPU→GPU data flows can
use implicit stream ordering.

**Verification:** Use `CUDA.@elapsed` (not Julia's `@elapsed`) to measure GPU-side
latency without CPU synchronization overhead.  The number of sync points in the hot
path should drop to ≤ 2 (one before reading the solution count, one at episode end).

---

## Bottleneck 11 — `MAX_CHUNKS = 4` Hard Limit

**File:** `ext/GPURewritingExt/solver/BitwiseDomain.jl`,
         `ext/GPURewritingExt/solver/DiveSolveKernel.jl`

**Diagnosis:**  
`MAX_CHUNKS = 4` limits the solver to worlds with at most 256 live elements per object
type.  This is enforced by the `MVector{MAX_CHUNKS, UInt64}` locals in
`dive_solve_kernel!`.  The `n_chunks` in `gpu_run_game_sched!` is now clamped to
`MAX_CHUNKS`, so worlds with more than 256 elements silently truncate their domains —
elements 257+ are never candidates.

**Fix — Parameterized MAX_CHUNKS:**  
Make `MAX_CHUNKS` a compile-time parameter threaded through the type system:

```julia
# BitwiseDomain.jl
const MAX_CHUNKS_OPTIONS = (1, 2, 4, 8, 16)  # 64, 128, 256, 512, 1024 elements

# DiveSolveKernel.jl — parameterized version
@kernel function dive_solve_kernel!(... NC_MAX::Val{NM}) where NM
    new_d     = MVector{NM, UInt64}(undef)
    reachable = MVector{NM, UInt64}(undef)
    ...
end
```

`gpu_run_game_sched!` selects the appropriate `Val{NC_MAX}` at compile time based on
`n_chunks`:

```julia
function _select_kernel(nc::Int)
    nc <= 1  && return Val(1)
    nc <= 2  && return Val(2)
    nc <= 4  && return Val(4)
    nc <= 8  && return Val(8)
    nc <= 16 && return Val(16)
    error("MAX_CHUNKS exceeded: $(nc) > 16 (max 1024 elements per type)")
end
```

Each `Val{N}` dispatches to a separately compiled kernel variant.  Compilation time
increases (5 variants instead of 1) but this is a one-time cost at `compile_schedule`
time.  Register pressure increases with larger NM — profile register usage for each.

**Alternative — Heap-allocated MArray fallback:**  
For nc > 4, fall back to a heap-allocated path where `new_d` and `reachable` are
columns of the `workspace` matrix (already heap-allocated).  This avoids recompilation
but adds register pressure and may require a different kernel structure.

**Verification:** Run `turbo_homomorphisms` on a graph with 300 vertices (requires
nc = 5).  Should return the correct homomorphism count with the nc=8 kernel variant.

---

## Bottleneck 12 — `IncrementalUpdate.jl` Entirely CPU-Side

**File:** `ext/GPURewritingExt/rewriting/IncrementalUpdate.jl`

**Diagnosis:**  
`incremental_match_update!` is not currently wired into the main scheduler (the main
scheduler re-solves from scratch every turn via `_gpu_solve_inplace!`).  But it exists
as infrastructure for a cache-based pipeline where match sets are maintained across
rewrites rather than re-computed.  In its current state it is entirely CPU-bound:
- Downloads `g.active[o]` for all types
- Runs `cpu_dive_solve` with a 1-chunk domain (max 64 elements)
- Has no multi-chunk support

**Fix — Full GPU incremental update:**  
This is the highest-complexity fix and the one that would enable the greatest long-term
throughput improvement by amortizing the O(n²) pattern-matching cost across episodes.

**Step 1 — GPU-resident match table:**  
Move `MatchTable.assignments` to a `CuMatrix{Int32}` so that solution candidates are
never downloaded to CPU for the update step.

**Step 2 — GPU forward-survive filter:**  
```julia
@kernel function filter_surviving_matches_kernel!(
    keep        :: AbstractVector{Bool},   # output: true = keep match m
    assignments :: AbstractMatrix{Int32},  # [n_vars × n_matches]
    active_flat :: AbstractVector{Bool},   # per-type active arrays, concatenated
    type_offsets :: AbstractVector{Int32}, # per-var type offset in active_flat
    n_vars      :: Int32,
    n_matches   :: Int32,
)
    m = @index(Global, Linear)
    m <= Int(n_matches) || return
    for v in 1:Int(n_vars)
        g_elem = Int(assignments[v, m])
        g_elem == 0 && continue
        off = Int(type_offsets[v])
        slot = off + g_elem
        slot > length(active_flat) && (keep[m] = false; return)
        active_flat[slot] || (keep[m] = false; return)
    end
end
```

**Step 3 — GPU compaction of surviving matches:**  
Prefix-sum on `keep` → `new_idx`; scatter `assignments[:, m]` to `assignments[:, new_idx[m]]`.

**Step 4 — GPU new-match discovery:**  
For each newly added element (pinned to one variable at a time), call `gpu_dive_solve`
with the pinned domain.  Append results to the match table.  This requires `gpu_dive_solve`
to write directly into a sub-range of the persistent `buf_solutions` matrix.

**Step 5 — Wire into scheduler:**  
`_gpu_solve_inplace!` becomes: read from the match table (constant time) → choose match
→ apply rewrite → call `incremental_update!` to update the table.  The per-turn solve
cost drops from O(|G|²) to O(|Δ|²) where |Δ| is the number of added elements.

This is a significant refactor requiring changes to `Scheduler.jl`, `IncrementalUpdate.jl`,
`GPUSchedulerState`, and the test suite.  Estimate: 3–5 days of implementation plus
extensive verification.

**Verification:** Compare match counts against `turbo_homomorphisms` after each rewrite
step for a known rule and world sequence.  Throughput benchmark should show super-linear
improvement for long episodes.

---

## Bottleneck 13 — Experience Pre-State Always `initial_world`

**File:** `ext/GPURewritingExt/GPURewritingExt.jl`

**Diagnosis:**  
In `gpu_run_game_sched!`, Experience records are constructed as:
```julia
state_pre  = GameState(initial_world, turn_n)
state_post = GameState(final_world, turn_n + 1)
```

The pre-state is hardcoded to `initial_world` regardless of which turn the event
occurred.  For multi-turn episodes the pre-state of turn 5 should be the world state
after turn 4's rewrite, not the episode-start state.

**Fix A — GPU state snapshots (expensive but correct):**  
Before each rewrite, `deepcopy(g)` the GPUACSet.  Store snapshots in the event log.
At episode end, download each snapshot for the corresponding Experience.

`deepcopy(g)` is already implemented (`Base.deepcopy` in `GPUACSet.jl`) and costs
approximately the same as `download_acset` in terms of GPU memory bandwidth.  For long
episodes (T_max = 1000) with many PLAYER_RULE firings this is expensive.

**Fix B — Trajectory-based reconstruction (recommended):**  
The `GPUTrajectoryLog` records additions and deletions as `DeltaEvent` records.  From
the full trajectory, each world state can be reconstructed by replaying deltas forward
from `initial_world`.  This requires:
1. Recording GPU-slot-level add/delete events (already done in `log_additions!` and
   `log_deletions!`)
2. A `replay_to_turn(log, initial_world, target_turn)` function that replays the delta
   sequence

This is cheaper than snapshotting because only the compact `DeltaEvent` stream is
stored.

**Fix C — Acknowledge the limitation and document it:**  
For RL training workflows where only the final outcome matters (sparse rewards at
episode end), the wrong pre-state is acceptable — the agent's `select_action` call
receives the correct world at decision time via `download_acset`.  Add a docstring
warning and a `track_pre_states::Bool = false` keyword argument that enables Fix A
only when needed.

---

## Bottleneck 14 — `download_acset` CPU-Side Compaction

**File:** `ext/GPURewritingExt/rewriting/GPUACSet.jl`

**Diagnosis:**  
`download_acset` runs a CPU-side compaction loop to eliminate tombstones:
```julia
for (old, alive) in enumerate(flags)
    alive || continue
    cursor += 1
    mapping[old] = cursor
end
```

For a world with 1000 elements (50% tombstoned) this loop iterates 1000 times per
object type.  The download itself (`Array(g.active[o])`, `Array(g.homs[h])`,
`Array(g.attrs[a])`) dominates when the world is large; the compaction loop is
secondary.

**Fix:**  
Run `compact_gpu_acset!` (Bottleneck 7, GPU-native version) before downloading.  After
compaction, all tombstones are removed and `download_acset` can do a direct array copy
without the compaction loop:

```julia
function download_acset_compact(g::GPUACSet, enc, world_type)
    compact_gpu_acset!(g, g.schema, backend)   # GPU-native after Bottleneck 7
    # Now: g has no tombstones, g.n_alloc[o] == g.n_live[o][]
    result = world_type()
    for o in g.schema.obj_types
        add_parts!(result, o, g.n_alloc[o])
    end
    for h in g.schema.homs
        host_fk = Array(g.homs[h])
        for i in 1:g.n_alloc[g.schema.hom_dom[h]]
            host_fk[i] > 0 && set_subpart!(result, i, h, host_fk[i])
        end
    end
    for a in g.schema.attrs
        host_av = Array(g.attrs[a])
        for i in 1:g.n_alloc[g.schema.attr_dom[a]]
            v = decode_value(enc, a, host_av[i])
            v !== nothing && set_subpart!(result, i, a, v)
        end
    end
    result
end
```

After compaction the CPU loop is O(n_live) instead of O(n_alloc), and there is no
ID-remapping step.

**Verification:** `download_acset(g, enc, world_type)` after compaction should produce
an ACSet identical to a round-trip `upload_acset(download_acset_no_compact(g,...),...)`.
Verify for a world after 50 deletion rewrites.

---

## Bottleneck 15 — `hom_fwd_offs` Recomputed and Re-Uploaded Every Solve

**File:** `ext/GPURewritingExt/control/Scheduler.jl`

**Diagnosis:**  
`_build_hom_fwd_gpu` recomputes `hom_fwd_offs` (a CPU-side integer array of length
`n_homs + 1`) on every call and uploads it to GPU via `KernelAbstractions.allocate` +
`copyto!`.  The offset values depend only on `g.n_alloc[h]` per morphism, which
changes only when new elements are added (reallocating a type's GPU arrays).

**Fix:**  
Cache `hom_fwd_offs` in `GPUSchedulerState`.  Rebuild it only when an object type's
`n_alloc` changes (i.e., after a rewrite that triggered array reallocation in
`apply_pushout!`).  Add a dirty flag per type:

```julia
mutable struct GPUSchedulerState
    # ...
    hf_offs_dirty :: Bool
    cached_hf_offs :: CuVector{Int32}
end
```

On first call or when `hf_offs_dirty = true`, recompute and re-upload.  Clear the flag
after upload.  `apply_pushout!` sets `hf_offs_dirty = true` whenever it reallocates a
type's arrays.

This saves one small `KernelAbstractions.allocate` + `copyto!` per solve in the common
case (no reallocation since last solve).

---

## Bottleneck 16 — `sort()` in `_build_domains_gpu`

**File:** `ext/GPURewritingExt/control/Scheduler.jl`

**Diagnosis:**  
```julia
type_bases = sort([(base, o) for (o, base) in pairs(csp.var_offset)], by=first)
```

This allocates a temporary vector and sorts it on every call.  The sort is O(n_types
log n_types) where n_types is typically 1–5.  It is negligible in absolute terms but
shows up in `@allocated` counts.

**Fix:**  
Pre-sort `var_offset` entries at `CSPProblem` construction time and store as a
`Vector{Pair{Int, Symbol}}` field:

```julia
struct CSPProblem
    # ...
    sorted_type_bases :: Vector{Pair{Int, Symbol}}  # [(base, o)] sorted by base
end
```

Populate in `lower_rule_to_csp` after building `var_offset`.  `_build_domains_gpu`
iterates `csp.sorted_type_bases` directly.

---

## Bottleneck 17 — `propagation_kernel!` Incomplete and Unused

**File:** `ext/GPURewritingExt/solver/PropagationKernel.jl`

**Diagnosis:**  
`propagation_kernel!` was written as a one-thread-per-instance batch propagation pass.
It has two critical limitations:

1. **Missing `PROP_FUNC`**: It does not implement the FK morphism constraint that is
   central to homomorphism finding.  It handles only `PROP_EQ`, `PROP_NEQ`,
   `PROP_ATTR_EQ`, and `DOMAIN_SIZE`.  This makes it useless for any schema with
   morphisms (i.e., every Graph-schema rule).

2. **Wrong parallelism model**: It assigns one thread per CSP *instance* (one candidate
   world state being evaluated).  This is not the within-block propagation that Turbo
   uses, where all threads in one block cooperate on *one* CSP instance by each
   handling a different bytecode.

The kernel is never called in the hot path.  The hot path inlines propagation inside
`dive_solve_kernel!`.

**Fix — Rewrite as a block-parallel `PROP_FUNC`-capable propagator (prerequisite for Bottleneck 9):**  
The Turbo paper's `propagate_blk` function assigns one warp per group of 32 bytecodes.
Each warp propagates its 32 bytecodes in parallel, then the block iterates until
fixpoint.  For our bitset domains, the propagation of one bytecode is:

- `PROP_FUNC`: iterate elements in var1's domain bitset, check if any element of
  `hom_fwd[w]` intersects var2's domain; if not, clear that element from var1.  This
  is inherently sequential *within one bytecode* but can be parallelized across
  bytecodes by different threads.
- `PROP_NEQ`, `PROP_EQ`: simple bitwise AND / AND-NOT; one thread.
- `PROP_ATTR_EQ`: bitwise AND with a precomputed mask; one thread.

The rewritten kernel should be `propagate_block!(domains_shared, bytecodes, hf_flat, hf_offs, nc, n_bc)`, operating on shared-memory domain arrays and returning `true` if no domain is empty after fixpoint.

This kernel becomes the inner loop of the Turbo multi-block solver (Bottleneck 9) and
replaces the inlined propagation in `dive_solve_kernel!`.

**Verification:** Unit-test `propagate_block!` against `cpu_propagate!` on a suite of
small graphs and rules.  The fixpoint result must be identical.  Then verify
`turbo_homomorphisms` still passes 230/230 when the new propagator is wired in.

---

## Implementation Ordering

Given the constraints of difficulty and inter-dependency, the recommended order is:

**Sprint 1 — Zero-allocation hot path (1–2 days):**
- Bottleneck 15 (hom_fwd_offs caching) — trivial
- Bottleneck 16 (pre-sort in CSPProblem) — trivial
- Bottleneck 1 (persistent GPU buffers) — high-impact, low-risk
- Bottleneck 6 (update_preserved! staging) — low-risk, small gain

**Sprint 2 — Rewrite path cleanup (1–2 days):**
- Bottleneck 2 (attr mask GPU kernel) — medium complexity
- Bottleneck 4 (dangling check collapsed) — low-complexity
- Bottleneck 10 (reduce sync points) — low-complexity

**Sprint 3 — Full GPU rewrite path (2–3 days):**
- Bottleneck 5 (pushout staging buffers)
- Bottleneck 3 (to_del mask GPU kernel)
- Bottleneck 7 (GPU-native stream compaction) — medium complexity, high impact
- Bottleneck 14 (compact before download)

**Sprint 4 — Player interface and correctness (1–2 days):**
- Bottleneck 8A (eliminate Catlab homomorphisms call) — immediate and high impact
- Bottleneck 13C (document pre-state limitation) — one line
- Bottleneck 13B (trajectory-based pre-state reconstruction)

**Sprint 5 — Turbo multi-block solver (5–8 days):**
This is the implementation that should have been there from the start.  All other
bottlenecks combined are secondary to this.

- Bottleneck 11 (parameterized MAX_CHUNKS / `Val{N}` kernel dispatch) — prerequisite
  for shared-memory domain sizing in the block-parallel kernel
- Bottleneck 17 (rewrite PropagationKernel to block-parallel, PROP_FUNC-capable) —
  prerequisite for Bottleneck 9
- Bottleneck 9 (Turbo multi-block EPS dive-and-solve) — the primary missing piece;
  implement in three sub-steps: (9a) block-parallel propagation, (9b) multi-block
  diving with binary-path subproblem indexing, (9c) within-block backtracking solve

Reference: `github.com/ptal/turbo/tree/aaai2026`, particularly
`include/gpu_dive_and_solve.hpp` (`dive`, `solve_problem`, `gpu_solve_kernel`).

**Sprint 6 — Incremental match cache (3–5 days):**
- Bottleneck 12 (GPU-resident incremental update) — most complex, largest long-term gain

---

## Expected Steady-State Performance After All Fixes

| Phase | Current latency (n=200, del_e) | After all fixes |
|-------|-------------------------------|-----------------|
| Domain + hom_fwd build | ~0.5 ms | ~0.05 ms (no alloc, no CPU) |
| CSP solve | ~40 ms | ~2–5 ms (parallel DFS) |
| Dangling check | ~0.2 ms | ~0.05 ms (1 kernel, 1 sync) |
| Deletion | ~0.1 ms | ~0.05 ms |
| Pushout | ~0.3 ms | ~0.1 ms (no small CuArray) |
| Stream compaction (amortized) | ~5 ms / 100 steps | ~0.5 ms / 100 steps |
| **Total per step** | **~74 ms** | **~3–8 ms** |

The parallelized CSP solver (Bottleneck 9) and the incremental match cache (Bottleneck 12)
together represent the path to O(|Δ|) per-step cost for long episodes with slowly
changing worlds — the ultimate target for RL training workloads.
