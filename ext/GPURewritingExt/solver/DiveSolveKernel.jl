"""
Dive-and-solve search kernel — chunked bitset version.

Domains are represented as flat `Vector{UInt64}` of length `n_vars * n_chunks`
where `domains[(v-1)*n_chunks + c]` is chunk `c` of variable `v`'s domain.
This lifts the 64-element-per-type cap: `n_chunks` chunks cover up to
`n_chunks * 64` elements.

The GPU kernel uses `MAX_CHUNKS = 4` (up to 256 elements per type) with
statically-sized MVector temporaries so that the KernelAbstractions compiler
can map them to registers.
"""

# ── Host-side DFS ─────────────────────────────────────────────────────────────

function cpu_dive_solve(csp::CSPProblem,
                        initial_domains::Vector{UInt64})::Vector{Vector{Int32}}
    solutions = Vector{Int32}[]
    assignment = zeros(Int32, Int(csp.n_vars))
    _dfs!(solutions, assignment, copy(initial_domains), csp.bytecodes,
          Int(csp.n_vars), csp.hom_forward, csp.n_chunks)
    return solutions
end

function _dfs!(solutions, assignment, domains::Vector{UInt64}, bytecodes,
               n_vars::Int,
               hom_forward::Vector{Vector{UInt64}} = Vector{UInt64}[],
               nc::Int = 1)

    cpu_propagate!(domains, bytecodes, hom_forward, nc) || return

    unbound = 0
    for v in 1:n_vars
        off  = (v - 1) * nc
        ones = 0
        for c in 1:nc; ones += count_ones(domains[off + c]); end
        ones == 0 && return
        ones > 1  && (unbound = v; break)
    end

    if unbound == 0
        # All variables fixed — record solution
        sol = Vector{Int32}(undef, n_vars)
        for v in 1:n_vars
            off  = (v - 1) * nc
            elem = Int32(0)
            for c in 1:nc
                ch = domains[off + c]
                ch != UInt64(0) || continue
                bi   = trailing_zeros(ch)
                elem = Int32((c - 1) * 64 + bi + 1)
                break
            end
            sol[v] = elem
        end
        push!(solutions, sol)
        return
    end

    # Branch on each element of the unbound variable
    off_ub = (unbound - 1) * nc
    for c in 1:nc
        chunk = domains[off_ub + c]
        while chunk != 0
            lsb = chunk & (-chunk); chunk &= ~lsb
            bi  = trailing_zeros(lsb)
            new_d = copy(domains)
            for ci in 1:nc; new_d[off_ub + ci] = UInt64(0); end
            new_d[off_ub + c] = lsb
            _dfs!(solutions, assignment, new_d, bytecodes, n_vars, hom_forward, nc)
        end
    end
end

# ── GPU kernel ────────────────────────────────────────────────────────────────

@kernel function dive_solve_kernel!(
    domains_in    :: AbstractVector{UInt64},   # [n_vars * n_chunks]
    bytecodes     :: AbstractVector{TCNBytecode},
    n_bc          :: Int,
    n_vars        :: Int,
    n_chunks      :: Int,
    solutions     :: AbstractMatrix{Int32},    # [n_vars × max_solutions]
    sol_count     :: AbstractVector{Int32},
    max_solutions :: Int,
    workspace     :: AbstractMatrix{UInt64},   # [n_vars * NM × 16]
    hom_fwd_flat  :: AbstractVector{UInt64},
    hom_fwd_offs  :: AbstractVector{Int32},
    ::Val{NM}
) where NM
    nc     = n_chunks
    nc_max = NM

    stack_vars = MVector{16, Int32}(undef)
    stack_next = MVector{16, Int32}(undef)
    new_d      = MVector{NM, UInt64}(undef)
    reachable  = MVector{NM, UInt64}(undef)

    # Level-1 init: copy domains_in into workspace
    for v in 1:n_vars
        off_w = (v - 1) * nc_max
        off_d = (v - 1) * nc
        for c in 1:nc
            workspace[off_w + c, 1] = domains_in[off_d + c]
        end
    end

    level  = 1
    state  = 1
    safety = 0

    while level > 0 && safety < 100_000_000
        safety += 1

        if state == 1
            # ── A. Propagate (inline AC-1) ────────────────────────────────────
            ok = true
            for _ in 1:8
                changed = false
                for i in 1:n_bc
                    bc = bytecodes[i]
                    v1 = Int(bc.var1); v1 == 0 && continue
                    off1 = (v1 - 1) * nc_max

                    if bc.op == PROP_FUNC && bc.var2 != 0
                        v2    = Int(bc.var2)
                        off2  = (v2 - 1) * nc_max
                        h_idx = Int(bc.param1)
                        n_homs = length(hom_fwd_offs) - 1
                        if 1 <= h_idx <= n_homs
                            off_h    = Int(hom_fwd_offs[h_idx])
                            n_elems_h = (Int(hom_fwd_offs[h_idx+1]) - off_h) ÷ nc

                            # Forward: build new domain for v1
                            for c in 1:nc; new_d[c] = UInt64(0); end
                            for c in 1:nc
                                chunk = workspace[off1 + c, level]
                                while chunk != 0
                                    lsb = chunk & (-chunk); chunk &= ~lsb
                                    bi  = trailing_zeros(lsb)
                                    w   = (c - 1) * 64 + bi + 1
                                    w > n_elems_h && continue
                                    off_w = off_h + (w - 1) * nc
                                    for ci in 1:nc
                                        (hom_fwd_flat[off_w + ci] &
                                         workspace[off2 + ci, level]) != 0 &&
                                            (new_d[c] |= lsb; break)
                                    end
                                end
                            end
                            for c in 1:nc
                                old_c = workspace[off1 + c, level]
                                workspace[off1 + c, level] = new_d[c]
                                old_c != new_d[c] && (changed = true)
                            end

                            # Backward: reachable set for v2
                            for c in 1:nc; reachable[c] = UInt64(0); end
                            for c in 1:nc
                                chunk = new_d[c]
                                while chunk != 0
                                    lsb = chunk & (-chunk); chunk &= ~lsb
                                    bi  = trailing_zeros(lsb)
                                    w   = (c - 1) * 64 + bi + 1
                                    w > n_elems_h && continue
                                    off_w = off_h + (w - 1) * nc
                                    for ci in 1:nc
                                        reachable[ci] |= hom_fwd_flat[off_w + ci]
                                    end
                                end
                            end
                            for c in 1:nc
                                new_c = workspace[off2 + c, level] & reachable[c]
                                new_c != workspace[off2 + c, level] && (changed = true)
                                workspace[off2 + c, level] = new_c
                            end
                        end

                    elseif bc.op == PROP_NEQ && bc.var2 != 0
                        v2   = Int(bc.var2)
                        off2 = (v2 - 1) * nc_max
                        ones2 = 0
                        for c in 1:nc; ones2 += count_ones(workspace[off2 + c, level]); end
                        if ones2 == 1
                            for c in 1:nc
                                new_c = workspace[off1 + c, level] &
                                        ~workspace[off2 + c, level]
                                new_c != workspace[off1 + c, level] && (changed = true)
                                workspace[off1 + c, level] = new_c
                            end
                        end

                    elseif bc.op == PROP_EQ && bc.var2 != 0
                        v2   = Int(bc.var2)
                        off2 = (v2 - 1) * nc_max
                        for c in 1:nc
                            old1 = workspace[off1 + c, level]
                            new1 = old1 & workspace[off2 + c, level]
                            workspace[off1 + c, level] = new1
                            old1 != new1 && (changed = true)
                        end
                    end
                end
                changed || break
            end

            # ── B. Consistency check ─────────────────────────────────────────
            for v in 1:n_vars
                off_v = (v - 1) * nc_max
                all_zero = true
                for c in 1:nc
                    workspace[off_v + c, level] != UInt64(0) && (all_zero = false; break)
                end
                if all_zero; ok = false; break; end
            end
            if !ok; level -= 1; state = 2; continue; end

            # ── C. Find first unbound variable ───────────────────────────────
            unbound = 0
            for v in 1:n_vars
                off_v = (v - 1) * nc_max
                ones  = 0
                for c in 1:nc; ones += count_ones(workspace[off_v + c, level]); end
                if ones > 1; unbound = v; break; end
            end

            if unbound == 0
                # Solution found
                idx = CUDA.atomic_add!(pointer(sol_count, 1), Int32(1)) + 1
                if idx <= max_solutions
                    for v in 1:n_vars
                        off_v = (v - 1) * nc_max
                        elem  = Int32(0)
                        for c in 1:nc
                            ch = workspace[off_v + c, level]
                            if ch != UInt64(0)
                                bi   = trailing_zeros(ch)
                                elem = Int32((c - 1) * 64 + bi + 1)
                                break
                            end
                        end
                        solutions[v, idx] = elem
                    end
                end
                level -= 1; state = 2; continue
            end

            # ── D. Set up branching ──────────────────────────────────────────
            stack_vars[level] = Int32(unbound)
            stack_next[level] = Int32(1)
            state = 2
        end  # state == 1

        if state == 2
            v   = Int(stack_vars[level])
            off = (v - 1) * nc_max
            ne  = Int(stack_next[level])
            found_next = false

            while ne <= nc * 64
                c_ne  = (ne - 1) >> 6 + 1
                bi_ne = (ne - 1) & 63
                if (workspace[off + c_ne, level] & (UInt64(1) << bi_ne)) != 0
                    if level < 16
                        stack_next[level] = Int32(ne + 1)
                        next_lv = level + 1
                        # Copy current workspace to next level
                        for vi in 1:n_vars
                            ofi = (vi - 1) * nc_max
                            for c in 1:nc
                                workspace[ofi + c, next_lv] = workspace[ofi + c, level]
                            end
                        end
                        # Pin branch variable to single element
                        for c in 1:nc; workspace[off + c, next_lv] = UInt64(0); end
                        workspace[off + c_ne, next_lv] = UInt64(1) << bi_ne
                        level = next_lv
                        state = 1
                        found_next = true
                        break
                    end
                    # level == 16: can't go deeper; skip this element
                end
                ne += 1
            end

            if !found_next
                stack_next[level] = Int32(1)
                level -= 1
                state = 2
            end
        end  # state == 2
    end  # while
end

"""
    gpu_dive_solve(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; max_solutions, scratch)

Variant that accepts pre-built, GPU-resident domain and hom-forward arrays.
`d_gpu`       — domain array of length `n_vars * nc` (will be consumed/modified).
`hf_flat_gpu` — flat hom-forward data.
`hf_offs_gpu` — per-morphism 0-based word offsets (length n_homs + 1).
`scratch`     — pre-allocated `GPUScratchBuffers`; when provided, reuses bytecode,
                solution, count, and workspace buffers with no CUDA allocations.
"""
function gpu_dive_solve(backend, csp::CSPProblem,
                        d_gpu, hf_flat_gpu, hf_offs_gpu;
                        max_solutions::Int = 10_000,
                        scratch = nothing)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks

    if scratch !== nothing
        # Use pre-allocated buffers from GPUScratchBuffers (B1: zero allocations)
        b_gpu    = scratch.buf_bytecodes
        sol_gpu  = scratch.buf_solutions
        cnt_gpu  = scratch.buf_sol_count
        work_gpu = scratch.buf_workspace

        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
        # sol_gpu and work_gpu are scratch; their stale values are overwritten by the kernel
    else
        b_gpu    = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        sol_gpu  = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
        cnt_gpu  = KernelAbstractions.allocate(backend, Int32, 1)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
        work_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars * MAX_CHUNKS, 16)
    end

    nc_max = _select_nc_max(nc)
    kernel = dive_solve_kernel!(backend)
    kernel(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, max_solutions,
           work_gpu, hf_flat_gpu, hf_offs_gpu, Val(nc_max); ndrange=1)
    KernelAbstractions.synchronize(backend)

    count = Int(Array(cnt_gpu)[1])
    count == 0 && return Vector{Int32}[]
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end

function gpu_dive_solve(backend, csp::CSPProblem,
                        initial_domains::Vector{UInt64};
                        max_solutions::Int = 10_000)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks

    @assert length(initial_domains) == n_vars * nc

    d_gpu   = KernelAbstractions.allocate(backend, UInt64, n_vars * nc)
    KernelAbstractions.copyto!(backend, d_gpu, initial_domains)
    b_gpu   = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
    cnt_gpu = KernelAbstractions.allocate(backend, Int32, 1)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))
    # workspace: n_vars * MAX_CHUNKS rows × 16 DFS levels
    work_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars * MAX_CHUNKS, 16)

    # Flatten hom_forward (already in chunked flat format)
    hom_fwd_flat = UInt64[]
    hom_fwd_offs = Int32[0]
    for fwd in csp.hom_forward
        append!(hom_fwd_flat, fwd)
        push!(hom_fwd_offs, Int32(length(hom_fwd_flat)))
    end
    isempty(hom_fwd_flat) && push!(hom_fwd_flat, UInt64(0))

    hf_flat_gpu = KernelAbstractions.allocate(backend, UInt64, length(hom_fwd_flat))
    KernelAbstractions.copyto!(backend, hf_flat_gpu, hom_fwd_flat)
    hf_offs_gpu = KernelAbstractions.allocate(backend, Int32, length(hom_fwd_offs))
    KernelAbstractions.copyto!(backend, hf_offs_gpu, hom_fwd_offs)

    nc_max = _select_nc_max(nc)
    kernel = dive_solve_kernel!(backend)
    kernel(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, max_solutions,
           work_gpu, hf_flat_gpu, hf_offs_gpu, Val(nc_max); ndrange=1)
    KernelAbstractions.synchronize(backend)

    count = Int(Array(cnt_gpu)[1])
    count == 0 && return Vector{Int32}[]
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end

# ── GPU branching-point detection kernels ────────────────────────────────────

"""
One thread per CSP variable: compute the popcount of each variable's bitset
domain and record the first unbound variable via atomicMin.

Output `ub_info` layout (length ≥ 3, all Int32):
  [1] = ub_var  — initialized to n_vars+1 (sentinel "none"); atomicMin wins
  [2] = n_subs  — filled by compact_domain_kernel! (0 here on entry)
  [3] = ok_flag — initialized to 1; set to 0 when any domain is empty

This kernel plus `compact_domain_kernel!` are launched on the same CUDA
stream so their GPU-memory dependency is automatically ordered; a single
synchronize suffices for both.
"""
@kernel function find_unbound_var_kernel!(
    ub_info :: AbstractVector{Int32},
    domains :: AbstractVector{UInt64},
    n_vars  :: Int,
    nc      :: Int,
)
    v = @index(Global, Linear)
    if v <= n_vars
        off  = (v - 1) * nc
        ones = 0
        for c in 1:nc
            ones += count_ones(domains[off + c])
        end
        if ones == 0
            CUDA.atomic_and!(pointer(ub_info, 3), Int32(0))
        elseif ones > 1
            CUDA.atomic_min!(pointer(ub_info, 1), Int32(v))
        end
    end
end

"""
Single-thread kernel: reads `ub_var` written by `find_unbound_var_kernel!`
directly from GPU memory, scatters the set bits of that variable's domain
into `ub_elems` as 1-based element indices, and writes the count to
`ub_info[2]`.  No-ops when ok==0 or ub_var is the "all-fixed" sentinel.
"""
@kernel function compact_domain_kernel!(
    ub_elems :: AbstractVector{Int32},
    ub_info  :: AbstractVector{Int32},
    domains  :: AbstractVector{UInt64},
    nc       :: Int,
)
    i = @index(Global, Linear)
    if i == 1
        ok     = Int(ub_info[3])
        ub_var = Int(ub_info[1])
        n_vars = length(domains) ÷ nc
        if ok == 0
            ub_info[2] = Int32(0)
        elseif ub_var > n_vars
            # All variables are singletons.  Use v1 as the branching variable so
            # turbo_eps_kernel! can run AC-1 to verify the assignment is consistent
            # (e.g. monic PROP_NEQ constraints may still reject it).
            off = 0   # v1 offset
            cnt = Int32(0)
            for c in 1:nc
                chunk = domains[off + c]
                if chunk != UInt64(0)
                    bi  = trailing_zeros(chunk)
                    cnt = Int32(1)
                    ub_elems[1] = Int32((c - 1) * 64 + bi + 1)
                    break
                end
            end
            ub_info[1] = Int32(1)   # ub_var ← v1
            ub_info[2] = cnt        # n_subs = 1 (or 0 if v1 somehow empty)
        else
            off = (ub_var - 1) * nc
            cnt = Int32(0)
            for c in 1:nc
                chunk = domains[off + c]
                while chunk != UInt64(0)
                    lsb   = chunk & (-chunk)
                    chunk &= ~lsb
                    bi    = trailing_zeros(lsb)
                    cnt  += Int32(1)
                    if cnt <= length(ub_elems)
                        ub_elems[cnt] = Int32((c - 1) * 64 + bi + 1)
                    end
                end
            end
            ub_info[2] = cnt
        end
    end
end

# ── Block-parallel AC-1 propagation device function (B17) ────────────────────
#
# Called from within @kernel functions.  All threads in the block cooperate on
# one AC-1 propagation pass over a shared domain array.
#
# dom_off: 0-based word offset of the current DFS level in the caller's
#          @localmem UInt64 array (dom).  The slice dom[dom_off+1 .. dom_off+n_vars*NM]
#          holds the active domain state.
#
# Parallelism: thread `tid` (1-based, blocksize total) handles bytecodes at
# positions tid, tid+blocksize, tid+2*blocksize, … Domains are written via
# Atomix.@atomic &= so concurrent narrowings by different threads are safe:
# AC-1 is monotone (domains only shrink), so stale reads produce looser
# (not incorrect) constraints that the outer fixpoint loop will tighten.
#
# changed_flag[1] is set to true if any domain word shrank.  Caller is
# responsible for resetting it to false before each pass and calling @synchronize
# after each call.
#
# Returns nothing; caller tests changed_flag[1] after @synchronize.
@inline function _propagate_block!(
    tid       :: Int,
    blocksize :: Int,
    dom       :: DOM,   # @localmem UInt64 array (concrete type enables specialization)
    dom_off   :: Int,
    changed_flag :: CF, # @localmem Bool [1] (concrete type enables specialization)
    bytecodes :: BCS,
    n_bc      :: Int,
    n_vars    :: Int,
    nc        :: Int,
    hf_flat   :: HFF,
    hf_offs   :: HFO,
    ::Val{NM}
) where {NM, DOM, CF, BCS, HFF, HFO}
    bc_idx = tid
    while bc_idx <= n_bc
        bc = bytecodes[bc_idx]
        v1 = Int(bc.var1)
        v1 == 0 && (bc_idx += blocksize; continue)
        off1 = dom_off + (v1 - 1) * NM

        if bc.op == PROP_FUNC && bc.var2 != 0
            v2    = Int(bc.var2)
            off2  = dom_off + (v2 - 1) * NM
            h_idx = Int(bc.param1)
            n_homs = length(hf_offs) - 1
            if 1 <= h_idx <= n_homs
                off_h     = Int(hf_offs[h_idx])
                n_elems_h = (Int(hf_offs[h_idx + 1]) - off_h) ÷ nc

                # Forward: build constrained domain for v1
                new_d1 = MVector{NM, UInt64}(undef)
                for c in 1:NM; new_d1[c] = UInt64(0); end
                for c in 1:nc
                    chunk = dom[off1 + c]
                    while chunk != UInt64(0)
                        lsb = chunk & (-chunk); chunk &= ~lsb
                        bi  = trailing_zeros(lsb)
                        w   = (c - 1) * 64 + bi + 1
                        w > n_elems_h && continue
                        off_w = off_h + (w - 1) * nc
                        for ci in 1:nc
                            if (hf_flat[off_w + ci] & dom[off2 + ci]) != UInt64(0)
                                new_d1[c] |= lsb; break
                            end
                        end
                    end
                end
                for c in 1:nc
                    old_c = dom[off1 + c]
                    new_c = old_c & new_d1[c]
                    if new_c != old_c
                        Atomix.@atomic dom[off1 + c] &= new_c
                        changed_flag[1] = true
                    end
                end

                # Backward: restrict v2 to reachable elements
                reachable = MVector{NM, UInt64}(undef)
                for c in 1:NM; reachable[c] = UInt64(0); end
                for c in 1:nc
                    chunk = new_d1[c]
                    while chunk != UInt64(0)
                        lsb = chunk & (-chunk); chunk &= ~lsb
                        bi  = trailing_zeros(lsb)
                        w   = (c - 1) * 64 + bi + 1
                        w > n_elems_h && continue
                        off_w = off_h + (w - 1) * nc
                        for ci in 1:nc; reachable[ci] |= hf_flat[off_w + ci]; end
                    end
                end
                for c in 1:nc
                    old_c = dom[off2 + c]
                    new_c = old_c & reachable[c]
                    if new_c != old_c
                        Atomix.@atomic dom[off2 + c] &= new_c
                        changed_flag[1] = true
                    end
                end
            end

        elseif bc.op == PROP_NEQ && bc.var2 != 0
            v2   = Int(bc.var2)
            off2 = dom_off + (v2 - 1) * NM
            ones2 = 0
            for c in 1:nc; ones2 += count_ones(dom[off2 + c]); end
            if ones2 == 1
                for c in 1:nc
                    old_c = dom[off1 + c]
                    new_c = old_c & ~dom[off2 + c]
                    if new_c != old_c
                        Atomix.@atomic dom[off1 + c] &= new_c
                        changed_flag[1] = true
                    end
                end
            end
            ones1 = 0
            for c in 1:nc; ones1 += count_ones(dom[off1 + c]); end
            if ones1 == 1
                for c in 1:nc
                    old_c = dom[off2 + c]
                    new_c = old_c & ~dom[off1 + c]
                    if new_c != old_c
                        Atomix.@atomic dom[off2 + c] &= new_c
                        changed_flag[1] = true
                    end
                end
            end

        elseif bc.op == PROP_EQ && bc.var2 != 0
            v2   = Int(bc.var2)
            off2 = dom_off + (v2 - 1) * NM
            for c in 1:nc
                old1 = dom[off1 + c]
                old2 = dom[off2 + c]
                new1 = old1 & old2
                new2 = old2 & old1
                if new1 != old1
                    Atomix.@atomic dom[off1 + c] &= new1
                    changed_flag[1] = true
                end
                if new2 != old2
                    Atomix.@atomic dom[off2 + c] &= new2
                    changed_flag[1] = true
                end
            end
        end

        bc_idx += blocksize
    end
    nothing
end

# ── EPS (Embarrassingly Parallel Search) Turbo kernel ────────────────────────

"""
EPS parallel solver kernel (B9).

Each thread handles one subproblem: a copy of the initial propagated domains
with `ub_var` pinned to one element from `ub_elements`.  Threads run
independent DFS instances in parallel, writing solutions atomically.

`workspace` must have at least `n_subs * n_vars * NM` rows and 16 columns.
Thread `sub` uses rows `(sub-1)*n_vars*NM + 1 : sub*n_vars*NM`.
"""
@kernel function turbo_eps_kernel!(
    domains_in    :: AbstractVector{UInt64},   # [n_vars * nc] propagated domains (read-only)
    bytecodes     :: AbstractVector{TCNBytecode},
    n_bc          :: Int,
    n_vars        :: Int,
    nc            :: Int,
    solutions     :: AbstractMatrix{Int32},    # [n_vars × max_solutions]
    sol_count     :: AbstractVector{Int32},
    max_solutions :: Int,
    workspace     :: AbstractMatrix{UInt64},   # [n_subs * n_vars * NM × 16]
    hom_fwd_flat  :: AbstractVector{UInt64},
    hom_fwd_offs  :: AbstractVector{Int32},
    ub_info_in    :: AbstractVector{Int32},    # [ub_var, n_subs, ok] written by compact_domain_kernel!
    ub_elements   :: AbstractVector{Int32},    # elements of ub_var's domain
    ::Val{NM}
) where NM
    sub    = @index(Global, Linear)
    n_subs = Int(ub_info_in[2])
    ub_var = Int(ub_info_in[1])
    if sub <= n_subs
        nc_max    = NM
        ws_stride = n_vars * nc_max
        ws_base   = (sub - 1) * ws_stride

        stack_vars = MVector{16, Int32}(undef)
        stack_next = MVector{16, Int32}(undef)
        new_d      = MVector{NM, UInt64}(undef)
        reachable  = MVector{NM, UInt64}(undef)

        # Initialise workspace level 1 from domains_in
        for v in 1:n_vars
            off_d = (v - 1) * nc
            off_w = ws_base + (v - 1) * nc_max
            for c in 1:nc
                workspace[off_w + c, 1] = domains_in[off_d + c]
            end
            for c in (nc + 1):nc_max
                workspace[off_w + c, 1] = UInt64(0)
            end
        end

        # Pin ub_var to ub_elements[sub]
        elem   = Int(ub_elements[sub])
        ci_e, bi_e = elem_to_chunk(elem)
        off_ub = ws_base + (ub_var - 1) * nc_max
        for c in 1:nc_max
            workspace[off_ub + c, 1] = UInt64(0)
        end
        if ci_e <= nc_max
            workspace[off_ub + ci_e, 1] = UInt64(1) << bi_e
        end

        level  = 1
        state  = 1
        safety = 0

        while level > 0 && safety < 100_000_000
            safety += 1

            if state == 1
                # A. Propagate (inline AC-1)
                ok = true
                for _ in 1:8
                    changed = false
                    for i in 1:n_bc
                        bc = bytecodes[i]
                        v1 = Int(bc.var1); v1 == 0 && continue
                        off1 = ws_base + (v1 - 1) * nc_max

                        if bc.op == PROP_FUNC && bc.var2 != 0
                            v2    = Int(bc.var2)
                            off2  = ws_base + (v2 - 1) * nc_max
                            h_idx = Int(bc.param1)
                            n_homs = length(hom_fwd_offs) - 1
                            if 1 <= h_idx <= n_homs
                                off_h    = Int(hom_fwd_offs[h_idx])
                                n_elems_h = (Int(hom_fwd_offs[h_idx+1]) - off_h) ÷ nc

                                for c in 1:nc; new_d[c] = UInt64(0); end
                                for c in 1:nc
                                    chunk = workspace[off1 + c, level]
                                    while chunk != 0
                                        lsb = chunk & (-chunk); chunk &= ~lsb
                                        bi  = trailing_zeros(lsb)
                                        w   = (c - 1) * 64 + bi + 1
                                        w > n_elems_h && continue
                                        off_w2 = off_h + (w - 1) * nc
                                        for ci in 1:nc
                                            (hom_fwd_flat[off_w2 + ci] &
                                             workspace[off2 + ci, level]) != 0 &&
                                                (new_d[c] |= lsb; break)
                                        end
                                    end
                                end
                                for c in 1:nc
                                    old_c = workspace[off1 + c, level]
                                    workspace[off1 + c, level] = new_d[c]
                                    old_c != new_d[c] && (changed = true)
                                end

                                for c in 1:nc; reachable[c] = UInt64(0); end
                                for c in 1:nc
                                    chunk = new_d[c]
                                    while chunk != 0
                                        lsb = chunk & (-chunk); chunk &= ~lsb
                                        bi  = trailing_zeros(lsb)
                                        w   = (c - 1) * 64 + bi + 1
                                        w > n_elems_h && continue
                                        off_w2 = off_h + (w - 1) * nc
                                        for ci in 1:nc
                                            reachable[ci] |= hom_fwd_flat[off_w2 + ci]
                                        end
                                    end
                                end
                                for c in 1:nc
                                    new_c = workspace[off2 + c, level] & reachable[c]
                                    new_c != workspace[off2 + c, level] && (changed = true)
                                    workspace[off2 + c, level] = new_c
                                end
                            end

                        elseif bc.op == PROP_NEQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = ws_base + (v2 - 1) * nc_max
                            ones2 = 0
                            for c in 1:nc; ones2 += count_ones(workspace[off2 + c, level]); end
                            if ones2 == 1
                                for c in 1:nc
                                    new_c = workspace[off1 + c, level] &
                                            ~workspace[off2 + c, level]
                                    new_c != workspace[off1 + c, level] && (changed = true)
                                    workspace[off1 + c, level] = new_c
                                end
                            end

                        elseif bc.op == PROP_EQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = ws_base + (v2 - 1) * nc_max
                            for c in 1:nc
                                old1 = workspace[off1 + c, level]
                                new1 = old1 & workspace[off2 + c, level]
                                workspace[off1 + c, level] = new1
                                old1 != new1 && (changed = true)
                            end
                        end
                    end
                    changed || break
                end

                # B. Consistency check
                for v in 1:n_vars
                    off_v = ws_base + (v - 1) * nc_max
                    all_zero = true
                    for c in 1:nc
                        workspace[off_v + c, level] != UInt64(0) && (all_zero = false; break)
                    end
                    if all_zero; ok = false; break; end
                end
                if !ok; level -= 1; state = 2; continue; end

                # C. Find first unbound variable
                unbound = 0
                for v in 1:n_vars
                    off_v = ws_base + (v - 1) * nc_max
                    ones  = 0
                    for c in 1:nc; ones += count_ones(workspace[off_v + c, level]); end
                    if ones > 1; unbound = v; break; end
                end

                if unbound == 0
                    idx = CUDA.atomic_add!(pointer(sol_count, 1), Int32(1)) + 1
                    if idx <= max_solutions
                        for v in 1:n_vars
                            off_v = ws_base + (v - 1) * nc_max
                            elem2 = Int32(0)
                            for c in 1:nc
                                ch = workspace[off_v + c, level]
                                if ch != UInt64(0)
                                    bi   = trailing_zeros(ch)
                                    elem2 = Int32((c - 1) * 64 + bi + 1)
                                    break
                                end
                            end
                            solutions[v, idx] = elem2
                        end
                    end
                    level -= 1; state = 2; continue
                end

                stack_vars[level] = Int32(unbound)
                stack_next[level] = Int32(1)
                state = 2
            end  # state == 1

            if state == 2
                v   = Int(stack_vars[level])
                off = ws_base + (v - 1) * nc_max
                ne  = Int(stack_next[level])
                found_next = false

                while ne <= nc * 64
                    c_ne  = (ne - 1) >> 6 + 1
                    bi_ne = (ne - 1) & 63
                    if (workspace[off + c_ne, level] & (UInt64(1) << bi_ne)) != 0
                        if level < 16
                            stack_next[level] = Int32(ne + 1)
                            next_lv = level + 1
                            for vi in 1:n_vars
                                ofi = ws_base + (vi - 1) * nc_max
                                for c in 1:nc
                                    workspace[ofi + c, next_lv] = workspace[ofi + c, level]
                                end
                            end
                            for c in 1:nc_max; workspace[off + c, next_lv] = UInt64(0); end
                            workspace[off + c_ne, next_lv] = UInt64(1) << bi_ne
                            level = next_lv
                            state = 1
                            found_next = true
                            break
                        end
                    end
                    ne += 1
                end

                if !found_next
                    stack_next[level] = Int32(1)
                    level -= 1
                    state = 2
                end
            end  # state == 2
        end  # while
    end  # if sub <= n_subs
end

# ── Block-parallel consistency check + unbound-variable detection ─────────────
#
# Called only from thread `tid`==1; writes:
#   ok_flag[1]  = true if all domains are non-empty, false on failure
#   binfo[1]    = first unbound variable (>1 element in domain), or 0 if all fixed
# No @synchronize — caller is responsible for issuing one after this returns.
@inline function _check_find_unbound!(
    dom     :: DOM,
    dom_off :: Int,
    ok_flag :: OF,
    binfo   :: BI,
    n_vars  :: Int,
    nc      :: Int,
    ::Val{NM}
) where {NM, DOM, OF, BI}
    ok_flag[1] = true
    binfo[1]   = Int32(0)
    for v in 1:n_vars
        off = dom_off + (v - 1) * NM
        all_zero = true
        for c in 1:nc
            dom[off + c] != UInt64(0) && (all_zero = false; break)
        end
        if all_zero; ok_flag[1] = false; return; end
    end
    for v in 1:n_vars
        off  = dom_off + (v - 1) * NM
        ones = 0
        for c in 1:nc; ones += count_ones(dom[off + c]); end
        if ones > 1; binfo[1] = Int32(v); return; end
    end
    nothing
end

# ── Turbo multi-block dive-and-solve kernel (B9) ─────────────────────────────

"""
Multi-block Turbo dive-and-solve kernel (B9).

Decomposes the CSP search tree into `2^D` subproblems via binary-path encoding.
CUDA blocks claim subproblems dynamically from the atomic counter `nextsub`.
All threads in a block cooperate on AC-1 propagation via `_propagate_block!`;
thread 1 drives branching decisions and the DFS stack.

Variable domains are stored in block-local shared memory (`dom`, 16 DFS levels).
No per-call workspace allocation: storage is statically sized via `Val{NM}`
(nc_max) and `Val{NVNM16}` (n_vars × nc_max × 16).

Launch with blocksize=32, n_blocks = min(2^D, 576).
"""
@kernel function turbo_block_kernel!(
    domains_root  :: AbstractVector{UInt64},
    bytecodes     :: AbstractVector{TCNBytecode},
    n_bc          :: Int,
    n_vars        :: Int,
    nc            :: Int,
    D             :: Int,
    nextsub       :: AbstractVector{Int32},
    solutions     :: AbstractMatrix{Int32},
    sol_count     :: AbstractVector{Int32},
    max_solutions :: Int,
    hf_flat       :: AbstractVector{UInt64},
    hf_offs       :: AbstractVector{Int32},
    ::Val{NM},
    ::Val{NVNM16},
) where {NM, NVNM16}

    tid       = @index(Local, Linear)   # 1-based thread within block
    blocksize = @groupsize()[1]
    n_vars_nm = n_vars * NM             # words per DFS level

    # ── Shared memory ─────────────────────────────────────────────────────────
    dom      = @localmem UInt64 (NVNM16,)  # domain workspace: 16 DFS levels
    stack_v  = @localmem Int32  (16,)      # branching variable at each solve-phase level
    stnext   = @localmem Int32  (16,)      # next candidate element index at each level
    ok_flag  = @localmem Bool   (1,)
    # binfo: [1]=ub_var (0=all fixed), [2]=branch_elem (1-based flat), [3]=found_next
    binfo    = @localmem Int32  (3,)
    # mysub_block: the subproblem index claimed by this block (one value for all 32 threads)
    mysub_b  = @localmem Int32  (1,)

    # ── Thread 1 claims first subproblem for the block ────────────────────────
    if tid == 1
        mysub_b[1] = CUDA.atomic_add!(pointer(nextsub, 1), Int32(1))
    end
    @synchronize()

    while mysub_b[1] < Int32(1 << D)
        mysub = Int(mysub_b[1])

        # ── 1. Copy root domains into dom level 1 (all threads cooperate) ──────
        j = tid
        while j <= n_vars_nm
            v_zero = (j - 1) ÷ NM
            c      = (j - 1) - v_zero * NM + 1
            dom[j] = c <= nc ? domains_root[v_zero * nc + c] : UInt64(0)
            j += blocksize
        end
        @synchronize()

        # ── 2. DIVE PHASE: binary-path decomposition for D levels ──────────────
        dive_ok = true

        for d in 1:D
            # Thread 1: inline AC-1 propagation on dom[1..n_vars_nm], then
            # consistency check, unbound-var detection, and domain pin.
            if tid == 1
                new_d     = MVector{NM, UInt64}(undef)
                reachable = MVector{NM, UInt64}(undef)
                for _ in 1:(n_bc * 8 + 1)
                    local_changed = false
                    for i in 1:n_bc
                        bc = bytecodes[i]
                        v1 = Int(bc.var1); v1 == 0 && continue
                        off1 = (v1 - 1) * NM

                        if bc.op == PROP_FUNC && bc.var2 != 0
                            v2    = Int(bc.var2)
                            off2  = (v2 - 1) * NM
                            h_idx = Int(bc.param1)
                            n_homs = length(hf_offs) - 1
                            if 1 <= h_idx <= n_homs
                                off_h     = Int(hf_offs[h_idx])
                                n_elems_h = (Int(hf_offs[h_idx + 1]) - off_h) ÷ nc
                                for c in 1:NM; new_d[c] = UInt64(0); end
                                for c in 1:nc
                                    chunk = dom[off1 + c]
                                    while chunk != UInt64(0)
                                        lsb = chunk & (-chunk); chunk &= ~lsb
                                        bi  = trailing_zeros(lsb)
                                        w   = (c - 1) * 64 + bi + 1
                                        w > n_elems_h && continue
                                        off_w = off_h + (w - 1) * nc
                                        for ci in 1:nc
                                            if (hf_flat[off_w + ci] & dom[off2 + ci]) != UInt64(0)
                                                new_d[c] |= lsb; break
                                            end
                                        end
                                    end
                                end
                                for c in 1:nc
                                    old_c = dom[off1 + c]
                                    dom[off1 + c] = new_d[c]
                                    old_c != new_d[c] && (local_changed = true)
                                end
                                for c in 1:NM; reachable[c] = UInt64(0); end
                                for c in 1:nc
                                    chunk = new_d[c]
                                    while chunk != UInt64(0)
                                        lsb = chunk & (-chunk); chunk &= ~lsb
                                        bi  = trailing_zeros(lsb)
                                        w   = (c - 1) * 64 + bi + 1
                                        w > n_elems_h && continue
                                        off_w = off_h + (w - 1) * nc
                                        for ci in 1:nc; reachable[ci] |= hf_flat[off_w + ci]; end
                                    end
                                end
                                for c in 1:nc
                                    new_c = dom[off2 + c] & reachable[c]
                                    new_c != dom[off2 + c] && (local_changed = true)
                                    dom[off2 + c] = new_c
                                end
                            end

                        elseif bc.op == PROP_NEQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = (v2 - 1) * NM
                            ones2 = 0
                            for c in 1:nc; ones2 += count_ones(dom[off2 + c]); end
                            if ones2 == 1
                                for c in 1:nc
                                    old_c = dom[off1 + c]
                                    new_c = old_c & ~dom[off2 + c]
                                    dom[off1 + c] = new_c
                                    old_c != new_c && (local_changed = true)
                                end
                            end
                            ones1 = 0
                            for c in 1:nc; ones1 += count_ones(dom[off1 + c]); end
                            if ones1 == 1
                                for c in 1:nc
                                    old_c = dom[off2 + c]
                                    new_c = old_c & ~dom[off1 + c]
                                    dom[off2 + c] = new_c
                                    old_c != new_c && (local_changed = true)
                                end
                            end

                        elseif bc.op == PROP_EQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = (v2 - 1) * NM
                            for c in 1:nc
                                old1 = dom[off1 + c]; old2 = dom[off2 + c]
                                new1 = old1 & old2;   new2 = old2 & old1
                                dom[off1 + c] = new1; dom[off2 + c] = new2
                                (old1 != new1 || old2 != new2) && (local_changed = true)
                            end
                        end
                    end
                    local_changed || break
                end

                ok_flag[1] = true
                for v in 1:n_vars
                    all_zero = true
                    for c in 1:nc
                        dom[(v - 1) * NM + c] != UInt64(0) && (all_zero = false; break)
                    end
                    if all_zero; ok_flag[1] = false; break; end
                end

                binfo[1] = Int32(0)
                if ok_flag[1]
                    for v in 1:n_vars
                        off  = (v - 1) * NM
                        ones = 0
                        for c in 1:nc; ones += count_ones(dom[off + c]); end
                        if ones > 1; binfo[1] = Int32(v); break; end
                    end

                    ub_v = Int(binfo[1])
                    if ub_v != 0
                        bit_d = (mysub >> (D - d)) & 1
                        off   = (ub_v - 1) * NM
                        if bit_d == 0
                            first_c  = Int32(0)
                            first_bi = Int32(0)
                            for c in 1:nc
                                if first_c == Int32(0)
                                    chunk = dom[off + c]
                                    if chunk != UInt64(0)
                                        first_c  = Int32(c)
                                        first_bi = Int32(trailing_zeros(chunk))
                                    end
                                end
                            end
                            for c in 1:NM; dom[off + c] = UInt64(0); end
                            if first_c != Int32(0)
                                dom[off + Int(first_c)] = UInt64(1) << Int(first_bi)
                            end
                        else
                            for c in 1:nc
                                chunk = dom[off + c]
                                if chunk != UInt64(0)
                                    dom[off + c] &= ~(chunk & (-chunk))
                                    break
                                end
                            end
                        end
                    end
                end
            end  # tid == 1
            @synchronize()

            if !ok_flag[1]; dive_ok = false; break; end

            if binfo[1] == Int32(0)
                # All variables fixed mid-dive: canonical subproblem records solution
                remaining      = D - d + 1
                remaining_mask = (1 << remaining) - 1
                if (mysub & remaining_mask) == 0 && tid == 1
                    tmp_idx = CUDA.atomic_add!(pointer(sol_count, 1), Int32(1)) + 1
                    if tmp_idx <= max_solutions
                        for v in 1:n_vars
                            off  = (v - 1) * NM
                            elem = Int32(0)
                            for c in 1:nc
                                ch = dom[off + c]
                                if ch != UInt64(0)
                                    bi   = trailing_zeros(ch)
                                    elem = Int32((c - 1) * 64 + bi + 1)
                                    break
                                end
                            end
                            solutions[v, tmp_idx] = elem
                        end
                    end
                end
                @synchronize()
                dive_ok = false; break
            end
        end  # for d in 1:D

        # ── 3. SOLVE PHASE: backtracking DFS from dive leaf ────────────────────
        # dom[0..n_vars_nm-1] holds the post-dive domain (DFS level 1).
        if dive_ok
            level  = 1
            state  = 1
            safety = 0

            while level > 0 && safety < 10_000_000
                safety += 1
                lev_off = (level - 1) * n_vars_nm

                if state == 1
                    if tid == 1
                        new_d     = MVector{NM, UInt64}(undef)
                        reachable = MVector{NM, UInt64}(undef)
                        for _ in 1:(n_bc * 8 + 1)
                            local_changed = false
                            for i in 1:n_bc
                                bc = bytecodes[i]
                                v1 = Int(bc.var1); v1 == 0 && continue
                                off1 = lev_off + (v1 - 1) * NM

                                if bc.op == PROP_FUNC && bc.var2 != 0
                                    v2    = Int(bc.var2)
                                    off2  = lev_off + (v2 - 1) * NM
                                    h_idx = Int(bc.param1)
                                    n_homs = length(hf_offs) - 1
                                    if 1 <= h_idx <= n_homs
                                        off_h     = Int(hf_offs[h_idx])
                                        n_elems_h = (Int(hf_offs[h_idx + 1]) - off_h) ÷ nc
                                        for c in 1:NM; new_d[c] = UInt64(0); end
                                        for c in 1:nc
                                            chunk = dom[off1 + c]
                                            while chunk != UInt64(0)
                                                lsb = chunk & (-chunk); chunk &= ~lsb
                                                bi  = trailing_zeros(lsb)
                                                w   = (c - 1) * 64 + bi + 1
                                                w > n_elems_h && continue
                                                off_w = off_h + (w - 1) * nc
                                                for ci in 1:nc
                                                    if (hf_flat[off_w + ci] & dom[off2 + ci]) != UInt64(0)
                                                        new_d[c] |= lsb; break
                                                    end
                                                end
                                            end
                                        end
                                        for c in 1:nc
                                            old_c = dom[off1 + c]
                                            dom[off1 + c] = new_d[c]
                                            old_c != new_d[c] && (local_changed = true)
                                        end
                                        for c in 1:NM; reachable[c] = UInt64(0); end
                                        for c in 1:nc
                                            chunk = new_d[c]
                                            while chunk != UInt64(0)
                                                lsb = chunk & (-chunk); chunk &= ~lsb
                                                bi  = trailing_zeros(lsb)
                                                w   = (c - 1) * 64 + bi + 1
                                                w > n_elems_h && continue
                                                off_w = off_h + (w - 1) * nc
                                                for ci in 1:nc; reachable[ci] |= hf_flat[off_w + ci]; end
                                            end
                                        end
                                        for c in 1:nc
                                            new_c = dom[off2 + c] & reachable[c]
                                            new_c != dom[off2 + c] && (local_changed = true)
                                            dom[off2 + c] = new_c
                                        end
                                    end

                                elseif bc.op == PROP_NEQ && bc.var2 != 0
                                    v2   = Int(bc.var2)
                                    off2 = lev_off + (v2 - 1) * NM
                                    ones2 = 0
                                    for c in 1:nc; ones2 += count_ones(dom[off2 + c]); end
                                    if ones2 == 1
                                        for c in 1:nc
                                            old_c = dom[off1 + c]
                                            new_c = old_c & ~dom[off2 + c]
                                            dom[off1 + c] = new_c
                                            old_c != new_c && (local_changed = true)
                                        end
                                    end
                                    ones1 = 0
                                    for c in 1:nc; ones1 += count_ones(dom[off1 + c]); end
                                    if ones1 == 1
                                        for c in 1:nc
                                            old_c = dom[off2 + c]
                                            new_c = old_c & ~dom[off1 + c]
                                            dom[off2 + c] = new_c
                                            old_c != new_c && (local_changed = true)
                                        end
                                    end

                                elseif bc.op == PROP_EQ && bc.var2 != 0
                                    v2   = Int(bc.var2)
                                    off2 = lev_off + (v2 - 1) * NM
                                    for c in 1:nc
                                        old1 = dom[off1 + c]; old2 = dom[off2 + c]
                                        new1 = old1 & old2;   new2 = old2 & old1
                                        dom[off1 + c] = new1; dom[off2 + c] = new2
                                        (old1 != new1 || old2 != new2) && (local_changed = true)
                                    end
                                end
                            end
                            local_changed || break
                        end

                        ok_flag[1] = true
                        for v in 1:n_vars
                            all_zero = true
                            off = lev_off + (v - 1) * NM
                            for c in 1:nc
                                dom[off + c] != UInt64(0) && (all_zero = false; break)
                            end
                            if all_zero; ok_flag[1] = false; break; end
                        end

                        binfo[1] = Int32(0)
                        if ok_flag[1]
                            for v in 1:n_vars
                                off  = lev_off + (v - 1) * NM
                                ones = 0
                                for c in 1:nc; ones += count_ones(dom[off + c]); end
                                if ones > 1; binfo[1] = Int32(v); break; end
                            end
                        end
                    end  # tid == 1
                    @synchronize()

                    if !ok_flag[1]; level -= 1; state = 2; continue; end

                    if binfo[1] == Int32(0)
                        if tid == 1
                            tmp_idx = CUDA.atomic_add!(pointer(sol_count, 1), Int32(1)) + 1
                            if tmp_idx <= max_solutions
                                for v in 1:n_vars
                                    off  = lev_off + (v - 1) * NM
                                    elem = Int32(0)
                                    for c in 1:nc
                                        ch = dom[off + c]
                                        if ch != UInt64(0)
                                            bi   = trailing_zeros(ch)
                                            elem = Int32((c - 1) * 64 + bi + 1)
                                            break
                                        end
                                    end
                                    solutions[v, tmp_idx] = elem
                                end
                            end
                        end
                        @synchronize()
                        level -= 1; state = 2; continue
                    end

                    if tid == 1
                        stack_v[level] = binfo[1]
                        stnext[level]  = Int32(1)
                    end
                    @synchronize()
                    state = 2
                end  # state == 1

                if state == 2
                    if tid == 1
                        binfo[3] = Int32(0)
                        v   = Int(stack_v[level])
                        off = lev_off + (v - 1) * NM
                        ne  = Int(stnext[level])
                        while ne <= nc * 64
                            c_ne  = (ne - 1) >> 6 + 1
                            bi_ne = (ne - 1) & 63
                            if (dom[off + c_ne] & (UInt64(1) << bi_ne)) != UInt64(0)
                                if level < 16
                                    binfo[2]      = Int32(ne)
                                    binfo[3]      = Int32(1)
                                    stnext[level] = Int32(ne + 1)
                                end
                                break
                            end
                            ne += 1
                        end
                    end
                    @synchronize()

                    if binfo[3] == Int32(1)
                        # Copy current level to next (all threads cooperate)
                        dst_off = level * n_vars_nm
                        j = tid
                        while j <= n_vars_nm
                            dom[dst_off + j] = dom[lev_off + j]
                            j += blocksize
                        end
                        @synchronize()

                        if tid == 1
                            v     = Int(stack_v[level])
                            ne    = Int(binfo[2])
                            c_ne  = (ne - 1) >> 6 + 1
                            bi_ne = (ne - 1) & 63
                            off   = dst_off + (v - 1) * NM
                            for c in 1:NM; dom[off + c] = UInt64(0); end
                            dom[off + c_ne] = UInt64(1) << bi_ne
                        end
                        @synchronize()
                        level += 1; state = 1
                    else
                        if tid == 1; stnext[level] = Int32(1); end
                        @synchronize()
                        level -= 1; state = 2
                    end
                end  # state == 2
            end  # while level > 0
        end  # if dive_ok

        # ── 4. Claim next subproblem for this block ────────────────────────────
        if tid == 1
            mysub_b[1] = CUDA.atomic_add!(pointer(nextsub, 1), Int32(1))
        end
        @synchronize()
    end  # while mysub_b[1] < 2^D
end

# Private helper: dispatch Val{nc_max} and Val{nvnm16} for turbo_block_kernel!.
# Called from gpu_turbo_solve and _gpu_turbo_fill_scratch!.
function _launch_turbo_block!(backend,
                               d_gpu, b_gpu,
                               n_bc::Int, n_vars::Int, nc::Int, D::Int,
                               nextsub, sol_gpu, cnt_gpu, max_solutions::Int,
                               hf_flat, hf_offs,
                               nc_max::Int, nvnm16::Int, n_blks::Int)
    turbo_block_kernel!(backend, 32)(
        d_gpu, b_gpu, n_bc, n_vars, nc, D,
        nextsub, sol_gpu, cnt_gpu, max_solutions,
        hf_flat, hf_offs, Val(nc_max), Val(nvnm16);
        ndrange = n_blks * 32
    )
end

# Private helper: three-step EPS pipeline for small (nc==1) problems.
#   1. find_unbound_var_kernel! — locate first branching variable
#   2. compact_domain_kernel!   — collect its domain elements into ub_elems
#   3. turbo_eps_kernel!        — one thread per domain element
#
# workspace must have at least nc_max*64 * n_vars * nc_max rows and 16 cols.
# Returns after launching all kernels; caller is responsible for synchronize.
function _launch_turbo_eps!(backend,
                             d_gpu, b_gpu,
                             n_bc::Int, n_vars::Int, nc::Int,
                             sol_gpu, cnt_gpu, max_solutions::Int,
                             hf_flat, hf_offs,
                             nc_max::Int,
                             ub_info,    # AbstractVector{Int32} length ≥ 3
                             ub_elems,   # AbstractVector{Int32} capacity ≥ nc_max*64
                             workspace)  # AbstractMatrix{UInt64} rows ≥ nc_max*64*n_vars, cols=16
    # Initialise: ub_info = [n_vars+1 (sentinel), 0 (n_subs), 1 (ok)]
    copyto!(ub_info, Int32[n_vars + 1, 0, 1])
    find_unbound_var_kernel!(backend, 64)(ub_info, d_gpu, n_vars, nc; ndrange = n_vars)
    compact_domain_kernel!(backend, 1)(ub_elems, ub_info, d_gpu, nc; ndrange = 1)
    # compact_domain_kernel! handles the all-fixed (singleton) edge case by setting
    # ub_var=1, n_subs=1 so turbo_eps_kernel! can verify constraints via AC-1.
    # We launch with a fixed ndrange = nc_max*64; each thread self-limits on n_subs
    # read from ub_info[2] at kernel entry, avoiding a CPU-GPU synchronize+download.
    max_ws_rows = nc_max * 64 * n_vars * nc_max
    turbo_eps_kernel!(backend, 64)(
        d_gpu, b_gpu, n_bc, n_vars, nc,
        sol_gpu, cnt_gpu, max_solutions,
        view(workspace, 1:max_ws_rows, :), hf_flat, hf_offs,
        ub_info, ub_elems, Val(nc_max);
        ndrange = nc_max * 64
    )
end

"""
    gpu_turbo_solve(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; kwargs) -> solutions

Multi-block Turbo solver (B9): decomposes the CSP search tree into 2^D
subproblems.  CUDA blocks claim subproblems from an atomic counter and cooperate
on AC-1 propagation within each block.  No per-call workspace allocation; all
domain state lives in shared memory.

Dispatch: nc==1 (≤64 elements/var) → EPS pipeline (find_unbound + compact + turbo_eps_kernel!,
one thread per domain element, global-memory workspace).
nc≥2 → multi-block Turbo (turbo_block_kernel!, shared-memory workspace, sub-problem counter
in scratch.buf_turbo_nextsub or a transient allocation).

Falls back to `gpu_dive_solve` for zero-variable or empty CSPs.
"""
function gpu_turbo_solve(backend, csp::CSPProblem,
                         d_gpu, hf_flat_gpu, hf_offs_gpu;
                         max_solutions::Int = 10_000,
                         scratch = nothing)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    nc_max = _select_nc_max(nc)

    # Empty pattern: unique empty match always exists.
    n_vars == 0 && return [Int32[]]

    if scratch !== nothing
        b_gpu   = scratch.buf_bytecodes
        sol_gpu = scratch.buf_solutions
        cnt_gpu = scratch.buf_sol_count
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
    else
        b_gpu   = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
        cnt_gpu = KernelAbstractions.allocate(backend, Int32, 1)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
    end

    if nc == 1
        # Small problem (≤64 elements/var): EPS — one thread per domain element.
        # Workspace rows needed: n_subs * n_vars * nc_max ≤ nc_max*64 * n_vars * nc_max.
        ws_cap = nc_max * 64 * n_vars * nc_max
        if scratch !== nothing
            if size(scratch.buf_workspace, 1) < ws_cap
                scratch.buf_workspace = CUDA.zeros(UInt64, ws_cap * 2, 16)
            end
            _launch_turbo_eps!(backend, d_gpu, b_gpu, n_bc, n_vars, nc,
                                sol_gpu, cnt_gpu, max_solutions,
                                hf_flat_gpu, hf_offs_gpu, nc_max,
                                scratch.buf_ub_info, scratch.buf_ub_elems,
                                scratch.buf_workspace)
        else
            ub_info   = CUDA.zeros(Int32, 3)
            ub_elems  = CUDA.zeros(Int32, nc_max * 64)
            workspace = CUDA.zeros(UInt64, ws_cap, 16)
            _launch_turbo_eps!(backend, d_gpu, b_gpu, n_bc, n_vars, nc,
                                sol_gpu, cnt_gpu, max_solutions,
                                hf_flat_gpu, hf_offs_gpu, nc_max,
                                ub_info, ub_elems, workspace)
        end
    else
        # Large problem (≥128 elements/var): multi-block Turbo.
        D      = clamp(ceil(Int, log2(max(nc * 64, 2))), 4, 14)
        n_blks = min(1 << D, 576)
        nvnm16 = n_vars * nc_max * 16
        if scratch !== nothing
            sub_gpu = scratch.buf_turbo_nextsub
            KernelAbstractions.fill!(sub_gpu, Int32(0))
        else
            sub_gpu = KernelAbstractions.allocate(backend, Int32, 1)
            KernelAbstractions.fill!(sub_gpu, Int32(0))
        end
        _launch_turbo_block!(backend, d_gpu, b_gpu, n_bc, n_vars, nc, D,
                              sub_gpu, sol_gpu, cnt_gpu, max_solutions,
                              hf_flat_gpu, hf_offs_gpu, nc_max, nvnm16, n_blks)
    end
    KernelAbstractions.synchronize(backend)

    count = Int(Array(cnt_gpu)[1])
    count == 0 && return Vector{Int32}[]
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end

"""
    _gpu_turbo_fill_scratch!(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; scratch) -> Int

Fills `scratch.buf_solutions` on-device and downloads only the 4-byte solution count.
Returns the number of solutions found (0 if none).

Dispatch: nc==1 → EPS pipeline; nc≥2 → multi-block Turbo (scratch.buf_turbo_nextsub).
Used by the `AbstractGPUPlayer` fast path so the player can inspect
`scratch.buf_solutions` without a bulk transfer.  No per-call GPU allocations.
"""
function _gpu_turbo_fill_scratch!(backend, csp::CSPProblem,
                                   d_gpu, hf_flat_gpu, hf_offs_gpu;
                                   max_solutions::Int = 10_000,
                                   scratch)::Int
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    nc_max = _select_nc_max(nc)

    # Empty pattern: exactly 1 empty match always exists.
    if n_vars == 0
        KernelAbstractions.fill!(scratch.buf_sol_count, Int32(1))
        return 1
    end

    b_gpu   = scratch.buf_bytecodes
    sol_gpu = scratch.buf_solutions
    cnt_gpu = scratch.buf_sol_count
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))

    if nc == 1
        # Small problem (≤64 elements/var): EPS pipeline.
        ws_cap = nc_max * 64 * n_vars * nc_max
        if size(scratch.buf_workspace, 1) < ws_cap
            scratch.buf_workspace = CUDA.zeros(UInt64, ws_cap * 2, 16)
        end
        _launch_turbo_eps!(backend, d_gpu, b_gpu, n_bc, n_vars, nc,
                            sol_gpu, cnt_gpu, max_solutions,
                            hf_flat_gpu, hf_offs_gpu, nc_max,
                            scratch.buf_ub_info, scratch.buf_ub_elems,
                            scratch.buf_workspace)
    else
        # Large problem (≥128 elements/var): multi-block Turbo.
        D      = clamp(ceil(Int, log2(max(nc * 64, 2))), 4, 14)
        n_blks = min(1 << D, 576)
        nvnm16 = n_vars * nc_max * 16
        sub_gpu = scratch.buf_turbo_nextsub
        KernelAbstractions.fill!(sub_gpu, Int32(0))
        _launch_turbo_block!(backend, d_gpu, b_gpu, n_bc, n_vars, nc, D,
                              sub_gpu, sol_gpu, cnt_gpu, max_solutions,
                              hf_flat_gpu, hf_offs_gpu, nc_max, nvnm16, n_blks)
    end
    KernelAbstractions.synchronize(backend)
    CUDA.@allowscalar min(Int(scratch.buf_sol_count[1]), max_solutions)
end

"""
Single-thread kernel: reads `sol_count` and `solutions[:,1]`, writes the
chosen match to `buf_match` and sets `buf_fired[1]` = 1 (match found) or 0
(no match).  Also zeros `buf_match` when no solution, ensuring stale data
from the previous step cannot leak into downstream guarded kernels.
"""
@kernel function write_match_from_sols_kernel!(
    buf_match :: AbstractVector{Int32},
    buf_fired :: AbstractVector{Int32},
    solutions :: AbstractMatrix{Int32},
    sol_count :: AbstractVector{Int32},
    n_vars    :: Int,
)
    i = @index(Global, Linear)
    if i == 1
        if sol_count[1] > Int32(0)
            buf_fired[1] = Int32(1)
            for v in 1:n_vars
                buf_match[v] = solutions[v, 1]
            end
        else
            buf_fired[1] = Int32(0)
            for v in 1:n_vars
                buf_match[v] = Int32(0)
            end
        end
    end
end

"""
    _find_first_unbound(d_host, n_vars, nc) -> (ub_var, elements)

CPU fallback used by `gpu_turbo_solve` when no scratch buffers are available.
Scans the domain array for the first variable with >1 active element.
Returns `(0, Int32[])` if any domain is empty or all are already fixed.
"""
function _find_first_unbound(d_host::Vector{UInt64}, n_vars::Int, nc::Int)
    for v in 1:n_vars
        off  = (v - 1) * nc
        ones = 0
        for c in 1:nc; ones += count_ones(d_host[off + c]); end
        ones == 0 && return (0, Int32[])
        if ones > 1
            elems = Int32[]
            for c in 1:nc
                chunk = d_host[off + c]
                while chunk != 0
                    lsb = chunk & (-chunk); chunk &= ~lsb
                    bi  = trailing_zeros(lsb)
                    push!(elems, Int32((c - 1) * 64 + bi + 1))
                end
            end
            return (v, elems)
        end
    end
    return (0, Int32[])
end
