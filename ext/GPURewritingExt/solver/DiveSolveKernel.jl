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
        if ok == 0 || ub_var > n_vars
            ub_info[2] = Int32(0)
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
    ub_var        :: Int,                      # first unbound variable (1-based)
    ub_elements   :: AbstractVector{Int32},    # elements of ub_var's domain
    ::Val{NM}
) where NM
    sub = @index(Global, Linear)
    n_subs = length(ub_elements)
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

"""
    gpu_turbo_solve(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; kwargs) -> solutions

EPS parallel solver (B9): runs one parallel DFS thread per element of the
first unbound variable in the initial propagated domain.  For a graph-matching
rule with N valid matches the solver explores N subproblems simultaneously.

When `scratch` is provided, the branching-point detection runs entirely on
GPU via two ordered kernels (`find_unbound_var_kernel!` and
`compact_domain_kernel!`) with a single subsequent synchronize and a 12-byte
scalar download — no domain array is transferred to the host.

Falls back to `gpu_dive_solve` when the domain is already fixed or empty.
"""
function gpu_turbo_solve(backend, csp::CSPProblem,
                         d_gpu, hf_flat_gpu, hf_offs_gpu;
                         max_solutions::Int = 10_000,
                         scratch = nothing)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    nc_max = _select_nc_max(nc)

    # ── Detect branching point on GPU ─────────────────────────────────────────
    if scratch !== nothing
        # Initialize ub_info: [ub_var=n_vars+1, n_subs=0, ok=1]
        copyto!(scratch.buf_ub_info, Int32[n_vars + 1, 0, 1])

        # Kernel 1: parallel popcount scan → atomicMin for ub_var, atomicAnd for ok
        find_unbound_var_kernel!(backend, 256)(
            scratch.buf_ub_info, d_gpu, n_vars, nc; ndrange = n_vars)

        # Kernel 2: single-thread bitset unpack — reads ub_var from GPU, writes elements
        # Ordered automatically by CUDA stream; no intermediate synchronize needed.
        compact_domain_kernel!(backend, 1)(
            scratch.buf_ub_elems, scratch.buf_ub_info, d_gpu, nc; ndrange = 1)

        KernelAbstractions.synchronize(backend)

        # Download 3 Int32s (12 bytes total)
        ub_info = Array(scratch.buf_ub_info)
        ub_var  = Int(ub_info[1])
        n_subs  = Int(ub_info[2])
        ok      = Int(ub_info[3])

        ok == 0        && return Vector{Int32}[]
        ub_var > n_vars && return gpu_dive_solve(backend, csp, d_gpu,
                                                  hf_flat_gpu, hf_offs_gpu;
                                                  max_solutions, scratch)
        n_subs == 0    && return Vector{Int32}[]

        ub_gpu = @view scratch.buf_ub_elems[1:n_subs]

        b_gpu   = scratch.buf_bytecodes
        sol_gpu = scratch.buf_solutions
        cnt_gpu = scratch.buf_sol_count
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
    else
        # CPU fallback: download domain, scan on host, upload element list.
        # Used only outside the scheduler hot path (e.g. turbo_homomorphisms).
        d_host = Array(d_gpu)
        ub_var, ub_elements = _find_first_unbound(d_host, n_vars, nc)
        (ub_var == 0 || isempty(ub_elements)) &&
            return gpu_dive_solve(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu;
                                  max_solutions, scratch)
        n_subs = length(ub_elements)
        ub_gpu = KernelAbstractions.allocate(backend, Int32, n_subs)
        KernelAbstractions.copyto!(backend, ub_gpu, ub_elements)

        b_gpu   = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
        n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
        sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
        cnt_gpu = KernelAbstractions.allocate(backend, Int32, 1)
        KernelAbstractions.fill!(cnt_gpu, Int32(0))
    end

    work_gpu = KernelAbstractions.allocate(backend, UInt64, n_subs * n_vars * nc_max, 16)

    turbo_eps_kernel!(backend)(
        d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, max_solutions,
        work_gpu, hf_flat_gpu, hf_offs_gpu, ub_var, ub_gpu, Val(nc_max);
        ndrange = n_subs)
    KernelAbstractions.synchronize(backend)

    count = Int(Array(cnt_gpu)[1])
    count == 0 && return Vector{Int32}[]
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end

"""
    _gpu_turbo_fill_scratch!(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; scratch) -> Int

Variant of `gpu_turbo_solve` that fills `scratch.buf_solutions` on-device but
downloads only the solution count (4 bytes) rather than all solutions.
Returns the number of solutions found (0 if none).

Used by the `AbstractGPUPlayer` fast path in `_gpu_solve_inplace!` so that
the player can inspect `scratch.buf_solutions` directly without a bulk transfer.
"""
function _gpu_turbo_fill_scratch!(backend, csp::CSPProblem,
                                   d_gpu, hf_flat_gpu, hf_offs_gpu;
                                   max_solutions::Int = 10_000,
                                   scratch)::Int
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    nc_max = _select_nc_max(nc)

    copyto!(scratch.buf_ub_info, Int32[n_vars + 1, 0, 1])
    find_unbound_var_kernel!(backend, 256)(
        scratch.buf_ub_info, d_gpu, n_vars, nc; ndrange = n_vars)
    compact_domain_kernel!(backend, 1)(
        scratch.buf_ub_elems, scratch.buf_ub_info, d_gpu, nc; ndrange = 1)
    KernelAbstractions.synchronize(backend)

    ub_info = Array(scratch.buf_ub_info)
    ub_var  = Int(ub_info[1])
    n_subs  = Int(ub_info[2])
    ok      = Int(ub_info[3])

    ok == 0     && return 0
    n_subs == 0 && return 0

    b_gpu   = scratch.buf_bytecodes
    sol_gpu = scratch.buf_solutions
    cnt_gpu = scratch.buf_sol_count
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))

    if ub_var > n_vars
        work_gpu = scratch.buf_workspace
        dive_solve_kernel!(backend)(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu,
            max_solutions, work_gpu, hf_flat_gpu, hf_offs_gpu, Val(nc_max); ndrange = 1)
    else
        ub_gpu   = @view scratch.buf_ub_elems[1:n_subs]
        work_gpu = KernelAbstractions.allocate(backend, UInt64, n_subs * n_vars * nc_max, 16)
        turbo_eps_kernel!(backend)(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu,
            max_solutions, work_gpu, hf_flat_gpu, hf_offs_gpu, ub_var, ub_gpu, Val(nc_max);
            ndrange = n_subs)
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
