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
    workspace     :: AbstractMatrix{UInt64},   # [n_vars * MAX_CHUNKS × 16]
    hom_fwd_flat  :: AbstractVector{UInt64},
    hom_fwd_offs  :: AbstractVector{Int32}
)
    nc     = n_chunks
    nc_max = MAX_CHUNKS

    stack_vars = MVector{16, Int32}(undef)
    stack_next = MVector{16, Int32}(undef)
    new_d      = MVector{MAX_CHUNKS, UInt64}(undef)
    reachable  = MVector{MAX_CHUNKS, UInt64}(undef)

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
    gpu_dive_solve(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu; max_solutions)

Variant that accepts pre-built, GPU-resident domain and hom-forward arrays.
Used by the GPU scheduler to avoid CPU↔GPU transfers during solve setup.
`d_gpu`       — domain array of length `n_vars * nc` (will be consumed/modified).
`hf_flat_gpu` — flat hom-forward data.
`hf_offs_gpu` — per-morphism 0-based word offsets (length n_homs + 1).
"""
function gpu_dive_solve(backend, csp::CSPProblem,
                        d_gpu, hf_flat_gpu, hf_offs_gpu;
                        max_solutions::Int = 10_000)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks

    b_gpu   = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
    cnt_gpu = KernelAbstractions.allocate(backend, Int32, 1)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))
    work_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars * MAX_CHUNKS, 16)

    kernel = dive_solve_kernel!(backend)
    kernel(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, max_solutions,
           work_gpu, hf_flat_gpu, hf_offs_gpu; ndrange=1)
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

    kernel = dive_solve_kernel!(backend)
    kernel(d_gpu, b_gpu, n_bc, n_vars, nc, sol_gpu, cnt_gpu, max_solutions,
           work_gpu, hf_flat_gpu, hf_offs_gpu; ndrange=1)
    KernelAbstractions.synchronize(backend)

    count = Int(Array(cnt_gpu)[1])
    count == 0 && return Vector{Int32}[]
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end
