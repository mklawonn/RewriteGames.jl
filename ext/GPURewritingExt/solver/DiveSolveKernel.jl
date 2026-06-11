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

# ── Host-side count-weighted random sampler (SampleSearch) ────────────────────
#
# Reference implementation of "take N" uniform-ish sampling, used as the CPU
# fallback and as the correctness oracle for the GPU sampling kernel.
#
# Principle (Gogate–Dechter): descend the search tree choosing, at each branching
# variable, a value with probability proportional to an estimate of the number of
# solutions in that branch's subtree.  We use the cheap relaxation weight
# w = ∏_v |D_v| (product of remaining domain sizes after one-step AC-1
# propagation) — an upper bound on completions.  On a dead end we backtrack and
# exclude the failed value (SampleSearch), so every returned assignment is a valid
# solution.  Uniformity is best-effort (the weight is an upper bound); validity is
# guaranteed.

# Extract the (single) element of each variable from a fully-determined domain set.
@inline function _extract_assignment(domains::Vector{UInt64}, n_vars::Int, nc::Int)
    sol = Vector{Int32}(undef, n_vars)
    for v in 1:n_vars
        off  = (v - 1) * nc
        elem = Int32(0)
        for c in 1:nc
            ch = domains[off + c]
            if ch != UInt64(0)
                bi   = trailing_zeros(ch)
                elem = Int32((c - 1) * 64 + bi + 1)
                break
            end
        end
        sol[v] = elem
    end
    return sol
end

# Product of domain sizes (∏_v popcount(D_v)) — subtree-size estimate.
@inline function _domain_product(domains::Vector{UInt64}, n_vars::Int, nc::Int)
    w = 1.0
    for v in 1:n_vars
        off  = (v - 1) * nc
        ones = 0
        for c in 1:nc; ones += count_ones(domains[off + c]); end
        w *= ones
    end
    return w
end

# One weighted random descent with backtracking. `domains` must already be
# arc-consistent and non-empty.  Returns a valid assignment or `nothing` if the
# subtree has no solution.
function _sample_descent(domains::Vector{UInt64}, bytecodes, hom_forward,
                         n_vars::Int, nc::Int, rng)
    # Find first unbound variable (popcount > 1).
    unbound = 0
    for v in 1:n_vars
        off  = (v - 1) * nc
        ones = 0
        for c in 1:nc; ones += count_ones(domains[off + c]); end
        ones == 0 && return nothing          # empty domain (defensive)
        ones > 1  && (unbound = v; break)
    end
    unbound == 0 && return _extract_assignment(domains, n_vars, nc)

    # Enumerate candidate values of the branching variable.
    off_ub = (unbound - 1) * nc
    cands  = Int[]
    for c in 1:nc
        chunk = domains[off_ub + c]
        while chunk != 0
            lsb = chunk & (-chunk); chunk &= ~lsb
            push!(cands, (c - 1) * 64 + trailing_zeros(lsb) + 1)
        end
    end

    # One-step lookahead: pin each candidate, propagate, weight by ∏|D_v|.
    children = Vector{Vector{UInt64}}(undef, length(cands))
    weights  = zeros(Float64, length(cands))
    for (k, e) in enumerate(cands)
        child = copy(domains)
        for ci in 1:nc; child[off_ub + ci] = UInt64(0); end
        cc = (e - 1) >> 6 + 1; bb = (e - 1) & 63
        child[off_ub + cc] = UInt64(1) << bb
        if cpu_propagate!(child, bytecodes, hom_forward, nc)
            weights[k]  = _domain_product(child, n_vars, nc)
            children[k] = child
        end
    end

    # Weighted sample without replacement; recurse, backtrack on failure.
    remaining = sum(weights)
    while remaining > 0
        r   = rand(rng) * remaining
        acc = 0.0; j = 0
        for k in eachindex(weights)
            weights[k] <= 0 && continue
            acc += weights[k]
            if r <= acc; j = k; break; end
        end
        j == 0 && (j = findlast(>(0.0), weights))   # floating-point guard
        res = _sample_descent(children[j], bytecodes, hom_forward, n_vars, nc, rng)
        res !== nothing && return res
        remaining -= weights[j]; weights[j] = 0.0    # exclude failed branch
    end
    return nothing
end

"""
    cpu_sample_solve(csp, initial_domains; take, rng) -> Vector{Vector{Int32}}

Return up to `take` distinct valid solutions sampled (best-effort uniformly) from
the CSP solution space via count-weighted random descent.  CPU reference for the
GPU sampling path and fallback when CUDA is unavailable.
"""
function cpu_sample_solve(csp::CSPProblem, initial_domains::Vector{UInt64};
                          take::Int, rng = default_rng())::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    nc     = csp.n_chunks
    solutions = Vector{Int32}[]
    n_vars == 0 && return [Int32[]]
    take <= 0 && return solutions

    root = copy(initial_domains)
    cpu_propagate!(root, csp.bytecodes, csp.hom_forward, nc) || return solutions

    seen = Set{Vector{Int32}}()
    # Cap attempts so that solution spaces smaller than `take` terminate; the
    # multiplier trades a few wasted descents for high coverage.
    max_attempts = take * 20 + 50
    attempts = 0
    while length(solutions) < take && attempts < max_attempts
        attempts += 1
        sol = _sample_descent(copy(root), csp.bytecodes, csp.hom_forward,
                              n_vars, nc, rng)
        sol === nothing && break                 # subtree has no solution at all
        if !(sol in seen)
            push!(seen, sol)
            push!(solutions, sol)
        end
    end
    return solutions
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

# ── Batched NAC/PAC existence kernel ──────────────────────────────────────────
#
# One work-item per candidate match.  Each work-item, entirely on-device:
#   1. copies the shared base domain `domains_in` (d0) into its own workspace slab,
#   2. pins its candidate's shared-L variables (slot read from `sol_mat[rv, cand]`),
#   3. runs the same dive-and-solve DFS as `dive_solve_kernel!` but stops at the
#      first solution (existence only) and writes `cnt_out[cand] = 1`.
#
# This replaces the host `for i in 1:N` loop that issued, per candidate, a
# `copyto!` + one `_pin_var!` launch per shared var + a `ndrange=1` dive — pure
# launch overhead.  Here a whole tile of candidates is one launch.  The DFS body
# is a faithful copy of `dive_solve_kernel!`; the only changes are the per-candidate
# workspace dimension `t`, the on-device pin, and existence early-exit.  Results
# (kept-set) are therefore identical to the serial filter.
@kernel function nac_exist_batch_kernel!(
    domains_in   :: AbstractVector{UInt64},   # [n_vars * n_chunks]  shared base (d0)
    bytecodes    :: AbstractVector{TCNBytecode},
    n_bc         :: Int,
    n_vars       :: Int,
    n_chunks     :: Int,
    pin_cv       :: AbstractVector{Int32},    # [n_pin]  condition variable to pin
    pin_rv       :: AbstractVector{Int32},    # [n_pin]  rule variable supplying the slot
    n_pin        :: Int,
    sol_mat      :: AbstractMatrix{Int32},    # [R × Ntot]  candidate solutions
    R            :: Int,
    Ntot         :: Int,
    base_cand    :: Int,                      # tile offset (0-based) into candidates
    keep_in      :: AbstractVector{Int32},    # [Ntot]  1 = still alive, 0 = already rejected
    cnt_out      :: AbstractVector{Int32},    # [Ntot]  existence flag (0/1)
    workspace    :: AbstractArray{UInt64, 3}, # [n_vars*nc_max × 16 × tile]
    hom_fwd_flat :: AbstractVector{UInt64},
    hom_fwd_offs :: AbstractVector{Int32},
    ::Val{NM}
) where NM
    nc     = n_chunks
    nc_max = NM

    stack_vars = MVector{16, Int32}(undef)
    stack_next = MVector{16, Int32}(undef)
    new_d      = MVector{NM, UInt64}(undef)
    reachable  = MVector{NM, UInt64}(undef)

    t    = @index(Global, Linear)             # 1..tile
    cand = base_cand + t                      # global candidate index

    @inbounds if cand <= Ntot && keep_in[cand] != Int32(0)
        # Level-1 init: copy shared base domain into this candidate's slab.
        for v in 1:n_vars
            off_w = (v - 1) * nc_max
            off_d = (v - 1) * nc
            for c in 1:nc
                workspace[off_w + c, 1, t] = domains_in[off_d + c]
            end
        end

        # Pin this candidate's shared-L variables (mirror of `_pin_var!`): for an
        # out-of-range slot (ci ∉ 1:nc) leave the variable's full domain, exactly
        # as the serial `_pin_var!` early-returns.
        for p in 1:n_pin
            cv   = Int(pin_cv[p])
            rv   = Int(pin_rv[p])
            slot = Int(sol_mat[rv, cand])
            ci   = ((slot - 1) >> 6) + 1
            bi   = (slot - 1) & 63
            if 1 <= ci <= nc
                off1 = (cv - 1) * nc_max
                for c in 1:nc
                    workspace[off1 + c, 1, t] = (c == ci) ?
                        (workspace[off1 + c, 1, t] & (UInt64(1) << bi)) : UInt64(0)
                end
            end
        end

        # ── Dive-and-solve DFS (existence): identical to dive_solve_kernel! ────
        found  = false
        level  = 1
        state  = 1
        safety = 0

        while level > 0 && safety < 100_000_000
            safety += 1

            if state == 1
                # ── A. Propagate (inline AC-1) ────────────────────────────────
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
                                    chunk = workspace[off1 + c, level, t]
                                    while chunk != 0
                                        lsb = chunk & (-chunk); chunk &= ~lsb
                                        bi  = trailing_zeros(lsb)
                                        w   = (c - 1) * 64 + bi + 1
                                        w > n_elems_h && continue
                                        off_w = off_h + (w - 1) * nc
                                        for ci in 1:nc
                                            (hom_fwd_flat[off_w + ci] &
                                             workspace[off2 + ci, level, t]) != 0 &&
                                                (new_d[c] |= lsb; break)
                                        end
                                    end
                                end
                                for c in 1:nc
                                    old_c = workspace[off1 + c, level, t]
                                    workspace[off1 + c, level, t] = new_d[c]
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
                                    new_c = workspace[off2 + c, level, t] & reachable[c]
                                    new_c != workspace[off2 + c, level, t] && (changed = true)
                                    workspace[off2 + c, level, t] = new_c
                                end
                            end

                        elseif bc.op == PROP_NEQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = (v2 - 1) * nc_max
                            ones2 = 0
                            for c in 1:nc; ones2 += count_ones(workspace[off2 + c, level, t]); end
                            if ones2 == 1
                                for c in 1:nc
                                    new_c = workspace[off1 + c, level, t] &
                                            ~workspace[off2 + c, level, t]
                                    new_c != workspace[off1 + c, level, t] && (changed = true)
                                    workspace[off1 + c, level, t] = new_c
                                end
                            end

                        elseif bc.op == PROP_EQ && bc.var2 != 0
                            v2   = Int(bc.var2)
                            off2 = (v2 - 1) * nc_max
                            for c in 1:nc
                                old1 = workspace[off1 + c, level, t]
                                new1 = old1 & workspace[off2 + c, level, t]
                                workspace[off1 + c, level, t] = new1
                                old1 != new1 && (changed = true)
                            end
                        end
                    end
                    changed || break
                end

                # ── B. Consistency check ─────────────────────────────────────
                for v in 1:n_vars
                    off_v = (v - 1) * nc_max
                    all_zero = true
                    for c in 1:nc
                        workspace[off_v + c, level, t] != UInt64(0) && (all_zero = false; break)
                    end
                    if all_zero; ok = false; break; end
                end
                if !ok; level -= 1; state = 2; continue; end

                # ── C. Find first unbound variable ───────────────────────────
                unbound = 0
                for v in 1:n_vars
                    off_v = (v - 1) * nc_max
                    ones  = 0
                    for c in 1:nc; ones += count_ones(workspace[off_v + c, level, t]); end
                    if ones > 1; unbound = v; break; end
                end

                if unbound == 0
                    found = true            # existence: first solution suffices
                    break
                end

                # ── D. Set up branching ──────────────────────────────────────
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
                    if (workspace[off + c_ne, level, t] & (UInt64(1) << bi_ne)) != 0
                        if level < 16
                            stack_next[level] = Int32(ne + 1)
                            next_lv = level + 1
                            for vi in 1:n_vars
                                ofi = (vi - 1) * nc_max
                                for c in 1:nc
                                    workspace[ofi + c, next_lv, t] = workspace[ofi + c, level, t]
                                end
                            end
                            for c in 1:nc; workspace[off + c, next_lv, t] = UInt64(0); end
                            workspace[off + c_ne, next_lv, t] = UInt64(1) << bi_ne
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

        cnt_out[cand] = found ? Int32(1) : Int32(0)
    end
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

# ── GPU count-weighted random-descent sampler ("take N") ──────────────────────
#
# One CUDA thread = one independent weighted random descent (= one sample).
# Mirrors `dive_solve_kernel!` but, at each branching variable, chooses a value
# with probability ∝ ∏_v |D_v| (subtree-size estimate after one-step AC-1) rather
# than enumerating exhaustively, and stops at the first leaf.  Full SampleSearch
# backtracking (try other weighted candidates on a deeper dead-end) guarantees a
# valid solution whenever one exists.  Each thread writes to its own output column
# `solutions[:, t]`, so no atomics are needed.  See `cpu_sample_solve` (the
# matching host reference / oracle).

# splitmix64 — advances `state`, returns `(new_state, output)`.  Chosen because it
# decorrelates structured / sequential seeds well (one xorshift step from a
# `seed ⊻ t·golden` seed biases the first draw, starving some branches).
@inline function _sm64_next(state::UInt64)
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z = z ⊻ (z >> 31)
    return state, z
end

# Single-thread AC-1 propagation on workspace column `col` (chunked, nc_max
# stride).  Returns true iff every variable domain is non-empty at fixpoint.
# Mirrors the inlined propagation in `dive_solve_kernel!`.  Attribute / domain-
# size constraints are pre-baked into the initial domains, so only PROP_FUNC /
# PROP_EQ / PROP_NEQ are handled here (as in the dive kernel).
@inline function _ac1_propagate_col!(ws, col::Int, bytecodes, n_bc::Int,
                                     n_vars::Int, nc::Int, nc_max::Int,
                                     hom_fwd_flat, hom_fwd_offs, new_d, reachable)
    n_homs = length(hom_fwd_offs) - 1
    for _ in 1:8
        changed = false
        for i in 1:n_bc
            bc = bytecodes[i]
            v1 = Int(bc.var1); v1 == 0 && continue
            off1 = (v1 - 1) * nc_max
            if bc.op == PROP_FUNC && bc.var2 != 0
                v2 = Int(bc.var2); off2 = (v2 - 1) * nc_max
                h_idx = Int(bc.param1)
                if 1 <= h_idx <= n_homs
                    off_h     = Int(hom_fwd_offs[h_idx])
                    n_elems_h = (Int(hom_fwd_offs[h_idx + 1]) - off_h) ÷ nc
                    for c in 1:nc; new_d[c] = UInt64(0); end
                    for c in 1:nc
                        chunk = ws[off1 + c, col]
                        while chunk != 0
                            lsb = chunk & (-chunk); chunk &= ~lsb
                            w   = (c - 1) * 64 + trailing_zeros(lsb) + 1
                            w > n_elems_h && continue
                            off_w = off_h + (w - 1) * nc
                            for ci in 1:nc
                                (hom_fwd_flat[off_w + ci] & ws[off2 + ci, col]) != 0 &&
                                    (new_d[c] |= lsb; break)
                            end
                        end
                    end
                    for c in 1:nc
                        old_c = ws[off1 + c, col]
                        ws[off1 + c, col] = new_d[c]
                        old_c != new_d[c] && (changed = true)
                    end
                    for c in 1:nc; reachable[c] = UInt64(0); end
                    for c in 1:nc
                        chunk = new_d[c]
                        while chunk != 0
                            lsb = chunk & (-chunk); chunk &= ~lsb
                            w   = (c - 1) * 64 + trailing_zeros(lsb) + 1
                            w > n_elems_h && continue
                            off_w = off_h + (w - 1) * nc
                            for ci in 1:nc; reachable[ci] |= hom_fwd_flat[off_w + ci]; end
                        end
                    end
                    for c in 1:nc
                        new_c = ws[off2 + c, col] & reachable[c]
                        new_c != ws[off2 + c, col] && (changed = true)
                        ws[off2 + c, col] = new_c
                    end
                end
            elseif bc.op == PROP_NEQ && bc.var2 != 0
                v2 = Int(bc.var2); off2 = (v2 - 1) * nc_max
                ones2 = 0
                for c in 1:nc; ones2 += count_ones(ws[off2 + c, col]); end
                if ones2 == 1
                    for c in 1:nc
                        new_c = ws[off1 + c, col] & ~ws[off2 + c, col]
                        new_c != ws[off1 + c, col] && (changed = true)
                        ws[off1 + c, col] = new_c
                    end
                end
            elseif bc.op == PROP_EQ && bc.var2 != 0
                v2 = Int(bc.var2); off2 = (v2 - 1) * nc_max
                for c in 1:nc
                    old1 = ws[off1 + c, col]
                    new1 = old1 & ws[off2 + c, col]
                    ws[off1 + c, col] = new1
                    old1 != new1 && (changed = true)
                end
            end
        end
        changed || break
    end
    for v in 1:n_vars
        off = (v - 1) * nc_max
        allz = true
        for c in 1:nc
            ws[off + c, col] != UInt64(0) && (allz = false; break)
        end
        allz && return false
    end
    return true
end

# ∏_v popcount(D_v) for workspace column `col` — subtree-size estimate.
@inline function _domain_product_col(ws, col::Int, n_vars::Int, nc::Int, nc_max::Int)
    w = 1.0
    for v in 1:n_vars
        off = (v - 1) * nc_max
        o = 0
        for c in 1:nc; o += count_ones(ws[off + c, col]); end
        w *= o
    end
    return w
end

# Copy all variable domains from column `src` to column `dst`.
@inline function _copy_col!(ws, dst::Int, src::Int, n_vars::Int, nc::Int, nc_max::Int)
    for v in 1:n_vars
        off = (v - 1) * nc_max
        for c in 1:nc; ws[off + c, dst] = ws[off + c, src]; end
    end
end

@kernel function sample_descent_kernel!(
    domains_in   :: AbstractVector{UInt64},   # [n_vars * nc] initial domains
    bytecodes    :: AbstractVector{TCNBytecode},
    n_bc         :: Int,
    n_vars       :: Int,
    nc           :: Int,
    max_levels   :: Int,                       # = n_vars + 1
    n_threads    :: Int,
    seed         :: UInt64,
    solutions    :: AbstractMatrix{Int32},     # [n_vars × n_threads], col t = thread t's sample
    ws           :: AbstractMatrix{UInt64},    # [n_vars*NM × n_threads*(max_levels+1)]
    hom_fwd_flat :: AbstractVector{UInt64},
    hom_fwd_offs :: AbstractVector{Int32},
    ::Val{NM},
) where {NM}
    t = @index(Global)
    if t <= n_threads
        nc_max    = NM
        new_d     = MVector{NM, UInt64}(undef)
        reachable = MVector{NM, UInt64}(undef)
        stack_var = MVector{16, Int32}(undef)

        cols_per_thread = max_levels + 1
        base   = (t - 1) * cols_per_thread
        tmpcol = base + cols_per_thread        # last per-thread column = candidate scratch

        # Level-1 column ← initial domains.
        col1 = base + 1
        for v in 1:n_vars
            offw = (v - 1) * nc_max
            offd = (v - 1) * nc
            for c in 1:nc; ws[offw + c, col1] = domains_in[offd + c]; end
        end

        rs = seed ⊻ (UInt64(t) * 0x9E3779B97F4A7C15)
        rs == UInt64(0) && (rs = 0x9E3779B97F4A7C15)

        lvl    = 1
        state  = 1
        safety = 0
        while lvl > 0 && safety < 2_000_000
            safety += 1
            col = base + lvl

            if state == 1
                ok = _ac1_propagate_col!(ws, col, bytecodes, n_bc, n_vars, nc, nc_max,
                                         hom_fwd_flat, hom_fwd_offs, new_d, reachable)
                if !ok
                    lvl -= 1; state = 2
                else
                    unbound = 0
                    for v in 1:n_vars
                        offv = (v - 1) * nc_max
                        ones = 0
                        for c in 1:nc; ones += count_ones(ws[offv + c, col]); end
                        if ones > 1; unbound = v; break; end
                    end
                    if unbound == 0
                        for v in 1:n_vars
                            offv = (v - 1) * nc_max
                            elem = Int32(0)
                            for c in 1:nc
                                ch = ws[offv + c, col]
                                if ch != UInt64(0)
                                    elem = Int32((c - 1) * 64 + trailing_zeros(ch) + 1)
                                    break
                                end
                            end
                            solutions[v, t] = elem
                        end
                        lvl = 0                       # leaf reached → thread done
                    else
                        stack_var[lvl] = Int32(unbound)
                        state = 2
                    end
                end

            else  # state == 2: weighted pick among the branching var's remaining values
                ub    = Int(stack_var[lvl])
                offub = (ub - 1) * nc_max

                # Pass A — total feasible weight over remaining candidate bits.
                W = 0.0
                for c in 1:nc
                    chunk = ws[offub + c, col]
                    while chunk != 0
                        lsb = chunk & (-chunk); chunk &= ~lsb
                        _copy_col!(ws, tmpcol, col, n_vars, nc, nc_max)
                        for cc in 1:nc; ws[offub + cc, tmpcol] = UInt64(0); end
                        ws[offub + c, tmpcol] = lsb
                        _ac1_propagate_col!(ws, tmpcol, bytecodes, n_bc, n_vars, nc, nc_max,
                                            hom_fwd_flat, hom_fwd_offs, new_d, reachable) &&
                            (W += _domain_product_col(ws, tmpcol, n_vars, nc, nc_max))
                    end
                end

                if W <= 0.0
                    lvl -= 1; state = 2               # node exhausted → backtrack
                else
                    rs, z = _sm64_next(rs)
                    r  = (Float64(z >> 11) * (1.0 / 9007199254740992.0)) * W
                    # Pass B — re-walk, pick the chosen candidate, leave child in tmpcol.
                    acc        = 0.0
                    chosen_c   = 0
                    chosen_lsb = UInt64(0)
                    for c in 1:nc
                        chunk = ws[offub + c, col]
                        while chunk != 0
                            lsb = chunk & (-chunk); chunk &= ~lsb
                            _copy_col!(ws, tmpcol, col, n_vars, nc, nc_max)
                            for cc in 1:nc; ws[offub + cc, tmpcol] = UInt64(0); end
                            ws[offub + c, tmpcol] = lsb
                            if _ac1_propagate_col!(ws, tmpcol, bytecodes, n_bc, n_vars, nc, nc_max,
                                                   hom_fwd_flat, hom_fwd_offs, new_d, reachable)
                                acc += _domain_product_col(ws, tmpcol, n_vars, nc, nc_max)
                                if acc >= r
                                    chosen_c = c; chosen_lsb = lsb; break
                                end
                            end
                        end
                        chosen_c != 0 && break
                    end
                    if chosen_c == 0
                        lvl -= 1; state = 2           # floating-point guard → backtrack
                    else
                        ws[offub + chosen_c, col] &= ~chosen_lsb    # exclude (mark tried)
                        if lvl < max_levels
                            _copy_col!(ws, base + lvl + 1, tmpcol, n_vars, nc, nc_max)
                            lvl += 1; state = 1
                        else
                            state = 2                 # depth cap: retry remaining here
                        end
                    end
                end
            end
        end
    end
end

"""
    gpu_turbo_sample(backend, csp, initial_domains; take, seed, oversample) -> Vector{Vector{Int32}}

Sample up to `take` distinct valid solutions via the count-weighted random-descent
kernel (one thread per sample, `take*oversample` threads launched, deduplicated on
host).  GPU analog of `cpu_sample_solve`.  Falls back to full solve + uniform
subsample when the pattern is too deep for the fixed 16-level thread stack.
"""
function gpu_turbo_sample(backend, csp::CSPProblem, initial_domains::Vector{UInt64};
                          take::Int, seed::Integer = 0,
                          oversample::Int = 4)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    n_vars == 0 && return [Int32[]]
    take  <= 0 && return Vector{Int32}[]
    @assert length(initial_domains) == n_vars * nc
    nc_max = _select_nc_max(nc)

    if n_vars + 1 > 16
        # Pattern deeper than the fixed thread stack: full solve + uniform subsample.
        return _subsample_solutions(gpu_dive_solve(backend, csp, initial_domains), take, seed)
    end

    max_levels = n_vars + 1
    K = max(take * oversample, take + 8)

    d_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars * nc)
    KernelAbstractions.copyto!(backend, d_gpu, initial_domains)
    b_gpu = KernelAbstractions.allocate(backend, TCNBytecode, max(n_bc, 1))
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, K)
    KernelAbstractions.fill!(sol_gpu, Int32(0))     # failed threads stay all-zero (invalid)
    ws_gpu  = KernelAbstractions.allocate(backend, UInt64, n_vars * nc_max, K * (max_levels + 1))

    hom_fwd_flat = UInt64[]; hom_fwd_offs = Int32[0]
    for fwd in csp.hom_forward
        append!(hom_fwd_flat, fwd); push!(hom_fwd_offs, Int32(length(hom_fwd_flat)))
    end
    isempty(hom_fwd_flat) && push!(hom_fwd_flat, UInt64(0))
    hf_flat_gpu = KernelAbstractions.allocate(backend, UInt64, length(hom_fwd_flat))
    KernelAbstractions.copyto!(backend, hf_flat_gpu, hom_fwd_flat)
    hf_offs_gpu = KernelAbstractions.allocate(backend, Int32, length(hom_fwd_offs))
    KernelAbstractions.copyto!(backend, hf_offs_gpu, hom_fwd_offs)

    sample_descent_kernel!(backend)(
        d_gpu, b_gpu, n_bc, n_vars, nc, max_levels, K, UInt64(seed),
        sol_gpu, ws_gpu, hf_flat_gpu, hf_offs_gpu, Val(nc_max); ndrange = K)
    KernelAbstractions.synchronize(backend)

    res  = Array(sol_gpu)                            # [n_vars × K]
    out  = Vector{Int32}[]
    seen = Set{Tuple}()
    for j in 1:K
        res[1, j] == Int32(0) && continue            # thread found no solution
        col = res[:, j]
        key = Tuple(col)
        key in seen && continue
        push!(seen, key); push!(out, col)
        length(out) >= take && break
    end
    return out
end

# Uniform random subsample (without replacement) of `take` from `sols`.
function _subsample_solutions(sols::Vector{Vector{Int32}}, take::Int, seed::Integer)
    length(sols) <= take && return sols
    return shuffle(Xoshiro(UInt64(seed)), sols)[1:take]
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
# positions tid, tid+blocksize, tid+2*blocksize, … Domains are narrowed via
# `CUDA.atomic_and!` (shared-memory atomic AND) so concurrent narrowings by
# different threads are safe: AC-1 is monotone (domains only shrink), so stale
# reads produce looser (not incorrect) constraints that the outer fixpoint loop
# tightens.  (Atomix.@atomic has no shared-memory method here, hence the
# explicit pointer-based atomic, matching this file's other CUDA.atomic_*! uses.)
#
# changed_flag[1] is set to true if any domain word shrank.  Caller is
# responsible for resetting it to false before each pass and calling @synchronize
# after each call.
#
# Returns nothing; caller tests changed_flag[1] after @synchronize.
@inline function _propagate_block!(
    tid, blocksize, dom, dom_off, changed_flag, bytecodes,
    n_bc, n_vars, nc, hf_flat, hf_offs, ::Val{NM}) where {NM}
    # Hoisted out of the bytecode loop: allocating MVectors inside the dynamic
    # `while bc_idx` loop prevents stack allocation on the GPU (dynamic alloc →
    # invalid IR).  Declared once, reset per use below.
    new_d1    = MVector{NM, UInt64}(undef)
    reachable = MVector{NM, UInt64}(undef)
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
                        CUDA.atomic_and!(pointer(dom, off1 + c), new_c)
                        changed_flag[1] = true
                    end
                end

                # Backward: restrict v2 to reachable elements
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
                        CUDA.atomic_and!(pointer(dom, off2 + c), new_c)
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
                        CUDA.atomic_and!(pointer(dom, off1 + c), new_c)
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
                        CUDA.atomic_and!(pointer(dom, off2 + c), new_c)
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
                    CUDA.atomic_and!(pointer(dom, off1 + c), new1)
                    changed_flag[1] = true
                end
                if new2 != old2
                    CUDA.atomic_and!(pointer(dom, off2 + c), new2)
                    changed_flag[1] = true
                end
            end
        end

        bc_idx += blocksize
    end
    nothing
end

# ── Single-thread AC-1 fixpoint (DEFAULT path; block-parallel is opt-in) ──────
#
# Same semantics as the block-parallel `_propagate_block!` fixpoint, but run
# entirely on the calling lane (no atomics, no @synchronize).  Call from a single
# thread (e.g. tid==1).  Operates on dom[dom_off+1 : dom_off+n_vars*NM].  Extracted
# from the (previously duplicated) inline dive- and solve-phase fixpoints.
@inline function _propagate_serial!(
    dom, dom_off::Int, bytecodes, n_bc::Int, n_vars::Int, nc::Int,
    hf_flat, hf_offs, ::Val{NM}) where {NM}
    new_d     = MVector{NM, UInt64}(undef)
    reachable = MVector{NM, UInt64}(undef)
    for _ in 1:(n_bc * 8 + 1)
        local_changed = false
        for i in 1:n_bc
            bc = bytecodes[i]
            v1 = Int(bc.var1); v1 == 0 && continue
            off1 = dom_off + (v1 - 1) * NM

            if bc.op == PROP_FUNC && bc.var2 != 0
                v2    = Int(bc.var2)
                off2  = dom_off + (v2 - 1) * NM
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
                off2 = dom_off + (v2 - 1) * NM
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
                off2 = dom_off + (v2 - 1) * NM
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
    nothing
end

# ── Intra-bytecode parallel PROP_FUNC (opt-in via RG_INTRA_PROP) ──────────────
#
# For game_full the turbo_block solves have FEW bytecodes (2-9) but HUGE domains
# (nc=45), so the win is parallelizing WITHIN a PROP_FUNC over its nc domain
# chunks across the 32 lanes (lane owns chunk c ⇒ race-free narrowing, no atomics)
# rather than across bytecodes (lever #1, too little width here).  Three sync-free
# phases (caller barriers between them; KA forbids @synchronize inside a device
# fn).  UNTYPED args (required for device fns called from a KA @kernel).
#
# Phase 1 — forward: lane strides over v1's chunks, writing new_d1_sh[c] = bits of
# v1 chunk c that have a hom-edge into v2's current domain.  No-op (early return)
# for an invalid hom index, matching the serial path; backward then also no-ops.
@inline function _intra_forward!(tid, blocksize, dom, dom_off, new_d1_sh, bc,
                                 nc, hf_flat, hf_offs, ::Val{NM}) where {NM}
    v1 = Int(bc.var1); v1 == 0 && return nothing; v2 = Int(bc.var2)
    h_idx = Int(bc.param1); n_homs = length(hf_offs) - 1
    (1 <= h_idx <= n_homs) || return nothing
    off1 = dom_off + (v1 - 1) * NM
    off2 = dom_off + (v2 - 1) * NM
    off_h = Int(hf_offs[h_idx])
    n_elems_h = (Int(hf_offs[h_idx + 1]) - off_h) ÷ nc
    c = tid
    while c <= nc
        acc = UInt64(0)
        chunk = dom[off1 + c]
        while chunk != UInt64(0)
            lsb = chunk & (-chunk); chunk &= ~lsb
            bi  = trailing_zeros(lsb)
            w   = (c - 1) * 64 + bi + 1
            if w <= n_elems_h
                off_w = off_h + (w - 1) * nc
                for ci in 1:nc
                    if (hf_flat[off_w + ci] & dom[off2 + ci]) != UInt64(0)
                        acc |= lsb; break
                    end
                end
            end
        end
        new_d1_sh[c] = acc
        c += blocksize
    end
    nothing
end

# Phase 2 — backward+narrow (after a barrier on new_d1_sh): lane owns chunk c and
# narrows BOTH v1 (= new_d1_sh[c]) and v2 (&= reachable[c], the OR over all
# surviving v1 bits of the hom column c).  Sets changed_flag if anything shrank.
@inline function _intra_backward!(tid, blocksize, dom, dom_off, new_d1_sh,
                                  changed_flag, bc, nc, hf_flat, hf_offs,
                                  ::Val{NM}) where {NM}
    v1 = Int(bc.var1); v1 == 0 && return nothing; v2 = Int(bc.var2)
    h_idx = Int(bc.param1); n_homs = length(hf_offs) - 1
    (1 <= h_idx <= n_homs) || return nothing
    off1 = dom_off + (v1 - 1) * NM
    off2 = dom_off + (v2 - 1) * NM
    off_h = Int(hf_offs[h_idx])
    n_elems_h = (Int(hf_offs[h_idx + 1]) - off_h) ÷ nc
    c = tid
    while c <= nc
        oldv1 = dom[off1 + c]
        nv1   = new_d1_sh[c]
        if nv1 != oldv1
            dom[off1 + c] = nv1
            changed_flag[1] = true
        end
        reach = UInt64(0)
        for cc in 1:nc
            ch = new_d1_sh[cc]
            while ch != UInt64(0)
                lsb = ch & (-ch); ch &= ~lsb
                bi  = trailing_zeros(lsb)
                w   = (cc - 1) * 64 + bi + 1
                if w <= n_elems_h
                    reach |= hf_flat[off_h + (w - 1) * nc + c]
                end
            end
        end
        oldv2 = dom[off2 + c]
        nv2   = oldv2 & reach
        if nv2 != oldv2
            dom[off2 + c] = nv2
            changed_flag[1] = true
        end
        c += blocksize
    end
    nothing
end

# Single-bytecode PROP_NEQ / PROP_EQ on the calling lane (cheap, O(nc), no hom
# scan).  Caller runs it under `tid == 1` and barriers after.
@inline function _intra_simple!(dom, dom_off, changed_flag, bc, nc,
                                ::Val{NM}) where {NM}
    v1 = Int(bc.var1); v1 == 0 && return nothing; v2 = Int(bc.var2)
    off1 = dom_off + (v1 - 1) * NM
    off2 = dom_off + (v2 - 1) * NM
    if bc.op == PROP_NEQ
        ones2 = 0
        for c in 1:nc; ones2 += count_ones(dom[off2 + c]); end
        if ones2 == 1
            for c in 1:nc
                o = dom[off1 + c]; nv = o & ~dom[off2 + c]
                if nv != o; dom[off1 + c] = nv; changed_flag[1] = true; end
            end
        end
        ones1 = 0
        for c in 1:nc; ones1 += count_ones(dom[off1 + c]); end
        if ones1 == 1
            for c in 1:nc
                o = dom[off2 + c]; nv = o & ~dom[off1 + c]
                if nv != o; dom[off2 + c] = nv; changed_flag[1] = true; end
            end
        end
    elseif bc.op == PROP_EQ
        for c in 1:nc
            o1 = dom[off1 + c]; o2 = dom[off2 + c]; n = o1 & o2
            if n != o1; dom[off1 + c] = n; changed_flag[1] = true; end
            if n != o2; dom[off2 + c] = n; changed_flag[1] = true; end
        end
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
(nc_max) and `Val{NVNM16}` (7 × nc_max × 16; one specialization per nc_max, covering n_vars ≤ 7).

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
    ::Val{PROP_MODE},
) where {NM, NVNM16, PROP_MODE}

    tid       = @index(Local, Linear)   # 1-based thread within block
    blocksize = @groupsize()[1]
    n_vars_nm = n_vars * NM             # words per DFS level

    # ── Shared memory ─────────────────────────────────────────────────────────
    dom      = @localmem UInt64 (NVNM16,)  # domain workspace: 16 DFS levels
    stack_v  = @localmem Int32  (16,)      # branching variable at each solve-phase level
    stnext   = @localmem Int32  (16,)      # next candidate element index at each level
    ok_flag  = @localmem Bool   (1,)
    changed_flag = @localmem Bool (1,)  # parallel AC-1 fixpoint change flag
    new_d1_sh = @localmem UInt64 (NM,)  # intra-bytecode: shared forward-narrowed v1 domain
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
            # AC-1 propagation fixpoint. PROP_MODE: 0=serial (default), 1=block-
            # parallel over bytecodes (RG_BLOCK_PROP), 2=intra-bytecode parallel
            # over nc domain chunks (RG_INTRA_PROP).
            if PROP_MODE == 2
                for _ in 1:(n_bc * 8 + 1)
                    if tid == 1; changed_flag[1] = false; end
                    @synchronize()
                    for i in 1:n_bc
                        bc = bytecodes[i]
                        if bc.op == PROP_FUNC
                            _intra_forward!(tid, blocksize, dom, 0, new_d1_sh, bc,
                                            nc, hf_flat, hf_offs, Val(NM))
                        end
                        @synchronize()
                        if bc.op == PROP_FUNC
                            _intra_backward!(tid, blocksize, dom, 0, new_d1_sh,
                                             changed_flag, bc, nc, hf_flat, hf_offs, Val(NM))
                        elseif tid == 1
                            _intra_simple!(dom, 0, changed_flag, bc, nc, Val(NM))
                        end
                        @synchronize()
                    end
                    changed_flag[1] || break
                end
            elseif PROP_MODE == 1
                for _ in 1:(n_bc * 8 + 1)
                    if tid == 1; changed_flag[1] = false; end
                    @synchronize()
                    _propagate_block!(tid, blocksize, dom, 0, changed_flag,
                                      bytecodes, n_bc, n_vars, nc, hf_flat, hf_offs, Val(NM))
                    @synchronize()
                    changed_flag[1] || break
                end
            elseif tid == 1
                _propagate_serial!(dom, 0, bytecodes, n_bc, n_vars, nc,
                                   hf_flat, hf_offs, Val(NM))
            end

            # Consistency check, unbound-var detection, and domain pin (tid 1).
            if tid == 1

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
                    # AC-1 propagation fixpoint. PROP_MODE: 0=serial, 1=block-
                    # parallel (RG_BLOCK_PROP), 2=intra-bytecode (RG_INTRA_PROP).
                    if PROP_MODE == 2
                        for _ in 1:(n_bc * 8 + 1)
                            if tid == 1; changed_flag[1] = false; end
                            @synchronize()
                            for i in 1:n_bc
                                bc = bytecodes[i]
                                if bc.op == PROP_FUNC
                                    _intra_forward!(tid, blocksize, dom, lev_off, new_d1_sh, bc,
                                                    nc, hf_flat, hf_offs, Val(NM))
                                end
                                @synchronize()
                                if bc.op == PROP_FUNC
                                    _intra_backward!(tid, blocksize, dom, lev_off, new_d1_sh,
                                                     changed_flag, bc, nc, hf_flat, hf_offs, Val(NM))
                                elseif tid == 1
                                    _intra_simple!(dom, lev_off, changed_flag, bc, nc, Val(NM))
                                end
                                @synchronize()
                            end
                            changed_flag[1] || break
                        end
                    elseif PROP_MODE == 1
                        for _ in 1:(n_bc * 8 + 1)
                            if tid == 1; changed_flag[1] = false; end
                            @synchronize()
                            _propagate_block!(tid, blocksize, dom, lev_off, changed_flag,
                                              bytecodes, n_bc, n_vars, nc, hf_flat, hf_offs, Val(NM))
                            @synchronize()
                            changed_flag[1] || break
                        end
                    elseif tid == 1
                        _propagate_serial!(dom, lev_off, bytecodes, n_bc, n_vars, nc,
                                           hf_flat, hf_offs, Val(NM))
                    end

                    if tid == 1

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

# AC-1 propagation mode for turbo_block_kernel!.  2 = intra-bytecode parallel over
# the nc domain chunks (DEFAULT) — each PROP_FUNC's chunk scan is split across the
# 32 lanes; ~3x faster on game_full (median 9.3 vs 27.4 s/turn, A40), whose solves
# have few bytecodes (n_bc=2-9) but huge domains (nc=45).  0 = single-thread serial
# (`RG_SERIAL_PROP`, the reference / kill-switch).  1 = block-parallel over bytecodes
# (`RG_BLOCK_PROP`) — a ~6-8% regression on game_full (too little width), kept for
# many-bytecode workloads.
_prop_mode() = haskey(ENV, "RG_SERIAL_PROP") ? 0 :
               (haskey(ENV, "RG_BLOCK_PROP") ? 1 : 2)

# BIGVAR routing: patterns with 8–14 variables go to the shared-memory block
# kernel whenever its @localmem workspace fits the 48 KB per-block limit, i.e.
# nc_max ≤ 16 (14×16×128+256 = 28 928 B; at nc_max ≥ 32 the 14-var band needs
# > 48 KB, so those patterns stay on the EPS pipeline).  The kernel body already
# supports this: indexing is runtime (n_vars_nm = n_vars*NM), the DFS stack
# holds 16 levels, and branch depth ≤ n_vars ≤ 14 < 16.  Exactly TWO NVNM16
# workspace shapes exist per nc_max bucket — 7*nc_max*16 (legacy band, keeps
# existing compiled specializations valid) and 14*nc_max*16 — bounding the
# kernel JIT tax (AGENT.md §JIT) to one extra cheap (nc_max ≤ 16) variant.
# RG_NO_BIGVAR restores the historical n_vars > 7 → EPS routing.
const _NV_BIG = 14
_nv_cap(nc_max::Int) =
    haskey(ENV, "RG_NO_BIGVAR") ? 7 :
    (_NV_BIG * nc_max * 128 + 256 <= 49152 ? _NV_BIG : 7)
_nvnm16_for(n_vars::Int, nc_max::Int) = (n_vars <= 7 ? 7 : _NV_BIG) * nc_max * 16
# Single source of truth for the EPS-vs-turbo_block routing decision, shared by
# both dispatch sites below and by the anchored-decomposition eligibility check
# in the Scheduler (which decomposes exactly the solves that would go to EPS).
function _would_use_eps(n_vars::Int, nc::Int)
    nc_max = _select_nc_max(nc)
    nc == 1 || n_vars > _nv_cap(nc_max) ||
        (n_vars <= 7 ? 7 : _NV_BIG) * nc_max * 128 + 256 > 49152
end

# Private helper: dispatch Val{nc_max}, Val{nvnm16}, Val{prop_mode} for
# turbo_block_kernel!.  Called from gpu_turbo_solve and _gpu_turbo_fill_scratch!.
function _launch_turbo_block!(backend,
                               d_gpu, b_gpu,
                               n_bc::Int, n_vars::Int, nc::Int, D::Int,
                               nextsub, sol_gpu, cnt_gpu, max_solutions::Int,
                               hf_flat, hf_offs,
                               nc_max::Int, nvnm16::Int, n_blks::Int,
                               prop_mode::Int = 0)
    haskey(ENV, "RG_SOLVE_DIAG") &&
        println(stderr, "TBLK nbc=$n_bc nv=$n_vars nc=$nc D=$D nblk=$n_blks")
    turbo_block_kernel!(backend, 32)(
        d_gpu, b_gpu, n_bc, n_vars, nc, D,
        nextsub, sol_gpu, cnt_gpu, max_solutions,
        hf_flat, hf_offs, Val(nc_max), Val(nvnm16), Val(prop_mode);
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
    haskey(ENV, "RG_SOLVE_DIAG") &&
        println(stderr, "TEPS nbc=$n_bc nv=$n_vars nc=$nc ncmax=$nc_max")
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

    # NOTE: an earlier "CPU AC fast-fail" downloaded d_gpu + hf_flat_gpu and ran
    # cpu_propagate! here to skip the GPU kernel for infeasible CSPs.  That broke
    # the "everything stays on the GPU" invariant (a GPU→CPU copy on the hot
    # path, once per solve).  It is removed: the turbo/EPS kernels run AC-1
    # propagation themselves and return zero solutions for infeasible CSPs, so
    # correctness is unchanged.  If infeasible-slot throughput ever needs a
    # pre-filter, add a GPU-resident feasibility kernel (no host round-trip).

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

    # Use EPS (global-memory) when nc==1, n_vars exceeds the routing cap
    # (7 legacy / 14 with BIGVAR, see _nv_cap), or when the block-kernel smem
    # workspace for the pattern's band would exceed the 48 KB CUDA per-block
    # limit: band_cap*nc_max*128 + ~256 bytes for stack_v/stnext/flags.
    # nc_max=48, n_vars≤7: 7×48×128+256 = 43264 < 49152 → turbo_block.
    # nc_max=48, n_vars=8–14 or nc_max=64: > 49152 → EPS.
    if _would_use_eps(n_vars, nc)
        # EPS pipeline: one thread per domain element, global-memory workspace.
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
        # Multi-block Turbo: block-cooperative AC-1 in shared memory.
        D      = clamp(ceil(Int, log2(max(nc * 64, 2))), 4, 14)
        n_blks = min(1 << D, 576)
        # Per-band NVNM16 (7 or 14 vars' worth of workspace) so all n_vars in a
        # band share one Val{NVNM16} specialization per nc_max.  @localmem
        # over-allocates slightly within a band; actual indexing uses
        # n_vars_nm = n_vars*NM so no out-of-bounds.
        nvnm16 = _nvnm16_for(n_vars, nc_max)
        if scratch !== nothing
            sub_gpu = scratch.buf_turbo_nextsub
            KernelAbstractions.fill!(sub_gpu, Int32(0))
        else
            sub_gpu = KernelAbstractions.allocate(backend, Int32, 1)
            KernelAbstractions.fill!(sub_gpu, Int32(0))
        end
        _launch_turbo_block!(backend, d_gpu, b_gpu, n_bc, n_vars, nc, D,
                              sub_gpu, sol_gpu, cnt_gpu, max_solutions,
                              hf_flat_gpu, hf_offs_gpu, nc_max, nvnm16, n_blks,
                              _prop_mode())
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

    # Same routing as gpu_turbo_solve: EPS when nc==1, n_vars over the cap
    # (7 legacy / 14 BIGVAR), or the pattern's band workspace exceeds 48 KB smem.
    if _would_use_eps(n_vars, nc)
        # EPS pipeline: one thread per domain element, global-memory workspace.
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
        # Multi-block Turbo: block-cooperative AC-1 in shared memory.
        D      = clamp(ceil(Int, log2(max(nc * 64, 2))), 4, 14)
        n_blks = min(1 << D, 576)
        nvnm16 = _nvnm16_for(n_vars, nc_max)  # per-band shape (7 or 14 vars), see _nv_cap
        sub_gpu = scratch.buf_turbo_nextsub
        KernelAbstractions.fill!(sub_gpu, Int32(0))
        _launch_turbo_block!(backend, d_gpu, b_gpu, n_bc, n_vars, nc, D,
                              sub_gpu, sol_gpu, cnt_gpu, max_solutions,
                              hf_flat_gpu, hf_offs_gpu, nc_max, nvnm16, n_blks,
                              _prop_mode())
    end
    KernelAbstractions.synchronize(backend)
    CUDA.@allowscalar min(Int(scratch.buf_sol_count[1]), max_solutions)
end

"""
    _gpu_turbo_sample_scratch!(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu;
                               take, seed, oversample, scratch) -> Int

Scratch-buffer "take N" sampler for the GPU player fast path.  Fills
`scratch.buf_solutions[:, 1:n]` with up to `take` DISTINCT valid solutions produced
by `sample_descent_kernel!` (count-weighted random descent) and returns `n` (≤take).
Mirrors `_gpu_turbo_fill_scratch!`; deduplication uses a tiny host round-trip over
the K = take·oversample per-thread output columns (cheap because `take` is small
when sampling is active).  Only valid for rules with no NAC/PAC conditions (same
guard as the fill-scratch fast path).
"""
function _gpu_turbo_sample_scratch!(backend, csp::CSPProblem,
                                    d_gpu, hf_flat_gpu, hf_offs_gpu;
                                    take::Int, seed::Integer = 0,
                                    oversample::Int = 4, scratch)::Int
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    nc     = csp.n_chunks
    nc_max = _select_nc_max(nc)

    if n_vars == 0
        KernelAbstractions.fill!(scratch.buf_sol_count, Int32(1))
        return 1
    end
    take <= 0 && return 0

    # Pattern deeper than the fixed 16-level thread stack: fall back to the full
    # solver (take ignored for these rare large patterns; none occur in Falcon).
    if n_vars + 1 > 16
        return _gpu_turbo_fill_scratch!(backend, csp, d_gpu, hf_flat_gpu, hf_offs_gpu;
                                        scratch = scratch)
    end

    max_levels = n_vars + 1
    K = min(max(take * oversample, take + 8), size(scratch.buf_solutions, 2))

    b_gpu = scratch.buf_bytecodes
    n_bc > 0 && KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)

    # Grow per-thread descent workspace if needed.
    ws_rows = n_vars * nc_max
    ws_cols = K * (max_levels + 1)
    if size(scratch.buf_sample_ws, 1) < ws_rows || size(scratch.buf_sample_ws, 2) < ws_cols
        scratch.buf_sample_ws = CUDA.zeros(UInt64, ws_rows, ws_cols)
    end

    sol_view = view(scratch.buf_solutions, 1:n_vars, 1:K)
    KernelAbstractions.fill!(sol_view, Int32(0))      # failed threads stay all-zero
    ws_view  = view(scratch.buf_sample_ws, 1:ws_rows, 1:ws_cols)

    sample_descent_kernel!(backend)(
        d_gpu, b_gpu, n_bc, n_vars, nc, max_levels, K, UInt64(seed),
        sol_view, ws_view, hf_flat_gpu, hf_offs_gpu, Val(nc_max); ndrange = K)
    KernelAbstractions.synchronize(backend)

    # Dedup the K sampled columns on host (tiny: K·n_vars Int32), compact to ≤ take.
    host     = Array(sol_view)                        # [n_vars × K]
    distinct = Vector{Int32}[]
    seen     = Set{Tuple}()
    for j in 1:K
        host[1, j] == Int32(0) && continue
        col = host[:, j]
        key = Tuple(col)
        key in seen && continue
        push!(seen, key); push!(distinct, col)
        length(distinct) >= take && break
    end
    n = length(distinct)
    n == 0 && return 0
    out = reduce(hcat, distinct)                      # [n_vars × n]
    KernelAbstractions.copyto!(backend, view(scratch.buf_solutions, 1:n_vars, 1:n), out)
    KernelAbstractions.synchronize(backend)
    return n
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
