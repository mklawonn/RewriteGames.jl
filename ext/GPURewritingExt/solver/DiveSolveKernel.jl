"""
Dive-and-solve search kernel.

After AC-1 propagation leaves some variables with more than one value in
their domain, this kernel explores the search tree by repeatedly picking the
first unbound variable and branching on each candidate value.  The search is
embarrassingly parallel: each GPU thread (or warp) independently explores one
branch, writing complete assignments to the solutions buffer.

For the host-side solver, `cpu_dive_solve!` implements the same algorithm
recursively.  It is used in tests without GPU hardware and as the ground-truth
reference for the equivalence tests.
"""

# ── Host-side recursive solver (used in tests + as reference) ────────────────

"""
    cpu_dive_solve(csp, initial_domains) -> Vector{Vector{Int32}}

Enumerate all valid assignments for `csp` starting from `initial_domains`.
Returns a vector of flat assignment arrays (one Int32 per variable, 1-based
world element index).

This is the CPU ground truth matched against the GPU Turbo engine in the
equivalence test suite.
"""
function cpu_dive_solve(csp::CSPProblem,
                        initial_domains::Vector{UInt64})::Vector{Vector{Int32}}
    solutions = Vector{Int32}[]
    assignment = zeros(Int32, csp.n_vars)
    _dfs!(solutions, assignment, copy(initial_domains), csp.bytecodes,
          Int(csp.n_vars))
    return solutions
end

function _dfs!(solutions, assignment, domains, bytecodes, n_vars)
    # Apply propagation
    cpu_propagate!(domains, bytecodes) || return

    # Find first unbound variable (domain size > 1)
    unbound = 0
    for v in 1:n_vars
        count_ones(domains[v]) > 1 && (unbound = v; break)
    end

    if unbound == 0
        # All variables uniquely bound — record solution
        sol = Int32[trailing_zeros(domains[v]) + Int32(1) for v in 1:n_vars]
        push!(solutions, sol)
        return
    end

    # Branch: try each value in domain of `unbound`
    d = domains[unbound]
    while d != UInt64(0)
        bit = d & (-d)              # lowest set bit
        val = trailing_zeros(bit) + 1
        d  &= d - UInt64(1)        # clear lowest bit

        new_domains        = copy(domains)
        new_domains[unbound] = bit  # fix this variable to `val`
        new_assign         = copy(assignment)
        new_assign[unbound] = Int32(val)

        _dfs!(solutions, new_assign, new_domains, bytecodes, n_vars)
    end
end

# ── GPU kernel (KernelAbstractions) ──────────────────────────────────────────

@kernel function dive_solve_kernel!(
    domains_in   :: AbstractMatrix{UInt64},   # [n_vars × n_instances] post-propagation
    bytecodes    :: AbstractVector{TCNBytecode},
    n_bc         :: Int,
    n_vars       :: Int,
    solutions    :: AbstractMatrix{Int32},    # [n_vars × max_solutions]  output
    sol_count    :: AbstractVector{Int32},    # [1] atomic solution counter
    max_solutions :: Int
)
    inst = @index(Global, Linear)
    if inst <= size(domains_in, 2)
        # Copy domains into thread-local registers (stack array)
        domains = MArray{Tuple{64}, UInt64}(undef)  # max 64 variables
        for v in 1:n_vars
            domains[v] = domains_in[v, inst]
        end

        _gpu_dfs_inline!(domains, bytecodes, n_bc, n_vars,
                         solutions, sol_count, max_solutions)
    end
end

function _gpu_dfs_inline!(domains, bytecodes, n_bc, n_vars,
                          solutions, sol_count, max_solutions)
    # Find first unbound variable
    unbound = 0
    for v in 1:n_vars
        count_ones(domains[v]) > 1 && (unbound = v; break)
    end

    if unbound == 0
        # Fully bound — write solution atomically
        idx = Atomix.@atomic sol_count[1] += Int32(1)
        idx > max_solutions && return
        for v in 1:n_vars
            solutions[v, idx] = Int32(trailing_zeros(domains[v]) + 1)
        end
        return
    end

    d = domains[unbound]
    while d != UInt64(0)
        bit = d & (-d)
        d  &= d - UInt64(1)

        # Clone domains and fix unbound → bit
        saved = MArray{Tuple{64}, UInt64}(undef)
        for v in 1:n_vars; saved[v] = domains[v]; end
        domains[unbound] = bit

        ok = true
        for _ in 1:(n_bc + n_vars)           # bounded AC-1 in kernel
            changed = false
            for bc_idx in 1:n_bc
                bc = bytecodes[bc_idx]
                v1 = Int(bc.var1); v1 == 0 && continue
                old = domains[v1]
                if bc.op == PROP_NEQ && bc.var2 != 0
                    v2 = Int(bc.var2)
                    d2 = domains[v2]
                    count_ones(d2) == 1 && (domains[v1] &= ~d2)
                    d1 = domains[v1]
                    count_ones(d1) == 1 && (domains[v2] &= ~d1)
                elseif bc.op == PROP_EQ && bc.var2 != 0
                    v2 = Int(bc.var2)
                    domains[v1] &= domains[v2]
                    domains[v2] &= old
                end
                domains[v1] != old && (changed = true)
            end
            # Check failures
            for v in 1:n_vars
                domains[v] == UInt64(0) && (ok = false; break)
            end
            !ok && break
            !changed && break
        end

        ok && _gpu_dfs_inline!(domains, bytecodes, n_bc, n_vars,
                               solutions, sol_count, max_solutions)

        # Restore domains for next branch
        for v in 1:n_vars; domains[v] = saved[v]; end
    end
end
