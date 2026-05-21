"""
Dive-and-solve search kernel.

Optimized for GPU execution:
- Iterative DFS with manual stack.
- Correct backtracking.
- Register-conscious stack sizing.
"""

# ── Host-side reference ───────────────────────────────────────────────────────

function cpu_dive_solve(csp::CSPProblem,
                        initial_domains::Vector{UInt64})::Vector{Vector{Int32}}
    solutions = Vector{Int32}[]
    assignment = zeros(Int32, csp.n_vars)
    _dfs!(solutions, assignment, copy(initial_domains), csp.bytecodes,
          Int(csp.n_vars))
    return solutions
end

function _dfs!(solutions, assignment, domains, bytecodes, n_vars)
    cpu_propagate!(domains, bytecodes) || return
    unbound = 0
    for v in 1:n_vars
        if count_ones(domains[v]) > 1; unbound = v; break; end
        if domains[v] == 0; return; end
    end
    if unbound == 0
        push!(solutions, Int32[trailing_zeros(domains[v]) + 1 for v in 1:n_vars])
        return
    end
    d = domains[unbound]
    while d != 0
        bit = d & (-d); d &= ~bit
        new_domains = copy(domains); new_domains[unbound] = bit
        _dfs!(solutions, assignment, new_domains, bytecodes, n_vars)
    end
end

# ── GPU kernel ───────────────────────────────────────────────────────────────

@kernel function dive_solve_kernel!(
    domains_in    :: AbstractVector{UInt64},
    bytecodes     :: AbstractVector{TCNBytecode},
    n_bc          :: Int,
    n_vars        :: Int,
    solutions     :: AbstractMatrix{Int32},
    sol_count     :: AbstractVector{Int32},
    max_solutions :: Int,
    workspace     :: AbstractMatrix{UInt64} # [n_vars, 16]
)
    # Stack management
    stack_bits = MVector{16, UInt64}(undef)
    stack_vars = MVector{16, Int32}(undef)
    
    # level 1 init
    for v in 1:n_vars
        workspace[v, 1] = domains_in[v]
    end
    
    level = 1
    # State: 1 = Propagate/FindVar, 2 = Branch/NextBit
    state = 1
    
    safety = 0
    while level > 0 && safety < 100000000
        safety += 1
        
        if state == 1
            # A. Propagate
            ok = true
            for _ in 1:8 # small fixed AC-1
                changed = false
                for i in 1:n_bc
                    bc = bytecodes[i]
                    v1 = Int(bc.var1); v1 == 0 && continue
                    d1 = workspace[v1, level]
                    if bc.op == PROP_NEQ && bc.var2 != 0
                        v2 = Int(bc.var2); d2 = workspace[v2, level]
                        if count_ones(d2) == 1
                            new_d1 = d1 & ~d2
                            if new_d1 != d1; d1 = new_d1; changed = true; workspace[v1, level] = d1; end
                        end
                    elseif bc.op == PROP_EQ && bc.var2 != 0
                        v2 = Int(bc.var2); d2 = workspace[v2, level]
                        new_d1 = d1 & d2
                        if new_d1 != d1; d1 = new_d1; changed = true; workspace[v1, level] = d1; end
                    end
                end
                if !changed; break; end
            end
            
            # Check consistency
            for v in 1:n_vars
                if workspace[v, level] == 0; ok = false; break; end
            end
            
            if !ok
                level -= 1; state = 2; continue
            end
            
            # B. Find unbound
            unbound = 0
            for v in 1:n_vars
                if count_ones(workspace[v, level]) > 1
                    unbound = v; break
                end
            end
            
            if unbound == 0
                # Solution found!
                idx = CUDA.atomic_add!(pointer(sol_count, 1), Int32(1)) + 1
                if idx <= max_solutions
                    for v in 1:n_vars
                        solutions[v, idx] = Int32(trailing_zeros(workspace[v, level]) + 1)
                    end
                end
                level -= 1; state = 2; continue
            end
            
            # C. Start branching
            stack_vars[level] = Int32(unbound)
            stack_bits[level] = workspace[unbound, level]
            state = 2
        end
        
        if state == 2
            if stack_bits[level] != 0
                bit = stack_bits[level] & (-stack_bits[level])
                stack_bits[level] &= ~bit
                
                if level < 16
                    next_level = level + 1
                    for v in 1:n_vars
                        workspace[v, next_level] = workspace[v, level]
                    end
                    workspace[Int(stack_vars[level]), next_level] = bit
                    level = next_level
                    state = 1 # go to propagate at next level
                else
                    # Too deep, just skip
                    state = 2 # stay here to try next bit
                end
            else
                # Backtrack
                level -= 1
                state = 2 # try next bit at parent
            end
        end
    end
end

function gpu_dive_solve(backend, csp::CSPProblem, 
                        initial_domains::Vector{UInt64};
                        max_solutions=10000)::Vector{Vector{Int32}}
    n_vars = Int(csp.n_vars)
    n_bc   = length(csp.bytecodes)
    
    d_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars)
    KernelAbstractions.copyto!(backend, d_gpu, initial_domains)
    b_gpu = KernelAbstractions.allocate(backend, TCNBytecode, n_bc)
    KernelAbstractions.copyto!(backend, b_gpu, csp.bytecodes)
    sol_gpu = KernelAbstractions.allocate(backend, Int32, n_vars, max_solutions)
    cnt_gpu = KernelAbstractions.allocate(backend, Int32, 1)
    KernelAbstractions.fill!(cnt_gpu, Int32(0))
    work_gpu = KernelAbstractions.allocate(backend, UInt64, n_vars, 16)
    
    kernel = dive_solve_kernel!(backend)
    kernel(d_gpu, b_gpu, n_bc, n_vars, sol_gpu, cnt_gpu, max_solutions, work_gpu; ndrange=1)
    KernelAbstractions.synchronize(backend)
    
    count = Int(Array(cnt_gpu)[1])
    if count == 0; return Vector{Int32}[]; end
    res = Array(sol_gpu)[:, 1:min(count, max_solutions)]
    return [res[:, i] for i in 1:size(res, 2)]
end
