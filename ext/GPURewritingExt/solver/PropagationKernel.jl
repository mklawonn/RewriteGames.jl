"""
Turbo AC-1 propagation kernel.

Each GPU thread handles one `TCNBytecode` packet.  A warp cooperates on one
CSP instance (one potential match being explored).  The kernel iterates until
no domain shrinks (fixed point), returning the domains for the dive-solve
phase.

Domain representation: `UInt64` bit-mask — bit `k` is set if world element
`k+1` is still in the domain of that variable (max 64 elements per type; see
`LargeDomainPropagation` for the sparse fallback when types exceed 64 parts).

For the host-side (CPU) solver used in tests without GPU hardware, the same
logic is implemented in `cpu_propagate!` below.
"""

# ── GPU kernel (KernelAbstractions) ─────────────────────────────────────────

@kernel function propagation_kernel!(
    domains   :: AbstractMatrix{UInt64},   # [n_vars × n_instances]
    bytecodes :: AbstractVector{TCNBytecode},
    n_bc      :: Int,
    changed   :: AbstractVector{Bool}      # [n_instances] — any domain shrank?
)
    inst = @index(Global, Linear)
    inst > size(domains, 2) && return

    # Use a local changed flag; write back once.
    local_changed = false

    for _ in 1:n_bc                        # AC-1: repeat until fixed point
        made_progress = false
        for bc_idx in 1:n_bc
            bc  = bytecodes[bc_idx]
            op  = bc.op
            v1  = Int(bc.var1)
            v2  = Int(bc.var2)
            p1  = Int(bc.param1)
            p2  = Int(bc.param2)

            v1 == 0 && continue            # unused field

            old = domains[v1, inst]

            if op == PROP_EQ
                # v1 and v2 must agree — intersect domains
                v2 == 0 && continue
                domains[v1, inst] &= domains[v2, inst]
                domains[v2, inst] &= old

            elseif op == PROP_NEQ
                # monic: if v2 is fixed (single bit) remove that value from v1
                v2 == 0 && continue
                d2 = domains[v2, inst]
                if count_ones(d2) == 1
                    domains[v1, inst] &= ~d2
                end
                d1 = domains[v1, inst]
                if count_ones(d1) == 1
                    domains[v2, inst] &= ~d1
                end

            elseif op == PROP_ATTR_EQ
                # v1 must map to world element whose attr column p1 == p2.
                # The kernel has no world access — attr constraints are applied
                # as domain masks pre-loaded before this kernel runs.
                # (This bytecode is a no-op here; domains already pre-masked.)
                nothing

            elseif op == DOMAIN_SIZE
                # Clamp domain to valid world element range [1..p1]
                if p1 < 64
                    mask = (UInt64(1) << p1) - UInt64(1)
                    domains[v1, inst] &= mask
                end
            end

            domains[v1, inst] != old && (made_progress = true)
        end
        made_progress || break
    end

    # Check for empty domain (failure)
    for v in 1:size(domains, 1)
        domains[v, inst] == UInt64(0) && (changed[inst] = false; return)
    end
    changed[inst] = true   # instance survived propagation
end

# ── CPU fallback (used in tests without GPU) ─────────────────────────────────

"""
    cpu_propagate!(domains, bytecodes) -> Bool

AC-1 propagation on host.  `domains` is a `Vector{UInt64}` of length n_vars.
Returns `true` if the domains are consistent (none empty), `false` on failure.
"""
function cpu_propagate!(domains::Vector{UInt64},
                        bytecodes::Vector{TCNBytecode})::Bool
    n  = length(domains)
    for _ in 1:(n + length(bytecodes))    # bounded AC-1 iterations
        made_progress = false
        for bc in bytecodes
            op = bc.op; v1 = Int(bc.var1); v2 = Int(bc.var2)
            p1 = Int(bc.param1); p2 = Int(bc.param2)
            v1 == 0 && continue
            old = domains[v1]

            if op == PROP_EQ
                v2 == 0 && continue
                domains[v1] &= domains[v2]
                domains[v2] &= old

            elseif op == PROP_NEQ
                v2 == 0 && continue
                d2 = domains[v2]
                count_ones(d2) == 1 && (domains[v1] &= ~d2)
                d1 = domains[v1]
                count_ones(d1) == 1 && (domains[v2] &= ~d1)

            elseif op == DOMAIN_SIZE
                if p1 < 64
                    mask = (UInt64(1) << p1) - UInt64(1)
                    domains[v1] &= mask
                end
            end

            domains[v1] != old && (made_progress = true)
        end
        made_progress || break
    end
    return all(d -> d != UInt64(0), domains)
end
