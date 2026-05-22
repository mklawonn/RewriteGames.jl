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
    if inst <= size(domains, 2)

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
        failed = false
        for v in 1:size(domains, 1)
            if domains[v, inst] == UInt64(0)
                failed = true
                break
            end
        end
        changed[inst] = !failed
    end
end

# ── CPU fallback (used in tests without GPU) ─────────────────────────────────

"""
    cpu_propagate!(domains, bytecodes, hom_forward, nc) -> Bool

AC-1 propagation on host.

`domains` is a flat `Vector{UInt64}` of length `n_vars * nc` where
`domains[(v-1)*nc + c]` is chunk `c` (1-based) of variable `v`'s domain.
`nc` defaults to 1 for backward compatibility.

Returns `true` if all variable domains are non-empty, `false` on failure.
"""
function cpu_propagate!(domains::Vector{UInt64},
                        bytecodes::Vector{TCNBytecode},
                        hom_forward::Vector{Vector{UInt64}} = Vector{UInt64}[],
                        nc::Int = 1)::Bool
    nv = length(domains) ÷ nc

    for _ in 1:(nv * nc + length(bytecodes))    # bounded AC-1 iterations
        made_progress = false
        for bc in bytecodes
            op = bc.op; v1 = Int(bc.var1); v2 = Int(bc.var2)
            p1 = Int(bc.param1); p2 = Int(bc.param2)
            v1 == 0 && continue

            off1 = (v1 - 1) * nc   # 0-based flat offset for var v1

            if op == PROP_FUNC
                v2 == 0 && continue
                h_idx = p1
                1 <= h_idx <= length(hom_forward) || continue
                hf  = hom_forward[h_idx]
                n_elems_h = length(hf) ÷ nc
                off2 = (v2 - 1) * nc

                # Forward: keep v1 elements w where hom(w) intersects domain(v2)
                new_d1 = zeros(UInt64, nc)
                for c in 1:nc
                    chunk = domains[off1 + c]
                    while chunk != 0
                        lsb = chunk & (-chunk); chunk &= ~lsb
                        bi  = trailing_zeros(lsb)
                        w   = (c - 1) * 64 + bi + 1
                        w > n_elems_h && continue
                        off_w = (w - 1) * nc
                        for ci in 1:nc
                            (hf[off_w + ci] & domains[off2 + ci]) != 0 &&
                                (new_d1[c] |= lsb; break)
                        end
                    end
                end

                changed_v1 = false
                for c in 1:nc
                    old_c = domains[off1 + c]
                    domains[off1 + c] = new_d1[c]
                    old_c != new_d1[c] && (changed_v1 = true)
                end

                # Backward: keep v2 elements reachable from new domain(v1)
                old2 = domains[off2+1:off2+nc]
                reachable = zeros(UInt64, nc)
                for c in 1:nc
                    chunk = new_d1[c]
                    while chunk != 0
                        lsb = chunk & (-chunk); chunk &= ~lsb
                        bi  = trailing_zeros(lsb)
                        w   = (c - 1) * 64 + bi + 1
                        w > n_elems_h && continue
                        off_w = (w - 1) * nc
                        for ci in 1:nc; reachable[ci] |= hf[off_w + ci]; end
                    end
                end
                changed_v2 = false
                for c in 1:nc
                    new_c = domains[off2 + c] & reachable[c]
                    new_c != domains[off2 + c] && (changed_v2 = true)
                    domains[off2 + c] = new_c
                end

                (changed_v1 || changed_v2) && (made_progress = true)
                continue

            elseif op == PROP_EQ
                v2 == 0 && continue
                off2 = (v2 - 1) * nc
                for c in 1:nc
                    old_c = domains[off1 + c]
                    domains[off1 + c] &= domains[off2 + c]
                    domains[off2 + c] &= old_c
                end

            elseif op == PROP_NEQ
                v2 == 0 && continue
                off2 = (v2 - 1) * nc
                ones2 = 0
                for c in 1:nc; ones2 += count_ones(domains[off2 + c]); end
                if ones2 == 1
                    for c in 1:nc; domains[off1 + c] &= ~domains[off2 + c]; end
                end
                ones1 = 0
                for c in 1:nc; ones1 += count_ones(domains[off1 + c]); end
                if ones1 == 1
                    for c in 1:nc; domains[off2 + c] &= ~domains[off1 + c]; end
                end

            elseif op == DOMAIN_SIZE
                # Clamp domain to valid range [1..p1]
                if p1 < nc * 64
                    last_chunk = cld(p1, 64)
                    partial    = p1 % 64
                    for c in (last_chunk + 1):nc
                        domains[off1 + c] = UInt64(0)
                    end
                    if last_chunk <= nc && partial > 0
                        domains[off1 + last_chunk] &= (UInt64(1) << partial) - UInt64(1)
                    end
                end
            end

            # Check if v1's domain changed (conservative: check all chunks)
            for c in 1:nc
                # domains[off1+c] was potentially modified above; we track progress
                # coarsely — PROP_FUNC continues, others fall through to here
            end
            made_progress = true   # conservative: re-run until stable
        end
        made_progress || break
    end

    # Check all domains non-empty
    for v in 1:nv
        off = (v - 1) * nc
        all_zero = true
        for c in 1:nc
            domains[off + c] != UInt64(0) && (all_zero = false; break)
        end
        all_zero && return false
    end
    return true
end
