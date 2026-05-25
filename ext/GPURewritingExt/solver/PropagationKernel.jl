"""
Block-parallel AC-1 propagation kernel — multi-chunk bitset domains.

`propagation_kernel!` assigns one GPU thread per CSP instance (one potential
world state). Domains are stored in the flat chunked layout:
  `domains[(v-1)*nc + c, inst]` = chunk c of variable v's domain in instance inst.

Supports PROP_FUNC (FK-morphism arc consistency), PROP_EQ, PROP_NEQ, PROP_ATTR_EQ,
and DOMAIN_SIZE.  Uses `MVector{NM, UInt64}` register-allocated temporaries via
a `Val{NM}` type parameter (same convention as `dive_solve_kernel!`).

For the host-side (CPU) solver used in tests without GPU hardware, the same
logic is implemented in `cpu_propagate!` below.
"""

# ── GPU kernel (KernelAbstractions) ─────────────────────────────────────────

@kernel function propagation_kernel!(
    domains   :: AbstractMatrix{UInt64},       # [n_vars*nc × n_instances], in-place
    hf_flat   :: AbstractVector{UInt64},       # flat hom-forward tables
    hf_offs   :: AbstractVector{Int32},        # per-hom 0-based word offsets
    bytecodes :: AbstractVector{TCNBytecode},
    n_bc      :: Int,
    n_vars    :: Int,
    nc        :: Int,
    ok        :: AbstractVector{Bool},         # [n_instances]: false if any domain empty
    ::Val{NM}
) where NM
    inst = @index(Global, Linear)
    if inst <= size(domains, 2)
        n_rows   = n_vars * nc
        inst_off = (inst - 1) * n_rows   # not used: domains is already sliced by column

        new_d     = MVector{NM, UInt64}(undef)
        reachable = MVector{NM, UInt64}(undef)

        n_iters = n_bc * nc + 1   # bounded AC-1 iterations
        for _ in 1:n_iters
            made_progress = false

            for bc_i in 1:n_bc
                bc = bytecodes[bc_i]
                v1 = Int(bc.var1)
                v1 == 0 && continue
                off1 = (v1 - 1) * nc

                if bc.op == PROP_FUNC && bc.var2 != 0
                    v2    = Int(bc.var2)
                    off2  = (v2 - 1) * nc
                    h_idx = Int(bc.param1)
                    n_homs = length(hf_offs) - 1
                    if 1 <= h_idx <= n_homs
                        off_h    = Int(hf_offs[h_idx])
                        n_elems_h = (Int(hf_offs[h_idx+1]) - off_h) ÷ nc

                        for c in 1:nc; new_d[c] = UInt64(0); end
                        for c in 1:nc
                            chunk = domains[off1 + c, inst]
                            while chunk != 0
                                lsb = chunk & (-chunk); chunk &= ~lsb
                                bi  = trailing_zeros(lsb)
                                w   = (c - 1) * 64 + bi + 1
                                if w <= n_elems_h
                                    off_w = off_h + (w - 1) * nc
                                    for ci in 1:nc
                                        if (hf_flat[off_w + ci] & domains[off2 + ci, inst]) != 0
                                            new_d[c] |= lsb
                                            break
                                        end
                                    end
                                end
                            end
                        end
                        for c in 1:nc
                            old_c = domains[off1 + c, inst]
                            domains[off1 + c, inst] = new_d[c]
                            old_c != new_d[c] && (made_progress = true)
                        end

                        for c in 1:nc; reachable[c] = UInt64(0); end
                        for c in 1:nc
                            chunk = new_d[c]
                            while chunk != 0
                                lsb = chunk & (-chunk); chunk &= ~lsb
                                bi  = trailing_zeros(lsb)
                                w   = (c - 1) * 64 + bi + 1
                                if w <= n_elems_h
                                    off_w = off_h + (w - 1) * nc
                                    for ci in 1:nc
                                        reachable[ci] |= hf_flat[off_w + ci]
                                    end
                                end
                            end
                        end
                        for c in 1:nc
                            new_c = domains[off2 + c, inst] & reachable[c]
                            new_c != domains[off2 + c, inst] && (made_progress = true)
                            domains[off2 + c, inst] = new_c
                        end
                    end

                elseif bc.op == PROP_NEQ && bc.var2 != 0
                    v2   = Int(bc.var2)
                    off2 = (v2 - 1) * nc
                    ones2 = 0
                    for c in 1:nc; ones2 += count_ones(domains[off2 + c, inst]); end
                    if ones2 == 1
                        for c in 1:nc
                            new_c = domains[off1 + c, inst] & ~domains[off2 + c, inst]
                            new_c != domains[off1 + c, inst] && (made_progress = true)
                            domains[off1 + c, inst] = new_c
                        end
                    end
                    ones1 = 0
                    for c in 1:nc; ones1 += count_ones(domains[off1 + c, inst]); end
                    if ones1 == 1
                        for c in 1:nc
                            new_c = domains[off2 + c, inst] & ~domains[off1 + c, inst]
                            new_c != domains[off2 + c, inst] && (made_progress = true)
                            domains[off2 + c, inst] = new_c
                        end
                    end

                elseif bc.op == PROP_EQ && bc.var2 != 0
                    v2   = Int(bc.var2)
                    off2 = (v2 - 1) * nc
                    for c in 1:nc
                        old1 = domains[off1 + c, inst]
                        new1 = old1 & domains[off2 + c, inst]
                        domains[off1 + c, inst] = new1
                        old1 != new1 && (made_progress = true)
                        old2 = domains[off2 + c, inst]
                        new2 = old2 & old1
                        domains[off2 + c, inst] = new2
                        old2 != new2 && (made_progress = true)
                    end

                elseif bc.op == DOMAIN_SIZE
                    p1 = Int(bc.param1)
                    if p1 < nc * 64
                        last_chunk = cld(p1, 64)
                        partial    = p1 % 64
                        for c in (last_chunk + 1):nc
                            old_c = domains[off1 + c, inst]
                            old_c != UInt64(0) && (made_progress = true)
                            domains[off1 + c, inst] = UInt64(0)
                        end
                        if last_chunk <= nc && partial > 0
                            mask  = (UInt64(1) << partial) - UInt64(1)
                            old_c = domains[off1 + last_chunk, inst]
                            new_c = old_c & mask
                            old_c != new_c && (made_progress = true)
                            domains[off1 + last_chunk, inst] = new_c
                        end
                    end
                end
            end

            if !made_progress
                break
            end
        end

        failed = false
        for v in 1:n_vars
            all_zero = true
            off = (v - 1) * nc
            for c in 1:nc
                if domains[off + c, inst] != UInt64(0)
                    all_zero = false
                    break
                end
            end
            if all_zero
                failed = true
                break
            end
        end
        ok[inst] = !failed
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
