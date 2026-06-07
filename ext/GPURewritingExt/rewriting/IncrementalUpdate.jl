"""
Incremental match update — GPU analog of `update_cache!` from
`src/engine/match_cache.jl`.

After a DPO rewrite, the match set for all other rules must be updated:
1. **Forward surviving matches**: existing matches whose images are all still
   active are forwarded. Matches that included a deleted element are dropped.
2. **Discover new matches**: run the Turbo solver restricted to sub-problems
   that include at least one of the newly added Δ⁺ elements.

B12 implementation: GPU-resident match table with GPU filter kernel and
GPU prefix-sum compaction.  `MatchTable` stores assignments in a flat
matrix; on CUDA-functional systems the GPU filter/compaction avoid all
GPU→CPU transfers except the scalar n_matches download.
"""

# ── GPU kernels ───────────────────────────────────────────────────────────────

@kernel function filter_surviving_matches_kernel!(
    keep         :: AbstractVector{Bool},    # [n_matches] output
    assignments  :: AbstractMatrix{Int32},   # [n_vars × max_matches]
    active_flat  :: AbstractVector{Bool},    # concatenated per-type active arrays
    type_offsets :: AbstractVector{Int32},   # per-var 0-based offset in active_flat
    n_vars       :: Int32,
    n_matches    :: Int32,
)
    m = @index(Global, Linear)
    if m <= Int(n_matches)
        ok = true
        for v in 1:Int(n_vars)
            g_elem = Int(assignments[v, m])
            if g_elem != 0
                off  = Int(type_offsets[v])
                slot = off + g_elem
                if slot < 1 || slot > length(active_flat) || !active_flat[slot]
                    ok = false
                    break
                end
            end
        end
        keep[m] = ok
    end
end

@kernel function scatter_matches_kernel!(
    dst       :: AbstractMatrix{Int32},  # [n_vars × max_matches]
    src       :: AbstractMatrix{Int32},  # [n_vars × max_matches]
    new_ids   :: AbstractVector{Int32},  # inclusive prefix-sum of keep (0 = deleted)
    keep      :: AbstractVector{Bool},
    n_vars    :: Int32,
    n_matches :: Int32,
)
    m = @index(Global, Linear)
    if m <= Int(n_matches) && keep[m]
        new_m = Int(new_ids[m])
        for v in 1:Int(n_vars)
            dst[v, new_m] = src[v, m]
        end
    end
end

# ── MatchTable ────────────────────────────────────────────────────────────────

"""
    MatchTable

Flat representation of a set of homomorphisms as a matrix of Int32 values.

`assignments[v, m]` = world element assigned to pattern variable `v` in match `m`.
`n_matches` = number of valid matches currently stored.

On CUDA-functional systems `assignments` is a `CuMatrix{Int32}`; on CPU it
is a `Matrix{Int32}`.
"""
mutable struct MatchTable
    assignments :: Any    # CuMatrix{Int32} on GPU, Matrix{Int32} on CPU
    n_matches   :: Int
    n_vars      :: Int
    max_matches :: Int
end

function MatchTable(n_vars::Int, max_matches::Int)
    if CUDA.functional()
        MatchTable(CUDA.zeros(Int32, n_vars, max_matches), 0, n_vars, max_matches)
    else
        MatchTable(zeros(Int32, n_vars, max_matches), 0, n_vars, max_matches)
    end
end

# ── GPU-native match compaction ───────────────────────────────────────────────

function _gpu_compact_matches!(table::MatchTable, keep_gpu, n_new::Int, backend)
    n_m    = table.n_matches
    n_vars = Int32(table.n_vars)
    n_m32  = Int32(n_m)

    keep_int = CuArray{Int32}(undef, n_m)
    @. keep_int = Int32(keep_gpu[1:n_m])
    new_ids = cumsum(keep_int)   # inclusive prefix-sum on GPU

    dst = CUDA.zeros(Int32, table.n_vars, table.max_matches)
    scatter_matches_kernel!(backend, 256)(
        dst, table.assignments, new_ids, keep_gpu, n_vars, n_m32; ndrange = n_m)
    KernelAbstractions.synchronize(backend)

    table.assignments = dst
    table.n_matches   = n_new
end

function _compact_matches!(table::MatchTable, keep::AbstractVector{Bool})
    host_asgn = table.assignments isa Matrix ? table.assignments : Array(table.assignments)
    src = 1; dst = 1
    while src <= table.n_matches
        if keep[src]
            dst != src && (host_asgn[:, dst] .= host_asgn[:, src])
            dst += 1
        end
        src += 1
    end
    table.n_matches = dst - 1
    if CUDA.functional()
        table.assignments = CuArray(host_asgn)
    else
        table.assignments = host_asgn
    end
end

# ── Build per-var type-offset array ──────────────────────────────────────────

function _var_type_offsets(csp::CSPProblem, schema::SchemaInfo,
                            g::GPUACSet)::Vector{Int32}
    n_vars  = Int(csp.n_vars)
    offsets = zeros(Int32, n_vars)
    cursor  = Int32(0)
    for o in schema.obj_types
        base = get(csp.var_offset, o, 0)
        n_alloc_o = Int32(g.n_alloc[o])
        if base == 0
            cursor += n_alloc_o
            continue
        end
        next_base = n_vars + 1
        for other in schema.obj_types
            ob = get(csp.var_offset, other, 0)
            ob > base && ob < next_base && (next_base = ob)
        end
        for v in base:(next_base - 1)
            v > n_vars && break
            offsets[v] = cursor
        end
        cursor += n_alloc_o
    end
    offsets
end

# ── Main update function ──────────────────────────────────────────────────────

"""
    incremental_match_update!(table, csp, cube, g, deleted_g, added_g, schema, enc)

Update `table` in-place after a rewrite.  Uses GPU filter + compaction when
CUDA is functional (no full GPU→CPU round-trip for surviving-match filter).
"""
function incremental_match_update!(table::MatchTable,
                                   csp::CSPProblem,
                                   cube::AdhesiveCube,
                                   g::GPUACSet,
                                   deleted_g::Dict{Symbol, Vector{Int32}},
                                   added_g::Dict{Symbol, Vector{Int32}},
                                   schema::SchemaInfo,
                                   enc::AttributeEncoder)
    n_m = table.n_matches
    n_m == 0 && return table

    n_vars  = Int32(csp.n_vars)
    backend = CUDA.functional() ? CUDA.CUDABackend() : CPU()

    # ── Step 1: filter surviving matches ──────────────────────────────────────
    if CUDA.functional() && n_m > 0
        active_flat_host = Bool[]
        for o in schema.obj_types
            n = g.n_alloc[o]
            n == 0 && continue
            append!(active_flat_host, Array(g.active[o])[1:n])
        end
        active_flat_gpu = CuArray(active_flat_host)
        offsets_gpu     = CuArray(_var_type_offsets(csp, schema, g))

        keep_gpu = CUDA.ones(Bool, n_m)
        filter_surviving_matches_kernel!(backend, 256)(
            keep_gpu,
            @view(table.assignments[:, 1:n_m]),
            active_flat_gpu, offsets_gpu,
            n_vars, Int32(n_m);
            ndrange = n_m)
        KernelAbstractions.synchronize(backend)

        n_new = Int(Array(CUDA.sum(keep_gpu))[1])
        n_new < n_m && _gpu_compact_matches!(table, keep_gpu, n_new, backend)
    else
        offsets_host = _var_type_offsets(csp, schema, g)
        host_active  = Dict(o => Array(g.active[o]) for o in schema.obj_types)
        host_asgn    = Array(table.assignments)

        keep = trues(n_m)
        for m in 1:n_m
            for v in 1:Int(n_vars)
                g_elem = Int(host_asgn[v, m])
                g_elem == 0 && continue
                o = _var_to_type(v, csp, schema)
                o === nothing && continue
                local_idx = g_elem - Int(offsets_host[v])
                if local_idx < 1 || local_idx > g.n_alloc[o] || !host_active[o][local_idx]
                    keep[m] = false; break
                end
            end
        end
        _compact_matches!(table, keep)
    end

    # ── Step 2: discover new matches (pinned to Δ⁺ elements) ─────────────────
    nc = csp.n_chunks
    for o in schema.obj_types
        new_elems = get(added_g, o, Int32[])
        isempty(new_elems) && continue

        base = get(csp.var_offset, o, 0)
        base == 0 && continue
        next_base = Int(csp.n_vars) + 1
        for other in schema.obj_types
            ob = get(csp.var_offset, other, 0)
            ob > base && ob < next_base && (next_base = ob)
        end

        for v in base:(next_base - 1)
            v > Int(csp.n_vars) && break
            for new_g_elem in new_elems
                domains = _init_domains_mc(csp, g, schema, nc)
                ci, bi  = elem_to_chunk(Int(new_g_elem))
                off_v   = (v - 1) * nc
                for c in 1:nc; domains[off_v + c] = UInt64(0); end
                ci <= nc && (domains[off_v + ci] = UInt64(1) << bi)
                _apply_attr_masks_mc!(domains, csp, g, schema, enc, nc)
                solutions = cpu_dive_solve(csp, domains)
                for sol in solutions
                    Int(sol[v]) == Int(new_g_elem) || continue
                    _push_match!(table, sol, backend)
                end
            end
        end
    end

    table
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _var_to_type(v::Int, csp::CSPProblem, schema::SchemaInfo)
    n_vars = Int(csp.n_vars)
    for o in schema.obj_types
        base = get(csp.var_offset, o, 0)
        base == 0 && continue
        next_base = n_vars + 1
        for other in schema.obj_types
            ob = get(csp.var_offset, other, 0)
            ob > base && ob < next_base && (next_base = ob)
        end
        v >= base && v < next_base && return o
    end
    return nothing
end

function _init_domains_mc(csp::CSPProblem, g::GPUACSet,
                           schema::SchemaInfo, nc::Int)::Vector{UInt64}
    nv      = Int(csp.n_vars)
    domains = zeros(UInt64, nv * nc)
    type_bases = csp.sorted_type_bases
    for (idx, (base, o)) in enumerate(type_bases)
        next_base = idx < length(type_bases) ? type_bases[idx+1][1] : nv + 1
        n_elems = min(g.n_alloc[o], nc * 64)
        host_active = Array(g.active[o])
        mask = zeros(UInt64, nc)
        for i in 1:n_elems
            (i <= length(host_active) && host_active[i]) || continue
            ci, bi = elem_to_chunk(i)
            ci <= nc && (mask[ci] |= UInt64(1) << bi)
        end
        for v in base:(next_base - 1)
            v > nv && break
            off = (v - 1) * nc
            for c in 1:nc; domains[off + c] = mask[c]; end
        end
    end
    domains
end

function _apply_attr_masks_mc!(domains::Vector{UInt64}, csp::CSPProblem,
                                g::GPUACSet, schema::SchemaInfo,
                                enc::AttributeEncoder, nc::Int)
    for bc in csp.bytecodes
        cmp = _attr_cmp_code(bc.op)
        cmp < 0 && continue
        v     = Int(bc.var1)
        a_idx = Int(bc.param1)
        req   = Int32(bc.param2)
        a     = schema.attrs[a_idx]
        owner = schema.attr_dom[a]
        host_av  = Array(g.attrs[a])
        host_act = Array(g.active[owner])
        mask     = zeros(UInt64, nc)
        n_elems  = min(g.n_alloc[owner], nc * 64)
        for i in 1:n_elems
            (i <= length(host_act) && host_act[i] &&
             i <= length(host_av)  && _attr_hit(host_av[i], req, cmp)) || continue
            ci, bi = elem_to_chunk(i)
            ci <= nc && (mask[ci] |= UInt64(1) << bi)
        end
        off = (v - 1) * nc
        for c in 1:nc; domains[off + c] &= mask[c]; end
    end
end

function _push_match!(table::MatchTable, sol::Vector{Int32}, backend)
    table.n_matches >= table.max_matches && return
    table.n_matches += 1
    m = table.n_matches
    if CUDA.functional()
        copyto!(@view(table.assignments[:, m]), CuArray(sol))
    else
        table.assignments[:, m] .= sol
    end
end
