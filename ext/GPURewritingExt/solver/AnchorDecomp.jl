"""
Anchored-fiber decomposition for big-codomain (would-be EPS) standard-path solves.

At full scenario scale the largest object types push `nc` past the shared-memory
limit, so 8-14-var rules route to the slow global-memory EPS pipeline even after
BIGVAR.  But when such a rule has exactly ONE pattern variable of some large
type (the ANCHOR — e.g. the kill chain's single Target), its matches partition
exactly by that variable's value.  We split the anchor's candidate slots into
contiguous CELLS, and per cell reuse the codomain-decomposition machinery
(`_decomp_gather` seeded with the cell + `_decomp_compact_solve`): restrict the
codomain to the cell's FK-closure, remap to a compact index space (small
`nc_local`), solve on the shared-memory turbo_block path, translate back.  The
union over cells is the EXACT global solution set, disjoint by construction —
no reconciliation, no dedup.  Only existing NM<=16 kernel specializations are
used (no new JIT shapes).

Soundness notes:
- closure over-approximates match support for any match whose anchor lies in
  the cell (same both-direction fixpoint argument as the pinned-agent decomp);
- in-CSP NAC/PAC witnesses are FK-connected to L, hence inside the closure;
  NacSpec post-filters run on the union against the FULL world as usual;
- types with no hom path to the anchor range over their full domain.

Adaptivity: if a cell's compact `nc_local` would still route to EPS (closure
blowup through hub objects), the whole attempt returns `nothing` and the caller
falls back to the global EPS solve — no per-rule allowlist needed.

Env knobs: RG_NO_ANCHOR_DECOMP kills the path; RG_ANCHOR_CELL (default 512)
sets the cell width in anchor slots; RG_ANCHOR_MIN (default 1024) is the
minimum anchor cardinality worth decomposing; RG_ANCHOR_FORCE=1 decomposes
every eligible solve regardless of the EPS-routing check (testing).
RG_ANCHOR_DIAG cross-checks solution sets in the Scheduler hook.
"""

const _ANCHOR_SOLVES    = Ref(0)   # standard-path solves that were decomposed
const _ANCHOR_FALLBACKS = Ref(0)   # eligible attempts abandoned (closure too big)

_anchor_cell_sz() = clamp(parse(Int, get(ENV, "RG_ANCHOR_CELL", "512")), 64, 4096)
_anchor_min_sz()  = parse(Int, get(ENV, "RG_ANCHOR_MIN", "1024"))

"""
    _anchor_for(csp, n_alloc) -> Union{Nothing, Tuple{Symbol,Int}}

Pick the anchor: the single-variable type with the largest allocated world
cardinality, provided it meets RG_ANCHOR_MIN.  Returns (type, var index).
"""
function _anchor_for(csp::CSPProblem, n_alloc::Dict{Symbol,Int})
    nvar = _decomp_nvars_per_type(csp)
    best = nothing; best_n = 0
    for (base, o) in csp.sorted_type_bases
        nvar[o] == 1 || continue
        n = get(n_alloc, o, 0)
        n > best_n && ((best, best_n) = ((o, base), n))
    end
    best === nothing && return nothing
    best_n < _anchor_min_sz() && return nothing
    best
end

# Mirrors gpu_turbo_solve's default cap: the standard path presents at most
# this many matches per solve, so the cell union stops there too (and the
# RG_ANCHOR_DIAG comparison skips cap-truncated solves — both sides are then
# arbitrary same-size subsets of the valid set, not comparable).
const _ANCHOR_MAX_SOLUTIONS = 10_000

"""
    anchored_decomposed_solve(backend, csp, schema, base_d_host, fk_cols, n_alloc)
        -> Union{Nothing, Vector{Vector{Int32}}}

Solve the CSP as a union of per-cell compact solves over the anchor's candidate
slots, in WORLD indices.  Returns `nothing` (caller falls back to the global
solve) when no anchor qualifies, the pattern exceeds the turbo_block var band,
or any cell's closure stays too big for shared memory.  The union stops at
`max_solutions` (matching the global solve's cap) — later cells are skipped.
"""
function anchored_decomposed_solve(backend, csp::CSPProblem, schema::SchemaInfo,
                                    base_d_host::Vector{UInt64},
                                    fk_cols::Dict{Symbol,Vector{Int32}},
                                    n_alloc::Dict{Symbol,Int};
                                    max_solutions::Int = _ANCHOR_MAX_SOLUTIONS)
    nv = Int(csp.n_vars)
    nv <= _NV_BIG || return nothing          # cells would still route to EPS
    a  = _anchor_for(csp, n_alloc)
    a === nothing && return nothing
    anchor_obj, v_anchor = a

    # Anchor candidates = set bits of the anchor VARIABLE's base domain (active
    # mask + attribute masks already applied), so dead slots never form cells.
    nc_w  = csp.n_chunks
    offw  = (v_anchor - 1) * nc_w
    slots = Int[]
    for w in 1:get(n_alloc, anchor_obj, 0)
        ci, bi = elem_to_chunk(w)
        ci <= nc_w && (base_d_host[offw + ci] & (UInt64(1) << bi)) != 0 && push!(slots, w)
    end
    isempty(slots) && return Vector{Vector{Int32}}()

    cell_sz = _anchor_cell_sz()
    cells   = [slots[i:min(i + cell_sz - 1, end)] for i in 1:cell_sz:length(slots)]

    # Gather every cell's closure first (cheap, host-only); abandon the attempt
    # before any solve if some cell can't reach the shared-memory path.
    nbhds = Vector{Dict{Symbol,Vector{Int}}}(undef, length(cells))
    for (i, cell) in enumerate(cells)
        nbhd  = _decomp_gather(schema, csp, fk_cols, n_alloc, anchor_obj, cell)
        k_max = maximum((length(ws) for ws in values(nbhd)); init = 1)
        nc_l  = max(cld(k_max, 64), 1)
        if nc_l > 1 && _would_use_eps(nv, nc_l)
            _ANCHOR_FALLBACKS[] += 1
            return nothing
        end
        nbhds[i] = nbhd
    end

    out = Vector{Vector{Int32}}()
    for nbhd in nbhds
        append!(out, _decomp_compact_solve(backend, csp, schema, nbhd,
                                           base_d_host, fk_cols;
                                           max_solutions = max_solutions - length(out)))
        length(out) >= max_solutions && break
    end
    length(out) > max_solutions && resize!(out, max_solutions)
    _ANCHOR_SOLVES[] += 1
    out
end
