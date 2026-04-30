# ── MatchCache ────────────────────────────────────────────────────────────────
#
# Generic incremental match cache for PlayerRuleApp boxes.
#
# Maintains the set of valid match morphisms for a Rule against an evolving
# ACSet world, updating it incrementally after each DPO rewrite rather than
# re-running the full homomorphism search from scratch.
#
# Design (inspired by AlgebraicRewriting PR #62):
#   - On initialisation: full get_matches call to populate the match set.
#   - After a DPO rewrite (via rewrite_match_maps):
#       1. Forward surviving matches through the pushout complement maps.
#          Matches whose image elements were deleted are dropped.
#       2. Re-check NAC / PAC conditions on forwarded matches.
#          Matches invalidated by newly-added material (NAC now satisfied) drop.
#       3. Discover new matches involving newly-added elements via pinned search.
#
# Limitations of current implementation (future work):
#   - Attribute value changes during rewrites are not tracked; the forwarding
#     step re-infers them from the new world via homomorphism search.
#   - Native RuleApp boxes that modify the world will silently stale any caches;
#     for correctness those should be converted to PlayerRuleApp or the cache
#     invalidated manually.

"""
    MatchCache

Incremental cache of valid match morphisms for a `Rule` against an evolving
ACSet world.

# Fields
- `rule`:        The rewrite rule whose match set is being maintained.
- `cat`:         The ACSet category (passed through to `get_matches` / `can_match`).
- `matches`:     Current set of valid match morphisms `L → world`.
- `match_limit`: Optional cap on the number of stored matches.
"""
mutable struct MatchCache
    rule        :: Any   # AbsRule (Rule{:DPO} in practice)
    cat         :: Any   # ACSetCategory or nothing
    matches     :: Vector{Any}   # Vector{ACSetTransformation}
    match_limit :: Union{Int, Nothing}
end

"""
    MatchCache(rule, cat, world::ACSet; match_limit=nothing) -> MatchCache

Initialise a cache with a homomorphism search against `world`.  When
`match_limit` is an `Int`, at most that many matches are stored.
"""
function MatchCache(rule, cat, world::ACSet;
                    match_limit::Union{Int,Nothing}=nothing)
    _cat = isnothing(cat) ? infer_acset_cat(world) : cat
    gen  = get_matches(rule, world; cat=_cat)
    ms   = match_limit === nothing ? collect(gen) :
           collect(Iterators.take(gen, match_limit))
    MatchCache(rule, cat, Vector{Any}(ms), match_limit)
end

"""
    update_cache!(cache::MatchCache, maps::Dict)

Incrementally update the match set after a DPO rewrite described by `maps`
(the output of `rewrite_match_maps`).

Expected keys in `maps`:
- `:kg` — monic morphism `K → G` (pushout complement → old world)
- `:kh` — monic morphism `K → H` (pushout complement → new world)
- `:rh` — morphism `R → H` (right-hand side → new world)
"""
function update_cache!(cache::MatchCache, maps::Dict)
    kg = maps[:kg]   # K → G  (monic)
    kh = maps[:kh]   # K → H  (monic)
    rh = maps[:rh]   # R → H

    H    = codom(rh)    # new world
    _cat = isnothing(cache.cat) ? infer_acset_cat(H) : cache.cat

    # ── Step 1 & 2: forward surviving matches and re-check conditions ──────────
    new_matches = Any[]
    for m in cache.matches
        m_fwd = _forward_match(m, kg, kh, _cat)
        m_fwd === nothing && continue                       # element deleted
        can_match(cache.rule, m_fwd; cat=_cat, homsearch=true) === nothing || continue  # cond violated
        push!(new_matches, m_fwd)
    end

    # ── Step 3: find new matches involving newly added elements ────────────────
    append!(new_matches, _find_new_matches(cache.rule, rh, kh, _cat))

    if cache.match_limit !== nothing && length(new_matches) > cache.match_limit
        resize!(new_matches, cache.match_limit)
    end
    cache.matches = new_matches
    return cache
end

# ── Internal helpers ───────────────────────────────────────────────────────────

"""
    _forward_match(m, kg, kh, cat) -> ACSetTransformation or nothing

Given an existing match `m : L → G` and the DPO complement maps `kg : K → G`
(monic) and `kh : K → H`, try to forward `m` to a match `m' : L → H`.

Each element `L` maps to must survive (lie in `image(kg)`).  If any element
was deleted, returns `nothing`.  Otherwise calls `homomorphism` with the
full object-component assignment to reconstruct the match including inferred
attribute values.
"""
function _forward_match(m, kg, kh, cat)
    L = dom(m)
    S = acset_schema(L)
    K = dom(kg)
    H = codom(kh)

    # Build G-element → K-element lookup (kg is monic, so preimage is unique)
    preim_kg = Dict{Symbol, Dict{Int,Int}}()
    for o in ob(S)
        d = Dict{Int,Int}()
        for k in parts(K, o)
            d[kg[o](k)] = k
        end
        preim_kg[o] = d
    end

    # Map each L-element through G → K → H
    fwd = Dict{Symbol, Dict{Int,Int}}()
    for o in ob(S)
        fwd_o = Dict{Int,Int}()
        for i in parts(L, o)
            g_i = m[o](i)
            haskey(preim_kg[o], g_i) || return nothing   # deleted
            fwd_o[i] = kh[o](preim_kg[o][g_i])
        end
        fwd[o] = fwd_o
    end

    # Reconstruct via homomorphism (handles attribute inference + validation)
    return homomorphism(L, H; cat=cat, initial=fwd)
end

"""
    _find_new_matches(rule, rh, kh, cat) -> Vector

Find all valid matches in the new world `H = codom(rh)` that involve at least
one element that was freshly added by the rewrite (i.e., in `image(rh)` but
not in `image(kh)`).

Uses pinned homomorphism search: for each new element `e` of each type that
appears in the pattern, enumerate all matches that send some pattern node to
`e`, then filter by the rule's NAC / PAC conditions.
"""
function _find_new_matches(rule, rh, kh, cat)
    R  = dom(rh)
    K  = dom(kh)
    H  = codom(rh)   # = codom(kh)
    L  = codom(left(rule))   # pattern of the rule
    S  = acset_schema(L)

    # Collect new H-elements: in image(rh) but not image(kh)
    new_elems = Dict{Symbol, Vector{Int}}()
    for o in ob(S)
        rh_img = Set(rh[o](r) for r in parts(R, o))
        kh_img = Set(kh[o](k) for k in parts(K, o))
        new_elems[o] = collect(setdiff(rh_img, kh_img))
    end

    result   = Any[]
    seen_key = Set{Any}()

    for o in ob(S)
        isempty(new_elems[o]) && continue
        nparts(L, o) == 0     && continue   # pattern has no o-elements

        for e_new in new_elems[o]
            for l_part in parts(L, o)
                # Pin pattern element l_part to new world element e_new
                init = Dict{Symbol, Any}(o => Dict(l_part => e_new))
                for m in homomorphisms(L, H; cat=cat, monic=rule.monic, initial=init)
                    can_match(rule, m; cat=cat, homsearch=true) === nothing || continue
                    # Deduplicate by full object assignment
                    key = Tuple(m[ob_](i) for ob_ in ob(S) for i in parts(L, ob_))
                    key ∈ seen_key && continue
                    push!(seen_key, key)
                    push!(result, m)
                end
            end
        end
    end

    return result
end
