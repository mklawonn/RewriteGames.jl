# ── Attribute threshold constraints (GPU-expressible CSP pre-filters) ──────────
#
# A side-channel for attaching *ordinal* attribute threshold constraints to a
# rule, beyond what the ACSet pattern `L` can express (which is equality only).
# The GPU Turbo lowering reads these and emits PROP_ATTR_LEQ / PROP_ATTR_GEQ
# bytecodes, applied as domain pre-filters at solve init (no dive-kernel cost).
#
# An `AttrThreshold` says: "L-element `idx` of object type `ob` must have
# attribute `attr` <op> `val`", where op ∈ (:leq, :lt, :geq, :gt) and `val` is a
# raw (pre-encoding) attribute value.  Only meaningful for ordinal (`Real` /
# custom-binned) attributes; ignored for nominal ones.

"""
    AttrThreshold(ob, idx, attr, op, val)

An ordinal attribute threshold constraint on element `idx` (1-based, within its
type) of object type `ob` in a rule's pattern `L`: its `attr` value must satisfy
`attr op val`, where `op ∈ (:leq, :lt, :geq, :gt)`.
"""
struct AttrThreshold
    ob   :: Symbol
    idx  :: Int
    attr :: Symbol
    op   :: Symbol
    val  :: Any
end

const _ATTR_THRESHOLDS = Base.IdDict{Any, Vector{AttrThreshold}}()

_as_threshold(t::AttrThreshold) = t
_as_threshold(t::NamedTuple)    = AttrThreshold(t.ob, t.idx, t.attr, t.op, t.val)
_as_threshold(t::Tuple)         = AttrThreshold(t[1], t[2], t[3], t[4], t[5])

"""
    set_attr_thresholds!(rule, thresholds) -> rule

Attach ordinal attribute threshold constraints to `rule` (an AlgebraicRewriting
`Rule`, a `PlayerRuleApp`, or any rule object the schedule stores).  The GPU
Turbo solver applies them as domain pre-filters at solve init.  Returns `rule`.

`thresholds` is an iterable of `AttrThreshold`, or of `NamedTuple`/`Tuple`
`(ob, idx, attr, op, val)`.  Registration is keyed by object identity, so attach
to the same rule object that goes into the schedule.
"""
function set_attr_thresholds!(rule, thresholds)
    _ATTR_THRESHOLDS[rule] = AttrThreshold[_as_threshold(t) for t in thresholds]
    rule
end

"""
    get_attr_thresholds(rules...) -> Vector{AttrThreshold}

Collect thresholds registered for any of the given rule objects, de-duplicated by
identity.  Callers pass both a wrapper and its inner rule (the schedule may wrap
a `Rule` in a `PlayerRuleApp`/`RuleApp`).
"""
function get_attr_thresholds(rules...)
    out  = AttrThreshold[]
    seen = Base.IdSet{Any}()
    for r in rules
        (r === nothing || r in seen) && continue
        push!(seen, r)
        ts = get(_ATTR_THRESHOLDS, r, nothing)
        ts === nothing || append!(out, ts)
    end
    out
end

# ── Affine attribute deltas (post-rewrite mutation: attr := matched_attr + d) ──
#
# The GPU rewrite path writes only static attribute values; it cannot evaluate an
# AlgebraicRewriting `expr` (a function of the matched value).  For the common
# case of an integer *counter* (identity-encoded ordinal: encoded == value + c),
# `attr := matched_attr + delta` is a pure on-device integer add on the encoded
# column, applied to the matched (preserved) element after the structural
# rewrite.  This is the Phase-2 companion to `AttrThreshold`: a rule becomes
# "structural + threshold(s) + delta(s)" instead of a token cloud.

"""
    AttrDelta(ob, idx, attr, delta)

A post-rewrite affine mutation of element `idx` (1-based, within its type) of
object type `ob`: after the rule fires on a match, the matched element's `attr`
(an identity-encoded ordinal) is incremented by integer `delta` (negative to
decrement), clamped to stay ≥ 1 (the 1 = smallest rank; 0 is the unset sentinel).
Use a threshold to guarantee the value stays in range (e.g. `fuel ≥ 1` before a
`-1`).
"""
struct AttrDelta
    ob    :: Symbol
    idx   :: Int
    attr  :: Symbol
    delta :: Int
end

const _ATTR_DELTAS = Base.IdDict{Any, Vector{AttrDelta}}()

_as_delta(d::AttrDelta) = d
_as_delta(d::NamedTuple) = AttrDelta(d.ob, d.idx, d.attr, d.delta)
_as_delta(d::Tuple)      = AttrDelta(d[1], d[2], d[3], d[4])

"""
    set_attr_deltas!(rule, deltas) -> rule

Attach post-rewrite affine attribute deltas to `rule`.  `deltas` is an iterable
of `AttrDelta` or `(ob, idx, attr, delta)`.  Keyed by object identity; attach to
the same rule object that goes into the schedule.
"""
function set_attr_deltas!(rule, deltas)
    _ATTR_DELTAS[rule] = AttrDelta[_as_delta(d) for d in deltas]
    rule
end

"""
    get_attr_deltas(rules...) -> Vector{AttrDelta}

Collect deltas registered for any of the given rule objects, de-duplicated by
identity (callers pass a wrapper and its inner rule).
"""
function get_attr_deltas(rules...)
    out  = AttrDelta[]
    seen = Base.IdSet{Any}()
    for r in rules
        (r === nothing || r in seen) && continue
        push!(seen, r)
        ds = get(_ATTR_DELTAS, r, nothing)
        ds === nothing || append!(out, ds)
    end
    out
end

"""
    attr_threshold_pred(thresholds) -> (L) -> ((m::ACSetTransformation) -> Bool)

Build a CPU match predicate enforcing `thresholds` against the codomain world,
for parity with the GPU domain pre-filter.  Used by the CPU match path so CPU
and GPU agree.  Returns a closure over the matched world `G = codom(m)`.
"""
function attr_threshold_pred(thresholds::Vector{AttrThreshold})
    isempty(thresholds) && return nothing
    function (m)
        G = codom(m)
        for t in thresholds
            # element idx of type t.ob in L maps to world element w
            w = m[t.ob](t.idx)
            v = subpart(G, w, t.attr)
            ok = t.op === :leq ? v <= t.val :
                 t.op === :lt  ? v <  t.val :
                 t.op === :geq ? v >= t.val :
                 t.op === :gt  ? v >  t.val : true
            ok || return false
        end
        return true
    end
end
