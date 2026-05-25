"""
    AttributeEncoder

Bidirectional mapping between Julia attribute values and GPU-safe Int32 IDs.

- **Nominal** attributes (`String`, `Symbol`, arbitrary discrete types): each
  distinct value is assigned a unique Int32 in encounter order (1-based).
- **Ordinal** attributes (`Real` subtypes): values are sorted and assigned
  ranks 1..n so that `<` and `≤` inequalities hold on the integer rank.
- **Custom** attributes: user-supplied `(encode_fn, decode_fn)` pair that
  completely overrides auto-detection for that attribute.

The zero value (Int32(0)) is reserved as "unset / wildcard".
"""
struct AttributeEncoder
    nominal :: Dict{Symbol, Dict{Any, Int32}}              # attr → value → id
    ordinal :: Dict{Symbol, Vector{Any}}                   # attr → sorted unique values
    custom  :: Dict{Symbol, Pair{Function, Function}}      # attr → (encode, decode)
end

AttributeEncoder() = AttributeEncoder(Dict(), Dict(), Dict())

"""
    build_encoder(world, schema; discretizers=Dict()) -> AttributeEncoder

Scan every attribute column in `world` and populate the encoder.
Attribute types are detected at runtime: `Real` subtypes go through the
ordinal path, everything else through the nominal path.

`discretizers` is an optional `Dict{Symbol, Pair{Function,Function}}` mapping
attribute names to `(encode_fn, decode_fn)` pairs.  When present for an
attribute, the custom functions are used instead of auto-detection.
`encode_fn :: Any → Int32`, `decode_fn :: Int32 → Any`.
"""
function build_encoder(world, schema::SchemaInfo;
                       discretizers::Dict{Symbol, Pair{Function,Function}} = Dict{Symbol, Pair{Function,Function}}())
    enc = AttributeEncoder(Dict(), Dict(), copy(discretizers))
    for a in schema.attrs
        haskey(discretizers, a) && continue   # custom encoder handles this attr
        owner = schema.attr_dom[a]
        vals  = [subpart(world, i, a) for i in parts(world, owner)]
        _register_attr!(enc, a, vals)
    end
    enc
end

function _register_attr!(enc::AttributeEncoder, attr::Symbol, vals::Vector)
    isempty(vals) && return
    concrete = filter(v -> !(v isa AttrVar), vals)
    isempty(concrete) && return

    if first(concrete) isa Real
        uniq = sort(unique(concrete))
        enc.ordinal[attr] = uniq
    else
        d = get!(enc.nominal, attr, Dict{Any,Int32}())
        for v in concrete
            haskey(d, v) || (d[v] = Int32(length(d) + 1))
        end
    end
end

"""
    encode_value(enc, attr, v) -> Int32

Encode a single attribute value.  Returns `Int32(0)` for `AttrVar`s or
values not yet seen (treated as wildcards during matching).
"""
function encode_value(enc::AttributeEncoder, attr::Symbol, v)::Int32
    v isa AttrVar && return Int32(0)
    if haskey(enc.custom, attr)
        return enc.custom[attr].first(v)
    end
    if haskey(enc.ordinal, attr)
        vals = enc.ordinal[attr]
        idx  = searchsortedfirst(vals, v)
        return (idx <= length(vals) && vals[idx] == v) ? Int32(idx) : Int32(0)
    else
        d = get(enc.nominal, attr, nothing)
        d === nothing && return Int32(0)
        return get(d, v, Int32(0))
    end
end

"""
    decode_value(enc, attr, i) -> Any

Recover the original attribute value from its integer encoding.
Returns `nothing` for `i == 0`.
"""
function decode_value(enc::AttributeEncoder, attr::Symbol, i::Int32)
    i == Int32(0) && return nothing
    if haskey(enc.custom, attr)
        return enc.custom[attr].second(i)
    end
    if haskey(enc.ordinal, attr)
        vals = enc.ordinal[attr]
        return (1 <= i <= length(vals)) ? vals[i] : nothing
    else
        d = get(enc.nominal, attr, nothing)
        d === nothing && return nothing
        for (v, id) in d
            id == i && return v
        end
        return nothing
    end
end

"""
    extend_encoder!(enc, world, schema)

Register any attribute values in `world` that were not seen when the encoder
was built (e.g. after a pushout adds new elements with new attribute values).
Custom-encoded attributes are skipped (their functions handle all values).
"""
function extend_encoder!(enc::AttributeEncoder, world, schema::SchemaInfo)
    for a in schema.attrs
        haskey(enc.custom, a) && continue
        owner = schema.attr_dom[a]
        vals  = [subpart(world, i, a) for i in parts(world, owner)]
        _register_attr!(enc, a, vals)
    end
    enc
end
