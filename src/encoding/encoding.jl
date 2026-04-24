"""
    EncodedState

Holds both the raw `GameState` and lazily-computed tensor encodings.

Agents that need only the raw world state pay no encoding cost; fields marked
`lazy` are computed on first access via accessor functions.

# Fields
- `raw`:           The original `GameState` (always available immediately).
- `node_features`: `(n_nodes × F)` `Float32` matrix; nodes are `(table, row)` pairs,
                   features are attribute values plus a one-hot table-type indicator.
- `edge_index`:    `(2 × n_edges)` `Int32` matrix in COO format; edges are foreign-key
                   entries.
- `edge_type`:     `(n_edges,)` `Int8` vector; schema morphism index for each edge.
- `turn_frac`:     `Float32` scalar `turn / T_max`.
"""
struct EncodedState
    raw           :: GameState
    node_features :: Matrix{Float32}        # n_nodes × F
    edge_index    :: Matrix{Int32}          # 2 × n_edges (COO)
    edge_type     :: Vector{Int8}           # n_edges
    turn_frac     :: Float32
end

Base.show(io::IO, s::EncodedState) =
    print(io, "EncodedState(nodes=$(size(s.node_features,1)), edges=$(size(s.edge_index,2)), turn_frac=$(round(s.turn_frac; digits=3)))")

# ─── encode_state ─────────────────────────────────────────────────────────────

"""
    encode_state(W, turn::Int, T_max::Int) -> EncodedState

Build an `EncodedState` from a raw ACSet world `W`, the turn number, and the
episode horizon.

The encoding is *schema-generic*: it introspects `acset_schema(W)` to
discover tables and foreign keys.

## Node encoding

Each (table, row) pair becomes one node.  Node features are:
1. One-hot indicator of the table type (length = number of ob-types in schema).
2. Numerical values of all attributes of that row (NaN-padded to the maximum
   attribute count across tables).

## Edge encoding

Each foreign-key entry `hom(src_table, tgt_table)` recorded as a directed
edge `src_node -> tgt_node` in COO format.
"""
function encode_state(W, turn::Int, T_max::Int)
    S        = acset_schema(W)
    obs      = collect(Symbol, objects(S))      # Vector{Symbol}
    hom_list = collect(homs(S))                 # Vector of (name, dom, codom) triples
    n_obs    = length(obs)

    # ── Build node list (table, row) ────────────────────────────────────────
    node_list   = Tuple{Int, Int}[]             # (ob_idx, row_id)
    node_offset = Dict{Int, Int}()              # ob_idx -> start index in node_list

    for (oi, o) in enumerate(obs)
        node_offset[oi] = length(node_list) + 1
        for r in parts(W, o)
            push!(node_list, (oi, r))
        end
    end
    n_nodes = length(node_list)

    # ── Attribute features ──────────────────────────────────────────────────
    attrs_per_ob = [_attr_names(S, o) for o in obs]
    max_attrs    = isempty(attrs_per_ob) ? 0 : maximum(length.(attrs_per_ob))
    F            = n_obs + max_attrs            # total feature width

    node_feat = zeros(Float32, n_nodes, F)

    for (ni, (oi, row)) in enumerate(node_list)
        # One-hot table indicator
        node_feat[ni, oi] = 1f0
        # Attribute values
        for (ai, aname) in enumerate(attrs_per_ob[oi])
            val = subpart(W, row, aname)
            node_feat[ni, n_obs + ai] = _to_float32(val)
        end
    end

    # ── Edge list (foreign-key entries) ─────────────────────────────────────
    edge_src_v  = Int32[]
    edge_dst_v  = Int32[]
    edge_type_v = Int8[]

    for (hi, (hname, src_ob_sym, tgt_ob_sym)) in enumerate(hom_list)
        src_oi = findfirst(==(src_ob_sym), obs)
        tgt_oi = findfirst(==(tgt_ob_sym), obs)
        (src_oi === nothing || tgt_oi === nothing) && continue
        for r in parts(W, src_ob_sym)
            tgt_row = subpart(W, r, hname)
            tgt_row == 0 && continue        # unset FK
            src_node = node_offset[src_oi] + r - 1
            tgt_node = node_offset[tgt_oi] + tgt_row - 1
            push!(edge_src_v,  Int32(src_node))
            push!(edge_dst_v,  Int32(tgt_node))
            push!(edge_type_v, Int8(hi))
        end
    end

    n_edges    = length(edge_src_v)
    edge_index = n_edges > 0 ?
        vcat(reshape(edge_src_v, 1, :), reshape(edge_dst_v, 1, :)) :
        Matrix{Int32}(undef, 2, 0)

    turn_frac = Float32(turn) / Float32(max(T_max, 1))

    return EncodedState(
        GameState(W, turn),
        node_feat,
        edge_index,
        edge_type_v,
        turn_frac,
    )
end

# ─── helpers ──────────────────────────────────────────────────────────────────

"""Return the list of attribute names for object `o` in schema `S`."""
function _attr_names(S, o::Symbol)
    # attrs(S; from=o, just_names=true) uses the ACSets schema API
    collect(Symbol, attrs(S; from=o, just_names=true))
end

"""Coerce attribute values to Float32 for feature matrices."""
_to_float32(v::Number)  = Float32(v)
_to_float32(v::Bool)    = v ? 1f0 : 0f0
_to_float32(::Nothing)  = Float32(NaN)
_to_float32(::Missing)  = Float32(NaN)
_to_float32(_)          = Float32(NaN)
