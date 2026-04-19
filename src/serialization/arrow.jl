using Arrow

"""
    write_experiences(path::String, experiences::Vector{Experience})

Serialise a vector of `Experience` values to an Arrow file at `path`.

Each record stores:
- `player`:        Symbol (stored as String)
- `turn_frac`:     Float32
- `n_nodes`:       Int32  (number of nodes in the pre-action state)
- `n_edges`:       Int32  (number of edges in the pre-action state)
- `node_features`: serialised as a flat Float32 vector (row-major)
- `rule_counters`: flat Int32 vector
- `action_rule_name`:  Symbol of chosen rule (stored as String; "nothing" if passed)
- `done`:          Bool
- `winner`:        String ("nothing" if no winner)
"""
function write_experiences(path::String, experiences::Vector{Experience})
    rows = map(experiences) do exp
        es = exp.state
        n_nodes, F = size(es.node_features)
        flat_feat   = vec(es.node_features)   # row-major flatten

        action_name = exp.action === nothing ? "nothing" :
                      String(exp.action.entry.name)

        winner_str  = exp.winner === nothing ? "nothing" : String(exp.winner)

        (
            player          = String(exp.player),
            turn_frac       = es.turn_frac,
            n_nodes         = Int32(n_nodes),
            n_feat          = Int32(F),
            n_edges         = Int32(size(es.edge_index, 2)),
            node_features   = flat_feat,
            rule_counters   = copy(es.rule_counters),
            action_rule_name = action_name,
            done            = exp.done,
            winner          = winner_str,
        )
    end

    Arrow.write(path, rows)
end

"""
    read_experiences(path::String) -> Vector{<:NamedTuple}

Read experiences previously written with `write_experiences`.

Returns a vector of named tuples containing the serialised fields.  Full
`Experience` structs cannot be reconstructed from disk without the game
definition; callers should use the raw tensors for training directly.
"""
function read_experiences(path::String)
    tbl = Arrow.Table(path)
    # Arrow.Tables is Arrow's re-exported Tables.jl – convert to vector of NamedTuples
    return Arrow.Tables.rowtable(tbl)
end
