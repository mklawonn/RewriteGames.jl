using Arrow
using Tables

"""
    write_experiences(path::String, experiences::Vector{Experience})

Serialise a vector of `Experience` values to an Arrow file at `path`.

Each record stores:
- `player`:           Symbol (stored as String)
- `turn`:             Int32 (turn number from the pre-action state)
- `action_rule_name`: Symbol of chosen rule (stored as String; "nothing" if passed)
- `done`:             Bool
- `winner`:           String ("nothing" if no winner)

Users who need tensor data for training should encode their states with a
custom encoder before writing.
"""
function write_experiences(path::String, experiences::Vector{Experience})
    rows = map(experiences) do exp
        action_name = exp.action === nothing ? "nothing" :
                      String(exp.action.entry.name)
        winner_str  = exp.winner === nothing ? "nothing" : String(exp.winner)

        (
            player           = String(exp.player),
            turn             = Int32(exp.state.turn),
            action_rule_name = action_name,
            done             = exp.done,
            winner           = winner_str,
        )
    end

    Arrow.write(path, rows)
end

"""
    read_experiences(path::String) -> Vector{<:NamedTuple}

Read experiences previously written with `write_experiences`.

Returns a vector of named tuples containing the serialised fields.
"""
function read_experiences(path::String)
    tbl = Arrow.Table(path)
    return Tables.rowtable(tbl)
end
