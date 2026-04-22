using JSON3
using ACSets: generate_json_acset, parse_json_acset

# ─── write_history ────────────────────────────────────────────────────────────

"""
    write_history(path::String, hist::GameHistory)

Serialise a `GameHistory` to a directory at `path`.

Layout:
```
<path>/
  world/<t>.json       — ACSet JSON for the world at time t
  chosen/<t>.json      — JSON for chosen rule's (L, K, R) ACSets at turn t
  available/<t>.json   — JSON for available rules' coproduct (L, K, R) at turn t
  scalars.json         — [{turn, player, schedule_path, winner}, …]
```

World and rule ACSets are written using ACSets.jl's `generate_json_acset`.
Scalar fields (player, schedule path, winner) are written as a JSON array.
"""
function write_history(path::String, hist::GameHistory)
    for dir in ("world", "chosen", "available")
        mkpath(joinpath(path, dir))
    end

    # World snapshots (includes t=0 initial world)
    for t in hist._world_turns
        w = get_world(hist, t)
        w === nothing && continue
        open(joinpath(path, "world", "$t.json"), "w") do io
            JSON3.write(io, generate_json_acset(w))
        end
    end

    # Per-turn action narratives
    for t in hist._step_turns
        ch = get_chosen(hist, t)
        if ch !== nothing
            open(joinpath(path, "chosen", "$t.json"), "w") do io
                JSON3.write(io, Dict(
                    "rule_name" => String(ch.rule_name),
                    "L"         => generate_json_acset(ch.L),
                    "K"         => generate_json_acset(ch.K),
                    "R"         => generate_json_acset(ch.R),
                ))
            end
        end

        av = get_available(hist, t)
        if av !== nothing
            open(joinpath(path, "available", "$t.json"), "w") do io
                JSON3.write(io, Dict(
                    "L" => generate_json_acset(av.L),
                    "K" => generate_json_acset(av.K),
                    "R" => generate_json_acset(av.R),
                ))
            end
        end
    end

    # Scalar narratives
    open(joinpath(path, "scalars.json"), "w") do io
        JSON3.write(io, [
            (
                turn          = t,
                player        = String(get_player(hist, t)),
                schedule_path = String.(get_path(hist, t)),
                winner        = let w = get_terminal(hist, t)
                                    w === nothing ? nothing : String(w)
                                end,
            )
            for t in hist._step_turns
        ])
    end
end

# ─── read_history ─────────────────────────────────────────────────────────────

"""
    read_history(path::String, WorldType::Type) -> GameHistory

Read a `GameHistory` previously written with `write_history`.

`WorldType` must be the concrete ACSet type used for the world narrative (the
same type the game was played with).  Rule span ACSets (chosen / available) are
also assumed to be of this type.

The match morphisms are not persisted and are returned as `nothing`.
"""
function read_history(path::String, WorldType::Type)
    # Discover recorded world turn indices from filenames
    world_dir  = joinpath(path, "world")
    world_turns = sort([
        parse(Int, splitext(f)[1])
        for f in readdir(world_dir) if endswith(f, ".json")
    ])

    isempty(world_turns) && error("No world snapshots found in $world_dir")

    # Reconstruct initial world to build the GameHistory
    t0_data    = JSON3.read(read(joinpath(world_dir, "$(world_turns[1]).json"), String))
    init_world = parse_json_acset(WorldType, t0_data)

    hist = GameHistory(init_world)

    # Load remaining world snapshots (skip t=0, already recorded by constructor)
    for t in world_turns[2:end]
        w = parse_json_acset(WorldType, JSON3.read(read(joinpath(world_dir, "$t.json"), String)))
        add!(hist.world_narrative, TemporalData.Interval(t), w)
        push!(hist._world_turns, t)
    end

    # Load scalar narratives
    scalars_path = joinpath(path, "scalars.json")
    if isfile(scalars_path)
        rows = JSON3.read(read(scalars_path, String))
        for row in rows
            t = row[:turn]
            hist.player_narrative[t]   = Symbol(row[:player])
            hist.path_narrative[t]     = Symbol.(row[:schedule_path])
            hist.terminal_narrative[t] = row[:winner] === nothing ? nothing :
                                         Symbol(row[:winner])
            hist.match_narrative[t]    = nothing  # not persisted
            push!(hist._step_turns, t)
        end
        sort!(hist._step_turns)
    end

    # Load chosen rule spans
    chosen_dir = joinpath(path, "chosen")
    if isdir(chosen_dir)
        for f in readdir(chosen_dir)
            endswith(f, ".json") || continue
            t    = parse(Int, splitext(f)[1])
            data = JSON3.read(read(joinpath(chosen_dir, f), String))
            hist.chosen_spans[t] = (
                rule_name = Symbol(data["rule_name"]),
                L         = parse_json_acset(WorldType, data["L"]),
                K         = parse_json_acset(WorldType, data["K"]),
                R         = parse_json_acset(WorldType, data["R"]),
            )
        end
    end

    # Load available rule coproduct spans
    avail_dir = joinpath(path, "available")
    if isdir(avail_dir)
        for f in readdir(avail_dir)
            endswith(f, ".json") || continue
            t    = parse(Int, splitext(f)[1])
            data = JSON3.read(read(joinpath(avail_dir, f), String))
            hist.avail_spans[t] = (
                L = parse_json_acset(WorldType, data["L"]),
                K = parse_json_acset(WorldType, data["K"]),
                R = parse_json_acset(WorldType, data["R"]),
            )
        end
    end

    return hist
end

export write_history, read_history
