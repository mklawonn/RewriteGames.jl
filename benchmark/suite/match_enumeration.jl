"""
Micro-benchmarks: match enumeration speed under three strategies.

For each board state (empty board, mid-game, nearly-full board), measures:
  baseline      – full get_matches (BacktrackingSearch + NAC evaluation)
  fast_path     – ttt_fast_matches via ACSet index lookup
  cache_update  – MatchCache.update_cache! (incremental DPO update)
"""

include(joinpath(@__DIR__, "ttt_setup.jl"))

const BENCH_MATCH = BenchmarkGroup()

# ── Representative board states ───────────────────────────────────────────────

function board_empty()
    create_board()
end

function board_mid()
    b = create_board()
    add_part!(b, :X, xsq=5)   # X centre
    add_part!(b, :O, osq=1)   # O top-left
    add_part!(b, :X, xsq=9)   # X bottom-right
    b
end

function board_nearly_full()
    b = create_board()
    for sq in [1,3,5,7,9]; add_part!(b, :X, xsq=sq); end
    for sq in [2,4,6];     add_part!(b, :O, osq=sq); end
    b   # 2 empty squares remain
end

const BOARDS = [("empty", board_empty()), ("mid", board_mid()), ("nearly_full", board_nearly_full())]

# ── Baseline ──────────────────────────────────────────────────────────────────

BENCH_MATCH["baseline"] = BenchmarkGroup()
for (name, board) in BOARDS
    BENCH_MATCH["baseline"][name] = @benchmarkable begin
        get_matches(mark_x, $board; cat=$𝒞_TTT)
    end
end

# ── Fast path ─────────────────────────────────────────────────────────────────

BENCH_MATCH["fast_path"] = BenchmarkGroup()
for (name, board) in BOARDS
    BENCH_MATCH["fast_path"][name] = @benchmarkable begin
        ttt_fast_matches(mark_x, $board, $𝒞_TTT)
    end
end

# ── Incremental cache update ──────────────────────────────────────────────────
# Simulate one DPO rewrite and measure update_cache! cost.

BENCH_MATCH["cache_update"] = BenchmarkGroup()
for (name, board) in BOARDS
    # Find the first empty square and pre-build the DPO maps for that move
    occupied = Set(vcat(subpart(board, :xsq), subpart(board, :osq)))
    empty_first = first(setdiff(1:nparts(board, :Sq), occupied))
    m = homomorphism(Sq_rep, board; cat=𝒞_TTT,
                     initial=Dict(:Sq => Dict(1 => empty_first)))
    maps = rewrite_match_maps(mark_x, m; cat=𝒞_TTT)

    BENCH_MATCH["cache_update"][name] = @benchmarkable begin
        cache = MatchCache(mark_x, $𝒞_TTT, $board)
        update_cache!(cache, $maps)
    end
end

# ── Initialisation cost ───────────────────────────────────────────────────────
# How long does the one-time cache population take?

BENCH_MATCH["cache_init"] = BenchmarkGroup()
for (name, board) in BOARDS
    BENCH_MATCH["cache_init"][name] = @benchmarkable begin
        MatchCache(mark_x, $𝒞_TTT, $board)
    end
end
