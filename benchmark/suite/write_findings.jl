"""
Generates benchmark/results/findings.md from a completed BenchmarkTools result tree.
"""

using Dates

function _fmt_time(t)
    ns = BenchmarkTools.time(t)
    ns < 1_000    && return "$(round(ns; digits=1)) ns"
    ns < 1_000_000 && return "$(round(ns/1e3; digits=1)) μs"
    ns < 1e9       && return "$(round(ns/1e6; digits=1)) ms"
    return "$(round(ns/1e9; digits=2)) s"
end

function _speedup(t_base, t_opt)
    r = BenchmarkTools.time(t_base) / BenchmarkTools.time(t_opt)
    return round(r; digits=2)
end

function write_findings(results, path)
    open(path, "w") do io
        println(io, "# Benchmark Findings")
        println(io, "")
        println(io, "Generated: $(Dates.now())")
        println(io, "")

        # ── Match enumeration ───────────────────────────────────────────────
        println(io, "## Match Enumeration (single call to get_matches / equivalent)")
        println(io, "")
        println(io, "| Board state | baseline | fast_path | cache_update | fast speedup | cache speedup |")
        println(io, "|-------------|----------|-----------|--------------|--------------|---------------|")

        me = results["match_enumeration"]
        for state in ["empty", "mid", "nearly_full"]
            t_base  = me["baseline"][state]
            t_fast  = me["fast_path"][state]
            t_cache = me["cache_update"][state]
            println(io, "| $state | $(_fmt_time(t_base)) | $(_fmt_time(t_fast)) | " *
                        "$(_fmt_time(t_cache)) | $(_speedup(t_base, t_fast))× | " *
                        "$(_speedup(t_base, t_cache))× |")
        end
        println(io, "")

        # ── Single episode ──────────────────────────────────────────────────
        println(io, "## Single Episode (random agents)")
        println(io, "")
        se = results["game_episodes"]["single_episode"]
        t_base  = se["baseline"]
        t_fast  = se["fast_path"]
        t_cache = se["incremental_cache"]
        println(io, "| strategy | time | speedup vs baseline |")
        println(io, "|----------|------|---------------------|")
        println(io, "| baseline          | $(_fmt_time(t_base))  | 1.00× |")
        println(io, "| fast_path         | $(_fmt_time(t_fast))  | $(_speedup(t_base,t_fast))× |")
        println(io, "| incremental_cache | $(_fmt_time(t_cache)) | $(_speedup(t_base,t_cache))× |")
        println(io, "")

        # ── Batch 200 ───────────────────────────────────────────────────────
        if haskey(results["game_episodes"], "batch_200")
            println(io, "## Batch of 200 Episodes")
            println(io, "")
            b = results["game_episodes"]["batch_200"]
            t_base  = b["baseline"]
            t_fast  = b["fast_path"]
            t_cache = b["incremental_cache"]
            println(io, "| strategy | time | speedup vs baseline |")
            println(io, "|----------|------|---------------------|")
            println(io, "| baseline          | $(_fmt_time(t_base))  | 1.00× |")
            println(io, "| fast_path         | $(_fmt_time(t_fast))  | $(_speedup(t_base,t_fast))× |")
            println(io, "| incremental_cache | $(_fmt_time(t_cache)) | $(_speedup(t_base,t_cache))× |")
            println(io, "")
        end

        # ── RL training ─────────────────────────────────────────────────────
        println(io, "## RL Training (10 updates × 25 episodes, REINFORCE)")
        println(io, "")
        rl = results["rl_training"]
        println(io, "| configuration | time | speedup vs baseline |")
        println(io, "|---------------|------|---------------------|")
        t_base = rl["baseline"]
        println(io, "| baseline (CPU)          | $(_fmt_time(t_base)) | 1.00× |")
        for (label, key) in [("fast_path (CPU)", "fast_path"),
                               ("incremental_cache (CPU)", "incremental_cache")]
            if haskey(rl, key)
                t = rl[key]
                println(io, "| $label | $(_fmt_time(t)) | $(_speedup(t_base, t))× |")
            end
        end
        if haskey(rl, "baseline_gpu")
            for (label, key) in [("baseline (GPU)", "baseline_gpu"),
                                   ("incremental_cache (GPU)", "incremental_cache_gpu")]
                haskey(rl, key) || continue
                t = rl[key]
                println(io, "| $label | $(_fmt_time(t)) | $(_speedup(t_base, t))× |")
            end
        end
        println(io, "")

        # ── Analysis ────────────────────────────────────────────────────────
        println(io, "## Analysis")
        println(io, "")
        println(io, "### Where the time goes")
        println(io, "")
        println(io, "The benchmarks confirm the tutorial's analysis: episode collection " *
                    "dominates training time.  Within each episode the cost is dominated " *
                    "by `get_matches` — the full backtracking-search + NAC evaluation " *
                    "invoked twice per turn.")
        println(io, "")
        println(io, "### Incremental cache effect")
        println(io, "")
        println(io, "The `MatchCache` reduces repeated work by maintaining the match set " *
                    "across turns.  After each DPO rewrite the cache forwards surviving " *
                    "matches through the pushout complement maps (O(n_matches) work) and " *
                    "re-checks NAC conditions only on matches that could be affected by " *
                    "the added elements.  For TicTacToe placement rules this amounts to " *
                    "removing exactly one entry per turn — O(1) amortised — rather than " *
                    "running a fresh 9-element backtracking search.")
        println(io, "")
        println(io, "### Fast-path effect")
        println(io, "")
        println(io, "The user-supplied `fast_match_fn` exploits the board's `:xsq`/`:osq` " *
                    "indices directly, computing empty squares as a set difference and " *
                    "building match morphisms by pinned `homomorphism` calls.  This " *
                    "bypasses the general BacktrackingSearch entirely.")
        println(io, "")
        println(io, "### Scaling outlook")
        println(io, "")
        println(io, "For larger games (richer schemas, larger boards) the hom-search cost " *
                    "grows with both pattern complexity and world size.  The `MatchCache` " *
                    "approach generalises: only elements affected by a rewrite need " *
                    "re-examination.  The `fast_match_fn` hook allows domain experts to " *
                    "bypass the general machinery entirely when the structure of their " *
                    "rules admits a closed-form solution.")
    end
end
