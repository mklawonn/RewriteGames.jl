# RewriteGames.jl — Agent Notes

## Ambient Category and `rewrite_match`

### Background

AlgebraicRewriting's `rewrite_match(rule, m)` calls `infer_acset_cat(m)` when no `cat=`
argument is supplied. That function returns `MADVarACSetCat` only when
`hasvar(dom(m)) || hasvar(codom(m))` — i.e., when the match morphism's domain (the
rule's `L`) or codomain (the world) contains AttrType parts (AttrVars). The world state
never contains AttrVars. So if a rule's `L` also has no AttrVars, `infer_acset_cat`
falls back to `MADACSetCat`, which is incompatible with matches produced by
`get_matches(...; cat=MADVarACSetCat)`.

### Engine Fix (`sched_runner.jl`)

`_exec_native_rule!` computes `_cat` at line 278 and passes it to `get_matches`, but
the `rewrite_match` call at line 292 originally omitted `cat=`. **Fixed:** that call now
passes `cat=_cat` explicitly so the same category is used for both match search and
rewrite application.

The `PlayerRuleApp` path (`_exec_player!`) is not affected: it uses
`rewrite_match_maps(box.rule, chosen.match; cat=_cat)` and derives the new world from
`codom(maps[:rh])`, never calling bare `rewrite_match`.

### What `Rule(l, r; cat=...)` Does NOT Do

`Rule` uses `cat` during construction for span validation but **does not store it as a
field**. Passing `cat=FALCON_CAT` to `Rule` has no effect on subsequent `rewrite_match`
calls. Always propagate the category explicitly through `get_matches` → `rewrite_match`.

### Rules for Engine Contributors

1. **Always pass `cat=` to `rewrite_match`.** Derive it from `box.cat`, the game
   schedule's `cat`, or a known ACSetCategory constant. Never rely on `infer_acset_cat`.

2. **`_cat` is already computed in both execution paths.** In `_exec_player!` it is
   `box.cat` (or `infer_acset_cat(world)` as fallback). In `_exec_native_rule!` it is
   `box.cat` (or `infer_acset_cat(L)` as fallback). Use it.

3. **The `infer_acset_cat(world)` and `infer_acset_cat(L)` fallbacks are fragile.**
   The world state has no AttrVars, so both return `MADACSetCat` if the caller did not
   supply `box.cat`. Games that use rules with AttrVars must set `cat=` on every
   `PlayerRuleApp`. See the game-specific AGENT.md (e.g., `FalconRewriteGame/AGENT.md`)
   for guidance on when a rule's L has or lacks AttrVars.

4. **Free-floating AttrVars are rejected by Catlab.** An AttrType part (Sym, Boo, Num)
   that is allocated but not referenced by any attribute in the same ACSet is
   "free-floating". `backtracking_search` will error with
   `Cannot search for morphisms with free-floating variables`. Adding a dummy AttrType
   part just to make `hasvar(L)` true is never the correct fix — pass `cat=` explicitly
   instead.

### `match_cache.jl`

`MatchCache` stores and propagates `cat` through all internal `homomorphism`,
`get_matches`, and `can_match` calls. No bare category inference occurs there.
The cache must be constructed with the same `cat` that the game uses for match search
(`MatchCache(rule, FALCON_CAT, world)`).
