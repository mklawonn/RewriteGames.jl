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

---

## Preferred Style for Rule Construction (follow the LV example)

The canonical reference is `lotka_volterra_example.jl` in this repository.

### Convenience aliases

Define at the top of every rule file:

```julia
const AV  = AttrVar
const AV1 = AttrVar(1)
```

### `@acset` over `add_parts!`

Construct K, L, R pattern ACSets declaratively:

```julia
# Good
K = @acset MyACSet begin
    Sym = 2; Boo = 1; Sheep = 1; V = 1
    sheep_loc = [1]; sheep_eng = [AV1]; sheep_dir = [AV(2)]
end

# Avoid
K = MyACSet()
add_parts!(K, :Sym, 2); add_part!(K, :Boo)
add_part!(K, :Sheep; sheep_loc=1, sheep_eng=AttrVar(1), sheep_dir=AttrVar(2))
add_part!(K, :V)
```

### `homomorphism` over `ACSetTransformation` + identity maps

Compute span morphisms and NAC inclusions with `homomorphism`:

```julia
# Good — AttrVars present, category inferred
l = homomorphism(K, L; monic=true)
r = homomorphism(K, R; monic=true)

# Good — no AttrVars in K/L, pass cat= explicitly
l = homomorphism(K, L; cat=MY_CAT, monic=true)

# Good — NAC as monic inclusion
L_nac = copy(L)
add_part!(L_nac, :SomeToken; some_fk=1)
nac = NAC(homomorphism(L, L_nac; monic=true))

# Avoid
comps = _id_comps(K)
l = ACSetTransformation(K, L; cat=MY_CAT, comps...)
nac = NAC(ACSetTransformation(L, L_nac; cat=MY_CAT, _id_comps(L)...))
```

### Representable presheaves (yoneda cache)

For minimal single-object patterns, use the yoneda cache:

```julia
gSheep, gWolf, gV, … = ob_generators(FinCat(SchLV))
yLV = yoneda_cache(LV; clear=true)
S = ob_map(yLV, gSheep)   # generic sheep — minimal ACSet with one Sheep + all attrs
W = ob_map(yLV, gWolf)
G = ob_map(yLV, gV)
```

Representables are the natural interfaces for agent morphisms and for patterns where
you want the most general match against a single entity.

### `expr` for attribute-modifying rules

When a rule changes an attribute value without adding or removing objects, use `expr`:

```julia
# Good: turn a sheep left without deleting and re-creating it
Rule(l, r;
     expr = (Dir = [((d,),) -> left(d)],),
     cat  = MY_CAT)
```

`expr` is a `NamedTuple` keyed by AttrType name. Each entry is a vector of functions
(one per R AttrVar) receiving inherited value tuples and returning the new value.
This avoids the spurious object deletion/recreation that would otherwise be needed
to change an attribute.

---

## Known Issue: `done` flag in `gpu_run_game_sched!`

`run_game_sched!` (CPU) sets `experience.done = true` on the final experience
whenever the terminal predicate fires or `turn > T_max`.

`gpu_run_game_sched!` can exit for a third reason: a schedule exit wire fires
(no active trace wire to continue). In that case the terminal predicate is not
consulted and the last experience is pushed with `done = false`.

**Impact:** Code that inspects `last(exps).done` to detect episode termination
will behave differently depending on which engine ran the episode.

**Possible fix:** In the GPU runner's exit-wire path, call the terminal predicate
(or unconditionally set `done = true`) on the final experience before returning,
mirroring the CPU runner's behaviour in `_exec_player!`.
