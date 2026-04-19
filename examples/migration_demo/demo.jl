"""
    Migration Demo

Demonstrates using GameMigration to migrate a game between two related schemas.

Scenario 1: Identity migration (Schema A → Schema A).
  Migrates worlds, rules, and the entire game using the identity functor.
  This is the simplest case and verifies the migration infrastructure.

Scenario 2: Inclusion migration (Schema A → Schema B).
  Schema B adds a Weight attribute to Schema A's edges.
  migrate_world works correctly; migrate_rules is shown but skipped for
  rules defined without attributes (a known limitation when new attribute
  types are introduced—the caller must re-define rules in the target schema).
"""

using Catlab
using AlgebraicRewriting
using RewriteGames

# ─── Schema A: plain directed graph ──────────────────────────────────────────
@present SchA(FreeSchema) begin
    V::Ob; E::Ob
    src::Hom(E,V); tgt::Hom(E,V)
end
@acset_type GraphA(SchA, index=[:src,:tgt])

# ─── Rewrite rules for Schema A ───────────────────────────────────────────────
I_empty = GraphA()
R_one_v = begin g = GraphA(); add_parts!(g, :V, 1); g end
rule_add_vertex = Rule(
    ACSetTransformation(I_empty, I_empty),
    ACSetTransformation(I_empty, R_one_v),
)

game_A = Game(
    SchA;
    players  = [:alice],
    rules    = Dict(:alice => [RuleEntry(rule_add_vertex; name=:add_vertex)]),
    terminal = (W) -> (nparts(W, :V) >= 5, nothing),
    initial  = () -> begin g = GraphA(); add_parts!(g, :V, 2); g end,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Identity migration SchA → SchA
# ═══════════════════════════════════════════════════════════════════════════════
println("── Scenario 1: Identity migration (SchA → SchA) ──")

cat_A = ACSetCategory(GraphA())
F_id  = Migrate(cat_A, SchA, GraphA)   # identity: maps every name to itself

migration_id = GameMigration(F_id, game_A, SchA)

# World migration
w_a = game_A.initial()
w_a2 = migrate_world(migration_id, w_a)
println("Original world: V=$(nparts(w_a,:V))")
println("Migrated world: V=$(nparts(w_a2,:V))  (should be same)")

# Rule migration
lib_a  = game_A.rules[:alice]
lib_a2 = migrate_rules(migration_id, lib_a)
println("Migrated library has $(length(lib_a2)) rule(s), names: " *
        join([e.name for e in lib_a2.entries], ", "))

# Full game migration
game_A2 = migrate_game(migration_id)
println("Migrated game players: $(game_A2.players)")

# Verify the migrated game plays correctly
agents = Dict{Symbol, AbstractAgent}(:alice => FunctionAgent((s,a) -> rand(a)))
exps   = run_game(game_A2, agents; T_max=20)
println("Episode length: $(length(exps)) steps, done=$(exps[end].done)")

# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Inclusion migration SchA → SchB (world only)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n── Scenario 2: Inclusion migration (SchA → SchB, world only) ──")

@present SchB(FreeSchema) begin
    V::Ob; E::Ob
    src::Hom(E,V); tgt::Hom(E,V)
    Weight::AttrType
    weight::Attr(E, Weight)
end
@acset_type WeightedGraphB(SchB, index=[:src,:tgt])

cat_A2 = ACSetCategory(GraphA())
F_incl = Migrate(
    cat_A2,
    Dict(:V => :V, :E => :E),
    Dict(:src => :src, :tgt => :tgt),
    SchA, GraphA,
    SchB, WeightedGraphB{Float64};
    delta = false,
)

# Build a world with edges so migration is interesting
w_with_edges = GraphA()
add_parts!(w_with_edges, :V, 3)
add_parts!(w_with_edges, :E, 2, src=[1,2], tgt=[2,3])

println("Source world: V=$(nparts(w_with_edges,:V)) E=$(nparts(w_with_edges,:E))")
migration_b = GameMigration(F_incl, game_A, SchB)
w_b = migrate_world(migration_b, w_with_edges)
println("Migrated world: V=$(nparts(w_b,:V)) E=$(nparts(w_b,:E))")
println("  weight attributes (AttrVar free vars): $(subpart(w_b, :weight))")
println("\nNote: migrate_rules with new attribute types requires rules")
println("      to be re-defined in the target schema by the caller.")

println("\nDemo complete.")


