"""
    GameMigration

Wraps a Catlab/AlgebraicRewriting `Migrate` functor together with a source
`Game` and a target schema, enabling world migration and rule migration across
related schemas.

# Fields
- `functor`:       An `AlgebraicRewriting.Migrate` object encoding the schema morphism.
- `source_game`:   The original `Game`.
- `target_schema`: The target schema presentation (optional metadata).
"""
struct GameMigration
    functor       :: Any   # AlgebraicRewriting.Migrate
    source_game   :: Game
    target_schema :: Any
end

Base.show(io::IO, m::GameMigration) =
    print(io, "GameMigration(players=$(m.source_game.players))")

"""
    migrate_world(migration::GameMigration, W) -> ACSet

Apply the schema functor to the ACSet world `W`, producing a world in the
target schema.
"""
function migrate_world(migration::GameMigration, W)
    return migration.functor(W)
end

"""
    migrate_rules(migration::GameMigration, library::RuleLibrary) -> RuleLibrary

Push each `Rule` in `library` forward through the schema functor via
AlgebraicRewriting's `(F::Migrate)(r::Rule)` implementation.

Returns a new `RuleLibrary` with migrated rules.  Budget and post-filter
settings are preserved.
"""
function migrate_rules(migration::GameMigration, library::RuleLibrary)
    F = migration.functor
    migrated_entries = map(library.entries) do entry
        new_rule = F(entry.rule)
        RuleEntry(new_rule; name=entry.name, budget=entry.budget,
                  post_filter=entry.post_filter)
    end
    return RuleLibrary(migrated_entries)
end

"""
    migrate_game(migration::GameMigration; initial=nothing, terminal=nothing) -> Game

Produce a new `Game` in the target schema by:
1. Migrating each player's `RuleLibrary` via `migrate_rules`.
2. Migrating each `AutoRule`'s underlying rule via the functor.
3. Preserving player order and the terminal predicate.

!!! note "Initial world factory"
    The `initial` factory of the source game produces ACSet instances in the **source**
    schema. After migration the returned game still references the original factory
    unless you override it via the `initial` keyword argument. When migrating to a
    different schema, always supply a new `initial` that returns a target-schema ACSet:

    ```julia
    migrate_game(m; initial = () -> TargetACSet())
    ```

The `terminal` keyword allows supplying a replacement terminal predicate for the same
reason: if the predicate references source-schema attribute names, it will not type-check
against a target-schema world.
"""
function migrate_game(migration::GameMigration;
                      initial::Union{Function,Nothing}  = nothing,
                      terminal::Union{Function,Nothing} = nothing) :: Game
    src = migration.source_game
    F   = migration.functor

    new_rules = Dict{Symbol, RuleLibrary}(
        p => migrate_rules(migration, src.rules[p]) for p in src.players
    )
    new_auto = AutoRule[
        AutoRule(F(ar.rule); name=ar.name, prob_attr=ar.prob_attr)
        for ar in src.auto
    ]

    return Game(
        migration.target_schema;
        players  = src.players,
        rules    = new_rules,
        auto     = new_auto,
        terminal = terminal !== nothing ? terminal : src.terminal,
        initial  = initial  !== nothing ? initial  : src.initial,
    )
end
