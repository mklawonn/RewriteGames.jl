"""
    GameMigration

Wraps a Catlab/AlgebraicRewriting `Migrate` functor together with a source
`Game` and a target schema, enabling world migration across related schemas.

Use `player_migrate` (from `src/schedule/player_rule_app.jl`) to migrate an
entire `GameSched` including player-tagged rule boxes.

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
