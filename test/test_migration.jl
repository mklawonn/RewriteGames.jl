using Test
using RewriteGames

@testset "Migration" begin
    @testset "GameMigration struct exists" begin
        @test isdefined(RewriteGames, :GameMigration)
        @test isdefined(RewriteGames, :migrate_world)
    end

    @testset "GameMigration constructs" begin
        game = Game(
            nothing;
            players  = [:p],
            terminal = (W) -> (false, nothing),
            initial  = () -> nothing,
        )
        migration = GameMigration(nothing, game, nothing)
        @test migration.source_game === game
        @test migration.functor     === nothing
        @test migration.target_schema === nothing
    end

    @testset "migrate_world applies functor" begin
        # Use identity as a stand-in functor (returns its argument)
        identity_functor = identity
        W = "fake_world"
        migration = GameMigration(
            identity_functor,
            Game(nothing; players=[:p], terminal=(W)->(false,nothing), initial=()->nothing),
            nothing,
        )
        @test migrate_world(migration, W) === W
    end
end
