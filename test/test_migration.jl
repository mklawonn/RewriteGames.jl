using Test
using RewriteGames

@testset "Migration stubs" begin
    @testset "GameMigration struct exists" begin
        @test isdefined(RewriteGames, :GameMigration)
        @test isdefined(RewriteGames, :migrate_world)
        @test isdefined(RewriteGames, :migrate_rules)
        @test isdefined(RewriteGames, :migrate_game)
    end

    @testset "GameMigration constructs" begin
        game = Game(
            nothing;
            players  = [:p],
            rules    = Dict(:p => RuleEntry[]),
            terminal = (W) -> (false, nothing),
            initial  = () -> nothing,
        )
        migration = GameMigration(nothing, game, nothing)
        @test migration.source_game === game
        @test migration.functor     === nothing
        @test migration.target_schema === nothing
    end
end
