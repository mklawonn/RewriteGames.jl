using Test

@testset "RewriteGames" begin
    include("test_core.jl")
    include("test_engine.jl")
    include("test_agents.jl")
    include("test_encoding.jl")
    include("test_serialization.jl")
    include("test_migration.jl")
end
