using Test

@testset "RewriteGames" begin
    include("test_core.jl")
    include("test_engine.jl")
    include("test_agents.jl")
    include("test_encoding.jl")
    include("test_serialization.jl")
    include("test_migration.jl")
    include("test_bug_fixes.jl")
    include("test_analysis.jl")
    include("test_dsl.jl")
    include("test_schedule.jl")
    include("test_json_serialization.jl")
    include("test_gpu_rewriting.jl")
    include("test_gpu_schedule.jl")
    include("test_gpu_agent_loop.jl")
    include("test_gpu_nac.jl")
    include("test_gpu_sampling.jl")
    include("test_gpu_download.jl")
end
