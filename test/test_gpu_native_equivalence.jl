using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

@testset "GPU-Native Equivalence Tests" begin
    if !CUDA.functional()
        @warn "CUDA not functional — skipping GPU equivalence tests"
        @test_skip true
    else
        @testset "Simple Loop: Vertex Addition" begin
            I = Graph()
            rule = Rule(homomorphism(I, I), homomorphism(I, Graph(1)))
            cat = ACSetCategory(I)
            alice_app = PlayerRuleApp(:add, rule, I, :alice; cat=cat)
            
            N = Names(Dict("I" => I))
            sched = mk_game_sched(NamedTuple(), (init=:I,), N,
                (r=tryrule(alice_app),),
                quote
                    out = r(init)
                    return out
                end; cat=cat)
            
            # Use deterministic seed for repeatable results
            agents = Dict(:alice => FunctionAgent((s,a)->first(a))) 
            
            # Verify CPU first
            cpu_exps = run_game_sched!(sched, I, agents; T_max=1)
            @test length(cpu_exps) > 0
            @test nparts(cpu_exps[1].next_state.world, :V) == 1
            
            # Verify GPU
            gpu_exps = gpu_run_game_sched!(sched, I, agents; T_max=1)
            @test length(gpu_exps) > 0
            if length(gpu_exps) > 0
                @test nparts(gpu_exps[1].next_state.world, :V) == 1
            else
                @test "GPU failed to produce experience" == ""
            end
        end
    end
end
