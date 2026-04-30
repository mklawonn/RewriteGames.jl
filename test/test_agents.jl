using Test
using RewriteGames
using Catlab

@testset "Agent tests" begin
    W     = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    state = GameState(W, 1)

    stub    = (name=:test, rule="stub")
    actions = [Action(stub, nothing), Action(stub, nothing), Action(stub, nothing)]

    @testset "FunctionAgent wraps callable" begin
        agent  = FunctionAgent((state, acts) -> acts[1])
        result = select_action(agent, state, actions)
        @test result === actions[1]
    end

    @testset "FunctionAgent random policy" begin
        agent   = FunctionAgent((state, acts) -> rand(acts))
        results = [select_action(agent, state, actions) for _ in 1:30]
        @test all(r -> r in actions, results)
    end

    @testset "select_action dispatches on FunctionAgent" begin
        called = Ref(false)
        agent  = FunctionAgent((state, acts) -> (called[] = true; acts[1]))
        select_action(agent, state, actions)
        @test called[]
    end

    @testset "agent receives GameState with correct fields" begin
        received = Ref{Any}(nothing)
        agent    = FunctionAgent((s, acts) -> (received[] = s; acts[1]))
        select_action(agent, state, actions)
        @test received[] isa GameState
        @test received[].world === W
        @test received[].turn == 1
    end
end
