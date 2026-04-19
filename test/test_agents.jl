using Test
using RewriteGames
using Catlab

@testset "Agent tests" begin
    # Build a minimal EncodedState for testing
    W = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
    enc = encode_state(W, Dict{Tuple{Symbol,Int},Int}(), 1, 50)

    # Build minimal Action instances
    rule = RuleEntry("stub"; name=:test)
    actions = [Action(rule, nothing), Action(rule, nothing), Action(rule, nothing)]

    @testset "FunctionAgent wraps callable" begin
        agent  = FunctionAgent((state, acts) -> acts[1])
        result = select_action(agent, enc, actions)
        @test result === actions[1]
    end

    @testset "FunctionAgent random policy" begin
        agent   = FunctionAgent((state, acts) -> rand(acts))
        results = [select_action(agent, enc, actions) for _ in 1:30]
        @test all(r -> r in actions, results)
    end

    @testset "select_action dispatches on FunctionAgent" begin
        called = Ref(false)
        agent  = FunctionAgent((state, acts) -> (called[] = true; acts[1]))
        select_action(agent, enc, actions)
        @test called[]
    end
end
