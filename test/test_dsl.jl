using Test
using RewriteGames
using Catlab
using AlgebraicRewriting

@testset "DSL @game macro" begin
    I_empty = Graph()
    R_one_v = Graph(1)
    rule_add_vertex = Rule(ACSetTransformation(I_empty, I_empty),
                           ACSetTransformation(I_empty, R_one_v))
    entry_v = RuleEntry(rule_add_vertex; name=:add_vertex)

    @testset "@game basic construction" begin
        g = @game nothing begin
            players:  alice, bob
            alice:    [entry_v]
            bob:      RuleEntry[]
            terminal: (W) -> (false, nothing)
            initial:  () -> nothing
        end

        @test g isa Game
        @test g.players == [:alice, :bob]
        @test length(g.rules[:alice]) == 1
        @test g.rules[:alice][1].name == :add_vertex
        @test isempty(g.rules[:bob])
    end

    @testset "@game player order inferred from rule clauses" begin
        g = @game nothing begin
            red:   [entry_v]
            blue:  RuleEntry[]
        end
        @test g.players == [:red, :blue]
    end

    @testset "@game matches equivalent Game(...) call" begin
        g_macro = @game nothing begin
            players:  p
            p:        [entry_v]
            terminal: (W) -> (nparts(W, :V) >= 5, nothing)
            initial:  () -> Graph(1)
        end
        g_hand = Game(
            nothing;
            players  = [:p],
            rules    = Dict(:p => [entry_v]),
            terminal = (W) -> (nparts(W, :V) >= 5, nothing),
            initial  = () -> Graph(1),
        )
        @test g_macro.players == g_hand.players
        @test length(g_macro.rules[:p]) == length(g_hand.rules[:p])
    end
end
