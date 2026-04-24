using Test
using RewriteGames
using Catlab

@testset "Encoding tests" begin
    W    = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
    turn = 1
    T_max = 100

    enc = encode_state(W, turn, T_max)

    @testset "EncodedState structure" begin
        @test enc isa EncodedState
        @test enc.raw isa GameState
        @test enc.raw.world === W
        @test enc.raw.turn == turn
    end

    @testset "node_features shape" begin
        nf = enc.node_features
        @test nf isa Matrix{Float32}
        # Total nodes = 3 V-nodes + 2 E-nodes = 5
        @test size(nf, 1) == 5
        # Features: 2 (one-hot ob type: V or E) + 0 attrs = 2
        @test size(nf, 2) == 2
        @test all(nf[1:3, 1] .== 1f0)
        @test all(nf[4:5, 2] .== 1f0)
    end

    @testset "edge_index shape" begin
        ei = enc.edge_index
        @test ei isa Matrix{Int32}
        @test size(ei, 1) == 2
        # 2 edges × 2 morphisms (src, tgt) = 4 entries
        @test size(ei, 2) == 4
    end

    @testset "edge_type length" begin
        @test length(enc.edge_type) == size(enc.edge_index, 2)
    end

    @testset "turn_frac" begin
        @test enc.turn_frac ≈ Float32(turn / T_max)
    end

    @testset "empty world" begin
        W_empty = Graph()
        enc_e   = encode_state(W_empty, 1, T_max)
        @test size(enc_e.node_features) == (0, 2)
        @test size(enc_e.edge_index, 2) == 0
    end
end
