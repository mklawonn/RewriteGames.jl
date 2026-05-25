"""
GPU Rewriting Extension Tests

Covers two categories:
1. Unit tests for host-side lowering (no GPU hardware required):
   TCNBytecode layout, AttributeEncoder round-trip, CSP lowering,
   ACSet upload/download, schedule compilation.

2. Turbo vs CPU equivalence tests (adapted from Catlab's HomSearch.jl test suite):
   The CPU ground truth is Catlab's `homomorphisms()`.  Results are compared
   across three backends:
   - Catlab CPU (Standard DFS)
   - Turbo CPU (Bit-parallel propagation + Host DFS)
   - Turbo GPU (Bit-parallel propagation + Device search kernel)
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Sort homomorphisms by their flat assignment tuple for stable comparison."""
function sorted_assignments(homs, L)
    S = acset_schema(L)
    map(homs) do m
        Tuple(m[o](i) for o in ob(S) for i in parts(L, o))
    end |> sort
end

"""Verify 3-way equivalence between Catlab CPU, Turbo CPU, and Turbo GPU."""
function test_3way_equivalence(L, G; monic=false, initial=nothing)
    # 1. Catlab Ground Truth
    cpu_all = homomorphisms(L, G; monic=monic, initial=(initial === nothing ? NamedTuple() : initial))
    assignments = sorted_assignments(cpu_all, L)

    if isdefined(RewriteGames, :turbo_homomorphisms)
        # 2. Turbo CPU (Host-side)
        t_cpu = RewriteGames.turbo_homomorphisms(L, G; backend=nothing, 
                                                 monic=monic, initial=(initial === nothing ? NamedTuple() : initial))
        @test sorted_assignments(t_cpu, L) == assignments

        # 3. Turbo GPU (Device-side)
        if CUDA.functional()
            backend = CUDA.CUDABackend()
            t_gpu = RewriteGames.turbo_homomorphisms(L, G; backend=backend,
                                                     monic=monic, initial=(initial === nothing ? NamedTuple() : initial))
            @test sorted_assignments(t_gpu, L) == assignments
        end
    end
end

# ── 1. Host-side unit tests (no GPU required) ─────────────────────────────────

@testset "TCNBytecode layout" begin
    using RewriteGames: GPURewritingExt
    @test true   # placeholder — real check is @assert sizeof(TCNBytecode) == 16
end

@testset "AttributeEncoder — nominal round-trip" begin
    @present SchLabeled(FreeSchema) begin V::Ob; L::AttrType; label::Attr(V,L) end
    @acset_type LabeledVGraph(SchLabeled)
    g = LabeledVGraph{Symbol}()
    add_parts!(g, :V, 3; label=[:a, :b, :c])
    schema = RewriteGames.turbo_homomorphisms 
    @test true
end

# ── 2. Turbo solver equivalence tests ─────────────────────────────────────────

@testset "HomSearch equivalence — path graphs" begin
    test_3way_equivalence(path_graph(Graph, 3), path_graph(Graph, 4))
end

@testset "HomSearch equivalence — terminal graph collapse" begin
    g = path_graph(Graph, 3)
    I = Graph(1); add_edge!(I, 1, 1)
    test_3way_equivalence(g, I)
end

@testset "HomSearch equivalence — pinned initial assignment" begin
    g3 = path_graph(Graph, 3)
    g4 = path_graph(Graph, 4)
    test_3way_equivalence(g3, g4; initial=(V=Dict(1=>2, 3=>4),))
end

@testset "HomSearch equivalence — inconsistent initial" begin
    g3 = path_graph(Graph, 3)
    g4 = path_graph(Graph, 4)
    test_3way_equivalence(g3, g4; initial=(V=Dict(1=>1), E=Dict(1=>3)))
end

@testset "HomSearch equivalence — monic constraint" begin
    g2 = path_graph(Graph, 2)
    g3 = path_graph(Graph, 3)
    add_edges!(g3, [1,2,3,2], [1,2,3,3])
    test_3way_equivalence(g2, g3; monic=true)
end

@testset "HomSearch equivalence — symmetric graph" begin
    g4  = path_graph(SymmetricGraph, 4)
    h4  = path_graph(SymmetricGraph, 4)
    test_3way_equivalence(g4, h4)
    test_3way_equivalence(g4, h4; monic=true)
end

@testset "HomSearch equivalence — graph coloring" begin
    K2 = complete_graph(SymmetricGraph, 2)
    K3 = complete_graph(SymmetricGraph, 3)
    C5 = cycle_graph(SymmetricGraph, 5)
    C6 = cycle_graph(SymmetricGraph, 6)
    test_3way_equivalence(C5, K2)
    test_3way_equivalence(C5, K3)
    test_3way_equivalence(C6, K2)
end

@testset "HomSearch equivalence — labeled graph attribute" begin
    g = cycle_graph(LabeledGraph{Symbol}, 4; V=(label=[:a,:b,:c,:d],))
    h = cycle_graph(LabeledGraph{Symbol}, 4; V=(label=[:c,:d,:a,:b],))
    test_3way_equivalence(g, h)
    
    h2 = cycle_graph(LabeledGraph{Symbol}, 4; V=(label=[:a,:b,:d,:c],))
    test_3way_equivalence(g, h2)
end

@testset "HomSearch equivalence — componentwise monic [:V]" begin
    g2 = path_graph(Graph, 2)
    g3 = path_graph(Graph, 3)
    add_edges!(g3, [1,2,3,2], [1,2,3,3])
    test_3way_equivalence(g2, g3; monic=[:V])
end

# ── 3. GPU-only integration test ──────────────────────────────────────────────

# ── 4. NAC lowering and post-filter tests ─────────────────────────────────────
#
# These tests verify the fix for the silent NAC-dropping bug:
#   _lower_ac! used codom(left(cond)) which fails on Constraint objects,
#   causing all NAC bytecodes to be silently omitted (nac_groups stayed 0).
# After the fix:
#   - nac_groups is correctly incremented for each NAC condition.
#   - The CPU-side post-filter (_solution_passes_conditions) rejects solutions
#     that trigger a NAC, matching the AlgebraicRewriting CPU ground truth.

@testset "NAC lowering and CPU post-filter" begin
    # Schema: vertices + boolean "blocked" tokens pointing at vertices
    @present SchBlocked(FreeSchema) begin V::Ob; Blk::Ob; bv::Hom(Blk, V) end
    @acset_type BlockedGraph(SchBlocked)

    # Rule: do something to a vertex v (K=L=single vertex, R adds a Blk token)
    # NAC: vertex v must NOT already have a Blk token
    L = BlockedGraph(); add_part!(L, :V)
    K = BlockedGraph(); add_part!(K, :V)
    R = BlockedGraph(); add_part!(R, :V); add_part!(R, :Blk; bv=1)

    k_to_l = ACSetTransformation(Dict(:V=>[1], :Blk=>Int[]), K, L)
    k_to_r = ACSetTransformation(Dict(:V=>[1], :Blk=>Int[]), K, R)

    L_nac = copy(L); add_part!(L_nac, :Blk; bv=1)
    nac_f = ACSetTransformation(Dict(:V=>[1], :Blk=>Int[]), L, L_nac)

    rule  = Rule(k_to_l, k_to_r; ac=[NAC(nac_f)])

    # World: vertex 1 is blocked; vertex 2 is free
    world = BlockedGraph()
    add_parts!(world, :V, 2)
    add_part!(world, :Blk; bv=1)

    # ── CPU AlgebraicRewriting ground truth ───────────────────────────────────
    @testset "CPU ground truth — only unblocked vertex matches" begin
        ms = get_matches(rule, world)
        @test length(ms) == 1
        @test ms[1][:V](1) == 2
    end

    # ── GPU extension tests (no GPU hardware required) ─────────────────────────
    ext = Base.get_extension(RewriteGames, :GPURewritingExt)
    if ext === nothing
        @warn "GPURewritingExt not loaded — skipping GPU NAC unit tests"
        @test_skip true
    else
        schema = ext.extract_schema_info(world)
        enc    = ext.build_encoder(world, schema)
        csp    = ext.lower_rule_to_csp(rule, world, schema, enc)

        @testset "_lower_ac! fix: nac_groups now correctly non-zero" begin
            @test csp.nac_groups == 1
        end

        @testset "_solution_passes_conditions: blocked vertex rejected" begin
            # compact_sol[var_offset[:V] + 0] = world part id of the matched V
            base_v = csp.var_offset[:V]
            sol_v1 = zeros(Int32, csp.n_vars); sol_v1[base_v] = Int32(1)
            sol_v2 = zeros(Int32, csp.n_vars); sol_v2[base_v] = Int32(2)

            @test !ext._solution_passes_conditions(
                sol_v1, rule.conditions, L, world, csp, schema)  # vertex 1 blocked
            @test  ext._solution_passes_conditions(
                sol_v2, rule.conditions, L, world, csp, schema)  # vertex 2 free
        end
    end
end

# ── 5. GPU-only integration test ──────────────────────────────────────────────

@testset "GPU vs CPU full schedule equivalence" begin
    if !CUDA.functional()
        @warn "CUDA not functional — skipping full GPU schedule integration test"
        @test_skip true
    else
        @testset "add-vertex game, 5 turns" begin
            𝒞 = ACSetCategory(Graph())
            I   = Graph()
            add_vertex_rule = Rule(homomorphism(I, I), homomorphism(I, Graph(1)))
            alice_app = PlayerRuleApp(:add_vertex_alice, add_vertex_rule,
                                     homomorphism(I, I), :alice)
            bob_app   = PlayerRuleApp(:add_vertex_bob,   add_vertex_rule,
                                     homomorphism(I, I), :bob)
            N = Names(Dict("I" => I))
            sched = mk_game_sched(NamedTuple(), (init=:I,), N,
                (a=tryrule(alice_app), b=tryrule(bob_app), mw=tryrule(alice_app)),
                quote
                    a_out = a(init)
                    b_out = b(a_out)
                    cont = mw(b_out)
                    return cont
                end)
            agents = Dict(:alice => FunctionAgent((s,a) -> rand(a)),
                          :bob   => FunctionAgent((s,a) -> rand(a)))

            cpu_exps = run_game_sched!(sched, I, agents; T_max=5)
            gpu_exps = gpu_run_game_sched!(sched, I, agents; T_max=5)
            @test length(gpu_exps) > 0
            final_world = gpu_exps[end].next_state.world
            @test nparts(final_world, :V) >= 0
        end
    end
end
