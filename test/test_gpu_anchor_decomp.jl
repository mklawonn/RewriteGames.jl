"""
Anchored-fiber decomposition equivalence tests.

Big-codomain (would-be EPS) solves with a single-variable anchor type are
solved as a union of per-cell compact solves (solver/AnchorDecomp.jl).  The
union must equal the global solution SET (repo convention: per-solve sets,
never trajectories).  To keep the suite free of expensive big-NM kernel JIT,
the big-world checks compare against Catlab `homomorphisms` (the decomposed
cells themselves only touch cheap nc ≤ 8 shapes), and the solver-vs-solver
check runs on a small world under RG_ANCHOR_FORCE.
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA
using Random

# Star schema: one big hub type A and seven fiber types, so 8-var patterns
# have a unique single-var anchor (the decomposition's eligible shape).
@present SchAnchorStar(FreeSchema) begin
    (A, B1, B2, B3, B4, B5, B6, B7)::Ob
    b1::Hom(B1, A); b2::Hom(B2, A); b3::Hom(B3, A); b4::Hom(B4, A)
    b5::Hom(B5, A); b6::Hom(B6, A); b7::Hom(B7, A)
end
@acset_type AnchorStar(SchAnchorStar)

_star_world(nA::Int) = begin
    W = AnchorStar()
    add_parts!(W, :A, nA)
    for (o, h) in ((:B1, :b1), (:B2, :b2), (:B3, :b3), (:B4, :b4),
                   (:B5, :b5), (:B6, :b6), (:B7, :b7))
        add_parts!(W, o, nA; Dict(h => collect(1:nA))...)
    end
    W
end

_star_pattern() = begin                      # 8 vars: a + one of each fiber
    L = AnchorStar()
    add_part!(L, :A)
    for (o, h) in ((:B1, :b1), (:B2, :b2), (:B3, :b3), (:B4, :b4),
                   (:B5, :b5), (:B6, :b6), (:B7, :b7))
        add_part!(L, o; Dict(h => 1)...)
    end
    L
end

@testset "Anchored-fiber decomposition" begin
    if !CUDA.functional()
        @warn "CUDA not functional — skipping anchored-decomposition tests"
        @test_skip true
    else
        ext     = Base.get_extension(RewriteGames, :GPURewritingExt)
        backend = CUDA.CUDABackend()

        hom_to_vec(h, csp) = begin
            v = zeros(Int32, Int(csp.n_vars))
            for (o, base) in pairs(csp.var_offset)
                comp = h[o]
                for i in 1:length(collect(comp))
                    v[base + i - 1] = Int32(comp(i))
                end
            end
            v
        end

        # One world upload + final domains for a pattern; returns what the
        # Scheduler hook sees.
        setup(W, L) = begin
            schema = ext.extract_schema_info(W)
            enc    = ext.build_encoder(W, schema)
            g      = ext.upload_acset(W, schema, enc)
            nch    = cld(maximum(nparts(W, o) for o in schema.obj_types), 64)
            csp    = ext.lower_pattern_to_csp(L, schema, enc;
                                              n_chunks = nch, n_alloc = g.n_alloc)
            hf, ho = ext._build_hom_fwd_gpu(backend, g, schema, csp.n_chunks)
            d      = ext._build_domains_gpu(backend, csp, g, schema)
            ext._apply_attr_masks_gpu_device!(d, csp, g, schema, enc)
            CUDA.synchronize()
            (; schema, enc, g, csp, hf, ho, d)
        end

        with_env(f; kv...) = begin
            saved = Dict(string(k) => get(ENV, string(k), nothing) for (k, _) in kv)
            for (k, v) in kv
                v === nothing ? delete!(ENV, string(k)) : (ENV[string(k)] = string(v))
            end
            try
                f()
            finally
                for (k, old) in saved
                    old === nothing ? delete!(ENV, k) : (ENV[k] = old)
                end
            end
        end

        @testset "big world: decomposed set == Catlab (nc=35, EPS-bound)" begin
            W  = _star_world(2200)            # nc = 35 → 8-var band over 48 KB
            L  = _star_pattern()
            s  = setup(W, L)
            n0 = ext._ANCHOR_SOLVES[]
            sols = ext._try_anchored_solve(backend, s.csp, s.g, s.schema, s.d)
            @test sols !== nothing            # eligible and decomposed
            @test ext._ANCHOR_SOLVES[] == n0 + 1
            expect = Set(hom_to_vec(h, s.csp) for h in homomorphisms(L, W))
            @test length(expect) == 2200
            @test Set(sols) == expect
            @test length(sols) == length(expect)   # disjoint cells: no dupes

            # Ragged last cell must not change the set.
            sols2 = with_env(; RG_ANCHOR_CELL = 300) do
                ext._try_anchored_solve(backend, s.csp, s.g, s.schema, s.d)
            end
            @test Set(sols2) == expect

            # Closure blowup (one whole-world cell) → clean fallback signal.
            nf = ext._ANCHOR_FALLBACKS[]
            @test with_env(; RG_ANCHOR_CELL = 4096) do
                ext._try_anchored_solve(backend, s.csp, s.g, s.schema, s.d)
            end === nothing
            @test ext._ANCHOR_FALLBACKS[] == nf + 1

            # Kill switch.
            @test with_env(; RG_NO_ANCHOR_DECOMP = 1) do
                ext._try_anchored_solve(backend, s.csp, s.g, s.schema, s.d)
            end === nothing

            # max_solutions cap: the union truncates like the global solve;
            # everything kept must still be a valid solution.
            fk = Dict{Symbol,Vector{Int32}}()
            for h in ext._decomp_relevant_homs(s.schema, s.csp)
                nh = s.g.n_alloc[s.schema.hom_dom[h]]
                fk[h] = nh > 0 ? Array(view(s.g.homs[h], 1:nh)) : Int32[]
            end
            capped = ext.anchored_decomposed_solve(backend, s.csp, s.schema,
                                                   Array(s.d), fk, s.g.n_alloc;
                                                   max_solutions = 50)
            @test length(capped) == 50
            @test all(v -> v in expect, capped)
        end

        @testset "ineligible shapes return nothing" begin
            # No single-var anchor: path pattern on a big Graph (V, E multi-var).
            rng = MersenneTwister(11)
            G   = Graph(2200)
            for _ in 1:3000
                add_edge!(G, rand(rng, 1:2200), rand(rng, 1:2200))
            end
            Lp = Graph(3); add_edge!(Lp, 1, 2); add_edge!(Lp, 2, 3)
            sg = setup(G, Lp)
            @test ext._try_anchored_solve(backend, sg.csp, sg.g, sg.schema, sg.d) === nothing

            # Small world (nc = 8): not EPS-bound → ineligible without FORCE.
            Ws = _star_world(500)
            ss = setup(Ws, _star_pattern())
            @test ext._try_anchored_solve(backend, ss.csp, ss.g, ss.schema, ss.d) === nothing
        end

        @testset "RG_ANCHOR_FORCE: decomposed == global solver (small world)" begin
            W = _star_world(500)              # nc = 8: cheap turbo_block shapes
            s = setup(W, _star_pattern())
            sols = with_env(; RG_ANCHOR_FORCE = 1, RG_ANCHOR_MIN = 256) do
                ext._try_anchored_solve(backend, s.csp, s.g, s.schema, s.d)
            end
            @test sols !== nothing
            ref = ext.gpu_turbo_solve(backend, s.csp, s.d, s.hf, s.ho;
                                      max_solutions = 4000)
            nv  = Int(s.csp.n_vars)
            @test Set(sols) == Set(Vector{Int32}(r[1:nv]) for r in ref)
        end
    end
end
