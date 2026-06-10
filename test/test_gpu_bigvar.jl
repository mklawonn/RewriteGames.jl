"""
BIGVAR routing equivalence tests.

Patterns with 8–14 variables route to the shared-memory `turbo_block_kernel!`
(when its workspace fits 48 KB, i.e. nc_max ≤ 16) instead of the global-memory
EPS pipeline.  Routing must not change the solution SET (repo convention:
compare per-solve sets, never trajectories).  Three-way check per pattern size:

  * default dispatch (BIGVAR → turbo_block for 8–14 vars),
  * `RG_NO_BIGVAR` (historical n_vars > 7 → EPS routing), and
  * Catlab `homomorphisms(L, W)` — `lower_pattern_to_csp` mirrors its
    non-monic semantics exactly, so the sets must be identical.
"""

using Test
using RewriteGames
using Catlab
using AlgebraicRewriting
using CUDA
using Random

@testset "BIGVAR routing equivalence (8–14 var patterns)" begin
    if !CUDA.functional()
        @warn "CUDA not functional — skipping BIGVAR tests"
        @test_skip true
    else
        ext     = Base.get_extension(RewriteGames, :GPURewritingExt)
        backend = CUDA.CUDABackend()

        # World: random sparse digraph, >64 vertices so nc ≥ 2 (the BIGVAR
        # band only exists on the nc ≥ 2 turbo path; nc == 1 is always EPS).
        rng = MersenneTwister(7)
        W   = Graph(150)
        for _ in 1:260
            add_edge!(W, rand(rng, 1:150), rand(rng, 1:150))
        end

        schema = ext.extract_schema_info(W)
        enc    = ext.build_encoder(W, schema)
        g      = ext.upload_acset(W, schema, enc)
        max_n  = maximum(nparts(W, o) for o in schema.obj_types)
        nch    = cld(max_n, 64)

        # Pattern: directed path with `nv_v` vertices ⇒ n_vars = 2*nv_v - 1.
        path_pattern(nv_v) = begin
            L = Graph(nv_v)
            for i in 1:nv_v-1; add_edge!(L, i, i + 1); end
            L
        end

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

        solve_with(csp; no_bigvar::Bool) = begin
            had = haskey(ENV, "RG_NO_BIGVAR")
            no_bigvar ? (ENV["RG_NO_BIGVAR"] = "1") : delete!(ENV, "RG_NO_BIGVAR")
            try
                hf_flat, hf_offs = ext._build_hom_fwd_gpu(backend, g, schema, csp.n_chunks)
                d_gpu = ext._build_domains_gpu(backend, csp, g, schema)
                ext._apply_attr_masks_gpu_device!(d_gpu, csp, g, schema, enc)
                CUDA.synchronize()
                ext.gpu_turbo_solve(backend, csp, d_gpu, hf_flat, hf_offs;
                                    max_solutions = 200_000)
            finally
                had ? (ENV["RG_NO_BIGVAR"] = "1") : delete!(ENV, "RG_NO_BIGVAR")
            end
        end

        # n_vars = 9, 13 exercise the BIGVAR band (8–14); 7 pins the legacy
        # band (must be untouched); 15 exceeds the cap (EPS on both settings).
        for nv_v in (4, 5, 7, 8)
            L   = path_pattern(nv_v)
            csp = ext.lower_pattern_to_csp(L, schema, enc;
                                           n_chunks = nch, n_alloc = g.n_alloc)
            n_vars = Int(csp.n_vars)
            @testset "n_vars = $n_vars" begin
                sols_default = solve_with(csp; no_bigvar = false)
                sols_eps     = solve_with(csp; no_bigvar = true)
                homs         = homomorphisms(L, W)
                expect       = Set(hom_to_vec(h, csp) for h in homs)

                @test !isempty(expect)                      # pattern actually matches
                @test Set(sols_default) == Set(sols_eps)    # routing-invariant set
                @test Set(sols_default) == expect           # exact Catlab semantics
                @test length(sols_default) == length(expect)  # no duplicate columns
            end
        end
    end
end

@testset "NAC tier ordering (batched-first at large N)" begin
    if !CUDA.functional()
        @test_skip true
    else
        ext = Base.get_extension(RewriteGames, :GPURewritingExt)
        # Self-loop rule with a NacSpec-eligible NAC, on a world with enough
        # candidates (≥ RG_NAC_BATCH_MIN = 32 default) that the batched tier-2
        # filter runs FIRST under the new ordering.  RG_NAC_DIAG cross-checks
        # tier-1, tier-2, and the CPU reference per solve; the kept sets must
        # agree under both orderings.
        K      = Graph(1)
        L      = Graph(1)
        R      = Graph(1); add_edge!(R, 1, 1)
        L_loop = Graph(1); add_edge!(L_loop, 1, 1)
        rule   = Rule(homomorphism(K, L; monic=true),
                      homomorphism(K, R; monic=true);
                      ac = [NAC(homomorphism(L, L_loop; monic=true))])
        pra    = PlayerRuleApp(:add_loop_if_none, rule, K, :alice)
        gs     = mk_game_sched((;), (init=:I,), Names(Dict("I" => K)), (r=pra,),
                               quote
                                   s, f = r(init)
                                   return s, f
                               end)
        agents = Dict{Symbol, AbstractAgent}(:alice => GPUFunctionPlayer((_, _c, _n, _t) -> 1))

        mk_world() = begin                  # 80 vertices, 40 pre-looped
            G = Graph(80)
            for v in 1:40; add_edge!(G, v, v); end
            G
        end

        run_once() = begin
            ext._NAC_DIAG_CHECKS[] = 0; ext._NAC_DIAG_MISM[] = 0
            ENV["RG_NAC_DIAG"] = "1"
            try
                exps = gpu_run_game_sched!(gs, mk_world(), agents; T_max = 1)
                w    = exps[end].next_state.world
                (count(e -> e.player == :alice, exps), nparts(w, :E),
                 ext._NAC_DIAG_CHECKS[], ext._NAC_DIAG_MISM[])
            finally
                delete!(ENV, "RG_NAC_DIAG")
            end
        end

        fired_new, ne_new, checks_new, mism_new = run_once()
        ENV["RG_NACSPEC_FIRST"] = "1"
        fired_old, ne_old, checks_old, mism_old = try
            run_once()
        finally
            delete!(ENV, "RG_NACSPEC_FIRST")
        end

        @test checks_new > 0 && checks_old > 0    # diag actually exercised
        @test mism_new == 0                       # tier-2-first: sets agree everywhere
        @test mism_old == 0                       # tier-1-first: sets agree everywhere
        @test fired_new == fired_old == 1         # one un-looped vertex gains a loop
        @test ne_new == ne_old == 41
    end
end
