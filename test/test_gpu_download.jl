"""
download_acset equivalence test (feat/fast-download).

The vectorized `download_acset` (bulk column writes + cumsum compaction) must
produce a byte-for-byte identical ACSet to the original per-element
implementation, including on tombstoned (deleted) parts and on a live source
whose target has been deleted (the 0-fallback path).  `download_acset_ref`
below is a verbatim copy of the original loop and serves as the oracle.
"""

using Test
using RewriteGames
using Catlab
using CUDA

const _ext = Base.get_extension(RewriteGames, :GPURewritingExt)

# ── Reference (original) implementation: per-element set_subpart! ─────────────
function download_acset_ref(g, enc, world_type)
    schema   = g.schema
    host_act = Dict(o => Array(g.active[o]) for o in schema.obj_types)
    host_hom = Dict(h => Array(g.homs[h])   for h in schema.homs)
    host_att = Dict(a => Array(g.attrs[a])  for a in schema.attrs)
    result = world_type()
    new_id = Dict{Symbol, Vector{Int}}()
    for o in schema.obj_types
        flags  = host_act[o]; n_live = sum(flags)
        add_parts!(result, o, n_live)
        mapping = zeros(Int, length(flags)); cursor = 0
        for (old, alive) in enumerate(flags)
            alive || continue; cursor += 1; mapping[old] = cursor
        end
        new_id[o] = mapping
    end
    for h in schema.homs
        owner = schema.hom_dom[h]; cod = schema.hom_cod[h]
        fks = host_hom[h]; flags = host_act[owner]
        for (old_i, (alive, tgt)) in enumerate(zip(flags, fks))
            alive || continue
            new_i = new_id[owner][old_i]
            new_tgt = tgt > 0 ? new_id[cod][tgt] : 0
            new_tgt > 0 && set_subpart!(result, new_i, h, new_tgt)
        end
    end
    for a in schema.attrs
        owner = schema.attr_dom[a]; avs = host_att[a]; flags = host_act[owner]
        for (old_i, (alive, enc_v)) in enumerate(zip(flags, avs))
            alive || continue
            new_i = new_id[owner][old_i]
            v = _ext.decode_value(enc, a, enc_v)
            v !== nothing && set_subpart!(result, new_i, a, v)
        end
    end
    result
end

# ── A small schema with homs + a Symbol attr + an Int attr ────────────────────
@present SchDL(FreeSchema) begin
    (X, Y)::Ob
    f::Hom(X, Y)
    (Name, Wt)::AttrType
    lbl::Attr(X, Name)
    w::Attr(Y, Wt)
end
@acset_type DLTest(SchDL, index=[:f])

function _build_world()
    W = DLTest{Symbol,Int}()
    add_parts!(W, :Y, 3)
    add_parts!(W, :X, 4)
    set_subpart!(W, :, :f, [1, 2, 3, 1])
    set_subpart!(W, :, :lbl, [:a, :b, :c, :d])
    set_subpart!(W, :, :w, [10, 20, 30])
    W
end

@testset "download_acset vectorized == reference" begin
    if !CUDA.functional()
        @test_skip "no GPU"
    else
        W      = _build_world()
        schema = _ext.extract_schema_info(W)
        enc    = _ext.build_encoder(W, schema)

        # (1) Fully-live world: exercises the bulk path for homs and attrs.
        g1 = _ext.upload_acset(W, schema, enc)
        new1 = _ext.download_acset(g1, enc, DLTest{Symbol,Int})
        ref1 = download_acset_ref(g1, enc, DLTest{Symbol,Int})
        @test new1 == ref1
        @test new1 == W                       # round-trips the original exactly

        # (2) Tombstone X#2 (a live source removed): exercises compaction.
        g2 = _ext.upload_acset(W, schema, enc)
        CUDA.@allowscalar g2.active[:X][2] = false
        new2 = _ext.download_acset(g2, enc, DLTest{Symbol,Int})
        ref2 = download_acset_ref(g2, enc, DLTest{Symbol,Int})
        @test new2 == ref2
        @test nparts(new2, :X) == 3           # one X removed

        # (3) Delete Y#1, which live X#1 and X#4 point at: live source → dead
        #     target, so f maps to 0 → per-element fallback path.
        g3 = _ext.upload_acset(W, schema, enc)
        CUDA.@allowscalar g3.active[:Y][1] = false
        new3 = _ext.download_acset(g3, enc, DLTest{Symbol,Int})
        ref3 = download_acset_ref(g3, enc, DLTest{Symbol,Int})
        @test new3 == ref3
        @test nparts(new3, :Y) == 2
        # X#1 (old) now has an undefined f (target deleted); X#2,X#3 keep theirs.
        @test new3 == ref3                    # structural identity is the contract

        # (4) Empty object (all X tombstoned): isempty(live) branch.
        g4 = _ext.upload_acset(W, schema, enc)
        CUDA.@allowscalar for i in 1:4; g4.active[:X][i] = false; end
        new4 = _ext.download_acset(g4, enc, DLTest{Symbol,Int})
        ref4 = download_acset_ref(g4, enc, DLTest{Symbol,Int})
        @test new4 == ref4
        @test nparts(new4, :X) == 0
    end
end
