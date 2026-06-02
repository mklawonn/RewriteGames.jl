# Microbench: vectorized download_acset vs the original per-element loop, at
# full-game-ish scale.  (DLTest has only 1 hom + 2 attrs; the real Falcon schema
# has many more, so this is a conservative lower bound on the speedup.)
#
# Usage: julia --project=. test/bench_download.jl   (ENV N default 3000)

using RewriteGames, Catlab, CUDA, Printf

const _ext = Base.get_extension(RewriteGames, :GPURewritingExt)

@present SchDL(FreeSchema) begin
    (X, Y)::Ob
    f::Hom(X, Y)
    (Name, Wt)::AttrType
    lbl::Attr(X, Name)
    w::Attr(Y, Wt)
end
@acset_type DLTest(SchDL, index=[:f])

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

const N = parse(Int, get(ENV, "N", "3000"))

function build_big(n)
    W = DLTest{Symbol,Int}()
    add_parts!(W, :Y, n)
    add_parts!(W, :X, n)
    set_subpart!(W, :, :f, rand(1:n, n))
    set_subpart!(W, :, :lbl, rand([:a,:b,:c,:d], n))
    set_subpart!(W, :, :w, rand(1:100, n))
    W
end

if !CUDA.functional()
    println("no GPU — skipping"); exit()
end

W      = build_big(N)
schema = _ext.extract_schema_info(W)
enc    = _ext.build_encoder(W, schema)
g      = _ext.upload_acset(W, schema, enc)

new1 = _ext.download_acset(g, enc, DLTest{Symbol,Int})
ref1 = download_acset_ref(g, enc, DLTest{Symbol,Int})
@assert new1 == ref1 "vectorized download differs from reference!"

const ITERS = 50
tn = @elapsed (for _ in 1:ITERS; _ext.download_acset(g, enc, DLTest{Symbol,Int}); end)
to = @elapsed (for _ in 1:ITERS; download_acset_ref(g, enc, DLTest{Symbol,Int}); end)
@printf("download_acset  N=%d (X+Y=%d parts, 1 hom + 2 attrs)  iters=%d\n", N, 2N, ITERS)
@printf("  new=%.3f ms/call   old=%.3f ms/call   speedup=%.2fx\n",
        1000*tn/ITERS, 1000*to/ITERS, to/tn)
