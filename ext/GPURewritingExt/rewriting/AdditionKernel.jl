"""
DPO pushout — GPU addition phase.

`apply_pushout!` adds R\\K elements to the GPUACSet using GPU scatter-write
kernels.  Arrays use 2× over-allocation so that additions within the spare
capacity require no reallocation.

`_update_preserved!` patches attributes and FKs of K elements that differ
in R, also via GPU scatter writes.

All indices in `match` are GPU-local (1-based within the type's slot array,
including tombstones).
"""

# ── GPU scatter-write kernels ─────────────────────────────────────────────────

@kernel function activate_slots_kernel!(active :: AbstractVector{Bool},
                                         slots  :: AbstractVector{Int32})
    i = @index(Global, Linear)
    if i <= length(slots)
        active[slots[i]] = true
    end
end

@kernel function write_fk_kernel!(fk     :: AbstractVector{Int32},
                                   slots  :: AbstractVector{Int32},
                                   values :: AbstractVector{Int32})
    i = @index(Global, Linear)
    if i <= length(slots)
        fk[slots[i]] = values[i]
    end
end

@kernel function write_attr_kernel!(attr   :: AbstractVector{Int32},
                                     slots  :: AbstractVector{Int32},
                                     values :: AbstractVector{Int32})
    i = @index(Global, Linear)
    if i <= length(slots)
        attr[slots[i]] = values[i]
    end
end

# ── apply_pushout! ────────────────────────────────────────────────────────────

function apply_pushout!(g::GPUACSet,
                        match::Vector{Int32},
                        cube::AdhesiveCube,
                        rule,
                        schema::SchemaInfo,
                        enc::AttributeEncoder;
                        scratch = nothing)::Dict{Symbol, Vector{Int32}}
    if rule === nothing || isempty(cube.new_r_fk) && cube.n_r_elems == 0
        return Dict(o => Int32[] for o in schema.obj_types)
    end

    # Determine new elements per type from precomputed cube data
    new_r_counts = Dict{Symbol, Int}(o => 0 for o in schema.obj_types)
    for (o, fk_o) in cube.new_r_fk
        new_r_counts[o] = isempty(fk_o) ? 0 :
            length(first(values(fk_o)))
    end
    # Fall back to counting from r_types if no FK data (pure-deletion rules)
    if all(v == 0 for v in values(new_r_counts)) && cube.n_r_elems > 0
        k_img_r_flat = Set{Int}(Int(x) for x in cube.k_to_r)
        for r_flat in 1:cube.n_r_elems
            r_flat ∈ k_img_r_flat && continue
            o = schema.obj_types[Int(cube.r_types[r_flat])]
            new_r_counts[o] += 1
        end
    end

    # 1. Assign GPU-slot indices; grow arrays when spare capacity exhausted
    r_to_local = Dict{Symbol, Vector{Int32}}()
    for o in schema.obj_types
        n_add = new_r_counts[o]
        if n_add == 0
            r_to_local[o] = Int32[]
            continue
        end
        n_cur  = g.n_alloc[o]
        n_next = n_cur + n_add
        cap    = length(g.active[o])

        globals = Int32[Int32(n_cur + j) for j in 1:n_add]
        r_to_local[o] = globals

        if n_next > cap
            new_cap = max(2 * cap, n_next)
            _z = CUDA.functional() ? (T, n) -> CUDA.zeros(T, n) : (T, n) -> zeros(T, n)
            new_active = _z(Bool, new_cap)
            n_cur > 0 && copyto!(new_active, 1, g.active[o], 1, n_cur)
            g.active[o] = new_active
            for h in schema.homs
                schema.hom_dom[h] == o || continue
                new_fk = _z(Int32, new_cap)
                n_cur > 0 && copyto!(new_fk, 1, g.homs[h], 1, n_cur)
                g.homs[h] = new_fk
            end
            for a in schema.attrs
                schema.attr_dom[a] == o || continue
                new_av = _z(Int32, new_cap)
                n_cur > 0 && copyto!(new_av, 1, g.attrs[a], 1, n_cur)
                g.attrs[a] = new_av
            end
        end
        g.n_alloc[o] = n_next
        g.n_live[o][] += n_add
    end

    # 2. GPU scatter: activate slots and write FKs + attrs using precomputed data
    backend  = CUDA.functional() ? CUDA.CUDABackend() : CPU()
    use_scratch = scratch !== nothing && CUDA.functional()

    for o in schema.obj_types
        n_add = new_r_counts[o]
        n_add == 0 && continue
        globals = r_to_local[o]

        # Upload slot indices — reuse staging buffer or allocate
        if use_scratch
            if length(scratch.buf_pushout_slots) < n_add
                scratch.buf_pushout_slots = CUDA.zeros(Int32, n_add * 2)
            end
            d_globals = @view scratch.buf_pushout_slots[1:n_add]
            copyto!(d_globals, globals)
        else
            d_globals = CUDA.functional() ? CuArray(globals) : globals
        end

        activate_slots_kernel!(backend, 256)(g.active[o], d_globals; ndrange=n_add)

        fk_o_pre   = get(cube.new_r_fk,   o, Dict{Symbol,Vector{Int32}}())
        attr_o_pre = get(cube.new_r_attr,  o, Dict{Symbol,Vector{Int32}}())

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            tgt_type = schema.hom_cod[h]
            fk_pre   = get(fk_o_pre, h, nothing)
            fk_vals  = zeros(Int32, n_add)
            if fk_pre !== nothing
                for j in 1:n_add
                    val = fk_pre[j]
                    if val < 0
                        # K-preserved target: match[k_to_l[-val]]
                        k_flat = Int(-val)
                        l_flat = Int(cube.k_to_l[k_flat])
                        fk_vals[j] = match[l_flat]
                    elseif val > 0
                        # New element target: r_to_local[tgt_type][val]
                        tgt_slots = r_to_local[tgt_type]
                        val <= length(tgt_slots) && (fk_vals[j] = tgt_slots[val])
                    end
                end
            end
            if use_scratch
                if length(scratch.buf_pushout_vals) < n_add
                    scratch.buf_pushout_vals = CUDA.zeros(Int32, n_add * 2)
                end
                d_vals = @view scratch.buf_pushout_vals[1:n_add]
                copyto!(d_vals, fk_vals)
            else
                d_vals = CUDA.functional() ? CuArray(fk_vals) : fk_vals
            end
            write_fk_kernel!(backend, 256)(g.homs[h], d_globals, d_vals; ndrange=n_add)
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            attr_pre  = get(attr_o_pre, a, nothing)
            attr_vals = attr_pre !== nothing ? copy(attr_pre) : zeros(Int32, n_add)
            if use_scratch
                if length(scratch.buf_pushout_vals) < n_add
                    scratch.buf_pushout_vals = CUDA.zeros(Int32, n_add * 2)
                end
                d_vals = @view scratch.buf_pushout_vals[1:n_add]
                copyto!(d_vals, attr_vals)
            else
                d_vals = CUDA.functional() ? CuArray(attr_vals) : attr_vals
            end
            write_attr_kernel!(backend, 256)(g.attrs[a], d_globals, d_vals; ndrange=n_add)
        end
    end

    r_to_local
end

# ── _update_preserved! ────────────────────────────────────────────────────────

"""
    _update_preserved!(g, match, cube, rule, schema, enc, r_to_local)

Patch attributes and foreign keys of preserved (K) elements that differ
between L and R.  Uses precomputed `cube.k_attr_pre` and `cube.k_fk_pre`
to avoid any access to the original Catlab rule ACSet at rewrite time.
`match[flat_l]` gives the GPU-local slot index for each L-variable.
"""
function _update_preserved!(g::GPUACSet, match::Vector{Int32},
                             cube::AdhesiveCube, rule,
                             schema::SchemaInfo, enc::AttributeEncoder,
                             r_to_local::Dict{Symbol, Vector{Int32}};
                             scratch = nothing)
    rule === nothing && return
    isempty(cube.k_attr_pre) && isempty(cube.k_fk_pre) && return

    attr_slots = Dict{Symbol, Vector{Int32}}()
    attr_vals  = Dict{Symbol, Vector{Int32}}()
    hom_slots  = Dict{Symbol, Vector{Int32}}()
    hom_vals   = Dict{Symbol, Vector{Int32}}()

    for (a, pairs) in cube.k_attr_pre
        for (k_flat, val) in pairs
            l_flat  = Int(cube.k_to_l[k_flat])
            g_local = Int(match[l_flat])
            g_local == 0 && continue
            push!(get!(attr_slots, a, Int32[]), Int32(g_local))
            push!(get!(attr_vals,  a, Int32[]), val)
        end
    end

    for (h, pairs) in cube.k_fk_pre
        tgt_type = schema.hom_cod[h]
        for (k_flat, enc_val) in pairs
            l_flat  = Int(cube.k_to_l[k_flat])
            g_local = Int(match[l_flat])
            g_local == 0 && continue
            g_tgt = if enc_val < 0
                k2 = Int(-enc_val)
                l2 = Int(cube.k_to_l[k2])
                Int(match[l2])
            elseif enc_val > 0
                slots = r_to_local[tgt_type]
                Int(enc_val) <= length(slots) ? Int(slots[Int(enc_val)]) : 0
            else
                0
            end
            g_tgt == 0 && continue
            push!(get!(hom_slots, h, Int32[]), Int32(g_local))
            push!(get!(hom_vals,  h, Int32[]), Int32(g_tgt))
        end
    end

    backend     = CUDA.functional() ? CUDA.CUDABackend() : CPU()
    use_scratch = scratch !== nothing && CUDA.functional()

    for (a, slots) in attr_slots
        isempty(slots) && continue
        n = length(slots)
        if use_scratch
            if length(scratch.buf_pushout_slots) < n
                scratch.buf_pushout_slots = CUDA.zeros(Int32, n * 2)
            end
            if length(scratch.buf_pushout_vals) < n
                scratch.buf_pushout_vals = CUDA.zeros(Int32, n * 2)
            end
            d_slots = @view scratch.buf_pushout_slots[1:n]
            d_vals  = @view scratch.buf_pushout_vals[1:n]
            copyto!(d_slots, slots)
            copyto!(d_vals, attr_vals[a])
        else
            d_slots = CUDA.functional() ? CuArray(slots)        : slots
            d_vals  = CUDA.functional() ? CuArray(attr_vals[a]) : attr_vals[a]
        end
        write_attr_kernel!(backend, 256)(g.attrs[a], d_slots, d_vals; ndrange=n)
    end
    for (h, slots) in hom_slots
        isempty(slots) && continue
        n = length(slots)
        if use_scratch
            if length(scratch.buf_pushout_slots) < n
                scratch.buf_pushout_slots = CUDA.zeros(Int32, n * 2)
            end
            if length(scratch.buf_pushout_vals) < n
                scratch.buf_pushout_vals = CUDA.zeros(Int32, n * 2)
            end
            d_slots = @view scratch.buf_pushout_slots[1:n]
            d_vals  = @view scratch.buf_pushout_vals[1:n]
            copyto!(d_slots, slots)
            copyto!(d_vals, hom_vals[h])
        else
            d_slots = CUDA.functional() ? CuArray(slots)       : slots
            d_vals  = CUDA.functional() ? CuArray(hom_vals[h]) : hom_vals[h]
        end
        write_fk_kernel!(backend, 256)(g.homs[h], d_slots, d_vals; ndrange=n)
    end
end
