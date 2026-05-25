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

# Guarded activation: skips activation when buf_fired[1] == 0 (rule did not fire)
@kernel function activate_slots_kernel_g!(
    active    :: AbstractVector{Bool},
    slots     :: AbstractVector{Int32},
    buf_fired :: AbstractVector{Int32},
)
    i = @index(Global, Linear)
    if i <= length(slots) && buf_fired[1] != Int32(0)
        active[slots[i]] = true
    end
end

# Guarded variants: skip writes when slot == 0 (deleted/invalid source)
@kernel function write_fk_safe_kernel!(fk     :: AbstractVector{Int32},
                                        slots  :: AbstractVector{Int32},
                                        values :: AbstractVector{Int32})
    i = @index(Global, Linear)
    if i <= length(slots) && slots[i] != Int32(0)
        fk[slots[i]] = values[i]
    end
end

@kernel function write_attr_safe_kernel!(attr   :: AbstractVector{Int32},
                                          slots  :: AbstractVector{Int32},
                                          values :: AbstractVector{Int32})
    i = @index(Global, Linear)
    if i <= length(slots) && slots[i] != Int32(0)
        attr[slots[i]] = values[i]
    end
end

# ── GPU FK-resolution kernels for GPU-resident match ─────────────────────────

"""
Resolve `new_r_fk` encoded values to concrete world slot indices using the
GPU-resident match.  Encoding:
  val < 0  → k_flat = -val → l_flat = k_to_l[k_flat] → match[l_flat]
  val > 0  → new element: n_cur_tgt + val
  val == 0 → FK unset / null
"""
@kernel function compute_fk_vals_kernel!(
    fk_vals    :: AbstractVector{Int32},
    fk_pre     :: AbstractVector{Int32},   # new_r_fk_gpu[o][h]
    d_match    :: AbstractVector{Int32},   # GPU-resident match (l_flat → slot)
    k_to_l    :: AbstractVector{Int32},   # cube k_to_l array
    n_cur_tgt  :: Int32,                   # pre_alloc[tgt_type] snapshot
)
    j = @index(Global, Linear)
    if j <= length(fk_vals)
        val = fk_pre[j]
        if val < Int32(0)
            k_flat = Int(-val)
            l_flat = Int(k_to_l[k_flat])
            fk_vals[j] = (l_flat > 0 && l_flat <= length(d_match)) ?
                         d_match[l_flat] : Int32(0)
        elseif val > Int32(0)
            fk_vals[j] = n_cur_tgt + val
        else
            fk_vals[j] = Int32(0)
        end
    end
end

"""
Gather source GPU-local slots from the match for preserved-K attr updates.
`l_flats[i]` is the pre-resolved flat L-element index for pair i.
"""
@kernel function gather_slots_kernel!(
    slots   :: AbstractVector{Int32},
    l_flats :: AbstractVector{Int32},
    d_match :: AbstractVector{Int32},
)
    i = @index(Global, Linear)
    if i <= length(l_flats)
        l = Int(l_flats[i])
        slots[i] = (l > 0 && l <= length(d_match)) ? d_match[l] : Int32(0)
    end
end

"""
Compute source slots and target FK values for preserved-K FK updates.
`l_flats[i]`  = pre-resolved flat L-element index of the source K element.
`enc_vals[i]` = encoded FK target (same encoding as compute_fk_vals_kernel!).
"""
@kernel function compute_preserved_fk_kernel!(
    src_slots :: AbstractVector{Int32},
    tgt_vals  :: AbstractVector{Int32},
    l_flats   :: AbstractVector{Int32},   # k_fk_l_gpu[h]
    enc_vals  :: AbstractVector{Int32},   # k_fk_enc_gpu[h]
    d_match   :: AbstractVector{Int32},
    k_to_l   :: AbstractVector{Int32},
    n_cur_tgt :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(l_flats)
        l = Int(l_flats[i])
        src_slots[i] = (l > 0 && l <= length(d_match)) ? d_match[l] : Int32(0)
        enc = enc_vals[i]
        if enc < Int32(0)
            k2 = Int(-enc)
            l2 = Int(k_to_l[k2])
            tgt_vals[i] = (l2 > 0 && l2 <= length(d_match)) ? d_match[l2] : Int32(0)
        elseif enc > Int32(0)
            tgt_vals[i] = n_cur_tgt + enc
        else
            tgt_vals[i] = Int32(0)
        end
    end
end

# ── apply_pushout! ────────────────────────────────────────────────────────────

function apply_pushout!(g::GPUACSet,
                        match::Vector{Int32},
                        cube::AdhesiveCube,
                        rule,
                        schema::SchemaInfo,
                        enc::AttributeEncoder;
                        scratch = nothing,
                        gpu_cube::Union{GPUAdhesiveCube, Nothing} = nothing,
                        d_match::Union{AbstractVector{Int32}, Nothing} = nothing)::Dict{Symbol, Vector{Int32}}
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

    # Snapshot pre-allocation counts before modifying g.n_alloc
    pre_alloc = Dict{Symbol, Int32}(o => Int32(g.n_alloc[o]) for o in schema.obj_types)

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
    backend    = CUDA.functional() ? CUDA.CUDABackend() : CPU()
    use_scratch = scratch !== nothing && CUDA.functional()
    use_gpu_match = gpu_cube !== nothing && d_match !== nothing && CUDA.functional()

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

        fk_o_gpu   = use_gpu_match ? get(gpu_cube.new_r_fk_gpu,  o, Dict{Symbol,Any}()) :
                                     Dict{Symbol,Any}()
        attr_o_gpu = use_gpu_match ? get(gpu_cube.new_r_attr_gpu, o, Dict{Symbol,Any}()) :
                                     Dict{Symbol,Any}()
        fk_o_pre   = get(cube.new_r_fk,   o, Dict{Symbol,Vector{Int32}}())
        attr_o_pre = get(cube.new_r_attr,  o, Dict{Symbol,Vector{Int32}}())

        for h in schema.homs
            schema.hom_dom[h] == o || continue
            tgt_type = schema.hom_cod[h]

            if use_gpu_match && haskey(fk_o_gpu, h)
                # Fully GPU path: resolve FK encoded values using d_match
                fk_pre_gpu = fk_o_gpu[h]
                n_pre = length(fk_pre_gpu)
                n_pre == 0 && continue
                d_vals = use_scratch ? begin
                    if length(scratch.buf_pushout_vals) < n_pre
                        scratch.buf_pushout_vals = CUDA.zeros(Int32, n_pre * 2)
                    end
                    @view scratch.buf_pushout_vals[1:n_pre]
                end : CUDA.zeros(Int32, n_pre)
                compute_fk_vals_kernel!(backend, 256)(
                    d_vals, fk_pre_gpu, d_match,
                    gpu_cube.k_to_l_gpu, pre_alloc[tgt_type];
                    ndrange = n_pre)
                write_fk_kernel!(backend, 256)(g.homs[h], d_globals, d_vals; ndrange=n_add)
            else
                # CPU fallback path
                fk_pre   = get(fk_o_pre, h, nothing)
                fk_vals  = zeros(Int32, n_add)
                if fk_pre !== nothing
                    for j in 1:n_add
                        val = fk_pre[j]
                        if val < 0
                            k_flat = Int(-val)
                            l_flat = Int(cube.k_to_l[k_flat])
                            fk_vals[j] = match[l_flat]
                        elseif val > 0
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
        end

        for a in schema.attrs
            schema.attr_dom[a] == o || continue
            if use_gpu_match && haskey(attr_o_gpu, a)
                attr_gpu = attr_o_gpu[a]
                n_pre = length(attr_gpu)
                n_pre == 0 && continue
                # Attr values are static (no match lookup needed) — already on GPU
                write_attr_kernel!(backend, 256)(g.attrs[a], d_globals, attr_gpu; ndrange=n_add)
            else
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
                             scratch = nothing,
                             gpu_cube::Union{GPUAdhesiveCube, Nothing} = nothing,
                             d_match::Union{AbstractVector{Int32}, Nothing} = nothing,
                             pre_alloc::Union{Dict{Symbol, Int32}, Nothing} = nothing)
    rule === nothing && return
    isempty(cube.k_attr_pre) && isempty(cube.k_fk_pre) && return

    backend      = CUDA.functional() ? CUDA.CUDABackend() : CPU()
    use_scratch  = scratch !== nothing && CUDA.functional()
    use_gpu_match = gpu_cube !== nothing && d_match !== nothing && CUDA.functional()

    # ── Attr updates ──────────────────────────────────────────────────────────
    for (a, _) in cube.k_attr_pre
        l_flats_gpu = use_gpu_match ? get(gpu_cube.k_attr_l_gpu, a, nothing) : nothing
        vals_gpu    = use_gpu_match ? get(gpu_cube.k_attr_v_gpu, a, nothing) : nothing

        if use_gpu_match && l_flats_gpu !== nothing && vals_gpu !== nothing
            n = length(l_flats_gpu)
            n == 0 && continue
            d_slots = use_scratch ? begin
                if length(scratch.buf_pushout_slots) < n
                    scratch.buf_pushout_slots = CUDA.zeros(Int32, n * 2)
                end
                @view scratch.buf_pushout_slots[1:n]
            end : CUDA.zeros(Int32, n)
            gather_slots_kernel!(backend, 256)(d_slots, l_flats_gpu, d_match; ndrange=n)
            write_attr_safe_kernel!(backend, 256)(g.attrs[a], d_slots, vals_gpu; ndrange=n)
        else
            pairs = cube.k_attr_pre[a]
            slots = Int32[]
            vals  = Int32[]
            for (k_flat, val) in pairs
                l_flat  = Int(cube.k_to_l[k_flat])
                g_local = Int(match[l_flat])
                g_local == 0 && continue
                push!(slots, Int32(g_local))
                push!(vals,  val)
            end
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
                copyto!(d_vals,  vals)
            else
                d_slots = CUDA.functional() ? CuArray(slots) : slots
                d_vals  = CUDA.functional() ? CuArray(vals)  : vals
            end
            write_attr_kernel!(backend, 256)(g.attrs[a], d_slots, d_vals; ndrange=n)
        end
    end

    # ── FK updates ────────────────────────────────────────────────────────────
    for (h, _) in cube.k_fk_pre
        tgt_type = schema.hom_cod[h]
        l_flats_gpu  = use_gpu_match ? get(gpu_cube.k_fk_l_gpu,   h, nothing) : nothing
        enc_vals_gpu = use_gpu_match ? get(gpu_cube.k_fk_enc_gpu,  h, nothing) : nothing

        if use_gpu_match && l_flats_gpu !== nothing && enc_vals_gpu !== nothing
            n = length(l_flats_gpu)
            n == 0 && continue
            n_cur_tgt = pre_alloc !== nothing ? pre_alloc[tgt_type] :
                        Int32(g.n_alloc[tgt_type])
            d_slots = use_scratch ? begin
                if length(scratch.buf_pushout_slots) < n
                    scratch.buf_pushout_slots = CUDA.zeros(Int32, n * 2)
                end
                @view scratch.buf_pushout_slots[1:n]
            end : CUDA.zeros(Int32, n)
            d_vals = use_scratch ? begin
                if length(scratch.buf_pushout_vals) < n
                    scratch.buf_pushout_vals = CUDA.zeros(Int32, n * 2)
                end
                @view scratch.buf_pushout_vals[1:n]
            end : CUDA.zeros(Int32, n)
            compute_preserved_fk_kernel!(backend, 256)(
                d_slots, d_vals,
                l_flats_gpu, enc_vals_gpu,
                d_match, gpu_cube.k_to_l_gpu, n_cur_tgt;
                ndrange = n)
            write_fk_safe_kernel!(backend, 256)(g.homs[h], d_slots, d_vals; ndrange=n)
        else
            pairs = cube.k_fk_pre[h]
            slots = Int32[]
            vals  = Int32[]
            for (k_flat, enc_val) in pairs
                l_flat  = Int(cube.k_to_l[k_flat])
                g_local = Int(match[l_flat])
                g_local == 0 && continue
                g_tgt = if enc_val < 0
                    k2 = Int(-enc_val)
                    l2 = Int(cube.k_to_l[k2])
                    Int(match[l2])
                elseif enc_val > 0
                    s = r_to_local[tgt_type]
                    Int(enc_val) <= length(s) ? Int(s[Int(enc_val)]) : 0
                else
                    0
                end
                g_tgt == 0 && continue
                push!(slots, Int32(g_local))
                push!(vals,  Int32(g_tgt))
            end
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
                copyto!(d_vals,  vals)
            else
                d_slots = CUDA.functional() ? CuArray(slots) : slots
                d_vals  = CUDA.functional() ? CuArray(vals)  : vals
            end
            write_fk_kernel!(backend, 256)(g.homs[h], d_slots, d_vals; ndrange=n)
        end
    end
end
