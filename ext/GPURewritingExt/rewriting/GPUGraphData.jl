"""
GPU-resident graph representation of El(world) — the category of elements of
the world ACSet viewed as a functor.

Nodes  = (obj_type, slot_id) pairs; global_id = obj_node_offset[type] + slot_id.
Edges  = one per (morphism h, slot k in dom(h)) where the FK is set.
Tombstone approach: node_active mirrors GPUACSet's active flags so that node
indices remain stable across deletions.  The GNN masks inactive nodes.

Node features: one Float32 per schema attribute (feat_dim = n_attrs).  Only the
attributes belonging to each node's type are non-zero; others are 0.0.

Construction is done via a thin CPU-round-trip on the active/homs/attrs arrays
(not the full world ACSet), then uploading the COO to GPU.  This is acceptable
for episodic construction (once per episode) or post-compaction rebuilds.

Incremental updates after rewrites use GPU kernels that update `node_active` and
`node_feat` for the affected slots, avoiding full rebuilds for every rewrite.
"""

mutable struct GPUGraphData
    # COO format edge list (live edges only — no tombstones at construction time)
    edge_src      :: CuVector{Int32}   # global node index of source
    edge_dst      :: CuVector{Int32}   # global node index of dest
    edge_type     :: CuVector{Int32}   # schema.hom_index value for this edge
    n_edges       :: Int               # number of live edges currently stored

    # Node arrays — indexed by global node id
    node_active   :: CuVector{Bool}    # mirrors GPUACSet active; stable across rewrites
    node_type     :: CuVector{Int32}   # schema.obj_index value for this node
    node_feat     :: CuMatrix{Float32} # [feat_dim × n_nodes_alloc]
    n_nodes_alloc :: Int               # total slots allocated (sum of g.n_alloc)
    feat_dim      :: Int               # = length(schema.attrs)

    # CPU-side offset table: obj_node_offset[i] = first global node id for type i (1-based)
    obj_node_offset :: Vector{Int32}
end

# ── Node feature update kernels ───────────────────────────────────────────────

@kernel function _graph_mark_active_kernel!(
    node_active  :: AbstractVector{Bool},
    active_type  :: AbstractVector{Bool},
    offset       :: Int32,
    n            :: Int32,
)
    k = @index(Global, Linear)
    if k <= Int(n)
        node_active[Int(offset) + k] = active_type[k]
    end
end

@kernel function _graph_write_features_kernel!(
    node_feat  :: AbstractMatrix{Float32},   # [feat_dim × n_nodes_alloc]
    attr_vals  :: AbstractVector{Int32},     # g.attrs[a][1:n] for one attribute
    offset     :: Int32,                     # obj_node_offset for this type
    feat_row   :: Int32,                     # which row of node_feat this attr goes in
    n          :: Int32,
)
    k = @index(Global, Linear)
    if k <= Int(n)
        node_feat[Int(feat_row), Int(offset) + k] = Float32(attr_vals[k])
    end
end

@kernel function _graph_set_node_inactive_kernel!(
    node_active :: AbstractVector{Bool},
    slots       :: AbstractVector{Int32},
    offset      :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(slots)
        node_active[Int(offset) + Int(slots[i])] = false
    end
end

@kernel function _graph_set_node_active_kernel!(
    node_active :: AbstractVector{Bool},
    slots       :: AbstractVector{Int32},
    offset      :: Int32,
)
    i = @index(Global, Linear)
    if i <= length(slots)
        node_active[Int(offset) + Int(slots[i])] = true
    end
end

# ── build_gpu_graph ───────────────────────────────────────────────────────────

"""
    build_gpu_graph(g, schema, enc; backend) -> GPUGraphData

Build a GPU-resident category-of-elements graph from `g`.

Downloads the thin `active`/`homs`/`attrs` arrays from GPU, builds the COO
edge list and node feature matrix on CPU, then uploads to GPU.
"""
function build_gpu_graph(g::GPUACSet, schema::SchemaInfo, enc::AttributeEncoder;
                          backend = CUDA.CUDABackend())::GPUGraphData
    feat_dim = length(schema.attrs)

    # Compute obj_node_offset (1-based cumulative sum of n_alloc)
    obj_node_offset = Vector{Int32}(undef, length(schema.obj_types))
    cumoff = Int32(0)
    for (i, o) in enumerate(schema.obj_types)
        obj_node_offset[i] = cumoff
        cumoff += Int32(g.n_alloc[o])
    end
    n_nodes_total = Int(cumoff)

    # Download active, homs, attrs (thin arrays, not the full world representation)
    host_active = Dict(o => Array(g.active[o]) for o in schema.obj_types)
    host_homs   = Dict(h => Array(g.homs[h])   for h in schema.homs)
    host_attrs  = Dict(a => Array(g.attrs[a])  for a in schema.attrs)

    # Build node_active and node_type on CPU
    node_active_cpu = zeros(Bool,  n_nodes_total)
    node_type_cpu   = zeros(Int32, n_nodes_total)
    for (i, o) in enumerate(schema.obj_types)
        off  = Int(obj_node_offset[i])
        n    = g.n_alloc[o]
        act  = host_active[o]
        for k in 1:n
            node_active_cpu[off + k] = act[k]
            node_type_cpu[off + k]   = Int32(i)
        end
    end

    # Build node feature matrix [feat_dim × n_nodes_total] on CPU
    node_feat_cpu = zeros(Float32, feat_dim, n_nodes_total)
    for (ai, a) in enumerate(schema.attrs)
        o   = schema.attr_dom[a]
        oi  = schema.obj_index[o]
        off = Int(obj_node_offset[oi])
        n   = g.n_alloc[o]
        av  = host_attrs[a]
        for k in 1:n
            node_feat_cpu[ai, off + k] = Float32(av[k])
        end
    end

    # Build COO edges on CPU (only live edges where active && FK is set)
    edge_src_cpu  = Int32[]
    edge_dst_cpu  = Int32[]
    edge_type_cpu = Int32[]
    for (hi, h) in enumerate(schema.homs)
        dom_o = schema.hom_dom[h]
        cod_o = schema.hom_cod[h]
        dom_i = schema.obj_index[dom_o]
        cod_i = schema.obj_index[cod_o]
        dom_off = Int(obj_node_offset[dom_i])
        cod_off = Int(obj_node_offset[cod_i])
        act_dom = host_active[dom_o]
        fk_col  = host_homs[h]
        n_dom   = g.n_alloc[dom_o]
        for k in 1:n_dom
            act_dom[k] || continue
            tgt = Int(fk_col[k])
            tgt == 0 && continue
            push!(edge_src_cpu,  Int32(dom_off + k))
            push!(edge_dst_cpu,  Int32(cod_off + tgt))
            push!(edge_type_cpu, Int32(hi))
        end
    end

    # Upload to GPU
    GPUGraphData(
        CuVector{Int32}(edge_src_cpu),
        CuVector{Int32}(edge_dst_cpu),
        CuVector{Int32}(edge_type_cpu),
        length(edge_src_cpu),
        CuVector{Bool}(node_active_cpu),
        CuVector{Int32}(node_type_cpu),
        CuMatrix{Float32}(node_feat_cpu),
        n_nodes_total,
        feat_dim,
        obj_node_offset,
    )
end

# ── Incremental updates ───────────────────────────────────────────────────────

"""
    update_graph_deletions!(graph, schema, deleted_slots; backend)

Mark deleted nodes as inactive in the graph.  Called after `_gpu_apply_inplace!`
reports which slots were deactivated.  Edge tombstoning is implicit: the caller
filters edges via `node_active` before GNN inference.
"""
function update_graph_deletions!(graph::GPUGraphData,
                                  schema::SchemaInfo,
                                  deleted_slots::Dict{Symbol, Vector{Int32}};
                                  backend = CUDA.CUDABackend())
    for (o, slots) in deleted_slots
        isempty(slots) && continue
        oi  = schema.obj_index[o]
        off = graph.obj_node_offset[oi]
        d_slots = CuVector{Int32}(slots)
        n = length(slots)
        _graph_set_node_inactive_kernel!(backend, 256)(
            graph.node_active, d_slots, Int32(off); ndrange=n)
    end
    KernelAbstractions.synchronize(backend)
end

"""
    update_graph_additions!(graph, g, schema, added_slots; backend)

Write node features and activate nodes for newly added slots.  Also appends
new COO edges for the new slots.  Called after `apply_pushout!` activates new
slots in `g`.
"""
function update_graph_additions!(graph::GPUGraphData,
                                  g::GPUACSet,
                                  schema::SchemaInfo,
                                  added_slots::Dict{Symbol, Vector{Int32}};
                                  backend = CUDA.CUDABackend())
    # Activate nodes and write features via GPU kernels (attrs already set in g)
    for (o, slots) in added_slots
        isempty(slots) && continue
        oi  = schema.obj_index[o]
        off = graph.obj_node_offset[oi]
        d_slots = CuVector{Int32}(slots)
        n = length(slots)

        # Activate node_active for the new slots
        _graph_set_node_active_kernel!(backend, 256)(
            graph.node_active, d_slots, Int32(off); ndrange=n)

        # Write node features: for each attribute of this type, scatter into node_feat
        n_alloc = g.n_alloc[o]
        for (ai, a) in enumerate(schema.attrs)
            schema.attr_dom[a] == o || continue
            _graph_write_features_kernel!(backend, 256)(
                graph.node_feat, @view(g.attrs[a][1:n_alloc]),
                Int32(off), Int32(ai), Int32(n_alloc); ndrange=n_alloc)
        end
    end
    KernelAbstractions.synchronize(backend)

    # Append new COO edges for added slots (download only the new slots' FK values)
    new_src  = Int32[]
    new_dst  = Int32[]
    new_type = Int32[]
    for (o, slots) in added_slots
        isempty(slots) && continue
        oi  = schema.obj_index[o]
        off = Int(graph.obj_node_offset[oi])
        for (hi, h) in enumerate(schema.homs)
            schema.hom_dom[h] == o || continue
            cod_o   = schema.hom_cod[h]
            cod_off = Int(graph.obj_node_offset[schema.obj_index[cod_o]])
            fk_cpu  = Array(@view(g.homs[h][1:g.n_alloc[o]]))
            for k in slots
                tgt = Int(fk_cpu[Int(k)])
                tgt == 0 && continue
                push!(new_src,  Int32(off + Int(k)))
                push!(new_dst,  Int32(cod_off + tgt))
                push!(new_type, Int32(hi))
            end
        end
        # Also add reverse edges: homs whose cod is o (other slots pointing TO new slots)
        for (hi, h) in enumerate(schema.homs)
            schema.hom_cod[h] == o || continue
            dom_o   = schema.hom_dom[h]
            dom_off = Int(graph.obj_node_offset[schema.obj_index[dom_o]])
            n_dom   = g.n_alloc[dom_o]
            act_dom = Array(g.active[dom_o])
            fk_cpu  = Array(@view(g.homs[h][1:n_dom]))
            slot_set = Set(Int.(slots))
            for k in 1:n_dom
                act_dom[k] || continue
                tgt = Int(fk_cpu[k])
                tgt ∈ slot_set || continue
                cod_off = Int(graph.obj_node_offset[schema.obj_index[o]])
                push!(new_src,  Int32(dom_off + k))
                push!(new_dst,  Int32(cod_off + tgt))
                push!(new_type, Int32(hi))
            end
        end
    end

    if !isempty(new_src)
        # Append to existing COO arrays
        new_n = length(new_src)
        old_n = graph.n_edges
        # Grow arrays if needed (CuArray does not support push!; reallocate)
        new_total = old_n + new_n
        new_edge_src  = CUDA.zeros(Int32, new_total)
        new_edge_dst  = CUDA.zeros(Int32, new_total)
        new_edge_type = CUDA.zeros(Int32, new_total)
        if old_n > 0
            copyto!(new_edge_src,  1, graph.edge_src,  1, old_n)
            copyto!(new_edge_dst,  1, graph.edge_dst,  1, old_n)
            copyto!(new_edge_type, 1, graph.edge_type, 1, old_n)
        end
        d_new_src  = CuVector{Int32}(new_src)
        d_new_dst  = CuVector{Int32}(new_dst)
        d_new_type = CuVector{Int32}(new_type)
        copyto!(new_edge_src,  old_n + 1, d_new_src,  1, new_n)
        copyto!(new_edge_dst,  old_n + 1, d_new_dst,  1, new_n)
        copyto!(new_edge_type, old_n + 1, d_new_type, 1, new_n)
        graph.edge_src  = new_edge_src
        graph.edge_dst  = new_edge_dst
        graph.edge_type = new_edge_type
        graph.n_edges   = new_total
    end
end

"""
    rebuild_gpu_graph!(graph, g, schema, enc; backend)

Full in-place rebuild.  Used after compaction (when slot indices shift) or as a
safe fallback when incremental tracking is impractical.
"""
function rebuild_gpu_graph!(graph::GPUGraphData,
                             g::GPUACSet,
                             schema::SchemaInfo,
                             enc::AttributeEncoder;
                             backend = CUDA.CUDABackend())
    new_gd = build_gpu_graph(g, schema, enc; backend)
    graph.edge_src        = new_gd.edge_src
    graph.edge_dst        = new_gd.edge_dst
    graph.edge_type       = new_gd.edge_type
    graph.n_edges         = new_gd.n_edges
    graph.node_active     = new_gd.node_active
    graph.node_type       = new_gd.node_type
    graph.node_feat       = new_gd.node_feat
    graph.n_nodes_alloc   = new_gd.n_nodes_alloc
    graph.obj_node_offset = new_gd.obj_node_offset
    nothing
end

# ── COO extraction for GNN libraries ─────────────────────────────────────────

"""
    live_coo(graph) -> (src, dst, edge_type, node_feat)

Return the subset of COO edges where both source and destination nodes are
active, along with the full node feature matrix.  Used to construct a GNNGraph.
Filters inactive-node edges on CPU (acceptable for inference-time use).
"""
function live_coo(graph::GPUGraphData)
    act    = Array(graph.node_active)
    src_h  = Array(graph.edge_src)
    dst_h  = Array(graph.edge_dst)
    etype_h = Array(graph.edge_type)

    mask = [act[Int(s)] && act[Int(d)] for (s, d) in zip(src_h, dst_h)]
    src_live   = CuVector{Int32}(src_h[mask])
    dst_live   = CuVector{Int32}(dst_h[mask])
    etype_live = CuVector{Int32}(etype_h[mask])

    src_live, dst_live, etype_live, graph.node_feat
end
