"""
RL training benchmarks.

Measures end-to-end REINFORCE self-play training (10 updates × 25 episodes)
under three match strategies and, when available, on GPU vs CPU.

The training budget is intentionally small — the goal is to stress the
engine's algebraic machinery, not to produce a strong player.
"""

include(joinpath(@__DIR__, "ttt_setup.jl"))

using Flux
using GraphNeuralNetworks

# ── GPU detection ─────────────────────────────────────────────────────────────

const USE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

const _device = USE_GPU ? Flux.gpu : Flux.cpu

if USE_GPU
    @info "GPU benchmark enabled: $(CUDA.name(CUDA.device()))"
else
    @info "No functional GPU found — benchmarking CPU only"
end

# ── GNN policy (identical to tutorial Part 6) ─────────────────────────────────

struct TTTGNNPolicy
    node_embed  :: Dense
    conv1       :: GCNConv
    conv2       :: GCNConv
    action_head :: Dense
end

Flux.@functor TTTGNNPolicy

function TTTGNNPolicy(; embed_dim=16, hidden_dim=32)
    TTTGNNPolicy(
        Dense(4, embed_dim, relu),
        GCNConv(embed_dim => hidden_dim, relu),
        GCNConv(hidden_dim => embed_dim),
        Dense(embed_dim, 1),
    )
end

function (p::TTTGNNPolicy)(g::GNNGraph, sq_indices::Vector{Int})
    x = p.node_embed(g.ndata.x)
    x = p.conv1(g, x)
    x = p.conv2(g, x)
    sq_feats = x[:, sq_indices]
    logits   = vec(p.action_head(sq_feats))
    return logits
end

# ── Board → GNNGraph encoding ─────────────────────────────────────────────────

function world_to_gnn(world::TicTacToe)
    n_sq = nparts(world, :Sq)
    n_e  = nparts(world, :E)
    n_x  = nparts(world, :X)
    n_o  = nparts(world, :O)
    n_nodes = n_sq + n_e + n_x + n_o

    sq_off = 0; e_off = n_sq; x_off = n_sq + n_e; o_off = n_sq + n_e + n_x

    nf = zeros(Float32, 4, n_nodes)
    for i in 1:n_sq; nf[1, sq_off + i] = 1f0; end
    for i in 1:n_e;  nf[2, e_off  + i] = 1f0; end
    for i in 1:n_x;  nf[3, x_off  + i] = 1f0; end
    for i in 1:n_o;  nf[4, o_off  + i] = 1f0; end

    srcs = Int[]; dsts = Int[]
    for xi in 1:n_x
        sq_j = subpart(world, xi, :xsq)
        push!(srcs, x_off+xi); push!(dsts, sq_off+sq_j)
        push!(srcs, sq_off+sq_j); push!(dsts, x_off+xi)
    end
    for oi in 1:n_o
        sq_j = subpart(world, oi, :osq)
        push!(srcs, o_off+oi); push!(dsts, sq_off+sq_j)
        push!(srcs, sq_off+sq_j); push!(dsts, o_off+oi)
    end
    for ei in 1:n_e
        sq_j = subpart(world, ei, :src)
        push!(srcs, e_off+ei); push!(dsts, sq_off+sq_j)
        push!(srcs, sq_off+sq_j); push!(dsts, e_off+ei)
    end
    for ei in 1:n_e
        sq_j = subpart(world, ei, :tgt)
        push!(srcs, e_off+ei); push!(dsts, sq_off+sq_j)
        push!(srcs, sq_off+sq_j); push!(dsts, e_off+ei)
    end
    g = GNNGraph(srcs, dsts; ndata=(; x=nf), num_nodes=n_nodes)
    return g, sq_off
end

# ── Helpers from tutorial Part 6 ─────────────────────────────────────────────

action_sq(a::Action) = collect(components(a.match)[:Sq])[1]

perspective_world(world, player::Symbol) = player === :O ? F_migrate(world) : world

function sample_categorical(probs::Vector{Float32})
    r = rand(Float32); cumulative = 0f0
    for (i, p) in enumerate(probs)
        cumulative += p
        cumulative >= r && return i
    end
    return length(probs)
end

struct StepRecord
    world  :: TicTacToe{Int}
    sq_ids :: Vector{Int}
    chosen :: Int
    player :: Symbol
end

function run_self_play_episode(model, game_sched)
    records = StepRecord[]

    function make_agent(player::Symbol)
        FunctionAgent(function (state::GameState, legal_actions::Vector{Action})
            isempty(legal_actions) && return nothing
            pw      = perspective_world(state.world, player)
            g, _    = world_to_gnn(pw)
            g_dev   = g |> _device
            sq_ids  = [action_sq(a) for a in legal_actions]
            logits  = model(g_dev, sq_ids) |> Flux.cpu
            probs   = softmax(logits)
            chosen  = sample_categorical(probs)
            push!(records, StepRecord(copy(pw), sq_ids, chosen, player))
            return legal_actions[chosen]
        end)
    end

    agents = Dict{Symbol, AbstractAgent}(
        :X => make_agent(:X),
        :O => make_agent(:O),
    )
    exps = run_game_sched!(game_sched, TTT_GAME, agents; T_max=20)
    winner = isempty(exps) ? nothing : exps[end].winner
    return records, winner
end

function reinforce_loss(model, batch, returns, graphs)
    total = 0f0
    for (rec, G, g) in zip(batch, returns, graphs)
        logits   = model(g, rec.sq_ids)
        log_probs = logits .- log(sum(exp.(logits)))
        total    -= log_probs[rec.chosen] * G
    end
    return total / length(batch)
end

"""
    train_self_play!(model, opt_state, game_sched; n_updates, n_episodes_per_update)

REINFORCE self-play loop.  Returns vector of win-rates at each update.
"""
function train_self_play!(model, opt_state, game_sched;
                           n_updates             = 10,
                           n_episodes_per_update = 25)
    for _ in 1:n_updates
        batch   = StepRecord[]
        returns = Float32[]

        for _ in 1:n_episodes_per_update
            recs, winner = run_self_play_episode(model, game_sched)
            for rec in recs
                G = winner === rec.player ? 1f0 :
                    winner === nothing     ? 0f0 : -1f0
                push!(batch, rec)
                push!(returns, G)
            end
        end

        graphs = GNNGraph[world_to_gnn(rec.world)[1] |> _device for rec in batch]
        _, grads = Flux.withgradient(m -> reinforce_loss(m, batch, returns, graphs), model)
        Flux.update!(opt_state, model, grads[1])
    end
end

# ── Benchmark groups ──────────────────────────────────────────────────────────

const BENCH_RL = BenchmarkGroup()

# We benchmark three configurations across the full training loop.
# Each @benchmarkable creates a fresh model + opt_state so runs are independent.

for (label, use_cache, use_fast) in [
        ("baseline",          false, false),
        ("fast_path",         false, true),
        ("incremental_cache", true,  false),
    ]
    sched = build_game_sched(use_cache=use_cache, use_fast=use_fast)
    BENCH_RL[label] = @benchmarkable begin
        Random.seed!(1)
        model     = TTTGNNPolicy(embed_dim=16, hidden_dim=32) |> $_device
        opt_state = Flux.setup(Adam(1e-3), model)
        train_self_play!(model, opt_state, $sched;
                          n_updates=10, n_episodes_per_update=25)
    end seconds=120
end

# GPU variant (only when available)
if USE_GPU
    for (label, use_cache, use_fast) in [
            ("baseline_gpu",          false, false),
            ("incremental_cache_gpu", true,  false),
        ]
        sched = build_game_sched(use_cache=use_cache, use_fast=use_fast)
        BENCH_RL[label] = @benchmarkable begin
            Random.seed!(1)
            model     = TTTGNNPolicy(embed_dim=16, hidden_dim=32) |> Flux.gpu
            opt_state = Flux.setup(Adam(1e-3), model)
            train_self_play!(model, opt_state, $sched;
                              n_updates=10, n_episodes_per_update=25)
        end seconds=120
    end
end
