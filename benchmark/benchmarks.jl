"""
Naive tutorial timing — times each major step of TicTacToeGameplay.qmd.

Run from the repo root:
    julia --project=benchmark benchmark/benchmarks.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__))

using Catlab, AlgebraicRewriting
using RewriteGames
using Random
using Dates
using Statistics: mean

println("=== RewriteGames Tutorial Timing ===")
println("Julia $(VERSION)\n")

# ── Part 1: Schema ───────────────────────────────────────────────────────────

print("Part 1 — @present + @acset_type + ACSetCategory ... ")
t_schema = @elapsed begin
    @present SchTTT(FreeSchema) begin
        Sq::Ob; E::Ob; X::Ob; O::Ob
        xsq::Hom(X, Sq); osq::Hom(O, Sq)
        src::Hom(E, Sq); tgt::Hom(E, Sq)
        SquareNum::AttrType
        num::Attr(Sq, SquareNum)
    end
    @acset_type TicTacToe(SchTTT, index=[:xsq, :osq])
    TTT = TicTacToe{Int}
    𝒞 = ACSetCategory(VarACSetCat(TTT()))
end
println("$(round(1000t_schema; digits=1)) ms")

function create_board()
    ttt = TTT()
    add_parts!(ttt, :Sq, 9; num=collect(1:9))
    for i in 0:2, j in 0:1
        add_part!(ttt, :E, src=3*i+j+1, tgt=3*i+j+2)
    end
    for i in 0:1, j in 0:2
        add_part!(ttt, :E, src=3*i+j+1, tgt=3*i+j+1+3)
    end
    return ttt
end

# ── Part 2: Yoneda cache + rules ─────────────────────────────────────────────

print("Part 2 — yoneda_cache + representables ... ")
t_yoneda = @elapsed begin
    gSq, gE, gX, gO = ob_generators(FinCat(SchTTT))
    yTTT   = yoneda_cache(TTT; clear=false)
    I      = TTT()
    Sq_rep = ob_map(yTTT, gSq)
    X_rep  = ob_map(yTTT, gX)
    O_rep  = ob_map(yTTT, gO)
    N      = Names(Dict("X" => X_rep, "O" => O_rep, "Sq" => Sq_rep, "" => I, "I" => I))
end
println("$(round(1000t_yoneda; digits=1)) ms")

print("Part 2 — mark_x rule + win-pattern rules ... ")
t_rules = @elapsed begin
    id_Sq    = id[𝒞](Sq_rep)
    mark_X_r = homomorphism(Sq_rep, X_rep; cat=𝒞)
    mark_x   = Rule(id_Sq, mark_X_r; monic=true,
                    ac=[NAC(homomorphism(Sq_rep, X_rep; cat=𝒞)),
                        NAC(homomorphism(Sq_rep, O_rep; cat=𝒞))])

    row_col_structural = TTT()
    add_parts!(row_col_structural, :SquareNum, 3)
    add_parts!(row_col_structural, :Sq, 3; num=AttrVar.(1:3))
    add_part!(row_col_structural, :E, src=1, tgt=2)
    add_part!(row_col_structural, :E, src=2, tgt=3)
    for i in 1:3; add_part!(row_col_structural, :X, xsq=i); end

    diag_structural_1 = TTT()
    add_parts!(diag_structural_1, :SquareNum, 5)
    add_parts!(diag_structural_1, :Sq, 5; num=AttrVar.(1:5))
    add_part!(diag_structural_1, :E, src=1, tgt=2); add_part!(diag_structural_1, :E, src=2, tgt=3)
    add_part!(diag_structural_1, :E, src=3, tgt=4); add_part!(diag_structural_1, :E, src=4, tgt=5)
    for i in [1,3,5]; add_part!(diag_structural_1, :X, xsq=i); end

    diag_structural_2 = TTT()
    add_parts!(diag_structural_2, :SquareNum, 5)
    add_parts!(diag_structural_2, :Sq, 5; num=AttrVar.(1:5))
    add_part!(diag_structural_2, :E, src=1, tgt=2); add_part!(diag_structural_2, :E, src=3, tgt=2)
    add_part!(diag_structural_2, :E, src=3, tgt=4); add_part!(diag_structural_2, :E, src=5, tgt=4)
    for i in [1,3,5]; add_part!(diag_structural_2, :X, xsq=i); end

    x_rows_rule  = Rule(id[𝒞](row_col_structural),  id[𝒞](row_col_structural); monic=true)
    x_diag1_rule = Rule(id[𝒞](diag_structural_1),   id[𝒞](diag_structural_1);  monic=true)
    x_diag2_rule = Rule(id[𝒞](diag_structural_2),   id[𝒞](diag_structural_2);  monic=true)

    x_rows_app  = RuleApp(:x_wins_rows,  x_rows_rule,  I; cat=𝒞)
    x_diag1_app = RuleApp(:x_wins_diag1, x_diag1_rule, I; cat=𝒞)
    x_diag2_app = RuleApp(:x_wins_diag2, x_diag2_rule, I; cat=𝒞)
end
println("$(round(1000t_rules; digits=1)) ms")

# ── Part 3: Migration functor ─────────────────────────────────────────────────

print("Part 3 — Migrate functor ... ")
t_migrate = @elapsed begin
    F      = Migrate(𝒞,
                     Dict(:X => :O, :O => :X, :Sq => :Sq, :E => :E, :SquareNum => :SquareNum),
                     Dict(:xsq => :osq, :osq => :xsq, :src => :src, :tgt => :tgt, :num => :num),
                     SchTTT, TTT)
    mark_o = F(mark_x)
end
println("$(round(1000t_migrate; digits=1)) ms")

# ── Part 4: Schedule construction ────────────────────────────────────────────

print("Part 4 — X win-check sub-schedule ... ")
t_win_x = @elapsed begin
    x_won_check_gs = mk_game_sched((;), (init=:I,), N,
        (r=x_rows_app, d1=x_diag1_app, d2=x_diag2_app, mw=merge_wires(I)),
        quote
            won_r,  not_r  = r(init)
            won_d1, not_d1 = d1(not_r)
            won_d2, not_d2 = d2(not_d1)
            won12 = mw(won_r,  won_d1)
            won   = mw(won12,  won_d2)
            return won, not_d2
        end)
end
println("$(round(1000t_win_x; digits=1)) ms")

print("Part 4 — X turn sub-schedule ... ")
t_x_sched = @elapsed begin
    mark_x_app = PlayerRuleApp(:mark_x, mark_x, I, :X; cat=𝒞)
    X_sched_gs = mk_game_sched((;), (init=:I,), N,
        (mx=mark_x_app,),
        quote
            moved, tie = mx(init)
            return moved, tie
        end)
end
println("$(round(1000t_x_sched; digits=1)) ms")

print("Part 4 — O schedules via player_migrate ... ")
t_o_sched = @elapsed begin
    O_sched_gs     = player_migrate(F, X_sched_gs, Dict(:X => :O); name_map=Dict(:mark_x => :mark_o))
    o_won_check_gs = player_migrate(F, x_won_check_gs, Dict(:X => :O))
end
println("$(round(1000t_o_sched; digits=1)) ms")

print("Part 4 — Full game schedule (baseline) ... ")
t_game_sched = @elapsed begin
    game_sched = mk_game_sched(
        (trace_arg=:I,), (init=:I,), N,
        (x=X_sched_gs, o=O_sched_gs, cx=x_won_check_gs, co=o_won_check_gs, mw=merge_wires(I)),
        quote
            x_moved, x_tie = x([init, trace_arg])
            x_won, x_cont  = cx(x_moved)
            o_moved, o_tie = o(x_cont)
            o_won, o_cont  = co(o_moved)
            tie = mw(x_tie, o_tie)
            return o_cont, x_won, o_won, tie
        end)
end
println("$(round(1000t_game_sched; digits=1)) ms")

print("Part 4 — Full game schedule (incremental cache) ... ")
t_game_sched_cached = @elapsed begin
    mark_x_app_cached = PlayerRuleApp(:mark_x, mark_x, I, :X; cat=𝒞, use_cache=true)
    X_sched_cached    = mk_game_sched((;), (init=:I,), N,
        (mx=mark_x_app_cached,),
        quote
            moved, tie = mx(init)
            return moved, tie
        end)
    O_sched_cached    = player_migrate(F, X_sched_cached, Dict(:X => :O); name_map=Dict(:mark_x => :mark_o))
    game_sched_cached = mk_game_sched(
        (trace_arg=:I,), (init=:I,), N,
        (x=X_sched_cached, o=O_sched_cached, cx=x_won_check_gs, co=o_won_check_gs, mw=merge_wires(I)),
        quote
            x_moved, x_tie = x([init, trace_arg])
            x_won, x_cont  = cx(x_moved)
            o_moved, o_tie = o(x_cont)
            o_won, o_cont  = co(o_moved)
            tie = mw(x_tie, o_tie)
            return o_cont, x_won, o_won, tie
        end)
end
println("$(round(1000t_game_sched_cached; digits=1)) ms")

# ── Part 5: Run game ──────────────────────────────────────────────────────────

game = Game(SchTTT;
    players        = [:X, :O],
    initial        = create_board,
    win_conditions = Dict{Symbol, Any}(:x_won => :X, :o_won => :O, :tie => nothing))

random_agents = Dict{Symbol, AbstractAgent}(
    :X => FunctionAgent((state, actions) -> rand(actions)),
    :O => FunctionAgent((state, actions) -> rand(actions)),
)

print("Part 5 — Single episode, baseline ... ")
Random.seed!(42)
t_episode = @elapsed run_game_sched!(game_sched, game, random_agents; T_max=20)
println("$(round(1000t_episode; digits=2)) ms")

print("Part 5 — Single episode, incremental cache ... ")
Random.seed!(42)
t_episode_cached = @elapsed run_game_sched!(game_sched_cached, game, random_agents; T_max=20)
println("$(round(1000t_episode_cached; digits=2)) ms  (speedup: $(round(t_episode/t_episode_cached; digits=2))×)")

print("Part 5 — 200 episodes, baseline ... ")
Random.seed!(42)
t_batch200 = @elapsed begin
    all_exps = [run_game_sched!(game_sched, game, random_agents; T_max=20) for _ in 1:200]
end
x_wins = count(e -> !isempty(e) && e[end].winner === :X, all_exps)
o_wins = count(e -> !isempty(e) && e[end].winner === :O, all_exps)
draws  = 200 - x_wins - o_wins
mean_len = mean(episode_length.(all_exps))
println("$(round(t_batch200; digits=2)) s  (X $(x_wins), O $(o_wins), draws $(draws), mean length $(round(mean_len; digits=1)))")

print("Part 5 — 200 episodes, incremental cache ... ")
Random.seed!(42)
t_batch200_cached = @elapsed begin
    [run_game_sched!(game_sched_cached, game, random_agents; T_max=20) for _ in 1:200]
end
println("$(round(t_batch200_cached; digits=2)) s  (speedup: $(round(t_batch200/t_batch200_cached; digits=2))×)")

# ── Part 6: GNN / RL ─────────────────────────────────────────────────────────

println("\nPart 6 — GNN / RL ...")
using Flux
using GraphNeuralNetworks

struct TTTGNNPolicy
    node_embed  :: Dense
    conv1       :: GCNConv
    conv2       :: GCNConv
    action_head :: Dense
end
Flux.@layer TTTGNNPolicy

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
    vec(p.action_head(sq_feats))
end

function world_to_gnn(world::TicTacToe)
    n_sq = nparts(world, :Sq); n_e = nparts(world, :E)
    n_x  = nparts(world, :X);  n_o = nparts(world, :O)
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
    GNNGraph(srcs, dsts; ndata=(; x=nf), num_nodes=n_nodes), sq_off
end

action_sq(a::Action) = collect(components(a.match)[:Sq])[1]

function sample_categorical(probs::Vector{Float32})
    r = rand(Float32); cumulative = 0f0
    for (i, p) in enumerate(probs)
        cumulative += p
        cumulative >= r && return i
    end
    length(probs)
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
            pw     = player === :O ? F(state.world) : state.world
            g, _   = world_to_gnn(pw)
            sq_ids = [action_sq(a) for a in legal_actions]
            logits = model(g, sq_ids)
            probs  = softmax(logits)
            chosen = sample_categorical(probs)
            push!(records, StepRecord(copy(pw), sq_ids, chosen, player))
            legal_actions[chosen]
        end)
    end
    agents = Dict{Symbol, AbstractAgent}(:X => make_agent(:X), :O => make_agent(:O))
    exps   = run_game_sched!(game_sched, game, agents; T_max=20)
    winner = isempty(exps) ? nothing : exps[end].winner
    records, winner
end

function reinforce_loss(model, batch, returns, graphs)
    total = 0f0
    for (rec, G, g) in zip(batch, returns, graphs)
        logits    = model(g, rec.sq_ids)
        log_probs = logits .- log(sum(exp.(logits)))
        total    -= log_probs[rec.chosen] * G
    end
    total / length(batch)
end

function train_self_play!(model, opt_state, game_sched;
                          n_updates=10, n_episodes_per_update=25)
    for _ in 1:n_updates
        batch   = StepRecord[]
        returns = Float32[]
        for _ in 1:n_episodes_per_update
            recs, winner = run_self_play_episode(model, game_sched)
            for rec in recs
                G = winner === rec.player ? 1f0 : winner === nothing ? 0f0 : -1f0
                push!(batch, rec); push!(returns, G)
            end
        end
        graphs = GNNGraph[world_to_gnn(rec.world)[1] for rec in batch]
        _, grads = Flux.withgradient(m -> reinforce_loss(m, batch, returns, graphs), model)
        Flux.update!(opt_state, model, grads[1])
    end
end

# Timing
sample_world = create_board()
add_part!(sample_world, :X, xsq=5)
add_part!(sample_world, :O, osq=1)

print("Part 6 — world_to_gnn encoding ... ")
t_encode = @elapsed world_to_gnn(sample_world)
println("$(round(1000t_encode; digits=2)) ms")

g_sample, _ = world_to_gnn(sample_world)
gnn_model   = TTTGNNPolicy(embed_dim=16, hidden_dim=32)

print("Part 6 — GNN forward pass (7 legal squares) ... ")
t_forward = @elapsed gnn_model(g_sample, [2, 3, 4, 6, 7, 8, 9])
println("$(round(1000t_forward; digits=2)) ms")

print("Part 6 — 50 self-play episodes, baseline ... ")
Random.seed!(1)
t_selfplay = @elapsed begin
    for _ in 1:50; run_self_play_episode(gnn_model, game_sched); end
end
println("$(round(t_selfplay; digits=2)) s  ($(round(1000t_selfplay/50; digits=1)) ms/episode)")

print("Part 6 — 50 self-play episodes, incremental cache ... ")
Random.seed!(1)
t_selfplay_cached = @elapsed begin
    for _ in 1:50; run_self_play_episode(gnn_model, game_sched_cached); end
end
println("$(round(t_selfplay_cached; digits=2)) s  ($(round(1000t_selfplay_cached/50; digits=1)) ms/episode, speedup: $(round(t_selfplay/t_selfplay_cached; digits=2))×)")

print("Part 6 — gradient update (batch from 50 episodes) ... ")
batch   = StepRecord[]; returns = Float32[]
Random.seed!(1)
for _ in 1:50
    recs, winner = run_self_play_episode(gnn_model, game_sched)
    for rec in recs
        push!(batch, rec)
        push!(returns, winner === rec.player ? 1f0 : winner === nothing ? 0f0 : -1f0)
    end
end
opt_state = Flux.setup(Adam(1e-3), gnn_model)
t_grad = @elapsed begin
    graphs = GNNGraph[world_to_gnn(rec.world)[1] for rec in batch]
    _, grads = Flux.withgradient(m -> reinforce_loss(m, batch, returns, graphs), gnn_model)
    Flux.update!(opt_state, gnn_model, grads[1])
end
println("$(round(1000t_grad; digits=1)) ms  (batch of $(length(batch)) steps)")

print("Part 6 — full training, baseline (10 updates × 25 episodes) ... ")
Random.seed!(1)
gnn_model2 = TTTGNNPolicy(embed_dim=16, hidden_dim=32)
opt_state2 = Flux.setup(Adam(1e-3), gnn_model2)
t_train = @elapsed train_self_play!(gnn_model2, opt_state2, game_sched;
                                    n_updates=10, n_episodes_per_update=25)
println("$(round(t_train; digits=2)) s")

print("Part 6 — full training, incremental cache (10 updates × 25 episodes) ... ")
Random.seed!(1)
gnn_model3 = TTTGNNPolicy(embed_dim=16, hidden_dim=32)
opt_state3 = Flux.setup(Adam(1e-3), gnn_model3)
t_train_cached = @elapsed train_self_play!(gnn_model3, opt_state3, game_sched_cached;
                                           n_updates=10, n_episodes_per_update=25)
println("$(round(t_train_cached; digits=2)) s  (speedup: $(round(t_train/t_train_cached; digits=2))×)")

# ── Summary ───────────────────────────────────────────────────────────────────

speedup5  = round(t_batch200    / t_batch200_cached;    digits=2)
speedup6  = round(t_train       / t_train_cached;        digits=2)
println("""

=== Summary ===
  Part 1  schema + ACSetCategory       $(lpad(round(1000t_schema;         digits=1), 8)) ms
  Part 2  yoneda cache                 $(lpad(round(1000t_yoneda;         digits=1), 8)) ms
  Part 2  rules                        $(lpad(round(1000t_rules;          digits=1), 8)) ms
  Part 3  migrate functor              $(lpad(round(1000t_migrate;        digits=1), 8)) ms
  Part 4  win-check sub-sched (X)      $(lpad(round(1000t_win_x;         digits=1), 8)) ms
  Part 4  X turn sub-sched             $(lpad(round(1000t_x_sched;       digits=1), 8)) ms
  Part 4  O schedules (migrate)        $(lpad(round(1000t_o_sched;       digits=1), 8)) ms
  Part 4  full game schedule           $(lpad(round(1000t_game_sched;    digits=1), 8)) ms
  Part 4  full game schedule (cached)  $(lpad(round(1000t_game_sched_cached; digits=1), 8)) ms

  Hom-search comparison (random agents)
                                        baseline      cached    speedup
  Part 5  single episode               $(lpad(round(1000t_episode;       digits=1), 8)) ms  $(lpad(round(1000t_episode_cached; digits=1), 8)) ms  $(round(t_episode/t_episode_cached; digits=2))×
  Part 5  200 episodes                 $(lpad(round(t_batch200;          digits=2), 8)) s   $(lpad(round(t_batch200_cached;    digits=2), 8)) s   $(speedup5)×

  Hom-search comparison (GNN self-play)
                                        baseline      cached    speedup
  Part 6  50 self-play episodes        $(lpad(round(t_selfplay;          digits=2), 8)) s   $(lpad(round(t_selfplay_cached;    digits=2), 8)) s   $(round(t_selfplay/t_selfplay_cached; digits=2))×
  Part 6  full training (10×25)        $(lpad(round(t_train;             digits=2), 8)) s   $(lpad(round(t_train_cached;       digits=2), 8)) s   $(speedup6)×

  Part 6  world_to_gnn                 $(lpad(round(1000t_encode;        digits=2), 8)) ms
  Part 6  GNN forward                  $(lpad(round(1000t_forward;       digits=2), 8)) ms
  Part 6  gradient update              $(lpad(round(1000t_grad;          digits=1), 8)) ms
""")

# Write findings
mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "findings.md"), "w") do io
    println(io, "# Benchmark Findings")
    println(io, "")
    println(io, "Generated: $(Dates.now())  Julia $(VERSION)")
    println(io, "")
    println(io, "## Setup")
    println(io, "")
    println(io, "| Step | Time |")
    println(io, "|------|------|")
    println(io, "| Part 1 — schema + ACSetCategory | $(round(1000t_schema; digits=1)) ms |")
    println(io, "| Part 2 — yoneda cache | $(round(1000t_yoneda; digits=1)) ms |")
    println(io, "| Part 2 — rules | $(round(1000t_rules; digits=1)) ms |")
    println(io, "| Part 3 — migrate functor | $(round(1000t_migrate; digits=1)) ms |")
    println(io, "| Part 4 — X win-check sub-schedule | $(round(1000t_win_x; digits=1)) ms |")
    println(io, "| Part 4 — X turn sub-schedule | $(round(1000t_x_sched; digits=1)) ms |")
    println(io, "| Part 4 — O schedules (player_migrate) | $(round(1000t_o_sched; digits=1)) ms |")
    println(io, "| Part 4 — full game schedule (baseline) | $(round(1000t_game_sched; digits=1)) ms |")
    println(io, "| Part 4 — full game schedule (cached) | $(round(1000t_game_sched_cached; digits=1)) ms |")
    println(io, "")
    println(io, "## Hom-search: baseline vs incremental cache")
    println(io, "")
    println(io, "| Benchmark | baseline | incremental cache | speedup |")
    println(io, "|-----------|----------|-------------------|---------|")
    println(io, "| Part 5 — single episode (random) | $(round(1000t_episode; digits=1)) ms | $(round(1000t_episode_cached; digits=1)) ms | $(round(t_episode/t_episode_cached; digits=2))× |")
    println(io, "| Part 5 — 200 episodes (random) | $(round(t_batch200; digits=2)) s | $(round(t_batch200_cached; digits=2)) s | $(speedup5)× |")
    println(io, "| Part 6 — 50 self-play episodes | $(round(t_selfplay; digits=2)) s | $(round(t_selfplay_cached; digits=2)) s | $(round(t_selfplay/t_selfplay_cached; digits=2))× |")
    println(io, "| Part 6 — full training (10×25) | $(round(t_train; digits=2)) s | $(round(t_train_cached; digits=2)) s | $(speedup6)× |")
    println(io, "")
    println(io, "## GNN components")
    println(io, "")
    println(io, "| Step | Time |")
    println(io, "|------|------|")
    println(io, "| Part 6 — world_to_gnn | $(round(1000t_encode; digits=2)) ms |")
    println(io, "| Part 6 — GNN forward pass | $(round(1000t_forward; digits=2)) ms |")
    println(io, "| Part 6 — gradient update | $(round(1000t_grad; digits=1)) ms |")
end
println("Results written to benchmark/results/findings.md")
