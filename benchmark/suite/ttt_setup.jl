"""
Shared TicTacToe game setup for all benchmark suites.

All suite files include this via:
    include(joinpath(@__DIR__, "ttt_setup.jl"))
"""

using Catlab, AlgebraicRewriting
using RewriteGames
using Random

# ── Schema ────────────────────────────────────────────────────────────────────

@present SchTTT(FreeSchema) begin
    Sq::Ob; E::Ob; X::Ob; O::Ob
    xsq::Hom(X, Sq); osq::Hom(O, Sq)
    src::Hom(E, Sq); tgt::Hom(E, Sq)
    SquareNum::AttrType
    num::Attr(Sq, SquareNum)
end

@acset_type TicTacToe(SchTTT, index=[:xsq, :osq])
const TTT = TicTacToe{Int}
const 𝒞_TTT = ACSetCategory(VarACSetCat(TTT()))

# ── Board factory ─────────────────────────────────────────────────────────────

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

# ── Yoneda representables ─────────────────────────────────────────────────────

const yTTT = yoneda_cache(TTT; clear=false)
const I    = TTT()
const gSq, gE, gX, gO = ob_generators(FinCat(SchTTT))
const Sq_rep = ob_map(yTTT, gSq)
const X_rep  = ob_map(yTTT, gX)
const O_rep  = ob_map(yTTT, gO)
const N_TTT  = Names(Dict("X" => X_rep, "O" => O_rep, "Sq" => Sq_rep, "" => I, "I" => I))

# ── Rules ─────────────────────────────────────────────────────────────────────

const id_Sq    = id[𝒞_TTT](Sq_rep)
const mark_X_r = homomorphism(Sq_rep, X_rep; cat=𝒞_TTT)
const mark_x   = Rule(id_Sq, mark_X_r; monic=true,
                       ac=[NAC(homomorphism(Sq_rep, X_rep; cat=𝒞_TTT)),
                           NAC(homomorphism(Sq_rep, O_rep; cat=𝒞_TTT))])

const mark_O_l = id[𝒞_TTT](O_rep)
const F_migrate = Migrate(
    𝒞_TTT,
    Dict(:X => :O, :O => :X, :Sq => :Sq, :E => :E, :SquareNum => :SquareNum),
    Dict(:xsq => :osq, :osq => :xsq, :src => :src, :tgt => :tgt, :num => :num),
    SchTTT, TTT)
const mark_o = F_migrate(mark_x)

# ── Fast-match function for placement rules ───────────────────────────────────

"""
    ttt_fast_matches(rule, world::TTT, cat) -> Vector

Bypasses homomorphism search for placement rules on TTT boards.
Computes empty squares directly from the ACSet index and constructs
match morphisms without running BacktrackingSearch.
"""
function ttt_fast_matches(rule, world, cat)
    occupied = Set(vcat(subpart(world, :xsq), subpart(world, :osq)))
    empty_sqs = setdiff(1:nparts(world, :Sq), occupied)
    L = codom(left(rule))   # = Sq_rep
    return [homomorphism(L, world; cat=cat, initial=Dict(:Sq => Dict(1 => s)))
            for s in empty_sqs]
end

# ── Win-detection patterns and rules ─────────────────────────────────────────

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

const x_rows_rule  = Rule(id[𝒞_TTT](row_col_structural),  id[𝒞_TTT](row_col_structural); monic=true)
const x_diag1_rule = Rule(id[𝒞_TTT](diag_structural_1),   id[𝒞_TTT](diag_structural_1);  monic=true)
const x_diag2_rule = Rule(id[𝒞_TTT](diag_structural_2),   id[𝒞_TTT](diag_structural_2);  monic=true)

const x_rows_app  = RuleApp(:x_wins_rows,  x_rows_rule,  I; cat=𝒞_TTT)
const x_diag1_app = RuleApp(:x_wins_diag1, x_diag1_rule, I; cat=𝒞_TTT)
const x_diag2_app = RuleApp(:x_wins_diag2, x_diag2_rule, I; cat=𝒞_TTT)

const x_won_check_gs = mk_game_sched((;), (init=:I,), N_TTT,
    (r=x_rows_app, d1=x_diag1_app, d2=x_diag2_app, mw=merge_wires(I)),
    quote
        won_r,  not_r  = r(init)
        won_d1, not_d1 = d1(not_r)
        won_d2, not_d2 = d2(not_d1)
        won12 = mw(won_r,  won_d1)
        won   = mw(won12,  won_d2)
        return won, not_d2
    end)

const o_won_check_gs = player_migrate(F_migrate, x_won_check_gs, Dict(:X => :O))

# ── Schedule builders ─────────────────────────────────────────────────────────

"""Build game schedule: baseline (no fast_match_fn, no cache)."""
function build_game_sched(; use_cache=false, use_fast=false)
    fast_fn = use_fast ? ttt_fast_matches : nothing
    mark_x_app = PlayerRuleApp(:mark_x, mark_x, I, :X; cat=𝒞_TTT,
                                fast_match_fn=fast_fn, use_cache=use_cache)
    X_sched_gs = mk_game_sched((;), (init=:I,), N_TTT,
        (mx=mark_x_app,),
        quote
            moved, tie = mx(init)
            return moved, tie
        end)

    O_sched_gs = player_migrate(F_migrate, X_sched_gs, Dict(:X => :O);
                                name_map=Dict(:mark_x => :mark_o))

    mk_game_sched(
        (trace_arg=:I,), (init=:I,), N_TTT,
        (x=X_sched_gs, o=O_sched_gs, cx=x_won_check_gs, co=o_won_check_gs,
         mw=merge_wires(I)),
        quote
            x_moved, x_tie = x([init, trace_arg])
            x_won, x_cont  = cx(x_moved)
            o_moved, o_tie = o(x_cont)
            o_won, o_cont  = co(o_moved)
            tie = mw(x_tie, o_tie)
            return o_cont, x_won, o_won, tie
        end)
end

# ── Game record ───────────────────────────────────────────────────────────────

const TTT_GAME = Game(SchTTT;
    players        = [:X, :O],
    initial        = create_board,
    win_conditions = Dict{Symbol, Any}(:x_won => :X, :o_won => :O, :tie => nothing))

const RANDOM_AGENTS = Dict{Symbol, AbstractAgent}(
    :X => FunctionAgent((state, actions) -> rand(actions)),
    :O => FunctionAgent((state, actions) -> rand(actions)),
)
