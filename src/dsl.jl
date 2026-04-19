"""
    @game schema begin ... end

A concise DSL for constructing a `Game`.  The body supports the following
clauses (each on its own line, in any order):

```julia
@game SchGraph begin
    players:  alice, bob
    alice:    [RuleEntry(r_add_vertex; name=:add_vertex)]
    bob:      [RuleEntry(r_add_edge; name=:add_edge, budget=3)]
    auto:     [AutoRule(r_env; prob_attr=:p)]
    terminal: (W) -> (nparts(W, :V) >= 10, nothing)
    initial:  () -> Graph(2)
end
```

**Clauses:**
- `players:` — comma-separated bare player names (symbols); optional if every
  player is named by a rule clause.
- `<player>:` — a Julia expression evaluating to a `Vector{RuleEntry}` for
  that player.
- `auto:` — a Julia expression evaluating to a `Vector{AutoRule}` (default: `[]`).
- `terminal:` — a function `W -> (Bool, Union{Symbol,Nothing})`.
- `initial:` — a zero-argument factory `() -> ACSet`.

The macro rewrites the block into a plain `Game(schema; ...)` call, so all
standard Julia expressions are valid inside each clause.
"""
macro game(schema, body)
    body isa Expr && body.head === :block ||
        error("@game: second argument must be a begin...end block")

    players_sym   = nothing          # will become Expr(:vect, ...)
    rules_pairs   = Pair{Symbol,Any}[]
    auto_expr     = :(AutoRule[])
    terminal_expr = :((W) -> (false, nothing))
    initial_expr  = :(() -> error("No initial world factory provided"))

    for stmt in body.args
        stmt isa LineNumberNode && continue

        # Julia parses `foo: bar` as `Expr(:call, :(:), :foo, bar)`
        local keysym::Symbol, val
        if stmt isa Expr && stmt.head === :call && length(stmt.args) == 3 &&
                stmt.args[1] === :((:))
            keysym = stmt.args[2] isa Symbol ? stmt.args[2] :
                error("@game: clause key must be a bare symbol, got $(stmt.args[2])")
            val = stmt.args[3]
        elseif stmt isa Expr && stmt.head === :(=)
            keysym = stmt.args[1] isa Symbol ? stmt.args[1] :
                error("@game: clause key must be a bare symbol, got $(stmt.args[1])")
            val = stmt.args[2]
        else
            error("@game: unrecognised clause:\n  $stmt")
        end

        if keysym === :players
            # `players: alice, bob`  →  val is a tuple or single symbol
            if val isa Symbol
                players_sym = Expr(:vect, QuoteNode(val))
            elseif val isa Expr && val.head === :tuple
                players_sym = Expr(:vect, QuoteNode.(val.args)...)
            else
                players_sym = esc(val)   # pass through arbitrary expression
            end
        elseif keysym === :auto
            auto_expr = val
        elseif keysym === :terminal
            terminal_expr = val
        elseif keysym === :initial
            initial_expr = val
        else
            push!(rules_pairs, keysym => val)
        end
    end

    if players_sym === nothing
        syms = [QuoteNode(p) for (p, _) in rules_pairs]
        players_sym = Expr(:vect, syms...)
    end

    rules_dict = Expr(:call, :Dict,
        [Expr(:call, :(=>), QuoteNode(p), esc(v)) for (p, v) in rules_pairs]...)

    quote
        Game(
            $(esc(schema));
            players  = $players_sym,
            rules    = $rules_dict,
            auto     = $(esc(auto_expr)),
            terminal = $(esc(terminal_expr)),
            initial  = $(esc(initial_expr)),
        )
    end
end
