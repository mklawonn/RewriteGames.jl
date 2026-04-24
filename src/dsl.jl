"""
    @game schema begin ... end

A concise DSL for constructing a `Game`.  The body supports the following
clauses (each on its own line, in any order):

```julia
@game SchGraph begin
    players:  alice, bob
    terminal: (W) -> (nparts(W, :V) >= 10, nothing)
    initial:  () -> Graph(2)
end
```

**Clauses:**
- `players:` — comma-separated bare player names (symbols); required.
- `terminal:` — a function `W -> (Bool, Union{Symbol,Nothing})`.
- `initial:` — a zero-argument factory `() -> ACSet`.

Rules now live inside `PlayerRuleApp` boxes in the wiring-diagram schedule;
use `mk_game_sched` to build the schedule separately.
"""
macro game(schema, body)
    body isa Expr && body.head === :block ||
        error("@game: second argument must be a begin...end block")

    players_sym   = nothing
    terminal_expr = :((W) -> (false, nothing))
    initial_expr  = :(() -> error("No initial world factory provided"))

    for stmt in body.args
        stmt isa LineNumberNode && continue

        local keysym::Symbol, val
        if stmt isa Expr && stmt.head === :call && length(stmt.args) == 3 &&
                stmt.args[1] === :((:))
            keysym = stmt.args[2] isa Symbol ? stmt.args[2] :
                error("@game: clause key must be a bare symbol, got $(stmt.args[2])")
            val = stmt.args[3]
        elseif stmt isa Expr && stmt.head === :tuple &&
                !isempty(stmt.args) &&
                stmt.args[1] isa Expr && stmt.args[1].head === :call &&
                length(stmt.args[1].args) == 3 &&
                stmt.args[1].args[1] === :((:))
            first_kv = stmt.args[1]
            keysym = first_kv.args[2] isa Symbol ? first_kv.args[2] :
                error("@game: clause key must be a bare symbol, got $(first_kv.args[2])")
            rest = stmt.args[2:end]
            val = isempty(rest) ? first_kv.args[3] :
                  Expr(:tuple, first_kv.args[3], rest...)
        elseif stmt isa Expr && stmt.head === :(=)
            keysym = stmt.args[1] isa Symbol ? stmt.args[1] :
                error("@game: clause key must be a bare symbol, got $(stmt.args[1])")
            val = stmt.args[2]
        else
            error("@game: unrecognised clause:\n  $stmt")
        end

        if keysym === :players
            if val isa Symbol
                players_sym = Expr(:vect, QuoteNode(val))
            elseif val isa Expr && val.head === :tuple
                players_sym = Expr(:vect, QuoteNode.(val.args)...)
            else
                players_sym = esc(val)
            end
        elseif keysym === :terminal
            terminal_expr = val
        elseif keysym === :initial
            initial_expr = val
        else
            error("@game: unrecognised clause key :$keysym. " *
                  "Valid clauses: players, terminal, initial")
        end
    end

    players_sym === nothing &&
        error("@game: missing required 'players:' clause")

    quote
        Game(
            $(esc(schema));
            players  = $players_sym,
            terminal = $(esc(terminal_expr)),
            initial  = $(esc(initial_expr)),
        )
    end
end
