## Lessons Learned: Algebraic Rewriting & Game Engine Integration

### Catlab & Variable ACSets
- **AttrVar Naturality:** In `MADVarACSetCat`, morphisms must be natural. Attribute variables in the domain must map to values or variables that preserve combinatorial consistency. Manual `ACSetTransformation` construction for attributes is safer than relying on defaults when variable indices are complex.
- **Homomorphism Search:** Use `homomorphism(X, Y; initial=..., cat=𝒞)` to bind agents to rules. The search engine handles the "variable-to-value" mapping required by `MADVarACSetCat` automatically.
- **Internal Objects:** Attribute morphism components in variable categories are often `CopairedFinDomFunction` objects. To evaluate them manually, use `f.val(x).val` to extract the underlying variable index or concrete value.

### AlgebraicRewriting & DPO Rules
- **Free-Floating Variables:** A DPO rule match will fail if `L` contains attribute variables with no preimage in `K`. All variables in `L` must be anchored to `K` to be bound during a search into a concrete world.
- **Attribute Modification:** In the `Rule(l, r)` constructor, `exprs` is reserved for attributes not mapped by `r`. If an attribute has a preimage in `K` and is mapped by `r`, it is considered "preserved" and cannot be modified via `exprs`.
- **Schedule Nesting:** Use `tryrule` to merge success/failure wires of a `RuleApp` into a single output before wrapping it in an `agent` loop. `agent` loops require a 1-to-1 input/output interface.

### State Modeling: Combinatorial vs. Attributed Updates
- **The Attribute Update Bottleneck:** Updating an attribute on a preserved entity in a variable ACSet (e.g., changing a status from a variable to a constant) can trigger implementation bugs in `AlgebraicRewriting.jl`. Specifically, the library may struggle with the "Naturality" of morphisms when an `AttrVar` in the rule must "collapse" into a concrete constant during a pushout.
- **The Combinatorial Token Pattern:** To ensure maximum rewrite stability, prefer **combinatorial tokens** over **attributes** for critical state flags (like "Alive", "Healthy", "Destroyed").
    - *Implementation:* Instead of a `status` attribute on a `Platform`, create a `Status` object and a `Hom(Status, Platform)`.
    - *Rewriting:* To "update" state, use DPO to delete the old `Status` part and add a new one. This converts a data-update problem into a structural graph-rewrite problem, which is significantly more robust in categorical engines.
- **When to use Attributes:** Use attributes for continuous or wide-ranging data (lat/lon, fuel levels, probabilities) where combinatorial tokens would lead to state-space explosion.
- **The `expr` pattern for Attributes:** If you must update an attribute on a preserved entity, use the `expr` parameter in the `Rule` constructor.
    - *Key Constraint:* The attribute must be represented by a **variable** in $K$, $L$, and $R$. Do NOT assign a constant in $R$ if the part is in $K$. Instead, keep the variable in $R$ and provide an expression in the `Rule` constructor to override its value (e.g., `expr=(Sym=[vs -> :new_value],)`).

### RewriteGames Engine
- **Parser IR:** The `mk_game_sched` AST parser is sensitive. Use tuple returns (`return a, b`) instead of vector returns (`return [a, b]`) to ensure the wiring diagram is correctly flattened into `BoxStep` IR.
- **Trace Wire & Parser Constraints:**
    - **Avoid Tuple Unpacking:** The Catlab AST parser used by `mk_game_sched` often fails with `MethodError: no method matching Wire(...)` when encountering tuple unpacking for multiple output ports (e.g., `s, f = rule(in)`). Prefer single-port returns or handle success/failure explicitly via `tryrule` to keep the AST simple.
    - **Implicit Port Names:** When a box has multiple input ports, ensure they are explicitly named or correctly indexed in a vector. The parser is prone to `AssertionError: length(arg_ports) == length(inputs)` if the AST connectivity does not perfectly match the box signature.
    - **Empty Trace Arguments:** If a schedule has no trace wires, pass `NamedTuple()` as the first argument to `mk_game_sched` to avoid method ambiguity.
- **Multi-Port Boxes:** When using a `Conditional` or other multi-port box, ensure the ports are merged back into a single output (e.g., via `merge_wires(AgentType)`) before wrapping in an `agent` loop. `agent` loops require a strict 1-to-1 port interface.
- **The `otimes` Error & `ret=:out`:** When constructing an `agent` loop for a schedule using `MADVarACSetCat`, `AlgebraicRewriting` may fail with a `MethodError: no method matching otimes(::Ob, ::Hom)`. This is often resolved by providing an explicit return wire name to the `agent` constructor (e.g., `agent(sub_sched; n=:Platform, ret=:out)`).
- **`tryrule` before `agent`:** Always wrap rule applications (both plain `RuleApp` and `PlayerRuleApp`) in `tryrule()` before passing them to `agent()`. This ensures the success and failure ports are merged, fulfilling the 1-to-1 interface requirement for the agent loop's internal trace operation.
- **Identity Morphisms in Variable Categories:** Avoid manual `Dict`-based `ACSetTransformation` construction for identity maps in variable ACSets, as raw indices may not align with the library's expectations for attribute variables. Instead, use a homomorphism search with pinned object components: `first(homomorphisms(L, L; cat=𝒞, initial=...))`.

