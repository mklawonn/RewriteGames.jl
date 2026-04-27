import ONNXRunTime

"""
    ONNXAgent(path; input_fn)

An agent that loads an ONNX model from `path` via ONNXRunTime.jl and runs
inference in-process.

# Arguments
- `path`:     Path to the `.onnx` file.
- `input_fn`: A function `(state::GameState) -> Dict{String, Array}` that
              converts a `GameState` to the named input arrays the model
              expects.  Use `state.world` for the raw ACSet or
              `elements_graph(state)` for the category-of-elements view.

The model is expected to return logits or probabilities over the legal action
list (in the same order as `legal_actions`).  The action with the highest
score is selected.

# Example
```julia
agent = ONNXAgent("policy.onnx"; input_fn = s -> Dict("x" => float.(nparts(s.world, :V))))
```
"""
struct ONNXAgent <: AbstractAgent
    session  :: Any            # ONNXRunTime session handle
    input_fn :: Function       # GameState -> Dict{String,Array}
end

function ONNXAgent(path::AbstractString; input_fn::Function)
    session = ONNXRunTime.load_inference(path)
    ONNXAgent(session, input_fn)
end

function select_action(agent::ONNXAgent, state::GameState,
                       legal_actions::Vector{Action})
    inputs  = agent.input_fn(state)
    outputs = agent.session(inputs)

    # Assume first output contains logits/probabilities over actions
    logits = first(values(outputs))
    n      = min(length(legal_actions), length(logits))

    best_idx = argmax(logits[1:n])
    return legal_actions[best_idx]
end
