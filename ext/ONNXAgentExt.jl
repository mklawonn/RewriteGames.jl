module ONNXAgentExt

import ONNXRunTime
using RewriteGames: AbstractAgent, EncodedState, Action, select_action

"""
    ONNXAgent(path; input_fn)

An agent that loads an ONNX model from `path` via ONNXRunTime.jl and runs
inference in-process.

# Arguments
- `path`:     Path to the `.onnx` file.
- `input_fn`: A function `(state::EncodedState) -> Dict{String, Array}` that
              converts an `EncodedState` to the named input arrays the model
              expects.

The model is expected to return logits or probabilities over the legal action
list (in the same order as `legal_actions`).  The action with the highest
score is selected.

# Example
```julia
using RewriteGames, ONNXRunTime
agent = ONNXAgent("policy.onnx"; input_fn = s -> Dict("x" => s.node_features))
```
"""
struct ONNXAgent <: AbstractAgent
    session  :: Any            # ONNXRunTime session handle
    input_fn :: Function       # EncodedState -> Dict{String,Array}
end

function ONNXAgent(path::AbstractString; input_fn::Function)
    session = ONNXRunTime.load_inference(path)
    ONNXAgent(session, input_fn)
end

function select_action(agent::ONNXAgent, state::EncodedState,
                       legal_actions::Vector{Action})
    inputs  = agent.input_fn(state)
    outputs = agent.session(inputs)

    logits = first(values(outputs))
    n      = min(length(legal_actions), length(logits))

    best_idx = argmax(logits[1:n])
    return legal_actions[best_idx]
end

end # module ONNXAgentExt
