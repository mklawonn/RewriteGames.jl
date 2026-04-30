# RewriteGames.jl

RewriteGames.jl is a Julia package leveraging Applied Category Theory and Double-Pushout (DPO) Rewriting—built upon `AlgebraicRewriting.jl` and `Catlab.jl`—for building, analyzing, and simulating complex, game-like systems. It enables representing complex game or world states as Attributed C-Sets (ACSets) and strictly models valid state transitions using algebraic rewrite rules.

## Motivation

The primary goal of RewriteGames.jl is to provide a declarative, categorically-founded framework for defining rules engines and game mechanics. Traditional game logic often relies on ad-hoc code that becomes difficult to maintain, verify, or analyze as complexity grows. By shifting to algebraic rewriting, we gain:
- **Composability**: Game rules and state structures can be cleanly composed.
- **Formal Verification**: Transitions can be analyzed algebraically.
- **AI Readiness**: Seamlessly integrate agents, from simple heuristics to neural networks (via `Flux`, `GraphNeuralNetworks`, or imported ONNX models), enabling robust reinforcement learning and simulations.

## Repository Organization

- `src/`: The core source code for the package.
  - `agents/`: Implementations of agents interacting with games, ranging from function-based agents to an ONNX-backed agent interface.
  - `core/`: Core data structures and abstractions for game states and rewrite rules.
  - `engine/`: The main execution engine for running matches, simulations, and driver logic.
  - `schedule/`: Logic for organizing player turns, context management, and scheduled application of rewrite rules.
  - `encoding/`: Utilities for representing and transforming game states.
  - `migration/`: Tooling for structural migration of game definitions.
  - `serialization/`: Data serialization functionality, such as Apache Arrow support.
  - `analysis.jl`: Tooling for game analysis.
  - `dsl.jl`: Domain-Specific Language (DSL) components for convenient game definition.
- `ext/`: Julia extensions, such as `ONNXAgentExt.jl` for optional ONNX runtime integration.
- `test/`: A comprehensive test suite covering the core modules.
- `tutorials/`: Examples and interactive materials, including an in-depth Tic-Tac-Toe implementation showcasing the engine's capabilities.
- `examples/`: Standalone scripts demonstrating specific features, such as migration.

## Roadmap

As RewriteGames.jl evolves, we are focused on the following key areas of improvement and expansion:

- **Cleanup Source**: We have been iterating quickly on the source and there are many artifacts of previous ideas still herein. If a function looks out of place it probably is.
- **Optimize Homomorphism Searches**: Investigate and improve the efficiency of homomorphism searches during rewrite rule matching. A key objective is to ensure that these searches operate effectively on the differential (the "diff") of the state rather than recalculating from scratch.
- **Enhance Multi-Move Scenarios**: Introduce better handling for multiple moves per turn. 
- **Documentation and Prose Improvements**: Continuously refine tutorial prose, explanations, and API documentation for clarity and accessibility. We want to show more functionality, e.g. how to add budgets to moves by making them explicit constraints within the algebraic rewrite rules themselves (e.g., ensuring a rule only matches if the appropriate "budget resource" exists in the world ACSet). We want to model more complex, narrative driven games like D&D-style tabletop RPGs.
- **Categorical Treatment of Games**: Right now the rewrite rules, world state, schedules, etc get a categorical treatment, but the games themselves have not. It should be possible to express some notion of morphism between games, and probably a category of them. This machinery could be useful to relate agents trained on related games.
- **Performance Optimization for RL**: We could potentially add support for distribution of game simulation so that a single training pipeline can be receiving data from multiple game instances playing out at once. We could also look into GPU accelerated homomorphism search so that game episodes play out on the same device that players are using to train.
