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

- **Enhance Action Budgets and Multi-Move Scenarios**: Introduce better handling for multiple moves per turn. We plan to refine move budgets, potentially making them explicit constraints within the algebraic rewrite rules themselves (e.g., ensuring a rule only matches if the appropriate "budget resource" exists in the world ACSet).
- **Optimize Homomorphism Searches**: Investigate and improve the efficiency of homomorphism searches during rewrite rule matching. A key objective is to ensure that these searches operate effectively on the differential (the "diff") of the state rather than recalculating from scratch.
- **Refine Categorical Topology of Examples**: Improve the foundational graph topology in examples like Tic-Tac-Toe. The goal is to dramatically reduce the number of rules required for checking complex conditions (e.g., capturing all geometric win conditions with just a couple of generalized rules).
- **Integrate Reinforcement Learning Workflows**: Expand and reintroduce reinforcement learning integrations within the tutorials to demonstrate the framework's readiness for Game AI training.
- **Expand Game Variety**: Adapt more complex, narrative-driven scenarios into the framework. We are exploring implementations of asymmetric scenarios like Lord of the Rings, as well as complex, stat-driven D&D-style tabletop RPG mechanics.
- **Documentation and Prose Improvements**: Continuously refine tutorial prose, explanations, and API documentation for clarity and accessibility.
