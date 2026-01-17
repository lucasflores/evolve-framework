# Evolve Framework API Reference

This directory contains module-level documentation for the Evolve Framework.

## Core Modules

### [`core`](core.md)
Core types, population management, and the evolution engine.
- `Individual`, `Fitness`, `Population`
- `EvolutionEngine`, `EvolutionConfig`, `EvolutionResult`
- Callbacks, stopping criteria

### [`representation`](representation.md)
Genome representations for different problem types.
- `VectorGenome` - Real-valued optimization
- `TreeGenome` - Genetic programming
- `GraphGenome` - Network evolution (NEAT)
- `PermutationGenome` - Combinatorial problems

### [`evaluation`](evaluation.md)
Fitness evaluation interfaces and implementations.
- `Evaluator` protocol
- `FunctionEvaluator` - Simple function wrapping
- Reference benchmark functions (Sphere, Rastrigin, etc.)

### [`core.operators`](operators.md)
Genetic operators for selection, crossover, and mutation.
- Selection: Tournament, Roulette, Rank
- Crossover: Single-point, Two-point, Uniform, SBX
- Mutation: Gaussian, Polynomial, Adaptive

## Advanced Modules

### [`multiobjective`](multiobjective.md)
Multi-objective optimization support.
- NSGA-II selection
- Pareto dominance and fronts
- Crowding distance

### [`diversity`](diversity.md)
Population diversity mechanisms.
- Island model parallelism
- Speciation (NEAT-style)
- Novelty search
- Fitness sharing

### [`rl`](rl.md)
Reinforcement learning integration.
- `GymAdapter` - OpenAI Gym/Gymnasium compatibility
- `RLEvaluator` - Episode-based fitness evaluation
- Policy decoders

### [`experiment`](experiment.md)
Experiment management and reproducibility.
- `ExperimentConfig` - Configuration management
- `Checkpoint`, `CheckpointManager` - State persistence
- `MetricTracker`, `LocalTracker` - Logging

### [`backends`](backends.md)
Computational backends for acceleration.
- Sequential evaluation (default)
- Parallel evaluation (multiprocessing)
- GPU evaluation (PyTorch, JAX)

## Utility Modules

### [`utils`](utils.md)
Shared utilities.
- Random number generation with seeding
- Logging configuration
