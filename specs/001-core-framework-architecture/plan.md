# Implementation Plan: Evolve Framework Core Architecture

**Branch**: `001-core-framework-architecture` | **Date**: 2026-01-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-core-framework-architecture/spec.md`

## Summary

Design and implement a research-grade evolutionary algorithms experimentation framework with five independently extensible layers: Evolution Core (backend-agnostic population dynamics), Representation (genome/phenotype abstractions), Evaluation (fitness computation with optional acceleration), Execution Backends (parallel/GPU/JIT), and Observability (experiment tracking and reproducibility). The framework supports classical GA, multi-objective optimization (NSGA-II), neuroevolution, reinforcement learning, island models, and speciation—all while enforcing strict separation of concerns and deterministic reproducibility.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: NumPy (core); Optional: PyTorch, JAX, MLflow, Ray  
**Storage**: File-based checkpoints (pickle/JSON), optional MLflow artifact store  
**Testing**: pytest with hypothesis for property-based tests  
**Target Platform**: Linux/macOS (primary), Windows (secondary)  
**Project Type**: Single library with optional extension packages  
**Performance Goals**: CPU reference implementations prioritize correctness; accelerated evaluators target 10x+ speedup  
**Constraints**: Zero ML framework imports in core modules; all randomness seeded; relative tolerance ≤1e-5 for CPU/GPU equivalence  
**Scale/Scope**: Research-grade library supporting populations up to 10k individuals, experiments up to 10k generations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: No hard dependencies on PyTorch, JAX, TensorFlow, or RL libraries in core modules
  - `evolve/core/`, `evolve/representation/`, `evolve/evaluation/`, `evolve/multiobjective/`, `evolve/diversity/` have zero ML imports
  - ML-dependent code isolated in `evolve/backends/accelerated/` and `evolve/memetic/`
- [x] **Separation of Concerns**: Evolutionary logic, representation, evaluation, and execution are decoupled
  - 6 distinct layers with explicit interfaces: Core → Representation → Evaluation → Backends
  - Operators don't access evaluators directly; evaluation is injected
- [x] **Optional Acceleration**: GPU/JIT features are opt-in with CPU reference implementations
  - `evolve/backends/sequential.py` is the default
  - `evolve/backends/accelerated/` requires explicit import and configuration
- [x] **Determinism**: All stochastic processes use explicit seeds; experiments are reproducible
  - `evolve/utils/random.py` provides seeded RNG utilities
  - All operators and evaluators accept seed parameters
  - Checkpoint includes RNG state (FR-049)
- [x] **Extensibility**: Clear extension points defined; optimization justified with profiling data
  - Abstract interfaces for all major components (Genome, Evaluator, Operator, Backend)
  - Plugin architecture for new operators without modifying core
- [x] **Multi-Domain Support**: Design accommodates classical EA, neuroevolution, multi-objective, causal discovery, RL
  - `evolve/multiobjective/` for Pareto-based methods
  - `evolve/rl/` for reinforcement learning
  - `evolve/representation/graph.py` for NEAT-style neuroevolution
  - Causal discovery supported via custom genomes/evaluators
- [x] **Observability**: Structured logging, metrics, and experiment tracking are integrated
  - `evolve/experiment/` provides config, logging, metrics, checkpoint, tracking
  - MLflow and W&B integrations in `evolve/experiment/tracking/`
- [x] **Clear Abstractions**: Type annotations, explicit interfaces, documented contracts
  - Python 3.10+ type hints throughout
  - Protocol classes define interfaces
- [x] **Composability**: No global state; components are independently testable
  - All dependencies injected via constructor
  - No module-level mutable state
- [x] **Test-First**: Tests written before implementation; reference implementations provided
  - `tests/` structure mirrors `evolve/` for comprehensive coverage
  - Property-based tests for invariants

**Violations requiring justification**: None

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
evolve/                          # Main package
├── __init__.py
├── core/                        # Layer A: Evolution Core (NO ML IMPORTS)
│   ├── __init__.py
│   ├── individual.py            # Individual, Fitness
│   ├── population.py            # Population container
│   ├── operators/               # Selection, Crossover, Mutation
│   │   ├── __init__.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   └── mutation.py
│   ├── engine.py                # EvolutionEngine (main loop)
│   ├── callbacks.py             # Event hooks
│   └── stopping.py              # Early stopping criteria
├── representation/              # Layer B: Representation
│   ├── __init__.py
│   ├── genome.py                # Abstract Genome interface
│   ├── phenotype.py             # Phenotype, Decoder interfaces
│   ├── vector.py                # Fixed-length vector genome
│   ├── sequence.py              # Variable-length sequence genome
│   └── graph.py                 # Graph-based genome (NEAT-style)
├── evaluation/                  # Layer C: Evaluation (acceleration boundary)
│   ├── __init__.py
│   ├── evaluator.py             # Abstract Evaluator interface
│   ├── fitness.py               # Fitness vector, constraints
│   └── reference/               # CPU reference implementations
│       ├── __init__.py
│       └── functions.py         # Benchmark functions (sphere, rastrigin)
├── multiobjective/              # Multi-objective optimization
│   ├── __init__.py
│   ├── dominance.py             # Pareto dominance checking
│   ├── ranking.py               # Non-dominated sorting, fronts
│   ├── crowding.py              # Crowding distance
│   └── selection.py             # NSGA-II selection
├── diversity/                   # Layer E: Diversity preservation
│   ├── __init__.py
│   ├── speciation.py            # Species, distance metrics
│   ├── niching.py               # Fitness sharing
│   ├── novelty.py               # Novelty search, archives
│   └── islands/                 # Island models
│       ├── __init__.py
│       ├── island.py            # Island abstraction
│       ├── migration.py         # Migration policies
│       └── topology.py          # Ring, fully-connected, custom
├── rl/                          # RL integration
│   ├── __init__.py
│   ├── environment.py           # Environment interface
│   ├── policy.py                # Policy (extends Phenotype)
│   ├── rollout.py               # Trajectory, episode handling
│   └── evaluator.py             # Episode-based fitness
├── backends/                    # Layer D: Execution backends (OPTIONAL)
│   ├── __init__.py
│   ├── base.py                  # ExecutionBackend interface
│   ├── sequential.py            # Default CPU sequential
│   ├── parallel.py              # Multiprocessing backend
│   └── accelerated/             # GPU/JIT backends
│       ├── __init__.py
│       ├── torch_evaluator.py   # PyTorch GPU evaluator
│       └── jax_evaluator.py     # JAX JIT evaluator
├── memetic/                     # Memetic extensions (OPTIONAL)
│   ├── __init__.py
│   ├── local_search.py          # LocalSearchOperator interface
│   └── gradient.py              # Gradient-based refinement
├── experiment/                  # Layer F: Observability
│   ├── __init__.py
│   ├── config.py                # Experiment configuration
│   ├── logging.py               # Structured logging
│   ├── metrics.py               # Metric collection
│   ├── checkpoint.py            # Checkpointing, resumption
│   └── tracking/                # Experiment tracking integrations
│       ├── __init__.py
│       ├── mlflow.py            # MLflow integration
│       └── wandb.py             # Weights & Biases integration
└── utils/                       # Utilities
    ├── __init__.py
    ├── random.py                # Seeded RNG utilities
    └── validation.py            # Input validation

tests/
├── unit/                        # Unit tests per module
│   ├── core/
│   ├── representation/
│   ├── evaluation/
│   ├── multiobjective/
│   ├── diversity/
│   └── rl/
├── integration/                 # Cross-layer integration tests
│   ├── test_simple_ga.py
│   ├── test_nsga2.py
│   ├── test_islands.py
│   └── test_checkpointing.py
├── property/                    # Property-based tests (hypothesis)
│   ├── test_determinism.py
│   └── test_pareto.py
└── benchmarks/                  # Performance benchmarks
    ├── test_cpu_gpu_equivalence.py
    └── test_scaling.py
```

**Structure Decision**: Single library package (`evolve/`) with layered architecture. Core modules (`core/`, `representation/`, `evaluation/`, `multiobjective/`, `diversity/`) have zero ML framework imports. Optional acceleration (`backends/accelerated/`) and memetic extensions (`memetic/`) are isolated packages that users opt into.

## Complexity Tracking

> **No violations identified** - Design follows all constitution principles

## Post-Design Constitution Re-Check ✅

**Date**: 2026-01-13 | **Phase 1 Artifacts Reviewed**: data-model.md, contracts/*, quickstart.md

| Principle | Status | Evidence |
|-----------|--------|----------|
| Model-Agnostic | ✅ PASS | All contracts use Protocol classes with NumPy/stdlib types only; Genome/Phenotype/Evaluator definitions have zero ML imports |
| Separation of Concerns | ✅ PASS | 8 distinct contract files define clear layer boundaries; Decoder separates genome→phenotype transformation |
| Optional Acceleration | ✅ PASS | `EvaluatorCapabilities` flags optional features; reference implementations in contracts use pure NumPy |
| Determinism | ✅ PASS | Every operator/evaluator signature includes `seed: int \| None`; RNG state in Checkpoint; seed derivation for parallel work |
| Extensibility | ✅ PASS | Protocol-based interfaces throughout; no concrete types in signatures except dataclasses |
| Multi-Domain | ✅ PASS | GraphGenome for NEAT; MultiObjectiveFitness for NSGA-II; Policy extends Phenotype for RL; QDArchive for MAP-Elites |
| Observability | ✅ PASS | ExperimentConfig, Checkpoint, MetricTracker protocols; MLflow/W&B adapters in contracts |
| Clear Abstractions | ✅ PASS | Type annotations on all protocol methods; docstrings explain invariants and edge cases |
| Composability | ✅ PASS | All components accept dependencies via constructor; no module-level state in contracts |
| Test-First | ✅ READY | `assert_evaluator_equivalence` and benchmark functions provide test infrastructure |

**Conclusion**: Phase 1 design artifacts comply with all constitution principles. Ready for Phase 2 task generation.
