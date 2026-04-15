# Evolve Framework

A research-grade evolutionary algorithms experimentation framework.

## Features

- **Declarative Experiment Configuration**: Define entire experiments via `UnifiedConfig` — operators, genomes, evaluators, callbacks, and advanced modes
- **Model-Agnostic Architecture**: No hard dependencies on ML frameworks in core modules
- **Deterministic Reproducibility**: All randomness uses explicit seeding
- **Optional GPU Acceleration**: PyTorch/JAX backends available as optional extras
- **Multi-Objective Optimization**: NSGA-II with Pareto ranking and crowding distance
- **Neuroevolution Support**: NEAT-style topology evolution
- **Evolvable Reproduction Protocols (ERP)**: Individual-level mating strategies with sexual selection
- **Experiment Tracking**: MLflow and Weights & Biases integrations
- **Registry System**: Extensible registries for operators, evaluators, callbacks, and genomes

## Installation

```bash
# Basic installation (NumPy only)
pip install evolve-framework

# With PyTorch GPU support
pip install evolve-framework[pytorch]

# With all optional dependencies
pip install evolve-framework[all]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

Every experiment starts with a `UnifiedConfig`, `create_engine()`, and `create_initial_population()`:

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine, create_initial_population
from evolve.evaluation.reference.functions import sphere

# Define your entire experiment declaratively
config = UnifiedConfig(
    name="sphere_optimization",
    population_size=50,
    max_generations=100,
    elitism=2,
    selection="tournament",
    selection_params={"tournament_size": 3},
    crossover="blend",
    crossover_rate=0.8,
    crossover_params={"alpha": 0.5},
    mutation="gaussian",
    mutation_rate=0.1,
    mutation_params={"sigma": 0.1},
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
    seed=42,
)

# Create engine and initial population from config
engine = create_engine(config, evaluator=sphere)
population = create_initial_population(config)

# Run evolution
result = engine.run(population)

print(f"Best fitness: {result.best.fitness.values[0]:.6f}")
print(f"Generations: {result.generations}")
```

### Adding Tracking

Enable MLflow tracking by adding a `TrackingConfig`:

```python
from evolve.config import UnifiedConfig
from evolve.config.tracking import TrackingConfig
from evolve.factory import create_engine, create_initial_population

config = UnifiedConfig(
    name="tracked_experiment",
    population_size=100,
    max_generations=50,
    selection="tournament",
    crossover="sbx",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
    tracking=TrackingConfig(
        enabled=True,
        backend="mlflow",
        experiment_name="my_experiments",
    ),
    seed=42,
)

engine = create_engine(config, evaluator=my_fitness_function)
population = create_initial_population(config)
result = engine.run(population)
```

### Multi-Objective Optimization

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine, create_initial_population

from evolve.config import ObjectiveSpec

config = UnifiedConfig(
    name="nsga2_example",
    population_size=100,
    max_generations=200,
    selection="crowded_tournament",
    crossover="sbx",
    mutation="polynomial",
    genome_type="vector",
    genome_params={"dimensions": 5, "bounds": (0.0, 1.0)},
    seed=42,
).with_multiobjective(
    objectives=(
        ObjectiveSpec(name="accuracy", direction="maximize"),
        ObjectiveSpec(name="complexity", direction="minimize"),
    ),
    reference_point=(0.0, 100.0),
)

engine = create_engine(config, evaluator=multi_objective_fn)
population = create_initial_population(config)
result = engine.run(population)
```

### Evolvable Reproduction Protocols (ERP)

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine, create_initial_population

config = UnifiedConfig(
    name="erp_experiment",
    population_size=100,
    max_generations=100,
    selection="tournament",
    crossover="blend",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
    seed=42,
).with_erp(
    protocol_mutation_rate=0.1,
    selectivity_pressure=0.3,
)

engine = create_engine(config, evaluator=my_fitness_function)
population = create_initial_population(config)
result = engine.run(population)
```

## Representations

The framework supports multiple genome types, configured via `genome_type` and `genome_params` in `UnifiedConfig`:

| Genome Type | `genome_type` | Description | Key `genome_params` |
|-------------|---------------|-------------|---------------------|
| **VectorGenome** | `"vector"` | Fixed-length real-valued arrays | `dimensions`, `bounds` |
| **SequenceGenome** | `"sequence"` | Variable-length symbolic sequences | `length`, `alphabet` |
| **GraphGenome** | `"graph"` | NEAT-style topology-evolving networks | `input_nodes`, `output_nodes` |
| **SCMGenome** | `"scm"` | Structural causal models | `num_variables` |
| **EmbeddingGenome** | `"embedding"` | LLM soft-prompt token embeddings | `n_tokens`, `embed_dim`, `model_id` |

## Built-in Registry

All operators, evaluators, callbacks, and genomes are registered by name and resolved automatically by `create_engine()` and `create_initial_population()`:

### Operators

| Category | Registry Names |
|----------|---------------|
| **Selection** | `tournament`, `roulette`, `rank`, `crowded_tournament` |
| **Crossover** | `single_point`, `two_point`, `uniform`, `sbx`, `blend` |
| **Mutation** | `gaussian`, `uniform`, `polynomial`, `creep` |

### Evaluators

| Registry Name | Description |
|---------------|-------------|
| `benchmark` | Standard benchmark functions (sphere, rastrigin, etc.) |
| `function` | Wraps any callable as an evaluator |
| `llm_judge` | LLM-based evaluation (requires optional deps) |
| `ground_truth` | Ground truth comparison evaluator |
| `scm` | Structural causal model evaluator |
| `rl` | Reinforcement learning evaluator (requires gymnasium) |

### Callbacks

| Registry Name | Description |
|---------------|-------------|
| `logging` | Structured logging callback |
| `checkpoint` | Periodic state checkpointing |
| `print` | Console output callback |
| `history` | In-memory history tracking |

### Custom Extensions

Register your own implementations:

```python
from evolve.registry import EvaluatorRegistry, CallbackRegistry

# Register a custom evaluator
EvaluatorRegistry.register("my_evaluator", my_evaluator_factory)

# Use in config
config = UnifiedConfig(evaluator="my_evaluator", evaluator_params={"param": "value"}, ...)
```

## Architecture

The framework is organized into independent layers:

1. **Config** (`evolve.config`): Declarative experiment specification via `UnifiedConfig`
2. **Factory** (`evolve.factory`): `create_engine()` resolves config into runnable engines
3. **Registry** (`evolve.registry`): Extensible registries for operators, evaluators, callbacks, genomes
4. **Evolution Core** (`evolve.core`): Backend-agnostic population dynamics
5. **Representation** (`evolve.representation`): Genome/phenotype abstractions
6. **Evaluation** (`evolve.evaluation`): Fitness computation interfaces
7. **Backends** (`evolve.backends`): Parallel/GPU/JIT execution
8. **Reproduction** (`evolve.reproduction`): Evolvable reproduction protocols (ERP) for sexual selection
9. **Observability** (`evolve.experiment`): Tracking and reproducibility

## Tutorials

See [docs/tutorials/](docs/tutorials/) for interactive Jupyter notebooks:

| Tutorial | Topic |
|----------|-------|
| [01 — UnifiedConfig](docs/tutorials/01_unified_config.ipynb) | **Start here.** Declarative config, factory, running experiments |
| [02 — Vector Genome](docs/tutorials/02_vector_genome.ipynb) | Continuous optimization with real-valued genomes |
| [03 — Sequence Genome](docs/tutorials/03_sequence_genome.ipynb) | Symbolic regression and genetic programming |
| [04 — Graph Genome (NEAT)](docs/tutorials/04_graph_genome_neat.ipynb) | Neural architecture search with topology evolution |
| [05 — RL Neuroevolution](docs/tutorials/05_rl_neuroevolution.ipynb) | Evolving RL policies (requires gymnasium) |
| [06 — SCM Multi-Objective](docs/tutorials/06_scm_multiobjective.ipynb) | Causal discovery with NSGA-II |
| [07 — Evolvable Reproduction](docs/tutorials/07_evolvable_reproduction.ipynb) | Sexual selection and speciation via ERP |

## Testing

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/ -m integration

# Run property-based tests
pytest tests/property/ -m property
```

## License

MIT License
