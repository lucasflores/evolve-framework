# Advanced Configuration Guide

This guide covers advanced `UnifiedConfig` features beyond the basics in the README.

## Table of Contents

1. [Custom Evaluators](#custom-evaluators)
2. [Custom Callbacks](#custom-callbacks)
3. [Multi-Objective Configuration](#multi-objective-configuration)
4. [Meta-Evolution Configuration](#meta-evolution-configuration)
5. [MLflow Tracking](#mlflow-tracking)
6. [Config Serialization & Reproducibility](#config-serialization--reproducibility)

---

## Custom Evaluators

Register a custom evaluator and reference it by name in `UnifiedConfig`:

```python
from evolve.config import UnifiedConfig
from evolve.evaluation.evaluator import Evaluator, FitnessResult
from evolve.factory import create_engine, create_initial_population
from evolve.registry.evaluators import get_evaluator_registry


class MyCustomEvaluator(Evaluator):
    """Custom evaluator that scores genomes on a domain-specific metric."""

    def __init__(self, target_sum: float = 10.0):
        self.target_sum = target_sum

    def evaluate(self, genome) -> FitnessResult:
        error = abs(sum(genome.genes) - self.target_sum)
        return FitnessResult(values=(-error,))


# Register with a factory function
def my_evaluator_factory(target_sum: float = 10.0, **kwargs) -> MyCustomEvaluator:
    return MyCustomEvaluator(target_sum=target_sum)


get_evaluator_registry().register("my_custom", my_evaluator_factory)

# Use in config — evaluator resolved by name
config = UnifiedConfig(
    name="custom_evaluator_demo",
    population_size=50,
    max_generations=100,
    selection="tournament",
    crossover="blend",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 5, "bounds": (-10.0, 10.0)},
    evaluator="my_custom",
    evaluator_params={"target_sum": 15.0},
    seed=42,
)

engine = create_engine(config)
population = create_initial_population(config)
result = engine.run(population)
```

### Built-in Evaluators

| Registry Name | Description |
|---------------|-------------|
| `benchmark` | Standard benchmark functions (sphere, rastrigin, etc.) |
| `function` | Wraps any callable as an evaluator |
| `llm_judge` | LLM-based evaluation |
| `ground_truth` | Ground truth comparison |
| `scm` | Structural causal model evaluator |
| `rl` | Reinforcement learning evaluator |

---

## Custom Callbacks

Register a custom callback and reference it in `UnifiedConfig.custom_callbacks`:

```python
from evolve.config import UnifiedConfig
from evolve.core.callbacks import Callback
from evolve.factory import create_engine, create_initial_population
from evolve.registry.callbacks import get_callback_registry


class ProgressPrinter(Callback):
    """Print progress every N generations."""

    def __init__(self, interval: int = 10):
        self.interval = interval

    def on_generation_end(self, generation, population, metrics):
        if generation % self.interval == 0:
            best = min(ind.fitness.values[0] for ind in population.individuals)
            print(f"Gen {generation}: best={best:.6f}")


# Register
def progress_factory(interval: int = 10, **kwargs) -> ProgressPrinter:
    return ProgressPrinter(interval=interval)


get_callback_registry().register("progress", progress_factory)

# Use in config
config = UnifiedConfig(
    name="callback_demo",
    population_size=50,
    max_generations=100,
    selection="tournament",
    crossover="blend",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 5, "bounds": (-5.0, 5.0)},
    custom_callbacks=[
        {"name": "progress", "params": {"interval": 10}},
    ],
    seed=42,
)

engine = create_engine(config, evaluator=my_fitness_function)
population = create_initial_population(config)
result = engine.run(population)
```

### Built-in Callbacks

| Registry Name | Description |
|---------------|-------------|
| `logging` | Structured logging callback |
| `checkpoint` | Periodic state checkpointing |
| `print` | Console output callback |
| `history` | In-memory history tracking |

---

## Multi-Objective Configuration

Enable NSGA-II multi-objective optimization with `with_multiobjective()`:

```python
from evolve.config import UnifiedConfig
from evolve.config.multiobjective import ObjectiveSpec
from evolve.factory import create_engine, create_initial_population

config = UnifiedConfig(
    name="multiobjective_demo",
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

The `with_multiobjective()` method:
- Configures NSGA-II with Pareto ranking and crowding distance
- Automatically enables `crowded_tournament` selection
- Sets up hypervolume tracking if a reference point is provided

---

## Meta-Evolution Configuration

Meta-evolution optimizes the hyperparameters of your evolutionary algorithm:

```python
from evolve.config import UnifiedConfig
from evolve.config.meta import MetaEvolutionConfig, ParameterSpec

config = UnifiedConfig(
    name="meta_evolution_demo",
    population_size=50,
    max_generations=100,
    selection="tournament",
    crossover="blend",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
    seed=42,
    meta_evolution=MetaEvolutionConfig(
        meta_population_size=10,
        meta_generations=20,
        parameter_specs=[
            ParameterSpec(name="mutation_rate", min_value=0.01, max_value=0.5),
            ParameterSpec(name="crossover_rate", min_value=0.5, max_value=1.0),
            ParameterSpec(name="selection_params.tournament_size", min_value=2, max_value=7),
        ],
    ),
)
```

Meta-evolution creates a population of *configs* and evolves the best hyperparameter combination.

---

## MLflow Tracking

Enable experiment tracking with `TrackingConfig`:

```python
from evolve.config import UnifiedConfig
from evolve.config.tracking import MetricCategory, TrackingConfig

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
        categories=(
            MetricCategory.FITNESS,
            MetricCategory.DIVERSITY,
            MetricCategory.OPERATORS,
        ),
    ),
    seed=42,
)
```

### Tracking Categories

| Category | Metrics Logged |
|----------|----------------|
| `FITNESS` | best/mean/worst fitness per generation |
| `DIVERSITY` | genotypic diversity, species count |
| `OPERATORS` | operator success rates, parameter values |
| `ERP` | protocol distributions, mating success, recovery events |
| `MULTIOBJECTIVE` | hypervolume, Pareto front size, spread |

---

## Config Serialization & Reproducibility

### Save/Load Configs

```python
# Save to JSON
config.to_file("experiment_config.json")

# Load from JSON
loaded_config = UnifiedConfig.from_file("experiment_config.json")

# Dict round-trip
config_dict = config.to_dict()
restored = UnifiedConfig.from_dict(config_dict)
```

### Config Hashing

Every config has a deterministic hash for reproducibility:

```python
hash_value = config.compute_hash()
print(f"Config hash: {hash_value}")

# Same config → same hash (regardless of creation order)
config2 = UnifiedConfig.from_dict(config.to_dict())
assert config.compute_hash() == config2.compute_hash()
```

Use config hashes to:
- Tag experiment runs for reproducibility
- Detect duplicate configurations
- Version experiment results
