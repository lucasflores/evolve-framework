# Experiment Module API

The `evolve.experiment` module provides experiment management, checkpointing, and metric tracking.

## Configuration

### `ExperimentConfig`

Complete configuration for reproducible experiments.

```python
from evolve.experiment import ExperimentConfig

config = ExperimentConfig(
    # Identification
    name="my_experiment",
    description="Optimize Rastrigin function",
    seed=42,
    
    # Population
    population_size=100,
    n_generations=200,
    
    # Operators
    selection_method="tournament",
    selection_params={"tournament_size": 3},
    crossover_method="uniform",
    crossover_rate=0.9,
    mutation_method="gaussian",
    mutation_rate=0.1,
    mutation_params={"sigma": 0.1},
    
    # Representation
    genome_type="vector",
    genome_params={"n_dims": 10},
    
    # Evaluation
    evaluator_type="function",
    evaluator_params={"function": "rastrigin"},
    
    # Multi-objective (optional)
    multi_objective=False,
    n_objectives=1,
    
    # Island model (optional)
    use_islands=False,
    n_islands=4,
    migration_rate=0.1,
    
    # Stopping criteria
    target_fitness=None,
    max_evaluations=None,
    stagnation_limit=50,
    
    # Output
    output_dir="./experiments",
    checkpoint_interval=10
)

# Validation
errors = config.validate()

# Serialization
config.to_json("config.json")
config = ExperimentConfig.from_json("config.json")

# Deterministic hash for deduplication
hash_value = config.hash()
```

---

## Checkpointing

### `Checkpoint`

Complete state for resuming experiments.

```python
from evolve.experiment import Checkpoint

checkpoint = Checkpoint(
    experiment_name="my_exp",
    config_hash="abc123...",
    generation=50,
    population=population.individuals,
    best_individual=best,
    rng_state=engine.get_rng_state(),
    fitness_history=engine.history
)

# Save/load
checkpoint.save("checkpoint.pkl")
checkpoint = Checkpoint.load("checkpoint.pkl")
```

### `CheckpointManager`

Automatic checkpoint management.

```python
from evolve.experiment import CheckpointManager

manager = CheckpointManager(
    output_dir="./checkpoints",
    checkpoint_interval=10,  # Every 10 generations
    keep_last_n=5            # Prune old checkpoints
)

# Check if should save
if manager.should_checkpoint(generation):
    checkpoint = Checkpoint(...)
    manager.save(checkpoint)

# Load for resume
latest = manager.load_latest()
specific = manager.load_generation(50)
```

---

## Metric Tracking

### `MetricTracker` Protocol

```python
from evolve.experiment import MetricTracker

class MyTracker:
    def start_run(self, config: ExperimentConfig) -> None:
        """Initialize tracking for new run."""
        ...
    
    def log_generation(self, generation: int, metrics: dict[str, float]) -> None:
        """Log per-generation metrics."""
        ...
    
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...
    
    def log_artifact(self, path: Path, name: str | None = None) -> None:
        """Save file as artifact."""
        ...
    
    def end_run(self) -> None:
        """Finalize tracking."""
        ...
```

### `LocalTracker`

File-based tracking (CSV + JSON).

```python
from evolve.experiment import LocalTracker

tracker = LocalTracker()
tracker.start_run(config)

for gen in range(100):
    # ... evolution ...
    tracker.log_generation(gen, {
        "best_fitness": best.fitness.values[0],
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "diversity": diversity_measure
    })

tracker.log_params({"custom_param": "value"})
tracker.end_run()

# Creates:
# output_dir/experiment_name/
#   config.json
#   metrics.csv
#   params.json
#   summary.json
#   artifacts/
```

### `CompositeTracker`

Log to multiple backends simultaneously.

```python
from evolve.experiment import CompositeTracker, LocalTracker
from evolve.experiment.tracking import MLflowTracker

tracker = CompositeTracker([
    LocalTracker(),
    MLflowTracker(experiment_name="evolve")
])
```

### `NullTracker`

No-op tracker for testing or disabled logging.

```python
from evolve.experiment import NullTracker

tracker = NullTracker()  # Silent
```

---

## MLflow Integration

```python
from evolve.experiment.tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="my_experiments",
    tracking_uri="http://localhost:5000"  # Optional
)

tracker.start_run(config)
# ... logs to MLflow ...
tracker.end_run()
```

Requires: `pip install mlflow`

---

## Weights & Biases Integration

```python
from evolve.experiment.tracking import WandbTracker

tracker = WandbTracker(
    project="evolve",
    entity="my-team"  # Optional
)

tracker.start_run(config)
# ... logs to W&B ...
tracker.end_run()
```

Requires: `pip install wandb`

---

## Experiment Runner

### `ExperimentRunner`

Orchestrates complete experiment execution.

```python
from evolve.experiment import ExperimentRunner

runner = ExperimentRunner(
    config=config,
    engine=engine,
    initial_population=population,
    tracker=LocalTracker(),
    checkpoint_manager=manager
)

# Run (optionally resume)
result = runner.run(resume=False)

# Resume from checkpoint
result = runner.run(resume=True)
```

---

## Experiment Comparison

```python
from evolve.experiment import ExperimentComparison
from pathlib import Path

comparison = ExperimentComparison({
    "baseline": Path("experiments/baseline"),
    "optimized": Path("experiments/optimized"),
    "with_islands": Path("experiments/islands"),
})

# Load all metrics
metrics = comparison.load_metrics()

# Summary table
summary = comparison.summarize()
# Returns: [{"name": "baseline", "final_best": 0.1, ...}, ...]
```

---

## Hyperparameter Sweeps

```python
from evolve.experiment import SweepConfig, ExperimentConfig

base_config = ExperimentConfig(
    name="sweep",
    seed=42,
    population_size=100,
    ...
)

sweep = SweepConfig(
    base_config=base_config,
    parameter_space={
        "population_size": [50, 100, 200],
        "mutation_rate": [0.01, 0.1, 0.5],
        "selection_params.tournament_size": [2, 3, 5],
    },
    num_seeds=3  # Repeat each config with different seeds
)

# Generate all configurations
configs = sweep.generate_configs()
# 3 pop_sizes × 3 mutation_rates × 3 tournament_sizes × 3 seeds = 81 configs

# Run all experiments
for config in configs:
    runner = ExperimentRunner(config=config, ...)
    runner.run()
```
