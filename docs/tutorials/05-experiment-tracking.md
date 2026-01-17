# Tutorial 5: Experiment Tracking

Learn how to track experiments for reproducibility, logging, and comparison.

## Overview

The experiment module provides:
- **Configuration management** - Store all hyperparameters
- **Checkpointing** - Save and resume runs
- **Metric tracking** - Log to files, MLflow, or W&B

## Basic Experiment Configuration

```python
from evolve.experiment import ExperimentConfig

config = ExperimentConfig(
    # Identification
    name="rastrigin_v1",
    description="Optimize 10D Rastrigin function",
    seed=42,
    
    # Evolution parameters
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
    
    # Output
    output_dir="./experiments",
    checkpoint_interval=20
)

# Validate configuration
errors = config.validate()
if errors:
    print(f"Configuration errors: {errors}")
```

## Configuration Serialization

```python
# Save to JSON
config.to_json("experiments/rastrigin_v1/config.json")

# Load from JSON
loaded_config = ExperimentConfig.from_json("experiments/rastrigin_v1/config.json")

# Get deterministic hash (for deduplication)
config_hash = config.hash()
print(f"Config hash: {config_hash[:12]}...")
```

## Local File Tracking

```python
from evolve.experiment import LocalTracker

tracker = LocalTracker()
tracker.start_run(config)

# During evolution
for generation in range(config.n_generations):
    # ... evolution step ...
    
    tracker.log_generation(generation, {
        "best_fitness": best.fitness.values[0],
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "diversity": diversity_measure
    })

# Log custom parameters
tracker.log_params({
    "problem": "rastrigin",
    "dimensions": 10,
    "custom_setting": "value"
})

tracker.end_run()
```

**Output files:**
```
experiments/rastrigin_v1/
├── config.json      # Full configuration
├── metrics.csv      # Per-generation metrics
├── params.json      # Additional parameters
├── summary.json     # Run summary
└── artifacts/       # Saved files
```

## Checkpointing

### Automatic Checkpointing

```python
from evolve.experiment import CheckpointManager, Checkpoint

manager = CheckpointManager(
    output_dir="./experiments/rastrigin_v1/checkpoints",
    checkpoint_interval=20,  # Every 20 generations
    keep_last_n=5            # Keep only last 5 checkpoints
)

for generation in range(config.n_generations):
    # ... evolution step ...
    
    if manager.should_checkpoint(generation):
        checkpoint = Checkpoint(
            experiment_name=config.name,
            config_hash=config.hash(),
            generation=generation,
            population=population.individuals,
            best_individual=best,
            rng_state=engine.get_rng_state(),
            fitness_history=engine.history
        )
        manager.save(checkpoint)
```

### Resuming from Checkpoint

```python
# Load latest checkpoint
checkpoint = manager.load_latest()

if checkpoint:
    print(f"Resuming from generation {checkpoint.generation}")
    
    # Restore population
    from evolve.core.population import Population
    population = Population(
        individuals=checkpoint.population,
        generation=checkpoint.generation
    )
    
    # Restore RNG state
    engine.set_rng_state(checkpoint.rng_state)
    
    # Continue from next generation
    start_generation = checkpoint.generation + 1
else:
    start_generation = 0
```

## MLflow Integration

```python
from evolve.experiment.tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="evolve_experiments",
    tracking_uri="http://localhost:5000"  # Optional
)

tracker.start_run(config)

for gen in range(100):
    # ... evolution ...
    tracker.log_generation(gen, {"best_fitness": best_fit})

tracker.end_run()
```

**View results:**
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

## Weights & Biases Integration

```python
from evolve.experiment.tracking import WandbTracker

tracker = WandbTracker(
    project="evolve",
    entity="my-team"  # Optional
)

tracker.start_run(config)

for gen in range(100):
    # ... evolution ...
    tracker.log_generation(gen, {"best_fitness": best_fit})

tracker.end_run()
```

## Multi-Backend Tracking

Log to multiple backends simultaneously:

```python
from evolve.experiment import CompositeTracker, LocalTracker
from evolve.experiment.tracking import MLflowTracker

tracker = CompositeTracker([
    LocalTracker(),
    MLflowTracker(experiment_name="evolve")
])

tracker.start_run(config)
# ... logs to both backends ...
tracker.end_run()
```

## Comparing Experiments

```python
from evolve.experiment import ExperimentComparison
from pathlib import Path

comparison = ExperimentComparison({
    "baseline": Path("experiments/baseline"),
    "larger_pop": Path("experiments/pop_200"),
    "higher_mutation": Path("experiments/mut_0.5"),
})

# Get summary
summary = comparison.summarize()
for exp in summary:
    print(f"{exp['name']}: final_best={exp['final_best']:.4f}")

# Load full metrics
metrics = comparison.load_metrics()
```

## Hyperparameter Sweeps

```python
from evolve.experiment import SweepConfig

sweep = SweepConfig(
    base_config=config,
    parameter_space={
        "population_size": [50, 100, 200],
        "mutation_rate": [0.01, 0.1, 0.5],
    },
    num_seeds=3  # 3 seeds per configuration
)

# Generate all configs
configs = sweep.generate_configs()  # 3 × 3 × 3 = 27 configs

# Run sweep
for cfg in configs:
    runner = ExperimentRunner(config=cfg, ...)
    runner.run()
```

## Complete Example

```python
from evolve.experiment import (
    ExperimentConfig, 
    ExperimentRunner,
    LocalTracker,
    CheckpointManager
)

# Configure
config = ExperimentConfig(
    name="rastrigin_full",
    seed=42,
    population_size=100,
    n_generations=200,
    selection_method="tournament",
    crossover_rate=0.9,
    mutation_rate=0.1,
    output_dir="./experiments",
    checkpoint_interval=20
)

# Setup tracking
tracker = LocalTracker()
manager = CheckpointManager(
    output_dir=f"./experiments/{config.name}/checkpoints",
    checkpoint_interval=20
)

# Run (with automatic resume support)
runner = ExperimentRunner(
    config=config,
    engine=engine,
    initial_population=population,
    tracker=tracker,
    checkpoint_manager=manager
)

result = runner.run(resume=True)  # Resumes if checkpoint exists

print(f"Best fitness: {result.best.fitness.values[0]}")
```

## Next Steps

- **[Tutorial 11](11-checkpointing.md)**: Advanced checkpointing patterns
- **[Tutorial 7](07-island-model.md)**: Checkpointing with island model
