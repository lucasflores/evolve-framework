# Quickstart: MLflow Metrics Tracking

**Feature**: 006-mlflow-metrics-tracking  
**Date**: March 17, 2026

## Overview

This guide shows how to enable comprehensive MLflow metrics tracking in the evolve framework, from basic fitness tracking to advanced analytics.

---

## 1. Basic Setup: Declarative Tracking with UnifiedConfig

The simplest way to enable tracking is to add a `TrackingConfig` to your `UnifiedConfig`:

```python
from evolve.config.unified import UnifiedConfig
from evolve.config.tracking import TrackingConfig
from evolve.factory import create_engine
from evolve.evaluation.reference.functions import sphere

# Define configuration with tracking enabled
config = UnifiedConfig(
    name="sphere_optimization",
    population_size=100,
    max_generations=50,
    selection="tournament",
    crossover="sbx",
    mutation="gaussian",
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    
    # Enable MLflow tracking with standard metrics
    tracking=TrackingConfig(
        backend="mlflow",
        experiment_name="evolve_experiments",
        run_name="sphere_run_001",
    ),
)

# Create engine - tracking is automatically wired
engine = create_engine(config, sphere)

# Run evolution - metrics logged to MLflow automatically
result = engine.run(initial_population)

# View results in MLflow UI: mlflow ui --port 5000
```

When you run this, you'll get:
- MLflow run created under "evolve_experiments"
- Configuration logged as parameters
- Per-generation metrics: `best_fitness`, `mean_fitness`, `std_fitness`

---

## 2. Metric Categories: Choose Your Detail Level

### Minimal (Core Only)

Best for: Production runs where you only need fitness convergence.

```python
tracking = TrackingConfig.minimal()
# Logs: best_fitness, mean_fitness, std_fitness
```

### Standard (Recommended)

Best for: Most experiments. Fitness + timing + extended stats.

```python
from evolve.config.tracking import TrackingConfig, MetricCategory

tracking = TrackingConfig(
    experiment_name="my_experiment",
    categories=frozenset({
        MetricCategory.CORE,
        MetricCategory.EXTENDED_POPULATION,
        MetricCategory.TIMING,
    }),
)
# Logs: worst_fitness, median_fitness, quartiles, generation_time_ms
```

### Comprehensive (All Metrics)

Best for: Research analysis, debugging convergence issues.

```python
tracking = TrackingConfig.comprehensive("detailed_analysis")
# Logs: Everything including diversity, entropy, selection pressure
```

---

## 3. Domain-Specific Metrics

### Multi-Objective Optimization

Multi-objective metrics are automatically enabled when `multiobjective` config is present:

```python
from evolve.config.multiobjective import MultiObjectiveConfig, ObjectiveSpec

config = UnifiedConfig(
    name="moo_experiment",
    multiobjective=MultiObjectiveConfig(
        objectives=[
            ObjectiveSpec(name="obj1", minimize=True),
            ObjectiveSpec(name="obj2", minimize=False),
        ],
    ),
    tracking=TrackingConfig(
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.MULTIOBJECTIVE,
        }),
        hypervolume_reference=(10.0, 0.0),  # Reference point for HV
    ),
)

# Metrics logged:
# - mo_hypervolume: Volume dominated by Pareto front
# - mo_pareto_front_size: Number of non-dominated solutions
# - mo_crowding_diversity: Distribution quality
```

### Evolvable Reproduction Protocols (ERP)

ERP metrics show mating dynamics:

```python
from evolve.config.erp import ERPSettings

config = UnifiedConfig(
    name="erp_experiment",
    erp=ERPSettings(
        compatibility_threshold=0.7,
        mating_type="assortative",
    ),
    tracking=TrackingConfig(
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.ERP,
        }),
    ),
)

# Metrics logged:
# - erp_mating_success_rate: Successful / attempted matings
# - erp_attempted_matings: Total mating attempts
# - erp_successful_matings: Offspring produced
# - erp_protocol_{name}_success_rate: Per-protocol breakdown
```

### Speciation

Speciation metrics track species dynamics:

```python
config = UnifiedConfig(
    # ... with speciation enabled ...
    tracking=TrackingConfig(
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.SPECIATION,
        }),
    ),
)

# Metrics logged:
# - spec_species_count: Number of species
# - spec_average_species_size: Mean individuals per species
# - spec_species_births: New species this generation
# - spec_species_extinctions: Species that went extinct
```

---

## 4. Fitness Metadata Extraction

If your evaluator returns rich metadata, you can automatically log it:

```python
from evolve.core.types import Fitness

def rl_evaluator(genome):
    """RL evaluator that returns episode metadata."""
    # Run episode...
    return Fitness(
        value=total_reward,
        metadata={
            "episode_reward": total_reward,
            "steps": episode_length,
            "collisions": collision_count,
            "goal_reached": 1 if reached_goal else 0,
        },
    )

config = UnifiedConfig(
    name="rl_experiment",
    tracking=TrackingConfig(
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.METADATA,
        }),
        metadata_threshold=0.5,  # Extract fields in >50% of individuals
        metadata_prefix="meta_",
    ),
)

# Metrics logged:
# - meta_episode_reward_best, meta_episode_reward_mean, meta_episode_reward_std
# - meta_steps_best, meta_steps_mean, meta_steps_std
# - meta_collisions_mean, etc.
```

---

## 5. Timing Instrumentation

Track where time is spent:

```python
tracking = TrackingConfig(
    categories=frozenset({
        MetricCategory.CORE,
        MetricCategory.TIMING,
    }),
    timing_breakdown=True,  # Enable fine-grained phase timing
)

# Metrics logged:
# - generation_time_ms: Total generation time
# - evaluation_time_ms: Time in fitness evaluation
# - selection_time_ms: Time in parent selection
# - crossover_time_ms: Time in crossover operations
# - mutation_time_ms: Time in mutation operations
```

---

## 6. Derived Analytics

Get computed insights:

```python
tracking = TrackingConfig(
    categories=frozenset({
        MetricCategory.CORE,
        MetricCategory.DERIVED,
    }),
)

# Metrics logged:
# - derived_selection_pressure: best / mean fitness ratio
# - derived_fitness_velocity: Rate of improvement
# - derived_population_entropy: Fitness distribution diversity
# - derived_elite_turnover_rate: New elites per generation
```

---

## 7. Remote MLflow Server

Connect to a remote tracking server:

```python
tracking = TrackingConfig(
    backend="mlflow",
    tracking_uri="http://mlflow.mycompany.com:5000",
    experiment_name="production_runs",
    
    # Resilience settings for unreliable connections
    buffer_size=500,      # Buffer up to 500 generations
    flush_interval=60.0,  # Retry every 60 seconds
)
```

If the server becomes unreachable:
1. Evolution continues uninterrupted
2. Metrics are buffered in memory
3. Periodic reconnection attempts
4. Buffered metrics flushed when connection restored

---

## 8. JSON Configuration

`TrackingConfig` is JSON-serializable for experiment reproducibility:

```python
# Save to JSON
config_dict = config.to_dict()
with open("experiment_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Load from JSON
with open("experiment_config.json") as f:
    data = json.load(f)
config = UnifiedConfig.from_dict(data)
```

JSON structure:
```json
{
  "name": "my_experiment",
  "population_size": 100,
  "tracking": {
    "enabled": true,
    "backend": "mlflow",
    "experiment_name": "evolve_experiments",
    "categories": ["core", "timing", "extended_population"],
    "log_interval": 1,
    "timing_breakdown": false
  }
}
```

---

## 9. Migrating from ExperimentRunner

If you're using the existing `ExperimentRunner` approach, you can continue doing so:

```python
# Old approach (still fully supported)
from evolve.experiment import ExperimentRunner, ExperimentConfig
from evolve.experiment.tracking import MLflowTracker

config = ExperimentConfig(name="my_exp", seed=42, ...)
runner = ExperimentRunner(
    config=config,
    engine=engine,
    initial_population=pop,
    tracker=MLflowTracker(experiment_name="evolve"),
)
result = runner.run()
```

Or migrate to the new declarative approach:

```python
# New approach with UnifiedConfig
config = UnifiedConfig(
    name="my_exp",
    seed=42,
    tracking=TrackingConfig(experiment_name="evolve"),
    # ... other settings from ExperimentConfig
)
engine = create_engine(config, evaluator)
result = engine.run(initial_population)
```

---

## 10. Viewing Results

### MLflow UI

```bash
# Start local MLflow UI
cd /path/to/mlruns
mlflow ui --port 5000
```

Open http://localhost:5000 to:
- Compare runs across experiments
- Visualize metric curves
- Download logged artifacts
- Query runs by parameters

### Programmatic Access

```python
import mlflow

# List recent runs
runs = mlflow.search_runs(experiment_names=["evolve_experiments"])
print(runs[["run_id", "metrics.best_fitness", "params.seed"]])

# Load metrics for a specific run
run = mlflow.get_run("abc123...")
print(run.data.metrics)  # Final values
history = mlflow.get_metric_history("abc123...", "best_fitness")
```

---

## Summary

| Use Case | Categories | Config |
|----------|------------|--------|
| Quick experiments | `{CORE}` | `TrackingConfig.minimal()` |
| Standard analysis | `{CORE, EXTENDED_POPULATION, TIMING}` | `TrackingConfig.standard()` |
| Research deep-dive | All categories | `TrackingConfig.comprehensive()` |
| Multi-objective | `{CORE, MULTIOBJECTIVE}` | + `hypervolume_reference` |
| RL experiments | `{CORE, METADATA}` | + evaluator with `Fitness.metadata` |
| Debugging performance | `{CORE, TIMING}` | + `timing_breakdown=True` |
