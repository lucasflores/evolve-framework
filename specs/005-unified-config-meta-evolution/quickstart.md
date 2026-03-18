# Quickstart: Unified Configuration & Meta-Evolution

**Feature Branch**: `005-unified-config-meta-evolution`

This guide gets you started with the unified configuration system in 5 minutes.

---

## Prerequisites

- Python 3.10+
- evolve framework installed

---

## Quick Examples

### 1. Define Experiment in JSON

Create `experiment.json`:

```json
{
  "name": "sphere_optimization",
  "population_size": 100,
  "max_generations": 50,
  
  "selection": "tournament",
  "selection_params": {"tournament_size": 5},
  
  "crossover": "sbx",
  "crossover_rate": 0.9,
  "crossover_params": {"eta": 20.0},
  
  "mutation": "gaussian",
  "mutation_rate": 1.0,
  "mutation_params": {"sigma": 0.1},
  
  "genome_type": "vector",
  "genome_params": {
    "dimensions": 10,
    "bounds": [-5.12, 5.12]
  },
  "minimize": true
}
```

Run it:

```python
from evolve.config import UnifiedConfig, create_engine

def sphere(x):
    return sum(xi ** 2 for xi in x)

config = UnifiedConfig.from_json("experiment.json")
engine = create_engine(config, sphere)
result = engine.run()

print(f"Best fitness: {result.best.fitness}")
```

### 2. One-Line Engine Creation

```python
from evolve.config import create_config, create_engine

# Create config programmatically
config = create_config(
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    population_size=100,
    max_generations=50,
    selection="tournament",
    selection_params={"tournament_size": 5},
    crossover="sbx",
    mutation="gaussian",
)

# Create engine with one call
engine = create_engine(config, sphere_function)
result = engine.run()
```

### 3. Switch to Multi-Objective

Just add the `multiobjective` section:

```json
{
  "name": "multi_objective_demo",
  "population_size": 100,
  "max_generations": 100,
  
  "selection": "tournament",
  "crossover": "sbx",
  "mutation": "gaussian",
  
  "genome_type": "vector",
  "genome_params": {"dimensions": 5, "bounds": [0, 1]},
  
  "multiobjective": {
    "objectives": [
      {"name": "f1", "direction": "minimize"},
      {"name": "f2", "direction": "minimize"}
    ],
    "reference_point": [11.0, 11.0]
  }
}
```

Engine automatically uses NSGA-II selection.

### 4. Enable ERP (Evolvable Reproduction Protocols)

Add the `erp` section:

```json
{
  "name": "erp_experiment",
  "population_size": 50,
  "max_generations": 100,
  
  "selection": "tournament",
  "crossover": "uniform",
  "mutation": "gaussian",
  
  "genome_type": "vector",
  "genome_params": {"dimensions": 10},
  
  "erp": {
    "step_limit": 1000,
    "recovery_threshold": 0.1,
    "protocol_mutation_rate": 0.1,
    "enable_intent": true,
    "enable_recovery": true
  }
}
```

Factory returns `ERPEngine` instead of `EvolutionEngine`.

### 5. Meta-Evolution (Hyperparameter Optimization)

```python
from evolve.config import UnifiedConfig, ParameterSpec, run_meta_evolution

# Base configuration (fixed parameters)
base_config = UnifiedConfig(
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    selection="tournament",
    crossover="sbx",
    mutation="gaussian",
)

# Define which parameters to evolve
param_specs = (
    ParameterSpec("population_size", "integer", bounds=(50, 500)),
    ParameterSpec("mutation_rate", "continuous", bounds=(0.1, 1.0)),
    ParameterSpec("selection_params.tournament_size", "integer", bounds=(2, 10)),
)

# Run meta-evolution
result = run_meta_evolution(
    base_config=base_config,
    param_specs=param_specs,
    evaluator=sphere_evaluator,
    outer_generations=20,
    trials_per_config=3,
)

print(f"Best configuration: {result.best_config.to_dict()}")
print(f"Best solution fitness: {result.best_solution.fitness}")

# Save best configuration for future use
result.export_best_config("optimal_config.json")
```

---

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema_version` | str | "1.0.0" | Schema version for compatibility |
| `name` | str | "" | Experiment identifier |
| `seed` | int | None | Random seed (None = random) |
| `population_size` | int | 100 | Population size |
| `max_generations` | int | 100 | Maximum generations |
| `elitism` | int | 1 | Elite count preserved |
| `minimize` | bool | true | Lower fitness is better |

### Operators

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selection` | str | "tournament" | Selection operator name |
| `selection_params` | dict | {} | Selection parameters |
| `crossover` | str | "uniform" | Crossover operator name |
| `crossover_rate` | float | 0.9 | Crossover probability |
| `crossover_params` | dict | {} | Crossover parameters |
| `mutation` | str | "gaussian" | Mutation operator name |
| `mutation_rate` | float | 1.0 | Mutation probability |
| `mutation_params` | dict | {} | Mutation parameters |

### Genome

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genome_type` | str | "vector" | vector, sequence, graph, scm |
| `genome_params` | dict | {} | Type-specific parameters |

### Built-in Operators

**Selection**: tournament, roulette, rank, crowded_tournament

**Crossover**: uniform, single_point, two_point, sbx, blend, neat

**Mutation**: gaussian, uniform, polynomial, boundary, neat

---

## Registry Extension

Register custom operators:

```python
from evolve.config import get_operator_registry

class DomainMutation:
    def __init__(self, rate: float = 0.1):
        self.rate = rate
    
    def mutate(self, genome, rng):
        # Custom mutation logic
        return modified_genome

# Register with genome compatibility
registry = get_operator_registry()
registry.register(
    "mutation", 
    "domain_specific", 
    DomainMutation,
    compatible_genomes={"vector", "sequence"}
)

# Now usable in config
config = UnifiedConfig(mutation="domain_specific", mutation_params={"rate": 0.2})
```

---

## Stopping Criteria

```json
{
  "stopping": {
    "max_generations": 500,
    "fitness_threshold": 0.001,
    "stagnation_generations": 50,
    "time_limit_seconds": 3600
  }
}
```

Evolution stops when ANY criterion is met.

---

## Callbacks

```json
{
  "callbacks": {
    "enable_logging": true,
    "log_level": "INFO",
    "enable_checkpointing": true,
    "checkpoint_dir": "./checkpoints",
    "checkpoint_frequency": 10
  }
}
```

Custom callbacks passed to factory:

```python
from evolve.core.callbacks import Callback

class MyCallback(Callback):
    def on_generation_end(self, generation, population, metrics):
        print(f"Gen {generation}: {metrics}")

engine = create_engine(config, evaluator, callbacks=[MyCallback()])
```

---

## Import Summary

```python
# Main imports
from evolve.config import (
    UnifiedConfig,
    create_engine,
    create_config,
)

# Registries
from evolve.config import (
    get_operator_registry,
    get_genome_registry,
)

# Meta-evolution
from evolve.config import (
    ParameterSpec,
    run_meta_evolution,
    MetaEvolutionResult,
)

# Configuration components
from evolve.config import (
    StoppingConfig,
    CallbackConfig,
    ERPSettings,
    MultiObjectiveConfig,
    ObjectiveSpec,
)
```

---

## Migration from Legacy Config

The unified configuration is backward compatible. Existing code continues to work:

```python
# Legacy approach (still works)
from evolve.core.engine import EvolutionEngine, EvolutionConfig
config = EvolutionConfig(population_size=100, max_generations=50)
engine = EvolutionEngine(config, evaluator, selection, crossover, mutation)

# New approach (simpler)
from evolve.config import UnifiedConfig, create_engine
config = UnifiedConfig(population_size=100, max_generations=50)
engine = create_engine(config, evaluator)  # Operators resolved automatically
```

---

## Next Steps

1. **Tutorials**: See `docs/tutorials/` for in-depth examples
2. **API Reference**: See `docs/api/config.md` for full API
3. **Examples**: See `examples/unified_config/` for complete scripts
