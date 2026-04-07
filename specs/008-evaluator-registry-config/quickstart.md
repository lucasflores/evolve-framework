# Quickstart: Evaluator Registry & UnifiedConfig Declarative Completeness

**Feature**: 008-evaluator-registry-config

## 1. Declare an evaluator in config and run

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine

config = UnifiedConfig(
    evaluator="benchmark",
    evaluator_params={"function_name": "sphere", "dimensions": 10},
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    population_size=50,
    max_generations=100,
    seed=42,
)

engine = create_engine(config)  # No evaluator argument needed
engine.run()
```

## 2. Register a custom evaluator

```python
from evolve.registry import get_evaluator_registry

def my_fitness_factory(target_value=0.0, penalty_weight=1.0):
    """Factory that returns an Evaluator-compatible object."""
    from evolve.evaluation.evaluator import FunctionEvaluator
    def fitness(genome):
        return -abs(sum(genome.genes) - target_value) * penalty_weight
    return FunctionEvaluator(fitness)

registry = get_evaluator_registry()
registry.register("my_fitness", my_fitness_factory)

config = UnifiedConfig(
    evaluator="my_fitness",
    evaluator_params={"target_value": 42.0, "penalty_weight": 2.0},
    genome_type="vector",
    genome_params={"dimensions": 5},
)

engine = create_engine(config)
engine.run()
```

## 3. Declare custom callbacks in config

```python
from evolve.registry import get_callback_registry

# Register a domain callback
def espo_callback_factory(log_diversity=True):
    from my_domain.callbacks import ESPOCallback
    return ESPOCallback(log_diversity=log_diversity)

get_callback_registry().register("espo", espo_callback_factory)

config = UnifiedConfig(
    evaluator="benchmark",
    evaluator_params={"function_name": "rastrigin", "dimensions": 20},
    genome_type="vector",
    genome_params={"dimensions": 20, "bounds": (-5.12, 5.12)},
    custom_callbacks=(
        {"name": "espo", "params": {"log_diversity": True}},
    ),
)

engine = create_engine(config)
engine.run()
```

## 4. Use runtime_overrides for non-serializable params

```python
from evolve.config import UnifiedConfig
from evolve.factory import create_engine

config = UnifiedConfig(
    evaluator="llm_judge",
    evaluator_params={"judge_model_id": "gpt-4", "temperature": 0.0},
)

# Decoder and task_spec are runtime objects, not JSON-serializable
engine = create_engine(
    config,
    runtime_overrides={"decoder": my_decoder, "task_spec": my_task_spec},
)
engine.run()
```

## 5. Share a complete experiment as JSON

```python
config = UnifiedConfig(
    evaluator="benchmark",
    evaluator_params={"function_name": "sphere", "dimensions": 10},
    genome_type="vector",
    genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    custom_callbacks=({"name": "history", "params": {}},),
    seed=42,
)

# Save to file
config.to_file("my_experiment.json")

# Colleague loads and runs
loaded = UnifiedConfig.from_file("my_experiment.json")
engine = create_engine(loaded)
engine.run()
```
