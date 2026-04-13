# Quickstart: Feature & Cleanup Backlog

## Verify Bug Fixes

```python
from evolve.core.population import Population
from evolve.core.types import Individual, Fitness

# Create population with known fitness values
inds = [
    Individual(genome=[1.0], fitness=Fitness.scalar(1.0)),
    Individual(genome=[2.0], fitness=Fitness.scalar(5.0)),
    Individual(genome=[3.0], fitness=Fitness.scalar(3.0)),
]
pop = Population(individuals=inds)

# Minimization: best = 1.0
stats_min = pop.statistics  # minimize=True by default
assert stats_min.best_fitness.values[0] == 1.0

# Maximization: best = 5.0
stats_max = pop.compute_statistics(minimize=False)
assert stats_max.best_fitness.values[0] == 5.0
assert stats_max.minimize == False
```

## Callbacks Persist

```python
from evolve.core.callbacks import HistoryCallback
from evolve.factory.engine import create_engine

history = HistoryCallback()
engine = create_engine(config, evaluator=my_eval, callbacks=[history])

# run() without passing callbacks — history still active
result = engine.run(initial_population)
assert len(history.records) > 0
```

## Population Dynamics Metrics

```python
# Metrics dict now includes fitness distribution, diversity, movement
result = engine.run(initial_population)
last_metrics = result.history[-1]

print(last_metrics["median_fitness"])
print(last_metrics["q1_fitness"])
print(last_metrics["unique_fitness_count"])
print(last_metrics["mean_pairwise_distance"])
print(last_metrics["centroid_drift"])
print(last_metrics["best_changed"])
```

## Callback Priority

```python
from evolve.core.callbacks import SimpleCallback

class MetricInjector(SimpleCallback):
    priority = 0  # runs before TrackingCallback (priority=1000)
    
    def on_generation_end(self, generation, population, metrics):
        metrics["custom_score"] = compute_something(population)
```

## Meta-Evolution MLflow Tracking

```python
# With tracking enabled, meta-evolution creates nested MLflow runs
config = UnifiedConfig(
    ...,
    tracking=TrackingConfig(enabled=True, experiment_name="meta-search"),
    meta=MetaEvolutionConfig(...)
)
# Parent run + child runs visible in MLflow UI
```

## UnifiedConfig Datasets & Tags

```python
from evolve.config.unified import UnifiedConfig, DatasetConfig

config = UnifiedConfig(
    ...,
    tags={"experiment": "ablation", "version": "v2"},
    training_data=DatasetConfig(name="train-set", path="/data/train.csv"),
    validation_data=DatasetConfig(name="val-set", path="/data/val.csv"),
)
# Tags appear in MLflow native Tags field + parameters
# Datasets appear in MLflow native Datasets field
```
