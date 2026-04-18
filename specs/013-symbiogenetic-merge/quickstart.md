# Quickstart: Symbiogenetic Merge Operator

**Feature**: 013-symbiogenetic-merge

---

## Enable merge in your experiment

Add a `merge` section to your `UnifiedConfig`:

```python
from evolve.config.unified import UnifiedConfig
from evolve.config.merge import MergeConfig

config = UnifiedConfig(
    population_size=100,
    max_generations=50,
    genome_type="graph",
    crossover="neat_crossover",
    mutation="neat_mutation",
    merge=MergeConfig(
        merge_rate=0.1,           # 10% of offspring undergo merge
        symbiont_source="cross_species",
        symbiont_fate="consumed",
    ),
)
```

## Track merge metrics

Enable the `SYMBIOGENESIS` metric category:

```python
from evolve.config.tracking import TrackingConfig, MetricCategory

config = UnifiedConfig(
    # ... other config ...
    merge=MergeConfig(merge_rate=0.1),
    tracking=TrackingConfig(
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.SYMBIOGENESIS,
        }),
    ),
)
```

## Use archive-based symbiont sourcing

Switch from cross-species to hall-of-fame archive:

```python
config = UnifiedConfig(
    # ... other config ...
    merge=MergeConfig(
        merge_rate=0.1,
        symbiont_source="archive",
        archive_size=50,           # keep top 50 individuals
        symbiont_fate="survives",  # symbionts remain in population
    ),
)
```

## Register a custom merge operator

```python
from evolve.registry.operators import OperatorRegistry

class MyCustomMerge:
    def merge(self, host, symbiont, rng, **kwargs):
        # Your custom merge logic
        return merged_genome

registry = OperatorRegistry()
registry.register("merge", "my_merge", MyCustomMerge, ["GraphGenome"])

config = UnifiedConfig(
    merge=MergeConfig(operator="my_merge"),
)
```

## Run the experiment

```python
from evolve.factory.engine import create_engine

engine = create_engine(config, evaluator=my_evaluator)
result = engine.run()

# Merged offspring are tagged with origin="symbiogenetic_merge"
for ind in result.population:
    if ind.metadata.origin == "symbiogenetic_merge":
        print(f"Merged individual {ind.id}: fitness={ind.fitness}")
```
