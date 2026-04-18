# API Contract: Symbiogenetic Merge Operator

**Feature**: 013-symbiogenetic-merge  
**Date**: 2026-04-17

---

## 1. Merge Operator Protocol

The primary public interface for merge operators.

```python
from typing import Protocol, TypeVar, runtime_checkable
from random import Random
from typing import Any

G = TypeVar("G")

@runtime_checkable
class SymbiogeneticMerge(Protocol[G]):
    """Permanently absorb a symbiont genome into a host genome."""

    def merge(
        self,
        host: G,
        symbiont: G,
        rng: Random,
        **kwargs: Any,
    ) -> G:
        """
        Merge symbiont into host, producing a single offspring.

        Args:
            host: The receiving genome (structure preserved as base).
            symbiont: The absorbed genome (remapped and integrated).
            rng: Explicit random number generator for determinism.
            **kwargs: Implementation-specific parameters.

        Returns:
            A new genome instance containing both host and symbiont material.

        Raises:
            ValueError: If host and symbiont are the same instance.
            TypeError: If genomes are incompatible types.
        """
        ...
```

### Invariants

- **Immutability**: `host` and `symbiont` are not modified. A new genome is returned.
- **Determinism**: Given the same `host`, `symbiont`, and `rng` state, the output is identical.
- **Complexity growth**: `complexity(result) >= complexity(host)` (offspring is at least as complex as host).
- **Identity check**: `host is not symbiont` — the operator MUST raise `ValueError` if they are the same instance.

---

## 2. Configuration Contract

```python
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(frozen=True)
class MergeConfig:
    """Configuration for the symbiogenetic merge phase."""

    operator: str = "graph_symbiogenetic"
    merge_rate: float = 0.0
    symbiont_source: Literal["cross_species", "archive"] = "cross_species"
    symbiont_fate: Literal["consumed", "survives"] = "consumed"
    archive_size: int = 50
    interface_count: int = 4
    interface_ratio: float = 0.5
    weight_method: Literal["mean", "host_biased", "random"] = "mean"
    weight_mean: float = 0.0
    weight_std: float = 1.0
    max_complexity: int | None = None
    operator_params: dict[str, Any] = field(default_factory=dict)
```

### Config integration

```python
# In UnifiedConfig:
merge: MergeConfig | None = None  # None = merge disabled
```

### YAML representation

```yaml
merge:
  operator: graph_symbiogenetic
  merge_rate: 0.15
  symbiont_source: cross_species
  symbiont_fate: consumed
  archive_size: 50
  interface_count: 4
  interface_ratio: 0.5
  weight_method: mean
  weight_mean: 0.0
  weight_std: 1.0
  max_complexity: null
  operator_params: {}
```

---

## 3. Registry Contract

### Registration

```python
# Category: "merge"
# Pattern: OperatorRegistry.register("merge", name, cls, compatible_genomes)

registry.register("merge", "graph_symbiogenetic", GraphSymbiogeneticMerge, ["GraphGenome"])
registry.register("merge", "sequence_symbiogenetic", SequenceSymbiogeneticMerge, ["SequenceGenome"])
registry.register("merge", "vector_symbiogenetic", VectorSymbiogeneticMerge, ["VectorGenome"])
registry.register("merge", "embedding_symbiogenetic", EmbeddingSymbiogeneticMerge, ["EmbeddingGenome"])
```

### Resolution

```python
merge_op = registry.get("merge", config.merge.operator, **config.merge.operator_params)
```

---

## 4. Metric Contract

### New MetricCategory value

```python
SYMBIOGENESIS = "symbiogenesis"
```

### Metrics emitted per generation (when enabled)

| Metric key | Type | Description |
|------------|------|-------------|
| `merge/count` | `int` | Number of merge operations performed |
| `merge/mean_genome_complexity` | `float` | Average gene count of merged offspring |
| `merge/complexity_delta` | `float` | Mean increase in complexity from merge |

---

## 5. Metadata Contract

Merged offspring carry the following metadata:

```python
IndividualMetadata(
    age=0,
    parent_ids=(host.id, symbiont.id),
    species_id=host.metadata.species_id,  # inherits host species
    origin="symbiogenetic_merge",
    source_strategy=config.merge.symbiont_source,  # "cross_species" or "archive"
)
```

---

## 6. Error Handling Contract

| Condition | Behavior | FR |
|-----------|----------|----|
| Host == Symbiont (same instance) | `ValueError` raised by operator | FR-015 |
| No compatible symbiont found | Skip merge for this host, emit `warnings.warn()` | FR-016 |
| Single-species population (cross_species) | Skip merge, emit warning | FR-016 |
| Empty archive (archive source) | Skip merge, emit warning | FR-016 |
| Merged genome exceeds `max_complexity` | Skip merge, emit warning | FR-016 |
| Interface count exceeds available nodes | Create as many as possible, emit warning | FR-016 |
| Incompatible genome types | `TypeError` from operator | — |
| merge_rate = 0.0 | Merge phase is a no-op | — |
