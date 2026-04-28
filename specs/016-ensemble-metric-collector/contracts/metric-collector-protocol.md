# Contract: MetricCollector Protocol

**Feature**: `016-ensemble-metric-collector`  
**Date**: 2026-04-27

`EnsembleMetricCollector` implements the `MetricCollector` protocol defined in `evolve/experiment/collectors/base.py`. This document records the protocol's interface contract as it applies to the new collector.

---

## Protocol Definition

```python
class MetricCollector(Protocol):
    def collect(self, context: CollectionContext) -> dict[str, float]: ...
    def reset(self) -> None: ...
```

---

## `EnsembleMetricCollector` Contract

### `collect(context: CollectionContext) -> dict[str, float]`

| Pre-condition | Behaviour |
|---------------|-----------|
| `len(context.population) == 0` | Returns `{}` immediately |
| All individuals have `None` fitness | Returns `{}` after logging DEBUG |
| `context.previous_elites is None` | `ensemble/expert_turnover` absent from result |
| `context.species_info is None` | `ensemble/specialization_index` absent from result |
| Any fitness value < 0 | Shift applied; `DEBUG` log emitted; result is still returned |

| Post-condition | Guarantee |
|----------------|-----------|
| All returned values are finite floats | `math.isfinite(v)` is `True` for every value |
| `ensemble/gini_coefficient` ∈ [0, 1] | Always |
| `ensemble/participation_ratio` ∈ [1, N] | Where N = non-None-fitness population size |
| `ensemble/top_k_concentration` ∈ [0, 1] | Always |
| `ensemble/expert_turnover` ∈ [0, 1] | When present |
| `ensemble/specialization_index` ∈ [0, 1] | When present |
| No writes to `context`, `population`, `individuals`, or `fitness` | Pure read |
| Does not raise for any valid `CollectionContext` | No exceptions on valid input |

### `reset() -> None`

No-op. Always succeeds. Has no observable effect.

---

## `CollectionContext` Fields Consumed

| Field | Access mode | Required |
|-------|-------------|----------|
| `context.population` | Read (iterate, index, `len()`) | Yes |
| `context.previous_elites` | Read (iterate for identity set construction via `ind.id` when available, else `id(ind)`) | No — metric omitted if `None` |
| `context.species_info` | Read (iterate over `dict.items()`) | No — metric omitted if `None` |

All other `CollectionContext` fields are ignored.

---

## Metric Key Naming Convention

All keys follow the `"{category}/{metric_name}"` convention established in `DerivedAnalyticsCollector`:

```
ensemble/gini_coefficient
ensemble/participation_ratio
ensemble/top_k_concentration
ensemble/expert_turnover
ensemble/specialization_index
```
