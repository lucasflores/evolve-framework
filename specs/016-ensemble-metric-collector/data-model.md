# Data Model: Ensemble Metric Collector

**Feature**: `016-ensemble-metric-collector`  
**Date**: 2026-04-27

---

## New Entities

### `MetricCategory.ENSEMBLE`

**Location**: `evolve/config/tracking.py`  
**Type**: New enum value added to existing `MetricCategory(Enum)`

| Field | Value | Notes |
|-------|-------|-------|
| `ENSEMBLE` | `"ensemble"` | String value consistent with all other categories |

**Constraints**:
- Not auto-enabled by `UnifiedConfig` derived configuration (unlike `ERP`, `MULTIOBJECTIVE`, `SYMBIOGENESIS`)
- Must be explicitly added to `TrackingConfig.categories` by the user
- Valid in `TrackingConfig.has_category()`, `with_category()`, `without_category()` without any changes to those methods

**State transitions**: `TrackingConfig` is frozen; enabling is done at construction time via `TrackingConfig(categories={MetricCategory.ENSEMBLE, ...})`.

---

### `EnsembleMetricCollector`

**Location**: `evolve/experiment/collectors/ensemble.py`  
**Type**: `@dataclass` implementing `MetricCollector` protocol  
**Base**: `MetricCollector` (protocol — no base class inheritance)

#### Fields

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `top_k_percent` | `float` | `10.0` | Percentage of population for top-k metrics |
| `elite_size` | `int \| None` | `None` | Override auto-derived elite count; if None uses `ceil(top_k_percent/100 * N)` |

No private mutable fields — collector is fully stateless. `reset()` is a no-op.

#### `collect(context: CollectionContext) -> dict[str, float]`

**Input validation**:
- If `len(context.population) == 0` → return `{}`
- Skip individuals with `None` fitness; log `DEBUG` warning if any skipped
- If remaining fitness count == 0 → return `{}`

**Fitness extraction**: `hasattr` dual-path pattern:
```python
if hasattr(ind.fitness, "values"):
    f = float(ind.fitness.values[0])
else:
    f = float(ind.fitness.value)
```

**Negative shift** (applied to `fitnesses` array before Gini, PR, Top-k only):
```python
min_f = np.min(fitnesses)
if min_f < 0:
    _logger.debug("Shifting %d fitness values by %.4f", len(fitnesses), abs(min_f))
    fitnesses = fitnesses - min_f
```

#### Always-present keys (when population is non-empty)

| Key | Formula | Range | Degenerate case |
|-----|---------|-------|-----------------|
| `ensemble/gini_coefficient` | O(N log N) Lorenz form | [0, 1] | `0.0` when `sum(f) == 0` |
| `ensemble/participation_ratio` | `sum(f)² / sum(f²)` | [1, N] | `float(N)` when `sum(f²) == 0` |
| `ensemble/top_k_concentration` | `sum(top-k f) / sum(f)` | [0, 1] | `0.0` when `sum(f) == 0`; top-k clamped to ≥ 1 individual |

#### Conditionally-present keys

| Key | Condition | Formula | Range | Degenerate case |
|-----|-----------|---------|-------|-----------------|
| `ensemble/expert_turnover` | `context.previous_elites is not None` | `\|elite_t \ elite_{t-1}\| / \|elite_t\|`, where elite membership uses a stable individual key: `ind.id` when present, otherwise `id(ind)` | [0, 1] | `1.0` when `previous_elites == []` |
| `ensemble/specialization_index` | `context.species_info is not None` | η² = SS_between / SS_total | [0, 1] | `0.0` when `SS_total == 0` |

#### `reset() -> None`

No-op. `EnsembleMetricCollector` carries no generation-to-generation state. Elite history is owned by the engine via `CollectionContext.previous_elites`.

---

## Modified Entities

### `evolve/config/tracking.py` — `MetricCategory`

**Change**: Add one enum value.

```python
# Before (last entry):
SYMBIOGENESIS = "symbiogenesis"

# After:
SYMBIOGENESIS = "symbiogenesis"
ENSEMBLE = "ensemble"
```

No changes to `TrackingConfig`, `has_category()`, `with_category()`, or `without_category()`.

---

### `evolve/experiment/collectors/__init__.py`

**Change**: Add import and `__all__` entry for `EnsembleMetricCollector`.

```python
from evolve.experiment.collectors.ensemble import EnsembleMetricCollector

__all__ = [
    ...existing...,
    "EnsembleMetricCollector",
]
```

---

### `evolve/core/engine.py`

**Change**: Add one conditional instantiation at `__init__` time and one dispatch call in the generation loop.

**Init guard** (following `ERPMetricCollector` and `SpeciationMetricCollector` pattern):
```python
self._ensemble_collector: EnsembleMetricCollector | None = (
    EnsembleMetricCollector()
    if self._tracking.has_category(MetricCategory.ENSEMBLE)
    else None
)
```

**Generation loop dispatch** (following existing collector dispatch pattern):
```python
if self._ensemble_collector is not None:
    metrics.update(self._ensemble_collector.collect(context))
```

---

## Validation Rules

| Rule | Location | Check |
|------|----------|-------|
| `top_k_percent` ∈ (0, 100] | `__post_init__` | `ValueError` if outside range |
| `elite_size`, if not None, must be ≥ 1 | `__post_init__` | `ValueError` if < 1 |
| Fitness values used only in read mode | `collect()` | No attribute writes to individuals |
| Result dict values are always finite floats | `collect()` | All `math.isfinite(v)` after computation |

---

## New Files

| File | Purpose |
|------|---------|
| `evolve/experiment/collectors/ensemble.py` | `EnsembleMetricCollector` implementation |
| `tests/unit/experiment/collectors/test_ensemble.py` | Unit tests (all edge cases from spec) |
| `docs/guides/metric-collectors-reference.md` | Comprehensive reference guide for all 10 collectors |

---

## State Transitions

```
TrackingConfig (without ENSEMBLE)
    → user adds MetricCategory.ENSEMBLE to categories
TrackingConfig (with ENSEMBLE)
    → engine.__init__ sees has_category(ENSEMBLE) == True
    → EnsembleMetricCollector() instantiated
    → each generation: collect(context) called
    → dict[str, float] merged into generation metrics
    → MLflow receives ensemble/gini_coefficient et al.
```
