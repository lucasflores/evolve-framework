# Quickstart: Ensemble Metric Collector

**Feature**: `016-ensemble-metric-collector`  
**Date**: 2026-04-27

---

## Enabling Ensemble Metrics

Add `MetricCategory.ENSEMBLE` to your `TrackingConfig`:

```python
from evolve.config import UnifiedConfig
from evolve.config.tracking import MetricCategory, TrackingConfig

config = UnifiedConfig(
    tracking=TrackingConfig(
        categories={
            MetricCategory.CORE,
            MetricCategory.ENSEMBLE,  # ← add this
        }
    ),
    # ... rest of your config
)
```

Once enabled, all applicable metrics appear automatically in every generation's logged data:

```
ensemble/gini_coefficient      → float in [0, 1]
ensemble/participation_ratio   → float in [1, N]
ensemble/top_k_concentration   → float in [0, 1]
ensemble/expert_turnover       → float in [0, 1]  (when previous_elites available)
ensemble/specialization_index  → float in [0, 1]  (when speciation active)
```

---

## Standalone Usage

`EnsembleMetricCollector` can be used directly without running a full engine:

```python
from evolve.experiment.collectors import EnsembleMetricCollector
from evolve.experiment.collectors.base import CollectionContext

collector = EnsembleMetricCollector(top_k_percent=10.0)
metrics = collector.collect(context)
```

---

## Custom Top-k Percentage

```python
# Track the top 20% instead of the default 10%
collector = EnsembleMetricCollector(top_k_percent=20.0)
```

---

## Custom Elite Size for Expert Turnover

```python
# Always use exactly 5 individuals as the elite set
collector = EnsembleMetricCollector(elite_size=5)
```

---

## Interpreting the Metrics

### Gini Coefficient (`ensemble/gini_coefficient`)

Measures fitness inequality across the generation.

- **0.0** — all individuals have equal fitness; the population has converged or not yet differentiated.
- **→ 1.0** — one individual monopolises nearly all fitness; extreme selection pressure.
- **Typical healthy range**: 0.2–0.6 in an active evolutionary run.

*Degenerate case*: returns `0.0` when all fitness values are zero.

---

### Participation Ratio (`ensemble/participation_ratio`)

The effective number of individuals contributing to the population's fitness.

- **= N** — all individuals contribute equally (uniform fitness distribution).
- **= 1** — a single individual holds all the fitness.
- **Formula**: $(∑_i f_i)^2 / ∑_i f_i^2$

*Degenerate case*: returns `float(N)` when total fitness is zero (all equally zero → all equally contributing).

---

### Top-k Concentration (`ensemble/top_k_concentration`)

Fraction of total population fitness held by the top `k`% of individuals.

- **= 1.0** — the top-k individuals hold all fitness; the rest contribute nothing.
- **= k/100** — fitness is perfectly uniform; the top-k hold exactly their proportional share.

*Degenerate case*: returns `0.0` when total fitness is zero.

---

### Expert Turnover (`ensemble/expert_turnover`)

Fraction of the elite set that changed identity since the previous generation. Only emitted when `context.previous_elites` is provided.

- **= 0.0** — the exact same individuals are elite; the top performers are stable.
- **= 1.0** — the elite set has completely changed; rapid exploration or instability.
- **Only present** when the engine populates `CollectionContext.previous_elites`.

*Edge case*: returns `1.0` when `previous_elites` is an empty list (no prior elites to match).

---

### Specialization Index (`ensemble/specialization_index`)

Fraction of total fitness variance explained by species membership (η²). Only emitted when speciation is active.

- **= 1.0** — species differences explain all fitness variance; strong species specialization.
- **= 0.0** — species membership explains none of the variance; species are indistinguishable by fitness.
- **Only present** when `context.species_info` is not `None`.

*Degenerate case*: returns `0.0` when total fitness variance is zero (converged population).

---

## MLflow Integration

With `MetricCategory.ENSEMBLE` enabled, all five metrics are automatically logged to MLflow via the standard experiment tracking pipeline. No additional MLflow configuration is needed.

To query them after a run:

```python
import mlflow

run = mlflow.get_run(run_id)
gini = run.data.metrics["ensemble/gini_coefficient"]
pr = run.data.metrics["ensemble/participation_ratio"]
```

---

## Reference Guide

For mathematical formulas, degenerate-case behavior, and coverage of all ten metric collectors, see:

```
docs/guides/metric-collectors-reference.md
```
