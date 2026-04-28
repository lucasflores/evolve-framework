# Research: Ensemble Metric Collector

**Feature**: `016-ensemble-metric-collector`  
**Date**: 2026-04-27  
**Status**: Complete — all NEEDS CLARIFICATION resolved

---

## 1. Fitness Access Pattern

**Decision**: Use `hasattr`-based dual-path access identical to `DerivedAnalyticsCollector` and `IslandsMetricCollector`.

**Rationale**: Two fitness types co-exist in the framework — multi-objective (`fitness.values[0]`) and scalar (`fitness.value`). Both existing collectors handle this with a try/except or `hasattr`. The ensemble collector must follow the same pattern for compatibility.

**Implementation**:
```python
def _get_scalar_fitness(ind: Individual[Any]) -> float | None:
    """Extract scalar fitness value from an individual."""
    if ind.fitness is None:
        return None
    if hasattr(ind.fitness, "values"):
        return float(ind.fitness.values[0])
    return float(ind.fitness.value)
```

Individuals with `None` fitness are skipped with a `logging.debug` warning, consistent with `DerivedAnalyticsCollector`.

**Alternatives considered**:
- Accessing `ind.fitness[0]` directly (used in `SpeciationMetricCollector`) — rejected because it is less explicit and may not work for all Fitness types.
- Requiring scalar Fitness only — rejected; the framework supports multi-objective runs where users might still want ensemble metrics on the first objective.

---

## 2. Numpy Vectorisation Strategy

**Decision**: Use `numpy` for all five metric computations. Import at module level inside `evolve/experiment/collectors/ensemble.py`.

**Rationale**: `DerivedAnalyticsCollector` already imports `numpy as np` at module level. Numpy is a project dependency (confirmed importable). Vectorised operations keep `collect()` well under 1 ms for populations ≤ 10k.

**Numpy patterns per metric**:

| Metric | Core numpy ops |
|--------|----------------|
| Gini Coefficient | `np.abs`, `np.subtract.outer`, `np.sum` |
| Participation Ratio | `np.sum(f)**2 / np.sum(f**2)` |
| Top-k Concentration | `np.partition` (partial sort), `np.sum` |
| Expert Turnover | Python sets on `id()` values — no numpy needed |
| Specialization Index | `np.mean`, groupwise sum-of-squares via loop over species |

**Negative-fitness shift**:
```python
if np.min(fitnesses) < 0:
    logging.getLogger(__name__).debug("Shifting fitness values by abs(min)...")
    fitnesses = fitnesses - np.min(fitnesses)
```
Applied once before Gini, PR, and Top-k; NOT applied before Specialization Index (variance is shift-invariant).

**Alternatives considered**:
- Pure-Python loops — rejected; O(N²) for Gini on large populations would be unacceptably slow.
- Using `scipy.stats` for Gini — rejected; adds a non-core dependency and scipy is not currently required by the framework.

---

## 3. Gini Coefficient Formula

**Decision**: $G = \frac{\sum_i\sum_j |f_i - f_j|}{2N\sum_i f_i}$ applied to the shifted non-negative array.

**Rationale**: This is the standard population-genetics / econometrics Gini formula. An efficient O(N log N) equivalent exists:

```python
# Equivalent O(N log N) form using sorted values
f_sorted = np.sort(fitnesses)
n = len(f_sorted)
cumsum = np.cumsum(f_sorted)
gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n  # Lorenz-curve form
```

Degenerate case: when `np.sum(fitnesses) == 0`, return `0.0` (uniform zero → perfect equality).

**Alternatives considered**:
- Direct double-sum O(N²): `np.abs(np.subtract.outer(f, f)).sum() / (2 * n * f.sum())` — correct but slower for large N; use O(N log N) form.

---

## 4. Participation Ratio Degenerate Cases

**Decision**: Return `float(N)` when `sum(f²) == 0` (which implies `sum(f) == 0`).

**Rationale**: When all fitnesses are zero, $PR = 0/0$. The uniform-fitness limit of the formula gives PR → N as all individuals become equal. This is numerically consistent with the case `f_i = ε ∀i` as ε → 0: `(Nε)²/(Nε²) = N`. Confirmed in Q2 of the spec clarification session.

---

## 5. Expert Turnover: Identity via `ind.id` (with `id()` fallback)

**Decision**: Use `ind.id` (UUID) when the individual has an `id` attribute — stable across `with_fitness()` copies — and fall back to Python `id()` only for non-framework objects (e.g., mocks without an `id` attribute).

**Rationale**: The original design used Python `id()` (object identity). In practice, `EvolutionEngine` produces new `Individual` objects every generation via `with_fitness()`, which preserves the UUID (`ind.id`) but returns a fresh Python object with a different `id()`. Using `id()` caused `expert_turnover` to always equal `1.0` because every generation produces new object references even for elites that survive. Using `ind.id` correctly identifies the same individual across generations.

**Elite set derivation**: `current_elite = sorted(population, key=fitness, reverse=True)[:elite_count]` where `elite_count = ceil(top_k_percent/100 * len(population))` if `elite_size` is None.

**Empty `previous_elites` convention**: If `previous_elites` is `[]` (not None), all current elites are "new" → turnover = 1.0.

---

## 6. Specialization Index: Eta-squared

**Decision**: $\eta^2 = SS_\text{between} / SS_\text{total}$ where between-species SS is computed by looping over species groups from `context.species_info`.

**Rationale**: The standard one-way ANOVA effect size. `species_info: dict[int, list[int]]` provides species_id → list of indices into `context.population`. The loop is at most `S` iterations where S = number of species (typically < 100), so performance is not a concern.

**Degenerate case**: When `SS_total == 0` (all individuals identical fitness), return `0.0` — confirmed in Q3 of the spec clarification session.

**Formula**:
```
grand_mean = mean(all_fitnesses)
SS_total = sum((f_i - grand_mean)^2 for all i)
SS_between = sum(n_s * (species_mean_s - grand_mean)^2 for each species s)
eta_squared = SS_between / SS_total  # 0.0 if SS_total == 0
```

---

## 7. Engine Dispatch Pattern

**Decision**: Follow exact same pattern as `ERPMetricCollector` and `SpeciationMetricCollector` — instantiate at `engine.__init__` time guarded by `tracking.has_category(MetricCategory.ENSEMBLE)`, call in per-generation loop.

**Rationale**: Confirmed in Q4 of the spec clarification session. No new registry abstraction. This keeps the engine's dispatcher predictable and consistent.

**Exact guard pattern** (to match existing):
```python
self._ensemble_collector: EnsembleMetricCollector | None = (
    EnsembleMetricCollector()
    if tracking.has_category(MetricCategory.ENSEMBLE)
    else None
)
```

Called in generation loop:
```python
if self._ensemble_collector is not None:
    metrics.update(self._ensemble_collector.collect(context))
```

---

## 8. Reference Guide Scope

**Decision**: All ten collector classes included, each with its actual enabling mechanism.

**The ten collectors** and their enabling mechanism:

| Collector | Module | Enabling Mechanism |
|-----------|--------|-------------------|
| `DerivedAnalyticsCollector` | `derived.py` | `MetricCategory.DERIVED` |
| `ERPMetricCollector` | `erp.py` | `MetricCategory.ERP` (auto when ERP enabled) |
| `FitnessMetadataCollector` | `metadata.py` | `MetricCategory.METADATA` |
| `IslandsMetricCollector` | `islands.py` | Auto when island model enabled |
| `MergeMetricCollector` | `merge.py` | Manual instantiation (no MetricCategory gate) |
| `MultiObjectiveMetricCollector` | `multiobjective.py` | `MetricCategory.MULTIOBJECTIVE` (auto when MO enabled) |
| `NEATMetricCollector` | `neat.py` | Manual instantiation (no MetricCategory gate) |
| `SpeciationMetricCollector` | `speciation.py` | `MetricCategory.SPECIATION` (auto when speciation enabled) |
| *(no separate symbiogenesis collector — `MergeMetricCollector` covers symbiogenesis metrics; see row above)* | | |
| `EnsembleMetricCollector` | `ensemble.py` *(new)* | `MetricCategory.ENSEMBLE` (explicit only) |

Confirmed in Q5 of the spec clarification session: include all ten, document actual enabling mechanism.

---

## Summary of All NEEDS CLARIFICATION Resolved

| Item | Resolution |
|------|-----------|
| Fitness access pattern | `hasattr` dual-path: `fitness.values[0]` / `fitness.value` |
| Numpy import strategy | Module-level, follows `DerivedAnalyticsCollector` |
| Gini edge case (zero total) | Return `0.0` |
| Participation Ratio edge case (zero total) | Return `float(N)` |
| Expert Turnover identity mechanism | Python `id()` set difference |
| Specialization Index edge case (zero variance) | Return `0.0` |
| Engine dispatch pattern | `has_category` guard at `__init__`, same as ERP/Speciation |
| Reference guide scope | All ten collectors |
| Negative fitness handling | Shift by `abs(min)` before Gini/PR/Top-k; log debug warning |
