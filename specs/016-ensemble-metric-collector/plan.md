# Implementation Plan: Ensemble Metric Collector & Reference Guide

**Branch**: `016-ensemble-metric-collector` | **Date**: 2026-04-27 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/016-ensemble-metric-collector/spec.md`

## Summary

Add `EnsembleMetricCollector` — a stateless `MetricCollector` that computes five population-level metrics (Gini coefficient, Participation Ratio, Top-k Concentration, Expert Turnover, Specialization Index) derived purely from individual fitness values and existing `CollectionContext` fields. Gated by a new `MetricCategory.ENSEMBLE` enum value instantiated in the engine at `__init__` time. Accompanied by `docs/guides/metric-collectors-reference.md` covering all ten metric collectors.

Technical approach: numpy-vectorised computation in a new `evolve/experiment/collectors/ensemble.py` dataclass, following the protocol pattern of `DerivedAnalyticsCollector` and `ERPMetricCollector`. No new external dependencies.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: `numpy` (already a project dependency; confirmed importable; no new deps added)  
**Storage**: N/A — pure in-memory computation per generation  
**Testing**: pytest 9.0.2; existing `MockPopulation` / `MockIndividual` / `MockFitness` patterns from `tests/unit/experiment/collectors/`  
**Target Platform**: Linux/macOS (same as rest of framework)  
**Project Type**: library  
**Performance Goals**: <1 ms per `collect()` call on populations up to 10 000 individuals; O(N log N) Gini; O(N) for all other metrics via numpy  
**Constraints**: `# NO ML FRAMEWORK IMPORTS ALLOWED`; no global mutable state; mypy strict compliance; `@dataclass` collector pattern  
**Scale/Scope**: five new metric keys; one new enum value; one new source file; one new test file; one new docs guide

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Verdict | Notes |
|-----------|---------|-------|
| I. Model-Agnostic Architecture | ✅ PASS | Pure Python + numpy; zero neural network or RL imports |
| II. Separation of Concerns | ✅ PASS | Collector is a pure read function; no writes to population, engine, or fitness |
| III. Declarative Completeness | ✅ PASS | `MetricCategory.ENSEMBLE` is a first-class enum value; user adds it to `TrackingConfig.categories` declaratively; engine wires collector automatically |
| IV. Acceleration as Optional | ✅ PASS | numpy CPU operations; no accelerated variant needed |
| V. Determinism and Reproducibility | ✅ PASS | All five metrics are deterministic functions of `CollectionContext` state |
| VI. Extensibility Over Optimization | ✅ PASS | Follows existing extension pattern exactly; no clever optimizations |
| VII. Multi-Domain Algorithm Support | ✅ PASS | Pure fitness-value metrics; works across all domain types (classical EA, neuroevolution, MO) |
| VIII. Observability and Experiment Tracking | ✅ PASS | This feature directly extends the observability principle |

**No violations. No complexity tracking required.**

*Post-design re-check*: All principles remain satisfied after Phase 1 design. The contracts document confirms `EnsembleMetricCollector` is a pure read; `MetricCategory.ENSEMBLE` integrates with the existing `TrackingConfig` without any changes to the frozen dataclass API.

## Project Structure

### Documentation (this feature)

```text
specs/016-ensemble-metric-collector/
├── plan.md              ← this file
├── research.md          ← Phase 0: resolved unknowns, formula choices, patterns
├── data-model.md        ← Phase 1: entity definitions, validation rules, modified files
├── quickstart.md        ← Phase 1: usage guide with MLflow integration notes
├── contracts/
│   └── metric-collector-protocol.md  ← MetricCollector protocol contract for EnsembleMetricCollector
└── tasks.md             ← Phase 2 output (created by /speckit.tasks — NOT this command)
```

### Source Code (repository root)

```text
evolve/
├── config/
│   └── tracking.py          ← MODIFY: add MetricCategory.ENSEMBLE = "ensemble"
├── experiment/
│   └── collectors/
│       ├── __init__.py      ← MODIFY: add EnsembleMetricCollector import + __all__ entry
│       └── ensemble.py      ← NEW: EnsembleMetricCollector implementation
└── core/
    └── engine.py            ← MODIFY: add ENSEMBLE guard at __init__ + collect() call

docs/
└── guides/
    └── metric-collectors-reference.md  ← NEW: reference guide for all 10 collectors

tests/
└── unit/
    └── experiment/
        └── collectors/
            └── test_ensemble.py  ← NEW: unit tests for all edge cases in spec
```

**Structure Decision**: Single-project library. Five targeted file modifications / additions. All new code lives in the established `evolve/experiment/collectors/` namespace.

## Complexity Tracking

> No constitution violations. This section is intentionally empty.
