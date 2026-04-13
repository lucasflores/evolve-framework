# Implementation Plan: Feature & Cleanup Backlog

**Branch**: `009-feature-cleanup-backlog` | **Date**: 2026-04-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/009-feature-cleanup-backlog/spec.md`

## Summary

Fix inverted fitness reporting, callback persistence, and statistics semantics (bugs). Add population dynamics metrics with representation-aware diversity via Genome protocol, numeric callback priority ordering, meta-evolution MLflow nested-run tracking, and UnifiedConfig dataset/tag extensions.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: numpy, mlflow, dataclasses (stdlib)
**Storage**: N/A (in-memory framework; MLflow for experiment tracking)
**Testing**: pytest (unit/, integration/, property/, benchmarks/)
**Target Platform**: Cross-platform Python library
**Project Type**: Library
**Performance Goals**: Diversity metrics gated by TrackingConfig categories; sampling for O(n) pairwise distance
**Constraints**: Frozen dataclasses throughout; backward compatibility required; no global state
**Scale/Scope**: ~15 files modified, ~5 new files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | ✅ PASS | `distance()` protocol on Genome keeps core model-agnostic; genome-specific impls are in representation modules |
| II. Separation of Concerns | ✅ PASS | Metrics computation in engine, logging in callbacks, config in UnifiedConfig — no cross-cutting |
| III. Declarative Completeness | ✅ PASS | New callback priority, dataset fields, and metric categories all expressed through UnifiedConfig/TrackingConfig |
| IV. Acceleration as Optional | ✅ PASS | No GPU/JIT dependencies introduced |
| V. Determinism and Reproducibility | ✅ PASS | Pairwise distance sampling uses explicit RNG; all new metrics are deterministic from population state |
| VI. Extensibility Over Premature Optimization | ✅ PASS | Protocol-based distance dispatch; numeric priorities are simple and extensible |
| VII. Multi-Domain Algorithm Support | ✅ PASS | Genome distance protocol generalizes across vector, sequence, graph, SCM representations |
| VIII. Observability and Experiment Tracking | ✅ PASS | This is the primary beneficiary — richer metrics, meta-evolution tracking, native MLflow integration |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
specs/009-feature-cleanup-backlog/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
evolve/
├── core/
│   ├── population.py        # PopulationStatistics minimize flag, _compute_statistics fix
│   ├── engine.py             # Callback persistence, _compute_metrics, dynamics metrics
│   ├── callbacks.py          # Priority field on Callback protocol/SimpleCallback
│   └── types.py              # Genome protocol distance() method (if added here)
├── representation/
│   ├── genome.py             # Genome protocol: add distance() method
│   ├── vector.py             # VectorGenome.distance() — L2
│   └── sequence.py           # SequenceGenome.distance() — edit distance
├── config/
│   ├── unified.py            # training_data, validation_data, DatasetConfig wrapper
│   └── tracking.py           # New MetricCategory entries (EXTENDED_POPULATION, DIVERSITY already exist)
├── meta/
│   └── evaluator.py          # MLflow nested runs for inner trials
├── experiment/
│   └── tracking/
│       └── callback.py       # Tags → mlflow.set_tags(), datasets → mlflow.log_input()
└── factory/
    └── engine.py             # TrackingCallback priority=1000, callback merge logic
tests/
├── unit/
│   ├── test_population_statistics.py  # minimize-aware stats
│   ├── test_engine_callbacks.py       # callback persistence + priority ordering
│   ├── test_engine_metrics.py         # fitness distribution, diversity, movement metrics
│   ├── test_genome_distance.py        # distance protocol for each genome type
│   └── test_unified_config.py         # dataset/tag fields
└── integration/
    ├── test_meta_mlflow.py            # meta-evolution MLflow hierarchy
    └── test_tracking_callback.py      # native Tags + Datasets logging
```

**Structure Decision**: All changes are within the existing `evolve/` package structure. No new top-level directories. New test files follow existing `tests/unit/` and `tests/integration/` conventions.
