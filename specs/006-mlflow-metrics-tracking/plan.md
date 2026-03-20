# Implementation Plan: MLflow Metrics Tracking Integration

**Branch**: `006-mlflow-metrics-tracking` | **Date**: March 17, 2026 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/006-mlflow-metrics-tracking/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Extend the evolve framework's experiment tracking to capture comprehensive observable metrics through MLflow integration. This bridges the gap between `UnifiedConfig`/`create_engine()` and the existing `ExperimentRunner` tracking infrastructure by:

1. Adding a `TrackingConfig` dataclass to `UnifiedConfig` for declarative tracking configuration
2. Extending `compute_generation_metrics()` with enhanced population statistics
3. Creating specialized `MetricCollector` protocols for ERP, multi-objective, speciation, islands, and NEAT
4. Adding timing instrumentation and derived analytics
5. Enabling automatic fitness metadata extraction

## Technical Context

**Language/Version**: Python 3.10+ (matches existing framework requirements in pyproject.toml)  
**Primary Dependencies**: 
- MLflow 2.0+ (optional, already defined in pyproject.toml `[project.optional-dependencies]`)
- NumPy >=1.24.0 (existing core dependency)
- NetworkX >=3.0 (existing core dependency for graph/NEAT metrics)

**Storage**: MLflow tracking server (remote) or local filesystem artifact store  
**Testing**: pytest (existing test framework in pyproject.toml)  
**Target Platform**: Cross-platform Python (Linux/macOS/Windows)  
**Project Type**: Single library project (Python package)  
**Performance Goals**: 
- Timing overhead MUST be <1% of total generation time for populations under 1,000 individuals (FR-012)
- Sampling-based diversity computation for populations >10,000 (edge case spec)

**Constraints**:
- No hard dependencies on MLflow in core modules (Constitution: Model-Agnostic Architecture)
- Graceful degradation when MLflow server is unreachable (FR-028)
- All enhanced metrics MUST be opt-in (FR-025/FR-026)

**Scale/Scope**: 
- Population sizes: 100 to 100,000+ individuals
- Generations: 100 to 10,000+
- Metrics per generation: ~10 (basic) to ~50+ (all enhanced metrics enabled)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: MLflow is an optional dependency in `[project.optional-dependencies]`. Core tracking protocols (`MetricTracker`, `MetricCollector`) have no MLflow imports. Import guards with clear `ImportError` messages at engine creation time (FR-005).
- [x] **Separation of Concerns**: Metrics collection is decoupled from evaluation logic. `MetricCollector` protocol separates specialized collectors (ERP, multi-objective, speciation) from core tracking. Engine orchestrates timing but doesn't perform fitness computation.
- [x] **Optional Acceleration**: No GPU/JIT features required. Diversity metrics use NumPy CPU implementations with sampling for large populations.
- [x] **Determinism**: Metrics computation is deterministic given population state. Sampling-based diversity uses seeded RNG from engine.
- [x] **Extensibility**: `MetricCollector` protocol enables third-party collectors. `TrackingConfig` is JSON-serializable (FR-027) for configuration experimentation.
- [x] **Multi-Domain Support**: Specialized collectors for ERP (reproduction), multi-objective (Pareto), speciation (diversity), islands (parallel), NEAT (neuroevolution). Core metrics work across all domains.
- [x] **Observability**: This feature IS the observability enhancement. Structured logging via MLflow, configurable metric categories, timing instrumentation.
- [x] **Clear Abstractions**: `TrackingConfig` dataclass with explicit fields. `MetricCollector` Protocol with typed `collect()` method. All public APIs have type hints.
- [x] **Composability**: Collectors are independently instantiable. `CompositeTracker` patterns already established. No global state.
- [x] **Test-First**: Unit tests for each metric collector, integration tests for `UnifiedConfig` + tracking path.

**Violations requiring justification**: None identified. Design aligns with all constitutional principles.

## Project Structure

### Documentation (this feature)

```text
specs/006-mlflow-metrics-tracking/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output - MLflow best practices, metric design
├── data-model.md        # Phase 1 output - TrackingConfig, MetricCollector schemas
├── quickstart.md        # Phase 1 output - Usage examples for tracking
├── contracts/           # Phase 1 output - API contracts
│   ├── tracking-config.yaml      # TrackingConfig schema
│   ├── metric-collector.yaml     # MetricCollector protocol
│   └── enhanced-metrics.yaml     # EnhancedGenerationMetrics schema
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
evolve/
├── config/
│   ├── unified.py           # MODIFY: Add tracking: TrackingConfig field
│   └── tracking.py          # NEW: TrackingConfig dataclass
├── experiment/
│   ├── metrics.py           # MODIFY: Extend compute_generation_metrics()
│   ├── collectors/          # NEW: Specialized metric collectors
│   │   ├── __init__.py
│   │   ├── base.py          # MetricCollector protocol
│   │   ├── erp.py           # ERPMetricCollector
│   │   ├── multiobjective.py # MultiObjectiveMetricCollector
│   │   ├── speciation.py    # SpeciationMetricCollector
│   │   ├── islands.py       # IslandsMetricCollector
│   │   ├── neat.py          # NEATMetricCollector
│   │   ├── metadata.py      # FitnessMetadataCollector
│   │   └── derived.py       # DerivedAnalyticsCollector
│   └── tracking/
│       └── mlflow_tracker.py # MODIFY: Add batch logging, reconnection
├── factory/
│   └── engine.py            # MODIFY: Wire tracking from UnifiedConfig
└── utils/
    └── timing.py            # NEW: TimingContext manager

tests/
├── unit/
│   └── experiment/
│       └── collectors/      # NEW: Tests for metric collectors
└── integration/
    └── test_tracking.py     # NEW: End-to-end tracking tests
```

**Structure Decision**: Single library project. New `collectors/` module under `experiment/` for specialized metric collection. `TrackingConfig` added to `config/` module to match existing config pattern (`StoppingConfig`, `CallbackConfig`, etc.).

## Complexity Tracking

> **No Constitution Check violations identified.**

| Decision | Rationale | Alternative Considered |
|----------|-----------|------------------------|
| Separate `collectors/` module | Specialized collectors (7+) are cohesive and benefit from shared base protocol | Inline in `metrics.py` - rejected due to file size and maintainability |
| MLflow 2.0+ only | MLflow 2.0 provides batch `log_metrics()` and system metrics; 1.x compat unnecessary for new feature | MLflow 1.x compatibility layer - rejected per FR-029 |
| Sampling for large populations | Required for <1% timing overhead with >10k individuals | Full diversity computation - rejected due to O(n²) complexity |

---

## Phase 0: Research Complete

See [research.md](research.md) for:
- MLflow 2.0 API best practices for batch logging
- Graceful degradation patterns for unreachable servers
- Metric aggregation strategies for evolutionary algorithms
- Performance benchmarks for diversity computation

## Phase 1: Design Complete

See:
- [data-model.md](data-model.md) - TrackingConfig, MetricCollector, EnhancedGenerationMetrics schemas
- [contracts/](contracts/) - API contracts in OpenAPI/YAML format
- [quickstart.md](quickstart.md) - Usage examples for declarative tracking
