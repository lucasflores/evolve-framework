# Implementation Plan: Dry-Run Statistics Tool

**Branch**: `014-dry-run-statistics` | **Date**: 2026-04-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/014-dry-run-statistics/spec.md`

## Summary

Provide a standalone `dry_run(config: UnifiedConfig) → DryRunReport` function that micro-benchmarks a single invocation of each atomic evolutionary operation through the real configured backend, then multiplies by the structural constants from config to produce a granular cost estimate. The tool covers the **full configured pipeline**: core phases (evaluation, selection, variation, merge), optional subsystems (ERP intent/matchability, NSGA-II ranking/crowding, genome decoding, tracking overhead), and meta-evolution (outer loop × inner run cost). The report includes per-phase timing breakdown with percentages, bottleneck identification, auto-detected compute resources, estimated memory usage, active subsystem inventory, caveats about unbenchmarkable overheads (e.g., remote MLflow latency, ERP recovery), and a formatted ASCII table summary.

## Technical Context

**Language/Version**: Python 3.10+, full type hints (mypy strict)  
**Primary Dependencies**: NumPy (numeric ops), stdlib for resource detection; no new external deps in core  
**Storage**: N/A (output is in-memory dataclass; no persistence)  
**Testing**: pytest; TDD (Red-Green-Refactor); tests in `tests/` mirroring `evolve/` structure  
**Target Platform**: Linux, macOS (same platforms as evolve-framework)  
**Project Type**: Library (new module within existing `evolve/` package)  
**Performance Goals**: Dry-run completes in <10 seconds for any config  
**Constraints**: Must not import ML framework dependencies (JAX/PyTorch) unconditionally; must use the same backend the actual run would use  
**Scale/Scope**: Single-function entry point + 5 frozen dataclasses (DryRunReport, PhaseEstimate, ComputeResources, MemoryEstimate, MetaEstimate); ~350-450 LOC implementation. Additional phases benchmarked conditionally: ERP (intent, matchability), NSGA-II (sort, crowding), decoder, tracking.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | ✅ PASS | Tool operates on abstract `Evaluator` protocol and generic operators; no hard deps on NN/JAX/PyTorch |
| II. Separation of Concerns | ✅ PASS | Dry-run is a standalone utility; does not modify engine, evaluator, or operator code |
| III. Declarative Completeness | ✅ PASS | Takes `UnifiedConfig` as sole input; resolves all components via existing factory/registry infrastructure |
| IV. Acceleration as Optional | ✅ PASS | Benchmarks through whatever backend is configured; no new acceleration deps |
| V. Determinism and Reproducibility | ✅ PASS | Uses explicit seed for sample genome creation; timing measurements are inherently non-deterministic but documented as such |
| VI. Extensibility Over Premature Optimization | ✅ PASS | Simple micro-benchmark approach; no premature optimization |
| VII. Multi-Domain Algorithm Support | ✅ PASS | Works with any genome type via registry; not domain-specific |
| VIII. Observability and Experiment Tracking | ✅ PASS | The tool itself is an observability feature; report is structured and inspectable |

**GATE RESULT**: ✅ All principles satisfied. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/014-dry-run-statistics/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (public API contract)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
evolve/
└── experiment/
    └── dry_run.py           # DryRunReport, PhaseEstimate, ComputeResources, MemoryEstimate, MetaEstimate, dry_run()

tests/
└── unit/
    └── experiment/
        └── test_dry_run.py  # Unit tests for dry-run module
```

**Structure Decision**: The dry-run tool lives in `evolve/experiment/` because it is an experiment-level utility (like `runner.py` and `metrics.py`). It is not a core engine component, not a factory, and not an evaluator — it consumes all of those to produce a pre-run estimate. Single file is sufficient for ~400 LOC. Tests mirror the structure under `tests/unit/experiment/`.

## Complexity Tracking

> No constitution violations. Table not needed.
