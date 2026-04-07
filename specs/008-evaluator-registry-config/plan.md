# Implementation Plan: Evaluator Registry & UnifiedConfig Declarative Completeness

**Branch**: `008-evaluator-registry-config` | **Date**: 2026-04-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/008-evaluator-registry-config/spec.md`

## Summary

Extend `UnifiedConfig` and the factory system so that evaluators and callbacks are first-class declarative components — specified by name and parameters in config, resolved through registries at factory time. Add `EvaluatorRegistry`, `CallbackRegistry`, new config fields (`evaluator`, `evaluator_params`, `custom_callbacks`), `runtime_overrides` on `create_engine`, backward-compatible hash updates, genome_params validation via signature introspection, and full serialization roundtrip support.

## Technical Context

**Language/Version**: Python >=3.10 (supports 3.10, 3.11, 3.12)  
**Primary Dependencies**: numpy>=1.24.0, networkx>=3.0, typing_extensions>=4.0.0  
**Storage**: N/A (in-memory registries, JSON file serialization)  
**Testing**: pytest>=7.0.0, pytest-cov>=4.0.0, hypothesis>=6.0.0  
**Target Platform**: Cross-platform (Linux, macOS, Windows)  
**Project Type**: Library  
**Performance Goals**: Registry lookup < 1μs (dict access); no measurable overhead on engine creation  
**Constraints**: Frozen dataclass for `UnifiedConfig`; no top-level ML imports in core; backward-compatible `compute_hash()`  
**Scale/Scope**: ~7 built-in evaluators, ~4 built-in callbacks; extensible via user registration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | **PASS** | Evaluator factories with ML deps use deferred imports. Core registry has no ML dependencies. |
| II. Separation of Concerns | **PASS** | Registries are independent of evolutionary logic. Evaluator resolution is factory concern, not engine concern. |
| III. Declarative Completeness | **PRIMARY DRIVER** | This feature directly implements Principle III. |
| IV. Acceleration as Optional | **PASS** | No acceleration involved. |
| V. Determinism and Reproducibility | **PASS** | `compute_hash()` update ensures hash reflects full experiment. Registries are deterministic. |
| VI. Extensibility Over Premature Optimization | **PASS** | Registries provide extension points. No abstract base registry class (would be premature). |
| VII. Multi-Domain Algorithm Support | **PASS** | Built-in registrations span domains: benchmark, SCM, RL, LLM, meta. |
| VIII. Observability and Experiment Tracking | **PASS** | Hash completeness enables proper experiment tracking. |
| Clear Abstractions | **PASS** | Full type annotations on all new public APIs. |
| Composable Components | **PASS** | Registries are singletons but resettable for testing. No global mutable state in config. |
| Test-First Development | **ENFORCE** | Tests written alongside implementation per dev-stack instructions. |

**Post-Phase 1 Re-check**: All gates still PASS. No violations introduced by data model or contracts.

## Project Structure

### Documentation (this feature)

```text
specs/008-evaluator-registry-config/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 entity definitions
├── quickstart.md        # Phase 1 usage examples
├── contracts/
│   └── public-api.md    # Phase 1 API contracts
├── checklists/
│   └── requirements.md  # Quality checklist
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
evolve/
├── registry/
│   ├── __init__.py          # MODIFY: export new registries
│   ├── operators.py         # REFERENCE ONLY (pattern to follow)
│   ├── genomes.py           # MODIFY: add signature introspection validation
│   ├── evaluators.py        # NEW: EvaluatorRegistry
│   └── callbacks.py         # NEW: CallbackRegistry
├── config/
│   ├── unified.py           # MODIFY: add evaluator/evaluator_params/custom_callbacks fields
│   └── callbacks.py         # REFERENCE ONLY
├── factory/
│   └── engine.py            # MODIFY: evaluator resolution, runtime_overrides, callback merge
└── evaluation/
    └── reference/
        └── functions.py     # REFERENCE ONLY (BENCHMARK_FUNCTIONS dict)

tests/
├── unit/
│   ├── test_evaluator_registry.py   # NEW
│   ├── test_callback_registry.py    # NEW
│   ├── test_unified_config_ext.py   # NEW (new fields, hash, serialization)
│   └── test_genome_params_validation.py  # NEW
└── integration/
    └── test_declarative_engine.py   # NEW (end-to-end create_engine from config)
```

**Structure Decision**: Single-project library layout using existing `evolve/` package structure. New registry modules placed alongside existing registries. Tests mirror the source layout under `tests/`.

## Complexity Tracking

No constitution violations to justify. All gates pass.

