# Implementation Plan: Unified Configuration & Meta-Evolution Framework

**Branch**: `005-unified-config-meta-evolution` | **Date**: March 12, 2026 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-unified-config-meta-evolution/spec.md`

## Summary

Create a unified, JSON-serializable configuration system that can define and spawn any experiment type (standard evolution, ERP, multi-objective, or meta-evolution) across all supported genome representations. The system includes operator and genome registries for name-based resolution, a one-line engine factory, and meta-evolution infrastructure for hyperparameter optimization.

## Technical Context

**Language/Version**: Python 3.10+ (supports 3.10, 3.11, 3.12)  
**Primary Dependencies**: numpy>=1.24.0, networkx>=3.0, typing_extensions (for <3.11)  
**Storage**: JSON file serialization; no database required  
**Testing**: pytest with hypothesis for property-based testing; mypy strict mode  
**Target Platform**: All platforms supporting Python 3.10+  
**Project Type**: Single library package with optional accelerated backends  
**Performance Goals**: Configuration loading <10ms; registry lookup O(1)  
**Constraints**: No breaking changes to existing APIs; CPU reference implementations required  
**Scale/Scope**: Configuration schemas for experiments with populations up to 10,000

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: Configuration schema uses abstract operator names and genome type strings; no PyTorch/JAX types in config
- [x] **Separation of Concerns**: Configuration (declarative data) is strictly separated from engines (execution), registries (resolution), and evaluators (fitness logic)
- [x] **Optional Acceleration**: Unified config is backend-agnostic; accelerated backends remain opt-in via separate configuration
- [x] **Determinism**: Configuration includes explicit seed field; factory threads seed through all components; inner runs in meta-evolution deterministically seeded
- [x] **Extensibility**: Operator and genome registries allow runtime registration; no framework modification required for custom operators
- [x] **Multi-Domain Support**: Configuration schema supports all existing domains (classical EA via vector, neuroevolution via graph, causal discovery via SCM, multi-objective via objectives array)
- [x] **Observability**: Configuration includes callback settings; logging and checkpointing callbacks configurable; metrics captured during meta-evolution
- [x] **Clear Abstractions**: All new types have type annotations; UnifiedConfig, OperatorRegistry, GenomeRegistry have explicit interfaces
- [x] **Composability**: Registries are explicitly instantiated singletons (not global state); configuration is immutable after creation
- [x] **Test-First**: Requirements specify testable acceptance scenarios; reference implementations will serve as executable specifications

**Violations requiring justification** (if any): None - design complies with all constitution principles.

## Project Structure

### Documentation (this feature)

```text
specs/005-unified-config-meta-evolution/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── unified_config.py    # Configuration schema
│   ├── registries.py        # Registry interfaces
│   └── meta_evolution.py    # Meta-evolution interfaces
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
evolve/
├── __init__.py
├── config/                      # NEW: Unified configuration module
│   ├── __init__.py
│   ├── unified.py               # UnifiedConfig class
│   ├── schema.py                # Schema versioning and validation
│   ├── stopping.py              # Stopping criteria configuration
│   └── callbacks.py             # Callback configuration
├── registry/                    # NEW: Operator and genome registries
│   ├── __init__.py
│   ├── operators.py             # OperatorRegistry
│   ├── genomes.py               # GenomeRegistry
│   └── builtin.py               # Built-in operator/genome registration
├── factory/                     # NEW: Engine factory
│   ├── __init__.py
│   └── engine.py                # create_engine() function
├── meta/                        # NEW: Meta-evolution module
│   ├── __init__.py
│   ├── codec.py                 # ConfigCodec
│   ├── evaluator.py             # MetaEvaluator
│   └── result.py                # MetaEvolutionResult
├── core/                        # EXISTING: Core evolution
│   ├── engine.py                # EvolutionEngine, EvolutionConfig (unchanged)
│   └── operators/               # Existing operators (unchanged)
├── reproduction/                # EXISTING: ERP engine
│   └── engine.py                # ERPEngine, ERPConfig (unchanged)
├── representation/              # EXISTING: Genome types (unchanged)
├── multiobjective/              # EXISTING: NSGA-II (unchanged)
└── experiment/                  # EXISTING: ExperimentConfig (unchanged)

tests/
├── unit/
│   ├── config/                  # NEW: UnifiedConfig tests
│   ├── registry/                # NEW: Registry tests
│   ├── factory/                 # NEW: Factory tests
│   └── meta/                    # NEW: Meta-evolution tests
├── integration/
│   ├── test_config_to_engine.py # NEW: End-to-end config tests
│   └── test_meta_evolution.py   # NEW: Meta-evolution integration
└── contract/                    # NEW: Contract tests
```

**Structure Decision**: Adds four new modules (`config/`, `registry/`, `factory/`, `meta/`) to the existing `evolve/` package structure. All existing modules remain unchanged to maintain backward compatibility.

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts created.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **Model-Agnostic Architecture** | ✅ PASS | UnifiedConfig uses string-based operator references; no ML framework types in data model |
| **Separation of Concerns** | ✅ PASS | Config (data-model.md) cleanly separated from Registries (registries.py), Factory (factory.py), and Evaluators (meta_evolution.py) |
| **Optional Acceleration** | ✅ PASS | ConfigCodec uses numpy but only for vector bounds; rest is pure Python |
| **Determinism** | ✅ PASS | `_compute_inner_seed()` in MetaEvaluator uses hash-based deterministic seeding |
| **Extensibility** | ✅ PASS | Both registries have `register()` methods for custom operators/genomes |
| **Multi-Domain Support** | ✅ PASS | genome_type supports all four representations; multiobjective config added |
| **Observability** | ✅ PASS | CallbackConfig supports logging/checkpointing; MetaEvolutionResult tracks metrics |
| **Clear Abstractions** | ✅ PASS | All dataclasses have type annotations and docstrings in contracts |
| **Composability** | ✅ PASS | Frozen dataclasses ensure immutability; `reset_*_registry()` functions enable testing |
| **Test-First** | ✅ PASS | `__post_init__` validators in contracts encode business rules for property testing |

**Conclusion**: Design artifacts comply with all 10 constitution principles. Ready to proceed to Phase 2 (task generation).

## Complexity Tracking

> No constitution violations. All principles satisfied by design.
