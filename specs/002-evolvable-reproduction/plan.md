# Implementation Plan: Evolvable Reproduction Protocols (ERP)

**Branch**: `002-evolvable-reproduction` | **Date**: January 28, 2026 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-evolvable-reproduction/spec.md`

## Summary

Enable individuals to encode, evolve, and execute their own reproductive compatibility logic and offspring construction strategies. This introduces a Reproduction Protocol Genome (RPG) that governs matchability functions, reproduction intent policies, and crossover protocols - all evolvable and heritable. The system maintains stability through sandboxed execution with step limits (1,000 steps default) and automatic recovery via immigration when reproduction fails.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: NumPy (existing), dataclasses (stdlib)  
**Storage**: N/A (in-memory, uses existing checkpoint infrastructure)  
**Testing**: pytest (existing test infrastructure)  
**Target Platform**: Cross-platform (Linux, macOS, Windows)  
**Project Type**: Single library (evolve/)  
**Performance Goals**: Protocol evaluation overhead less than 20% of reproduction phase time  
**Constraints**: 1,000 step limit per protocol execution; no global mutable state access  
**Scale/Scope**: Support 10,000+ generations with adversarial protocols

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: RPG uses abstract genome representations; no ML framework dependencies
- [x] **Separation of Concerns**: Reproduction protocols are decoupled from evaluation, selection, and execution
- [x] **Optional Acceleration**: CPU reference implementation first; JIT optimization deferred to future work
- [x] **Determinism**: All protocol execution uses explicit RNG; reproducible from seed
- [x] **Extensibility**: Protocol types defined via Protocol interfaces; new crossover types require single interface implementation
- [x] **Multi-Domain Support**: RPG works with any genome type (vector, graph, sequence)
- [x] **Observability**: Protocol execution emits ReproductionEvent; metrics for acceptance rates, reproduction success
- [x] **Clear Abstractions**: Type annotations throughout; explicit interface protocols (MatchabilityEvaluator, IntentEvaluator, CrossoverExecutor)
- [x] **Composability**: No global state; protocols are independently testable; StepCounter for resource limiting
- [x] **Test-First**: Tests written before implementation; reference implementations for all protocol types

**Post-Design Re-Check (Phase 1 Complete)**:
- [x] data-model.md entities are model-agnostic (no PyTorch/JAX types)
- [x] contracts/protocols.py uses Python Protocol for interfaces (not concrete classes)
- [x] MateContext and IntentContext are immutable (frozen dataclasses)
- [x] StepCounter provides explicit resource limiting
- [x] ReproductionProtocol.to_dict()/from_dict() enable serialization for checkpointing

**Violations requiring justification**: None

## Project Structure

### Documentation (this feature)

```text
specs/002-evolvable-reproduction/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── protocols.py     # Protocol interface contracts
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
evolve/
├── reproduction/                    # NEW MODULE
│   ├── __init__.py
│   ├── protocol.py                  # ReproductionProtocol, RPG dataclass
│   ├── matchability.py              # MatchabilityFunction protocol + implementations
│   ├── intent.py                    # ReproductionIntentPolicy protocol + implementations
│   ├── crossover_protocol.py        # CrossoverProtocol wrapper + inheritance logic
│   ├── sandbox.py                   # Sandboxed execution with step limits
│   ├── recovery.py                  # Immigration/recovery when reproduction fails
│   ├── mutation.py                  # Protocol mutation operators (I3 fix)
│   └── engine.py                    # ERPEngine (extends EvolutionEngine)
├── core/
│   └── types.py                     # Extended Individual with optional RPG
└── representation/
    └── (no new files)               # RPG representation lives in reproduction/protocol.py

tests/
├── unit/
│   └── reproduction/                # NEW
│       ├── test_matchability.py
│       ├── test_intent.py
│       ├── test_crossover_protocol.py
│       ├── test_sandbox.py
│       └── test_recovery.py
├── integration/
│   ├── test_erp_basic.py            # NEW
│   ├── test_erp_stability.py        # NEW (adversarial protocols)
│   └── test_erp_nsga2.py            # NEW (multi-objective integration)
└── property/
    └── test_erp_determinism.py      # NEW
```

**Structure Decision**: New `evolve/reproduction/` module to keep ERP functionality isolated from core engine. Core types extended minimally to support optional RPG attachment.

## Complexity Tracking

No constitution violations requiring justification.
