# Tasks: Evolvable Reproduction Protocols (ERP)

**Input**: Design documents from `/specs/002-evolvable-reproduction/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1-US6) this task belongs to

## Path Conventions

Based on plan.md structure:
- **Source**: `evolve/reproduction/` (new module), `evolve/core/`, `evolve/representation/`
- **Tests**: `tests/unit/reproduction/`, `tests/integration/`, `tests/property/`

---

## Phase 1: Setup

**Purpose**: Create module structure and base dependencies

- [X] T001 Create `evolve/reproduction/` directory structure with `__init__.py`
- [X] T002 [P] Create `evolve/reproduction/protocol.py` with ReproductionProtocol dataclass from contracts
- [X] T003 [P] Create `tests/unit/reproduction/` directory with `__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement StepCounter and StepLimitExceeded in `evolve/reproduction/sandbox.py`
- [X] T005 [P] Implement MateContext dataclass in `evolve/reproduction/protocol.py`
- [X] T006 [P] Implement IntentContext dataclass in `evolve/reproduction/protocol.py`
- [X] T007 [P] Implement CrossoverType enum in `evolve/reproduction/protocol.py`
- [X] T008 Implement MatchabilityFunction dataclass with to_dict/from_dict in `evolve/reproduction/protocol.py`
- [X] T009 Implement ReproductionIntentPolicy dataclass with to_dict/from_dict in `evolve/reproduction/protocol.py`
- [X] T010 Implement CrossoverProtocolSpec dataclass with to_dict/from_dict in `evolve/reproduction/protocol.py`
- [X] T011 Implement full ReproductionProtocol with default() factory in `evolve/reproduction/protocol.py`
- [X] T012 Extend Individual in `evolve/core/types.py` to include optional `protocol: ReproductionProtocol | None` field
- [X] T013 [P] Create unit tests for all protocol dataclasses in `tests/unit/reproduction/test_protocol.py`

**Checkpoint**: Foundation complete - user story implementation can begin

---

## Phase 3: User Story 1 - Evolvable Mating Compatibility (Priority: P1) 🎯 MVP

**Goal**: Individuals encode matchability functions that determine mate acceptability

**Independent Test**: Create population with different matchability functions, verify mating pairs determined by individual compatibility

### Implementation for User Story 1

- [X] T014 [US1] Create MatchabilityEvaluator protocol interface in `evolve/reproduction/matchability.py`
- [X] T015 [P] [US1] Implement AcceptAllMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T016 [P] [US1] Implement RejectAllMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T017 [P] [US1] Implement DistanceThresholdMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T018 [P] [US1] Implement SimilarityThresholdMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T019 [P] [US1] Implement FitnessRatioMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T020 [P] [US1] Implement DifferentNicheMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T021 [P] [US1] Implement ProbabilisticMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T022 [US1] Create MatchabilityRegistry to map type strings to evaluators in `evolve/reproduction/matchability.py`
- [X] T023 [US1] Implement evaluate_matchability() function with sandboxed execution in `evolve/reproduction/matchability.py`
- [X] T024 [US1] Create unit tests for all matchability evaluators in `tests/unit/reproduction/test_matchability.py`
- [X] T025 [US1] Create integration test verifying asymmetric matchability in `tests/integration/test_erp_basic.py`

**Checkpoint**: US1 complete - matchability functions work independently

---

## Phase 4: User Story 4 - System Stability (Priority: P1) 🎯 MVP

**Goal**: System remains stable under adversarial/degenerate protocols

**Independent Test**: Inject malformed protocols, verify system continues with fallbacks

### Implementation for User Story 4

- [X] T026 [US4] Implement sandboxed_execute() wrapper in `evolve/reproduction/sandbox.py`
- [X] T027 [US4] Add step limit enforcement to matchability evaluation in `evolve/reproduction/matchability.py`
- [X] T028 [US4] Implement safe_evaluate_matchability() with try/except fallback in `evolve/reproduction/matchability.py`
- [X] T029 [US4] Implement offspring validation in `evolve/reproduction/crossover_protocol.py`
- [X] T030 [US4] Create unit tests for step limit enforcement in `tests/unit/reproduction/test_sandbox.py`
- [X] T031 [US4] Create stability test with adversarial protocols in `tests/integration/test_erp_stability.py`

**Checkpoint**: US4 complete - system is robust against bad protocols

---

## Phase 5: User Story 2 - Evolve Crossover Strategies (Priority: P2)

**Goal**: Offspring construction methods are encoded and evolve

**Independent Test**: Run evolution with different crossover protocols, verify inheritance and mutation

### Implementation for User Story 2

- [X] T032 [US2] Create CrossoverExecutor protocol interface in `evolve/reproduction/crossover_protocol.py`
- [X] T033 [P] [US2] Implement SinglePointCrossoverExecutor in `evolve/reproduction/crossover_protocol.py`
- [X] T034 [P] [US2] Implement TwoPointCrossoverExecutor in `evolve/reproduction/crossover_protocol.py`
- [X] T035 [P] [US2] Implement UniformCrossoverExecutor in `evolve/reproduction/crossover_protocol.py`
- [X] T036 [P] [US2] Implement BlendCrossoverExecutor (BLX-alpha) in `evolve/reproduction/crossover_protocol.py`
- [X] T037 [P] [US2] Implement CloneCrossoverExecutor (no-op fallback) in `evolve/reproduction/crossover_protocol.py`
- [X] T038 [US2] Create CrossoverRegistry to map types to executors in `evolve/reproduction/crossover_protocol.py`
- [X] T039 [US2] Implement execute_crossover() with sandboxed execution in `evolve/reproduction/crossover_protocol.py`
- [X] T040 [US2] Implement crossover protocol inheritance (50/50 single-parent) in `evolve/reproduction/crossover_protocol.py`
- [X] T041 [US2] Create unit tests for all crossover executors in `tests/unit/reproduction/test_crossover_protocol.py`
- [X] T042 [US2] Create integration test for protocol inheritance in `tests/integration/test_erp_basic.py`

**Checkpoint**: US2 complete - crossover protocols work and inherit

---

## Phase 6: User Story 3 - Reproduction Intent Policies (Priority: P2)

**Goal**: Individuals encode when they attempt reproduction

**Independent Test**: Create individuals with different intent policies, verify intent evaluated before matchability

### Implementation for User Story 3

- [X] T043 [US3] Create IntentEvaluator protocol interface in `evolve/reproduction/intent.py`
- [X] T044 [P] [US3] Implement AlwaysWillingIntent evaluator in `evolve/reproduction/intent.py`
- [X] T045 [P] [US3] Implement NeverWillingIntent evaluator in `evolve/reproduction/intent.py`
- [X] T046 [P] [US3] Implement FitnessThresholdIntent evaluator in `evolve/reproduction/intent.py`
- [X] T047 [P] [US3] Implement FitnessRankThresholdIntent evaluator in `evolve/reproduction/intent.py`
- [X] T048 [P] [US3] Implement ResourceBudgetIntent evaluator (with state tracking) in `evolve/reproduction/intent.py`
- [X] T049 [P] [US3] Implement AgeDependentIntent evaluator in `evolve/reproduction/intent.py`
- [X] T050 [P] [US3] Implement ProbabilisticIntent evaluator in `evolve/reproduction/intent.py`
- [X] T051 [US3] Create IntentRegistry to map type strings to evaluators in `evolve/reproduction/intent.py`
- [X] T052 [US3] Implement evaluate_intent() function with sandboxed execution in `evolve/reproduction/intent.py`
- [X] T053 [US3] Create unit tests for all intent evaluators in `tests/unit/reproduction/test_intent.py`
- [X] T054 [US3] Create integration test verifying intent before matchability in `tests/integration/test_erp_basic.py`

**Checkpoint**: US3 complete - intent policies work independently

---

## Phase 7: User Story 6 - Multi-Objective Integration (Priority: P2)

**Goal**: ERP works with NSGA-II, Pareto ranking, crowding distance

**Independent Test**: Run NSGA-II with ERP enabled, verify selection operates on Pareto fronts while ERP governs mating

### Implementation for User Story 6

- [X] T055 [US6] Add crowding_distance field to MateContext in `evolve/reproduction/protocol.py`
- [X] T056 [US6] Implement DiversitySeekingMatchability evaluator in `evolve/reproduction/matchability.py`
- [X] T057 [US6] Ensure ERPEngine respects selection authority over survival in `evolve/reproduction/engine.py`
- [X] T058 [US6] Create integration test for ERP + NSGA-II in `tests/integration/test_erp_nsga2.py`

**Checkpoint**: US6 complete - multi-objective evolution works with ERP

---

## Phase 8: User Story 5 - Neutral Drift via Junk Code (Priority: P3)

**Goal**: Protocol genomes support inactive logic and dormant strategies

**Independent Test**: Create protocols with inactive regions, verify mutations can activate dormant logic

### Implementation for User Story 5

- [X] T059 [US5] Ensure junk_data field is preserved through copy/serialize in `evolve/reproduction/protocol.py`
- [X] T060 [US5] Implement ProtocolMutator that can activate dormant parameters in `evolve/reproduction/mutation.py`
- [X] T061 [US5] Create mutation operators for protocol parameters in `evolve/reproduction/mutation.py`
- [X] T062 [US5] Implement junk_data mutation (add/remove/modify dormant params) in `evolve/reproduction/mutation.py`
- [X] T063 [US5] Create unit tests for protocol mutation in `tests/unit/reproduction/test_mutation.py`
- [X] T064 [US5] Create integration test for dormant logic activation in `tests/integration/test_erp_basic.py`

**Checkpoint**: US5 complete - junk code and neutral drift work

---

## Phase 9: Engine Integration

**Purpose**: Integrate all components into ERPEngine

- [X] T065 Implement ReproductionEvent dataclass for observability in `evolve/reproduction/protocol.py`
- [X] T066 Implement ImmigrationRecovery strategy in `evolve/reproduction/recovery.py`
- [X] T067 Create ERPConfig extending EvolutionConfig in `evolve/reproduction/engine.py`
- [X] T068 Implement ERPEngine._attempt_mating() with intent+matchability checks in `evolve/reproduction/engine.py`
- [X] T069 Implement ERPEngine._step() overriding base reproduction logic in `evolve/reproduction/engine.py`
- [X] T070 Implement protocol inheritance in offspring creation in `evolve/reproduction/engine.py`
- [X] T071 Implement recovery mechanism when zero matings occur in `evolve/reproduction/engine.py`
- [X] T072 Add ReproductionEvent emission for observability in `evolve/reproduction/engine.py`
- [X] T073 Create unit tests for recovery mechanism in `tests/unit/reproduction/test_recovery.py`
- [X] T074 Create property test for determinism in `tests/property/test_erp_determinism.py`

**Checkpoint**: Full ERP engine operational

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, exports, final validation

- [X] T075 [P] Export all public classes from `evolve/reproduction/__init__.py`
- [X] T076 [P] Add docstrings to all public functions and classes
- [X] T077 [P] Create ERPMetricsCallback for tracking protocol statistics in `evolve/reproduction/callbacks.py`
- [X] T078 Update `evolve/__init__.py` to include reproduction module
- [X] T079 Run full test suite and verify SC-004 (10,000+ generations stability)
- [X] T080 Run performance benchmark and verify SC-005 (<20% overhead)

**Checkpoint**: Feature complete and validated

---

## Phase 11: Extensibility Verification (C1 Fix)

**Purpose**: Verify FR-026, FR-027, FR-028 extensibility requirements are met by interface design

- [X] T081 [P] Verify MatchabilityEvaluator interface supports RL-trained policies (FR-026) - document in ADR
- [X] T082 [P] Verify CrossoverExecutor interface supports memetic local search post-crossover (FR-027) - document in ADR
- [X] T083 [P] Verify ReproductionProtocol serialization supports protocol-aware migration (FR-028) - document in ADR
- [X] T084 Create `docs/adr/002-erp-extensibility.md` documenting extension points for future RL/memetic/island features

**Checkpoint**: Extensibility requirements verified via interface audit

---

## Dependencies Graph

```
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundation) ──────────────────────────────────────┐
    │                                                       │
    ├──────────────┬──────────────┬──────────────┐         │
    ▼              ▼              ▼              ▼         │
Phase 3 (US1)  Phase 4 (US4)  Phase 5 (US2)  Phase 6 (US3) │
[Matchability] [Stability]   [Crossover]    [Intent]       │
    │              │              │              │         │
    └──────────────┴──────────────┴──────────────┘         │
                        │                                   │
                        ▼                                   │
              Phase 7 (US6) [Multi-Obj]                    │
                        │                                   │
                        ▼                                   │
              Phase 8 (US5) [Junk Code]                    │
                        │                                   │
                        ▼                                   │
              Phase 9 (Engine Integration) ◄───────────────┘
                        │
                        ▼
              Phase 10 (Polish)
```

## Parallel Execution Opportunities

**Within Phase 2**: T005, T006, T007 can run in parallel  
**Within Phase 3**: T015-T021 (all matchability evaluators) can run in parallel  
**Within Phase 5**: T033-T037 (all crossover executors) can run in parallel  
**Within Phase 6**: T044-T050 (all intent evaluators) can run in parallel  
**Within Phase 10**: T075, T076, T077 can run in parallel

## Summary

| Metric | Value |
|--------|-------|
| Total tasks | 84 |
| Setup tasks | 3 |
| Foundational tasks | 10 |
| US1 (P1 Matchability) tasks | 12 |
| US4 (P1 Stability) tasks | 6 |
| US2 (P2 Crossover) tasks | 11 |
| US3 (P2 Intent) tasks | 12 |
| US6 (P2 Multi-Obj) tasks | 4 |
| US5 (P3 Junk Code) tasks | 6 |
| Engine Integration tasks | 10 |
| Polish tasks | 6 |
| Extensibility Verification tasks | 4 |
| Parallelizable tasks | 38 |

**MVP Scope**: Phases 1-4 (Setup + Foundation + US1 + US4) = 31 tasks
