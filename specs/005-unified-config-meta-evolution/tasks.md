# Tasks: Unified Configuration & Meta-Evolution Framework

**Input**: Design documents from `/specs/005-unified-config-meta-evolution/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1-US6) this task belongs to

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directory structure for new modules

- [X] T001 Create `evolve/config/` module directory with `__init__.py`
- [X] T002 [P] Create `evolve/registry/` module directory with `__init__.py`
- [X] T003 [P] Create `evolve/factory/` module directory with `__init__.py`
- [X] T004 [P] Create `evolve/meta/` module directory with `__init__.py`
- [X] T005 [P] Create test directories `tests/unit/config/`, `tests/unit/registry/`, `tests/unit/factory/`, `tests/unit/meta/`
- [X] T006 [P] Create `tests/integration/` directory for end-to-end tests

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core dataclasses that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 [FR-009→FR-013] Implement `StoppingConfig` frozen dataclass in `evolve/config/stopping.py` with validation for all stopping criteria
- [X] T008 [P] [FR-038→FR-042] Implement `CallbackConfig` frozen dataclass in `evolve/config/callbacks.py` with log level, output destination, checkpoint directory, and frequency fields
- [X] T009 [P] Implement `ERPSettings` frozen dataclass in `evolve/config/erp.py` with validation
- [X] T010 [P] Implement `ObjectiveSpec` and `ConstraintSpec` dataclasses in `evolve/config/multiobjective.py`
- [X] T011 [P] Implement `MultiObjectiveConfig` frozen dataclass in `evolve/config/multiobjective.py`
- [X] T012 [P] Implement `ParameterSpec` frozen dataclass in `evolve/config/meta.py` with validation
- [X] T013 Implement `MetaEvolutionConfig` frozen dataclass in `evolve/config/meta.py` (depends on T012)
- [X] T014a [FR-006] Implement schema version parsing and `SchemaVersion` dataclass in `evolve/config/schema.py`
- [X] T014b [FR-007] Implement older schema loading with deprecation warnings in `evolve/config/schema.py`
- [X] T014c [FR-008] Implement newer schema rejection with version mismatch error in `evolve/config/schema.py`

**Checkpoint**: Foundation complete - user story implementation can now begin

---

## Phase 3: User Story 1 + 2 - JSON Configuration & One-Line Engine (Priority: P1) 🎯 MVP

**Goal**: Researchers can define experiments in JSON and create engines with one function call

**Independent Test**: Create JSON config → load → call `create_engine()` → run one generation

### Implementation for User Stories 1 & 2

- [X] T015 [US1] Implement `UnifiedConfig` frozen dataclass in `evolve/config/unified.py` with all fields from data-model.md
- [X] T016 [US1] Implement `to_dict()` method for JSON serialization in `evolve/config/unified.py`
- [X] T017 [US1] Implement `from_dict()` classmethod for JSON deserialization in `evolve/config/unified.py`
- [X] T018 [US1] Implement `to_json()` and `from_json()` methods in `evolve/config/unified.py`
- [X] T019 [US1] Implement `compute_hash()` method for deterministic config hashing in `evolve/config/unified.py`
- [X] T020 [US1] Implement `with_params()` method for creating modified copies in `evolve/config/unified.py`
- [X] T021 [US1] Implement `__post_init__` validation in `UnifiedConfig`
- [X] T022 [P] [US2] Implement `OperatorRegistry` class in `evolve/registry/operators.py` with lazy initialization
- [X] T023 [P] [US2] Implement `GenomeRegistry` class in `evolve/registry/genomes.py` with lazy initialization
- [X] T024 [US2] Implement `_register_builtin_operators()` in `evolve/registry/operators.py` for selection operators
- [X] T025 [US2] Extend `_register_builtin_operators()` to register all crossover operators
- [X] T026 [US2] Extend `_register_builtin_operators()` to register all mutation operators
- [X] T027 [US2] Implement `_register_builtin_genomes()` in `evolve/registry/genomes.py` for vector, sequence, graph, scm
- [X] T028 [US2] Implement `get_operator_registry()` singleton accessor in `evolve/registry/operators.py`
- [X] T029 [US2] Implement `get_genome_registry()` singleton accessor in `evolve/registry/genomes.py`
- [X] T030 [US2] Implement `create_engine()` factory function in `evolve/factory/engine.py`
- [X] T031 [US2] Implement `_validate_operator_compatibility()` skeleton in `evolve/factory/engine.py` *(full validation requires T047-T049 compatibility metadata)*
- [X] T032 [US2] [FR-041, FR-042] Implement `_build_callbacks()` helper in `evolve/factory/engine.py` with log level/destination and checkpoint dir/frequency support
- [X] T033 [US2] Implement `_build_stopping_criteria()` helper in `evolve/factory/engine.py`
- [X] T034 [US2] Implement `_create_standard_engine()` helper in `evolve/factory/engine.py`
- [X] T035 [US2] Implement `_create_erp_engine()` helper in `evolve/factory/engine.py`
- [X] T036 Export public API from `evolve/config/__init__.py` (UnifiedConfig, StoppingConfig, etc.)
- [X] T037 Export public API from `evolve/registry/__init__.py` (get_operator_registry, get_genome_registry)
- [X] T038 Export public API from `evolve/factory/__init__.py` (create_engine)

**Checkpoint**: User Stories 1 & 2 complete - JSON config → engine works end-to-end

---

## Phase 4: User Story 3 - Register Custom Operators (Priority: P2)

**Goal**: Researchers can register custom operators and use them by name in configuration

**Independent Test**: Register custom mutation → reference in config → verify engine uses it

### Implementation for User Story 3

- [X] T039 [US3] Implement `register()` method in `OperatorRegistry` with genome compatibility metadata
- [X] T040 [US3] Implement `is_compatible()` method in `OperatorRegistry` for genome validation
- [X] T041 [US3] Implement `get_compatibility()` method to query compatible genome types
- [X] T042 [US3] Implement `list_operators()` and `list_all()` methods in `OperatorRegistry`
- [X] T043 [P] [US3] Implement `register()` method in `GenomeRegistry` for custom genomes
- [X] T044 [P] [US3] Implement `list_types()` and `is_registered()` methods in `GenomeRegistry`
- [X] T045 [US3] Implement `reset_operator_registry()` for testing isolation
- [X] T046 [P] [US3] Implement `reset_genome_registry()` for testing isolation

**Checkpoint**: Custom operators can be registered and used via configuration

---

## Phase 5: User Story 4 - Switch Genome Representations (Priority: P2)

**Goal**: Researchers can switch genome types by changing `genome_type` in configuration

**Independent Test**: Create configs with different genome_type values → verify each produces correct genome

### Implementation for User Story 4

- [X] T047 [US4] Add compatibility metadata to all built-in crossover operator registrations
- [X] T048 [US4] Add compatibility metadata to all built-in mutation operator registrations
- [X] T049 [US4] Update `create_engine()` to call `_validate_operator_compatibility()` for all operators
- [X] T050 [US4] Implement descriptive error messages for operator-genome incompatibility
- [X] T051 [US4] Update `_create_standard_engine()` to create correct genome factory from registry

**Checkpoint**: Genome types can be switched via configuration with proper validation

---

## Phase 6: User Story 6 - Multi-Objective Configuration (Priority: P2)

**Goal**: Researchers can configure multi-objective optimization declaratively

**Independent Test**: Create config with objectives array → verify engine uses NSGA-II selection

### Implementation for User Story 6

- [X] T052 [US6] Implement `_create_multiobjective_engine()` helper in `evolve/factory/engine.py`
- [X] T053 [US6] Update `create_engine()` to detect `multiobjective` config and call appropriate builder
- [X] T054 [US6] Configure CrowdedTournamentSelection when multi-objective enabled
- [X] T055 [US6] Handle `reference_point` configuration for hypervolume tracking
- [X] T056a [US6] [FR-034] Implement constraint specification parsing (named constraints returning violation magnitude)
- [X] T056b [US6] [FR-035] Add constraint violation as additional minimization objective
- [X] T056c [US6] [FR-036, FR-037] Implement constraint dominance: feasible > infeasible; lower violation ranks higher
- [X] T056d [US6] Integration test for constrained multi-objective workflow

**Checkpoint**: Multi-objective optimization configurable via JSON

---

## Phase 7: User Story 5 - Meta-Evolution (Priority: P3)

**Goal**: Researchers can evolve hyperparameters with the framework handling inner loops

**Independent Test**: Specify evolvable params with bounds → run meta-evolution → get best config and solution

### Implementation for User Story 5

- [X] T057 [US5] Implement `ConfigCodec` class in `evolve/meta/codec.py`
- [X] T058 [US5] Implement `_compute_bounds()` method in `ConfigCodec`
- [X] T059 [US5] Implement `encode()` method in `ConfigCodec` for config → vector conversion
- [X] T060 [US5] Implement `decode()` method in `ConfigCodec` for vector → config conversion
- [X] T061 [US5] Implement `_get_param()` helper for dot-notation path access
- [X] T062 [US5] Implement `_set_param_update()` helper for nested parameter updates
- [X] T063 [US5] Implement `MetaEvaluator` class in `evolve/meta/evaluator.py`
- [X] T064 [US5] Implement `evaluate()` method in `MetaEvaluator` that runs inner evolution
- [X] T065 [US5] Implement `_compute_inner_seed()` for deterministic seeding
- [X] T066 [US5] Implement `_aggregate_fitness()` for trial aggregation (mean, median, best)
- [X] T067 [US5] Implement solution caching in `MetaEvaluator` with `get_cached_solution()`
- [X] T068 [US5] Implement `MetaEvolutionResult` frozen dataclass in `evolve/meta/result.py`
- [X] T069 [US5] Implement `get_pareto_configs()` method in `MetaEvolutionResult`
- [X] T070 [US5] Implement `export_best_config()` method in `MetaEvolutionResult`
- [X] T071 [US5] Implement `run_meta_evolution()` function in `evolve/meta/evaluator.py`
- [X] T072 [US5] Export meta-evolution API from `evolve/meta/__init__.py`

**Checkpoint**: Meta-evolution fully functional with all aggregation methods

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, integration tests, and final validation

- [X] T073 [P] Add module-level docstrings to all new `__init__.py` files
- [X] T074 [P] Create unit tests for `UnifiedConfig` serialization in `tests/unit/config/test_unified.py`
- [X] T075 [P] Create unit tests for `OperatorRegistry` in `tests/unit/registry/test_operators.py`
- [X] T076 [P] Create unit tests for `GenomeRegistry` in `tests/unit/registry/test_genomes.py`
- [X] T077 [P] Create unit tests for `create_engine()` in `tests/unit/factory/test_engine.py`
- [X] T078 [P] Create unit tests for `ConfigCodec` in `tests/unit/meta/test_codec.py`
- [X] T079 [P] Create unit tests for `MetaEvaluator` in `tests/unit/meta/test_evaluator.py`
- [X] T080 Create integration test `tests/integration/test_config_to_engine.py` for JSON → engine workflow
- [X] T081 Create integration test `tests/integration/test_meta_evolution.py` for meta-evolution
- [X] T082 Validate all scenarios in `quickstart.md` work correctly
- [X] T083 Update `evolve/__init__.py` to expose unified config public API
- [X] T084 Verify all existing tests pass (backward compatibility)
- [X] T085 [P] Test edge case: invalid operator name produces descriptive error
- [X] T086 [P] Test edge case: conflicting config flags (ERP + incompatible genome) detected at build time
- [X] T087 [P] Test edge case: inner loop failure in meta-evolution assigns worst-case fitness
- [X] T088 [P] Test edge case: partial JSON with missing optional sections applies defaults correctly

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ──────────────────────────────────────────────────────┐
                                                                       │
Phase 2 (Foundational) ◄──────────────────────────────────────────────┘
        │
        ├──► Phase 3 (US1+US2) ──┬──► Phase 4 (US3)
        │                        │
        │                        ├──► Phase 5 (US4)
        │                        │
        │                        └──► Phase 6 (US6)
        │
        └──► Phase 7 (US5) ──────────► [Depends on Phase 3 completion]
                                                                       
Phase 8 (Polish) ◄──── All previous phases complete
```

### User Story Dependencies

| User Story | Depends On | Can Start After |
|------------|------------|-----------------|
| US1 (P1) | Phase 2 | Foundational complete |
| US2 (P1) | US1 (T015-T021) | UnifiedConfig complete |
| US3 (P2) | US2 (T022-T029) | Registries complete |
| US4 (P2) | US2 (T022-T029) | Registries complete |
| US5 (P3) | US1+US2 complete | Engine factory works |
| US6 (P2) | US2 (T030-T035) | Engine factory complete |

### Parallel Opportunities

**Within Phase 2** (after Setup):
- T008, T009, T010, T011, T012 can run in parallel
- T007 and T014 are independent

**Within Phase 3** (US1+US2):
- T022 and T023 can run in parallel
- T015-T021 must be sequential
- After T022 complete: T024-T026 sequential
- After T023 complete: T027 independent

**User Story Phases 4, 5, 6** can run in parallel once Phase 3 complete

**Phase 8** tests:
- All unit tests (T074-T079) can run in parallel

---

## Summary

| Phase | Tasks | Parallel Tasks | Focus |
|-------|-------|----------------|-------|
| 1. Setup | T001-T006 | 5 | Directory structure |
| 2. Foundational | T007-T014 | 5 | Core dataclasses |
| 3. US1+US2 (MVP) | T015-T038 | 3 | Config + Factory |
| 4. US3 | T039-T046 | 3 | Custom operators |
| 5. US4 | T047-T051 | 0 | Genome switching |
| 6. US6 | T052-T056d | 0 | Multi-objective |
| 7. US5 | T057-T072 | 0 | Meta-evolution |
| 8. Polish | T073-T088 | 11 | Tests + docs |

**Total Tasks**: 91  
**MVP Scope (US1+US2)**: Tasks T001-T038 (40 tasks, includes T014a-c)  
**Full Feature**: All 91 tasks
