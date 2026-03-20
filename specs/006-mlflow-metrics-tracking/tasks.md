# Tasks: MLflow Metrics Tracking Integration

**Input**: Design documents from `/specs/006-mlflow-metrics-tracking/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and core abstractions

- [X] T001 Create collectors package structure at evolve/experiment/collectors/__init__.py
- [X] T002 Create MetricCategory enum in evolve/config/tracking.py
- [X] T003 [P] Create CollectionContext dataclass in evolve/experiment/collectors/base.py
- [X] T004 [P] Create MetricCollector Protocol in evolve/experiment/collectors/base.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement TrackingConfig dataclass with validation in evolve/config/tracking.py (FR-001)
- [X] T006 Implement TrackingConfig.to_dict() and from_dict() for JSON serialization (FR-027)
- [X] T007 [P] Implement TrackingConfig factory methods (minimal, standard, comprehensive) in evolve/config/tracking.py
- [X] T008 Create TimingContext context manager in evolve/utils/timing.py (FR-010)
- [X] T009 [P] Create GenerationTimer class for phase timing in evolve/utils/timing.py (FR-011)
- [X] T010 Implement ResilientMLflowTracker with buffering in evolve/experiment/tracking/mlflow_tracker.py (FR-028)
- [X] T011 Add batch log_metrics() and reconnection logic to ResilientMLflowTracker (FR-029)
- [X] T012 [P] Add unit tests for TrackingConfig in tests/unit/config/test_tracking.py
- [X] T013 [P] Add unit tests for TimingContext in tests/unit/utils/test_timing.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Declarative Tracking with UnifiedConfig (Priority: P1) 🎯 MVP

**Goal**: Enable MLflow tracking through UnifiedConfig without requiring ExperimentRunner

**Independent Test**: Create a UnifiedConfig with `tracking=TrackingConfig(...)`, run evolution via create_engine(), verify MLflow run exists with logged metrics

### Implementation for User Story 1

- [X] T014 [US1] Add `tracking: TrackingConfig | None` field to UnifiedConfig in evolve/config/unified.py (FR-002)
- [X] T015 [US1] Update UnifiedConfig.to_dict() and from_dict() to handle tracking field
- [X] T016 [US1] Implement MLflow import guard with clear ImportError message in evolve/experiment/tracking/mlflow_tracker.py (FR-005)
- [X] T017 [US1] Add tracking callback wiring to create_engine() in evolve/factory/engine.py (FR-003)
- [X] T018 [US1] Implement automatic MLflow run creation with experiment_name and run_name
- [X] T019 [US1] Log UnifiedConfig parameters to MLflow run on start
- [X] T020 [US1] Implement per-generation metric logging in engine._step() callback
- [X] T021 [US1] Add graceful no-op when tracking=None (default) to avoid overhead (FR-004)
- [X] T022 [P] [US1] Add unit tests for UnifiedConfig with tracking in tests/unit/config/test_unified.py
- [X] T023 [US1] Add integration test for create_engine() + tracking in tests/integration/test_tracking.py

**Checkpoint**: At this point, basic MLflow tracking via UnifiedConfig should work independently

---

## Phase 4: User Story 2 - Full Population Statistics per Generation (Priority: P1)

**Goal**: Provide comprehensive population health metrics beyond best/mean fitness

**Independent Test**: Run evolution with enhanced metrics enabled, verify diversity_score, worst_fitness, fitness_range, and species_count (when speciation enabled) appear in logs

### Implementation for User Story 2

- [X] T024 [US2] Extend compute_generation_metrics() with worst_fitness, median_fitness, quartiles in evolve/experiment/metrics.py (FR-006)
- [X] T025 [US2] Add fitness_range computation (max - min) to metrics.py
- [X] T026 [US2] Implement diversity_score computation with configurable distance function in evolve/experiment/metrics.py (FR-007)
- [X] T027 [US2] Add sampling-based diversity for populations > diversity_sample_size using engine's seeded RNG for determinism (performance requirement)
- [X] T028 [P] [US2] Create SpeciationMetricCollector in evolve/experiment/collectors/speciation.py (FR-008, FR-015)
- [X] T029 [US2] Implement species_count, average_species_size, largest_species_fitness in SpeciationMetricCollector
- [X] T030 [US2] Add species_births, species_extinctions, stagnation_count tracking
- [X] T031 [US2] Implement elite_turnover_rate computation in evolve/experiment/metrics.py (FR-009)
- [X] T032 [P] [US2] Add unit tests for extended metrics in tests/unit/experiment/test_metrics.py
- [X] T033 [P] [US2] Add unit tests for SpeciationMetricCollector in tests/unit/experiment/collectors/test_speciation.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work - users get comprehensive population stats logged to MLflow

---

## Phase 5: User Story 3 - Timing Instrumentation (Priority: P2)

**Goal**: Provide per-phase timing breakdown for performance analysis

**Independent Test**: Enable timing metrics, verify generation_time_ms, evaluation_time_ms, selection_time_ms, variation_time_ms appear in logs

### Implementation for User Story 3

- [X] T034 [US3] Wrap evaluation phase with timing_context in engine._step() (FR-010)
- [X] T035 [US3] Wrap selection phase with timing_context in engine._step()
- [X] T036 [US3] Wrap crossover and mutation phases with timing_context in engine._step()
- [X] T037 [US3] Add total generation_time_ms metric
- [X] T038 [US3] Distinguish wall_clock_time and cpu_time for parallel backends (FR-011)
- [X] T039 [US3] Add timing_breakdown config flag to control fine-grained vs total-only timing
- [X] T040 [US3] Verify timing overhead <1% for populations under 1000 (FR-012)
- [X] T041 [P] [US3] Add unit tests for timing instrumentation in tests/unit/experiment/test_timing.py
- [X] T042 [P] [US3] Add benchmark test for timing overhead in tests/benchmarks/test_timing_overhead.py

**Checkpoint**: Users can now identify performance bottlenecks through logged timing data

---

## Phase 6: User Story 4 - ERP Mating Statistics (Priority: P2)

**Goal**: Expose ERP mating dynamics for debugging reproduction behavior

**Independent Test**: Run ERP evolution with tracking, verify mating_success_rate, attempted_matings, successful_matings appear in logs

### Implementation for User Story 4

- [X] T043 [P] [US4] Create MatingStats dataclass in evolve/experiment/collectors/base.py
- [X] T044 [US4] Create ERPMetricCollector in evolve/experiment/collectors/erp.py (FR-013)
- [X] T045 [US4] Implement mating_success_rate, attempted_matings, successful_matings
- [X] T046 [US4] Add per-protocol success rates (erp_protocol_{name}_success_rate)
- [X] T047 [US4] Wire MatingStats from ERP engine to CollectionContext
- [X] T048 [US4] Add warning/event logging when mating_success_rate drops to zero
- [X] T049 [P] [US4] Add unit tests for ERPMetricCollector in tests/unit/experiment/collectors/test_erp.py

**Checkpoint**: ERP users can now debug mating dynamics through logged metrics

---

## Phase 7: User Story 5 - Multi-Objective Metrics (Priority: P2)

**Goal**: Track Pareto front quality metrics for multi-objective optimization

**Independent Test**: Run MO optimization with tracking, verify hypervolume, pareto_front_size, spread appear in logs

### Implementation for User Story 5

- [X] T050 [P] [US5] Create MultiObjectiveMetricCollector in evolve/experiment/collectors/multiobjective.py (FR-014)
- [X] T051 [US5] Implement pareto_front_size metric
- [X] T052 [US5] Implement hypervolume computation for 2-3 objectives
- [X] T053 [US5] Add configurable reference point via hypervolume_reference config
- [X] T054 [US5] Implement crowding_diversity metric using existing crowding_distance
- [X] T055 [US5] Implement spread metric for front distribution
- [X] T056 [US5] Add fallback to approximate indicators for >3 objectives
- [X] T057 [US5] Auto-enable MULTIOBJECTIVE category when multiobjective config present
- [X] T058 [P] [US5] Add unit tests for MultiObjectiveMetricCollector in tests/unit/experiment/collectors/test_multiobjective.py

**Checkpoint**: MOO users can now track Pareto front quality over generations

---

## Phase 8: User Story 6 - Fitness Metadata Extraction (Priority: P3)

**Goal**: Automatically extract and aggregate domain-specific data from Fitness.metadata

**Independent Test**: Create evaluator that populates Fitness.metadata, verify meta_* fields appear in logs

### Implementation for User Story 6

- [X] T059 [P] [US6] Create FitnessMetadataCollector in evolve/experiment/collectors/metadata.py (FR-018)
- [X] T060 [US6] Implement numeric field extraction from Fitness.metadata
- [X] T061 [US6] Implement aggregation (best, mean, std) with metadata_prefix (FR-019)
- [X] T062 [US6] Handle missing fields with metadata_threshold majority policy (FR-020)
- [X] T063 [US6] Skip non-numeric fields with debug logging
- [X] T064 [US6] Handle nested metadata structures with flattened keys
- [X] T065 [P] [US6] Add unit tests for FitnessMetadataCollector in tests/unit/experiment/collectors/test_metadata.py

**Checkpoint**: Users with rich evaluators now see domain-specific data logged automatically

---

## Phase 9: User Story 7 - Derived Analytics Metrics (Priority: P3)

**Goal**: Provide computed analytics that combine multiple raw metrics

**Independent Test**: Run evolution, verify selection_pressure, fitness_improvement_velocity, population_entropy appear in logs

### Implementation for User Story 7

- [X] T066 [P] [US7] Create DerivedAnalyticsCollector in evolve/experiment/collectors/derived.py
- [X] T067 [US7] Implement selection_pressure (best/mean ratio) (FR-021)
- [X] T068 [US7] Implement fitness_improvement_velocity with configurable window (FR-022)
- [X] T069 [US7] Implement population_entropy using fitness histogram binning (FR-023)
<!-- T070 removed: elite_turnover_rate implemented in T031 under US2 (FR-009) -->
- [X] T071 [US7] Maintain history state across generations for velocity computation
- [X] T072 [US7] Implement reset() to clear history between runs
- [X] T073 [P] [US7] Add unit tests for DerivedAnalyticsCollector in tests/unit/experiment/collectors/test_derived.py

**Checkpoint**: Advanced users now have actionable derived insights logged automatically

---

## Phase 10: Additional Collectors (Supporting)

**Purpose**: Complete remaining specialized collectors from plan.md

- [X] T074 [P] Create IslandsMetricCollector in evolve/experiment/collectors/islands.py (FR-016)
- [X] T075 Implement inter_island_variance, intra_island_variance, migration_events
- [X] T076 [P] Create NEATMetricCollector in evolve/experiment/collectors/neat.py (FR-017)
- [X] T077 Implement average_node_count, average_connection_count, topology_innovations
- [X] T078 [P] Add unit tests for IslandsMetricCollector in tests/unit/experiment/collectors/test_islands.py
- [X] T079 [P] Add unit tests for NEATMetricCollector in tests/unit/experiment/collectors/test_neat.py

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T080 [P] Update evolve/experiment/collectors/__init__.py with all collector exports
- [X] T081 [P] Update evolve/config/__init__.py to export TrackingConfig and MetricCategory
- [X] T082 [P] Update evolve/utils/__init__.py to export timing utilities
- [X] T083 Add comprehensive docstrings to all public APIs
- [X] T084 [P] Add type hints validation (mypy check) - tracking code passes, 2 pre-existing issues in unrelated files
- [X] T085 Implement category auto-detection (auto-enable ERP when ERP config, MO when MO config, etc.) (FR-025)
- [X] T086 Verify default TrackingConfig logs only core metrics for minimal overhead (FR-026)
- [X] T087 [P] Run quickstart.md examples as integration validation (tests/integration/test_quickstart_validation.py - 14 tests)
- [X] T088 Performance optimization: verify batch logging and sampling for large populations (tests/integration/test_tracking_performance.py - 29 tests)
- [X] T089 [P] Add end-to-end integration test covering all user stories in tests/integration/test_tracking_comprehensive.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phases 3-9)**: All depend on Foundational (Phase 2) completion
  - US1 & US2 (P1): Can proceed in parallel after Foundational
  - US3, US4, US5 (P2): Can proceed in parallel after Foundational
  - US6, US7 (P3): Can proceed in parallel after Foundational
- **Additional Collectors (Phase 10)**: Can proceed after Phase 1 Setup
- **Polish (Phase 11)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on other stories - core tracking path
- **User Story 2 (P1)**: Depends on US1 tracking infrastructure for metric logging
- **User Story 3 (P2)**: Depends on US1 tracking infrastructure - timing flows through same path
- **User Story 4 (P2)**: Depends on US1 tracking infrastructure - ERP metrics logged same way
- **User Story 5 (P2)**: Depends on US1 tracking infrastructure - MO metrics logged same way
- **User Story 6 (P3)**: Depends on US1 tracking infrastructure
- **User Story 7 (P3)**: Depends on US2 for access to core metrics for derived computations

### Within Each User Story

- Core implementation before integration
- Collectors before wiring
- Wiring before tests
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 2 (Foundational)**:
```
T007 (factory methods) ‖ T008 (TimingContext) ‖ T009 (GenerationTimer)
T012 (TrackingConfig tests) ‖ T013 (timing tests)
```

**Phase 3-4 (P1 Stories)**:
```
After T021 completes:
  T022 (config tests) ‖ T028 (SpeciationCollector) ‖ T032 (metrics tests) ‖ T033 (speciation tests)
```

**Phase 5-7 (P2 Stories)**:
```
After Foundational:
  US3 (timing) ‖ US4 (ERP) ‖ US5 (multi-objective) can all proceed in parallel
  T041 ‖ T042 (timing tests)
  T043 ‖ T049 (ERP)
  T050 ‖ T058 (MO)
```

**Phase 8-9 (P3 Stories)**:
```
After Foundational:
  US6 (metadata) ‖ US7 (derived) can proceed in parallel
  T059 ‖ T065 (metadata)
  T066 ‖ T073 (derived)
```

---

## Implementation Strategy

### MVP First (User Stories 1-2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 - Declarative Tracking
4. Complete Phase 4: User Story 2 - Population Statistics
5. **STOP and VALIDATE**: Test MLflow tracking with comprehensive population stats
6. Deploy/demo if ready - users can now get observability via UnifiedConfig

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 (Declarative Tracking) → Test → **MVP Release!**
3. Add US2 (Population Stats) → Test → Release with enhanced metrics
4. Add US3 (Timing) → Test → Release with performance profiling
5. Add US4-5 (ERP + MO) → Test → Release for specialized domains
6. Add US6-7 (Metadata + Derived) → Test → Full feature release

### Parallel Team Strategy

With 3 developers after Foundational:
- **Dev A**: US1 → US3 → US6
- **Dev B**: US2 → US4 → US7
- **Dev C**: US5 → Phase 10 (Islands/NEAT) → Phase 11 (Polish)

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tasks** | 88 |
| **Setup Phase** | 4 tasks |
| **Foundational Phase** | 9 tasks |
| **User Story 1** | 10 tasks |
| **User Story 2** | 10 tasks |
| **User Story 3** | 9 tasks |
| **User Story 4** | 7 tasks |
| **User Story 5** | 9 tasks |
| **User Story 6** | 7 tasks |
| **User Story 7** | 7 tasks |
| **Additional Collectors** | 6 tasks |
| **Polish Phase** | 10 tasks |
| **Parallelizable Tasks** | 35 (marked [P]) |
| **MVP Scope** | Phases 1-4 (33 tasks) |

---

## Notes

- [P] tasks = different files, no dependencies within same phase
- [US#] label maps task to specific user story for traceability
- Each user story is independently completable and testable after Foundational phase
- FR-### references map to Functional Requirements in spec.md
- MLflow 2.0+ required per FR-029 (no 1.x compatibility)
- All enhanced metrics opt-in per FR-025/FR-026
