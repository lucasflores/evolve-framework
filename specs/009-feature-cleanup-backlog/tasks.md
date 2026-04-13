# Tasks: Feature & Cleanup Backlog

**Input**: Design documents from `/specs/009-feature-cleanup-backlog/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Tests**: Included per constitution requirement (Test-First Development).

**Organization**: Tasks grouped by user story for independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1–US8)
- Exact file paths included

---

## Phase 1: Setup

**Purpose**: No new project structure needed — all changes are within existing `evolve/` package. This phase handles foundational shared changes.

- [x] T001 Add `minimize: bool = True` field to `PopulationStatistics` frozen dataclass in `evolve/core/population.py`
- [x] T002 [P] Add `priority: int` property (default 0) to `Callback` protocol and `SimpleCallback` base class in `evolve/core/callbacks.py`
- [x] T003 [P] Add `distance(self, other) -> float` method to `Genome` protocol in `evolve/representation/genome.py`

**Checkpoint**: Core protocol/dataclass changes in place. All downstream tasks can reference these.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared callback infrastructure changes that multiple user stories depend on.

- [x] T004 Store `_creation_callbacks: list[Callback]` on `EvolutionEngine.__init__()` in `evolve/core/engine.py` — capture callbacks passed to constructor
- [x] T005 Modify `EvolutionEngine.run()` in `evolve/core/engine.py` to merge `self._creation_callbacks` with `callbacks` parameter instead of replacing; sort merged list by `priority` (ascending, stable)
- [x] T006 [P] Write unit tests for callback persistence and priority ordering in `tests/unit/test_engine_callbacks.py` — test creation-time callbacks persist through `run()`, merge behavior, priority sort, backward compatibility with no callbacks

**Checkpoint**: Callback persistence + ordering infrastructure ready. All user stories can now build on this.

---

## Phase 3: User Story 1 — Correct Best/Worst Fitness Reporting (Priority: P1) 🎯 MVP

**Goal**: Fix `_compute_statistics()` to respect `minimize` flag so `best_fitness`/`worst_fitness` report correct values.

**Independent Test**: Create population with known fitnesses, verify stats with `minimize=True` and `minimize=False`.

- [x] T007 [P] [US1] Write failing tests for minimize-aware statistics in `tests/unit/test_population_statistics.py` — test `best_fitness`=min when `minimize=True`, `best_fitness`=max when `minimize=False`, multi-objective fallback, empty population
- [x] T008 [US1] Modify `Population._compute_statistics()` in `evolve/core/population.py` to accept `minimize: bool` parameter; use `np.argmin` for `minimize=True`, `np.argmax` for `minimize=False`; pass `minimize` to `PopulationStatistics` constructor
- [x] T009 [US1] Update `Population` class in `evolve/core/population.py` to accept `minimize: bool = True` in constructor and store it; pass stored `minimize` to `_compute_statistics()`; invalidate cached statistics when `minimize` changes
- [x] T010 [US1] Run tests in `tests/unit/test_population_statistics.py` and verify all pass

**Checkpoint**: Population statistics correctly respect minimize flag.

---

## Phase 4: User Story 2 — Callbacks Persist Through `engine.run()` (Priority: P1)

**Goal**: Engine-creation callbacks persist; `run()` merges instead of replacing.

**Independent Test**: Create engine with HistoryCallback, call `run()` without callbacks, verify callback received events.

- [x] T011 [US2] Run tests in `tests/unit/test_engine_callbacks.py` (written in T006) and verify all pass after T004/T005 changes

**Checkpoint**: Callbacks persist through `run()`, merge correctly, ordered by priority.

---

## Phase 5: User Story 3 — PopulationStatistics Minimize-Aware (Priority: P1)

**Goal**: `PopulationStatistics` stores `minimize` flag, making `best_fitness`/`worst_fitness` self-describing.

**Independent Test**: Inspect `stats.minimize` field to determine optimization direction.

- [x] T012 [P] [US3] Write tests in `tests/unit/test_population_statistics.py` verifying `statistics.minimize` field is accessible and correct for both directions
- [x] T013 [US3] Verify `PopulationStatistics.minimize` field (added in T001) is populated correctly by `_compute_statistics()` (done in T008) — run tests

**Checkpoint**: PopulationStatistics is self-describing for optimization direction.

---

## Phase 6: User Story 4 — Correct best_fitness/worst_fitness in Metrics Dict (Priority: P1)

**Goal**: `_compute_metrics()` emits correct `best_fitness`/`worst_fitness` reflecting minimize flag.

**Independent Test**: Run engine with `minimize=False`, verify metrics dict `best_fitness` = max.

- [x] T014 [P] [US4] Write failing tests in `tests/unit/test_engine_metrics.py` for minimize-aware metrics dict — test `best_fitness`/`worst_fitness` values for both minimize=True and minimize=False
- [x] T015 [US4] Update `EvolutionEngine._compute_metrics()` in `evolve/core/engine.py` to read `stats.best_fitness`/`stats.worst_fitness` (which are now minimize-aware from T008) and add `worst_fitness` to metrics dict
- [x] T016 [US4] Run tests in `tests/unit/test_engine_metrics.py` and verify all pass

**Checkpoint**: Metrics dict correctly reflects optimization direction for all consumers.

---

## Phase 7: User Story 5 — Native Population Dynamics Metrics (Priority: P2)

**Goal**: Engine computes fitness distribution, genome diversity, and search movement metrics.

**Independent Test**: Run multi-generation evolution, verify new metric keys present with valid values.

### Fitness Distribution

- [x] T017 [P] [US5] Write failing tests for fitness distribution metrics in `tests/unit/test_engine_metrics.py` — median, Q1, Q3, min, max, fitness_range, unique_fitness_count
- [x] T018 [US5] Add fitness distribution computation to `EvolutionEngine._compute_metrics()` in `evolve/core/engine.py` — compute median, quartiles, min/max (direction-aware), range, unique count from population fitness values; gate by `MetricCategory.EXTENDED_POPULATION`; ensure `EXTENDED_POPULATION` and `DIVERSITY` categories in `MetricCategory` enum exist in `evolve/config/tracking.py` (add if missing)

### Genome Distance Implementations

- [x] T019 [P] [US5] Implement `VectorGenome.distance()` using L2 norm in `evolve/representation/vector.py`
- [x] T020 [P] [US5] Implement `SequenceGenome.distance()` using Levenshtein edit distance in `evolve/representation/sequence.py`
- [x] T021 [P] [US5] Write tests for genome distance methods in `tests/unit/test_genome_distance.py` — L2 for vectors, edit distance for sequences, type mismatch handling

### Genome Diversity Metrics

- [x] T022 [P] [US5] Write failing tests for genome diversity metrics in `tests/unit/test_engine_metrics.py` — mean_gene_std, mean_distance_from_centroid, mean_pairwise_distance
- [x] T023 [US5] Add genome diversity computation to `EvolutionEngine._compute_metrics()` in `evolve/core/engine.py` — mean per-gene std (for VectorGenome), mean distance from centroid, sampled pairwise distance via `Genome.distance()` protocol; gate by `MetricCategory.DIVERSITY`; skip gracefully if genome has no `distance()` method

### Search Movement Metrics

- [x] T024 [P] [US5] Write failing tests for search movement metrics in `tests/unit/test_engine_metrics.py` — centroid_drift, best_genome_similarity, best_changed
- [x] T025 [US5] Add `_prev_centroid` and `_prev_best_genome` state to `EvolutionEngine` in `evolve/core/engine.py`; compute centroid drift, cosine similarity between consecutive best genomes, and boolean `best_changed` flag in `_compute_metrics()`; gate by `MetricCategory.DIVERSITY`
- [x] T026 [US5] Run all tests in `tests/unit/test_engine_metrics.py` and `tests/unit/test_genome_distance.py` and verify all pass

**Checkpoint**: Full population dynamics metrics available in metrics dict.

---

## Phase 8: User Story 6 — Callback Priority / Ordering Guarantees (Priority: P2)

**Goal**: Numeric priority on callbacks; lower runs first; TrackingCallback defaults to 1000.

**Independent Test**: Two callbacks with different priorities — injecting callback runs before tracking callback.

- [x] T027 [P] [US6] Write tests for callback priority ordering in `tests/unit/test_engine_callbacks.py` — verify metric-injecting callback (priority=0) runs before TrackingCallback (priority=1000), equal-priority registration-order stability
- [x] T028 [US6] Set `priority = 1000` on `TrackingCallback` in `evolve/experiment/tracking/callback.py`
- [x] T029 [US6] Update `_build_callbacks()` in `evolve/factory/engine.py` so factory-built callbacks (TrackingCallback) preserve priority=1000 and user custom_callbacks get default priority=0
- [x] T030 [US6] Run tests in `tests/unit/test_engine_callbacks.py` and verify all pass

**Checkpoint**: Callback ordering deterministic by priority.

---

## Phase 9: User Story 7 — Meta-Evolution MLflow Tracking (Priority: P2)

**Goal**: Meta-evolution logs to MLflow with parent run + nested child runs + tags.

**Independent Test**: Run small meta-evolution with tracking, verify nested MLflow run structure.

- [x] T031 [P] [US7] Write integration tests for meta-evolution MLflow tracking in `tests/integration/test_meta_mlflow.py` — verify parent run created, child runs nested, tags present (meta_generation, meta_parent_run_id, config_hash), best config logged as artifact
- [x] T032 [US7] Modify `MetaEvaluator.evaluate()` in `evolve/meta/evaluator.py` to create child MLflow runs (via `mlflow.start_run(nested=True)`) for each inner evolution trial; set tags `meta_generation`, `meta_parent_run_id`, `config_hash` on child runs
- [x] T033 [US7] Modify `run_meta_evolution()` in `evolve/meta/evaluator.py` to create a parent MLflow run for the outer loop; log outer-loop metrics (generation, best_config_fitness, parameter stats) per generation; wire TrackingCallback into inner `engine.run()` calls
- [x] T034 [US7] Log best configuration as JSON artifact on parent run at completion in `evolve/meta/evaluator.py`
- [x] T035 [US7] Run integration tests in `tests/integration/test_meta_mlflow.py` and verify all pass

**Checkpoint**: Meta-evolution fully observable in MLflow UI.

---

## Phase 10: User Story 8 — UnifiedConfig Datasets and Tags in MLflow (Priority: P3)

**Goal**: Typed dataset fields on UnifiedConfig; native MLflow Datasets + Tags logging.

**Independent Test**: Create config with datasets and tags, run tracked evolution, verify MLflow fields populated.

- [x] T036 [P] [US8] Create `DatasetConfig` frozen dataclass in `evolve/config/unified.py` with fields: name (str), path (str|None), data (Any|None), context (str); add `to_dict()`/`from_dict()` methods
- [x] T037 [US8] Add `training_data: DatasetConfig | None = None` and `validation_data: DatasetConfig | None = None` fields to `UnifiedConfig` in `evolve/config/unified.py`; update `to_dict()`/`from_dict()`/`compute_hash()` to handle new fields
- [x] T038 [P] [US8] Write unit tests for DatasetConfig and UnifiedConfig dataset fields in `tests/unit/test_unified_config.py` — serialization round-trip, hash includes datasets, None defaults
- [x] T039 [US8] Update `TrackingCallback.on_run_start()` in `evolve/experiment/tracking/callback.py` to call `mlflow.set_tags()` with UnifiedConfig tags (in addition to existing parameter logging)
- [x] T040 [US8] Update `TrackingCallback.on_run_start()` in `evolve/experiment/tracking/callback.py` to call `mlflow.log_input()` for training_data and validation_data when present; convert DatasetConfig.data to appropriate MLflow dataset type (numpy/pandas/dict)
- [x] T041 [P] [US8] Write integration tests in `tests/integration/test_tracking_callback.py` — verify MLflow native Tags populated, MLflow native Datasets populated, backward compatibility (tags still in params)
- [x] T042 [US8] Run all tests in `tests/unit/test_unified_config.py` and `tests/integration/test_tracking_callback.py` and verify all pass

**Checkpoint**: MLflow native Datasets and Tags fields populated from UnifiedConfig.

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all user stories.

- [x] T043 Run full test suite (`pytest tests/`) to verify backward compatibility (FR-016) — all pre-existing tests pass without modification
- [x] T044 Run quickstart.md validation — execute code snippets from `specs/009-feature-cleanup-backlog/quickstart.md` to verify end-to-end
- [x] T045 Verify lint passes (`ruff check evolve/ tests/` or project lint command)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — T001, T002, T003 all parallel
- **Phase 2 (Foundational)**: Depends on T002 (priority on Callback) — T004, T005 sequential; T006 parallel with T005
- **Phase 3 (US1)**: Depends on T001 (minimize on PopulationStatistics) — can start after Phase 1
- **Phase 4 (US2)**: Depends on Phase 2 (T004, T005) — inherits from foundational
- **Phase 5 (US3)**: Depends on Phase 3 (T008, T009 for minimize-aware stats)
- **Phase 6 (US4)**: Depends on Phase 3 (US1 fixes) and Phase 5 (US3 for self-describing stats)
- **Phase 7 (US5)**: Depends on T003 (distance protocol) and Phase 6 (correct metrics path)
- **Phase 8 (US6)**: Depends on Phase 2 (priority infrastructure)
- **Phase 9 (US7)**: Independent of US1-US6; depends only on Phase 1/2
- **Phase 10 (US8)**: Independent of US1-US7; depends only on Phase 1
- **Phase 11 (Polish)**: Depends on all previous phases

### User Story Dependencies

- **US1 (P1)**: → US3, US4 (downstream)
- **US2 (P1)**: Independent (foundational callback fix)
- **US3 (P1)**: ← US1
- **US4 (P1)**: ← US1, US3
- **US5 (P2)**: ← US4, T003
- **US6 (P2)**: ← Phase 2
- **US7 (P2)**: Independent (meta module)
- **US8 (P3)**: Independent (config + tracking)

### Parallel Opportunities

- T001, T002, T003 all parallel (different files)
- T019, T020, T021 all parallel (different genome files)
- T017, T022, T024 all parallel (different test sections)
- US7 and US8 can run in parallel with each other and with US5/US6
- T036, T038, T041 parallel within US8

---

## Implementation Strategy

**MVP (Phase 1–6)**: Bug fixes (US1–US4) deliver immediately correct fitness reporting and callback persistence. This is the minimum viable delivery.

**Increment 2 (Phase 7–8)**: Population dynamics metrics + callback ordering. High research value.

**Increment 3 (Phase 9–10)**: Meta-evolution tracking + config datasets/tags. Observability improvements.

**Total tasks**: 45
