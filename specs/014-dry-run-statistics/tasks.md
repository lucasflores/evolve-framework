# Tasks: Dry-Run Statistics Tool

**Input**: Design documents from `/specs/014-dry-run-statistics/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/public-api.md, quickstart.md

**Tests**: Included per TDD requirement from `agents.md` and `.github/copilot-instructions.md`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Create the module file and foundational dataclasses

- [X] T001 Create `evolve/experiment/dry_run.py` with module docstring and `from __future__ import annotations` import
- [X] T002 [P] Create `tests/unit/experiment/__init__.py` (if not exists) and `tests/unit/experiment/test_dry_run.py` with initial test structure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define all frozen dataclasses that every user story depends on. These are the data structures from `data-model.md` and `contracts/public-api.md`.

**⚠️ CRITICAL**: No user story work can begin until all dataclasses are defined and tested.

- [X] T003 Implement `PhaseEstimate` frozen dataclass with fields (`name`, `measured_time_ms`, `operations_per_generation`, `estimated_total_ms`, `percentage`, `is_bottleneck`) in `evolve/experiment/dry_run.py`
- [X] T004 [P] Write unit tests for `PhaseEstimate` construction, immutability, and field validation in `tests/unit/experiment/test_dry_run.py`
- [X] T005 Implement `ComputeResources` frozen dataclass with fields (`cpu_count`, `total_memory_bytes`, `gpu_available`, `gpu_name`, `gpu_memory_bytes`, `backend_name`, `backend_workers`) in `evolve/experiment/dry_run.py`
- [X] T006 [P] Write unit tests for `ComputeResources` construction and GPU-consistency validation in `tests/unit/experiment/test_dry_run.py`
- [X] T007 Implement `MemoryEstimate` frozen dataclass with fields (`genome_bytes`, `individual_overhead_bytes`, `population_bytes`, `history_bytes`, `total_bytes`) in `evolve/experiment/dry_run.py`
- [X] T008 [P] Write unit tests for `MemoryEstimate` construction and `total_bytes` consistency in `tests/unit/experiment/test_dry_run.py`
- [X] T009 Implement `MetaEstimate` frozen dataclass with fields (`inner_run_estimate_ms`, `outer_generations`, `trials_per_config`, `total_inner_runs`, `total_estimated_ms`) in `evolve/experiment/dry_run.py`
- [X] T010 [P] Write unit tests for `MetaEstimate` construction and `total_inner_runs` consistency in `tests/unit/experiment/test_dry_run.py`
- [X] T011 Implement `DryRunReport` frozen dataclass with fields (`config_hash`, `phase_estimates`, `total_estimated_ms`, `estimated_generations`, `resources`, `memory`, `seed_used`, `early_stop_possible`, `active_subsystems`, `meta_estimate`, `caveats`) in `evolve/experiment/dry_run.py`
- [X] T012 Write unit tests for `DryRunReport` construction, `summary()` method stub, and `__str__()` delegation in `tests/unit/experiment/test_dry_run.py`

**Checkpoint**: All 5 frozen dataclasses defined, tested, and importable via `from evolve.experiment.dry_run import DryRunReport, PhaseEstimate, ComputeResources, MemoryEstimate, MetaEstimate`

---

## Phase 3: User Story 1 — Quick Cost Estimate Before a Long Run (Priority: P1) 🎯 MVP

**Goal**: Given a `UnifiedConfig`, micro-benchmark each core atomic operation (evaluation, crossover, mutation, selection, merge) and produce a `DryRunReport` with per-phase timing breakdown, percentages, and bottleneck identification.

**Independent Test**: Provide a `UnifiedConfig` with known parameters and verify the report contains per-phase estimates, structural multipliers, and a total duration.

### Tests for User Story 1

- [X] T013 [US1] Write failing test: `dry_run()` accepts a valid `UnifiedConfig` and returns a `DryRunReport` with `phase_estimates` containing at least `initialization`, `evaluation`, `selection`, `variation` in `tests/unit/experiment/test_dry_run.py`
- [X] T014 [P] [US1] Write failing test: `dry_run()` with merge enabled includes a `merge` phase in `phase_estimates` in `tests/unit/experiment/test_dry_run.py`
- [X] T015 [P] [US1] Write failing test: each `PhaseEstimate` in the report has `percentage` values that sum to ~100% and exactly one has `is_bottleneck=True` in `tests/unit/experiment/test_dry_run.py`
- [X] T016 [P] [US1] Write failing test: `dry_run()` raises `ValueError` when config is invalid (e.g., missing evaluator) in `tests/unit/experiment/test_dry_run.py`
- [X] T017 [P] [US1] Write failing test: `dry_run()` reports `operations_per_generation` matching structural constants derived from config (`population_size`, `elitism`, `crossover_rate`, `mutation_rate`) in `tests/unit/experiment/test_dry_run.py`

### Implementation for User Story 1

- [X] T018 [US1] Implement `_validate_config(config)` helper that checks for required fields (evaluator, genome_type, population_size, max_generations) and raises `ValueError` with actionable messages in `evolve/experiment/dry_run.py`
- [X] T019 [US1] Implement `_create_sample_population(config, seed)` helper that creates sample individuals using `GenomeRegistry.create()` with `config.genome_params` — sample size = `max(3, 2 × n_workers)` for parallel backends, 3 otherwise — in `evolve/experiment/dry_run.py`
- [X] T020 [US1] Implement `_derive_structural_constants(config)` helper that computes per-generation operation counts (eval: `population_size`, selection: `(population_size - elitism) * 2`, crossover: `population_size - elitism`, mutation: `population_size - elitism`, merge: `population_size * merge_rate` if enabled) in `evolve/experiment/dry_run.py`
- [X] T021 [US1] Implement `_benchmark_phase(callable, timeout)` helper that times a single invocation with wall-clock timing, returns `measured_time_ms`, and aborts with a timeout marker if exceeded in `evolve/experiment/dry_run.py`
- [X] T022 [US1] Implement core benchmarking logic in `_benchmark_core_phases(config, sample_population, evaluator, seed, timeout_per_phase)` that benchmarks initialization, evaluation (via backend batch interface), selection, crossover, mutation, and merge (if enabled) — returning a list of `PhaseEstimate` objects in `evolve/experiment/dry_run.py`
- [X] T023 [US1] Implement `_compute_percentages_and_bottleneck(phase_estimates)` helper that calculates percentage of total for each phase and marks the highest as `is_bottleneck=True` — handle edge case of equal percentages (first wins) in `evolve/experiment/dry_run.py`
- [X] T024 [US1] Implement the top-level `dry_run(config, evaluator, seed, timeout_per_phase) -> DryRunReport` function that orchestrates validation, sample creation, core benchmarking, percentage computation, and report assembly in `evolve/experiment/dry_run.py`
- [X] T025 [US1] Implement `DryRunReport.summary()` method returning the formatted ASCII table with box-drawing characters, showing phase/time/percentage/bottleneck columns (with `★` marker on the bottleneck row), resource line, and memory line in `evolve/experiment/dry_run.py`

**Checkpoint**: `dry_run(config)` works for standard configs (sequential/parallel backends, with/without merge). Returns a complete `DryRunReport` with ASCII table output. All T013–T017 tests pass.

---

## Phase 4: User Story 2 — Auto-Detect Computational Resources (Priority: P2)

**Goal**: Auto-detect CPU count (container-aware), system memory, GPU presence/type/memory, and backend worker count. Populate `ComputeResources` in the report.

**Independent Test**: Invoke resource detection on the current machine and verify CPU count, memory, and GPU status are correctly reported.

### Tests for User Story 2

- [X] T026 [P] [US2] Write failing test: `_detect_resources(config)` returns a `ComputeResources` with `cpu_count >= 1` and `backend_name` matching the config in `tests/unit/experiment/test_dry_run.py`
- [X] T027 [P] [US2] Write failing test: when `gpu_available` is False, `gpu_name` and `gpu_memory_bytes` are None in `tests/unit/experiment/test_dry_run.py`

### Implementation for User Story 2

- [X] T028 [US2] Implement `_detect_cpu_count()` using `os.sched_getaffinity` (Linux) with fallback to `os.cpu_count()`, respecting container cgroup limits via `/proc` inspection, in `evolve/experiment/dry_run.py`
- [X] T029 [US2] Implement `_detect_memory()` using `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` with platform fallback, returning `int | None` in `evolve/experiment/dry_run.py`
- [X] T030 [US2] Implement `_detect_gpu()` using conditional imports of `torch.cuda` and `jax.devices` (via `check_dependency()` from `evolve/utils/dependencies.py`), returning `(gpu_available, gpu_name, gpu_memory_bytes)` in `evolve/experiment/dry_run.py`
- [X] T031 [US2] Implement `_detect_resources(config)` that composes CPU, memory, and GPU detection into a `ComputeResources` instance, including `backend_name` and `backend_workers` from config in `evolve/experiment/dry_run.py`
- [X] T032 [US2] Integrate `_detect_resources()` into the `dry_run()` function so `DryRunReport.resources` is populated in `evolve/experiment/dry_run.py`

**Checkpoint**: `dry_run(config).resources` returns accurate hardware detection. All T026–T027 tests pass.

---

## Phase 5: User Story 3 — Identify Computational Bottlenecks (Priority: P2)

**Goal**: Report per-phase percentage of total estimated time and flag the dominant phase as the bottleneck. Include the bottleneck indicator in the ASCII table.

**Independent Test**: Run dry-run on a config where evaluation dominates and verify the report correctly identifies evaluation as the bottleneck.

### Tests for User Story 3

- [X] T033 [P] [US3] Write failing test: `DryRunReport.summary()` output contains a `★` marker on the bottleneck row in `tests/unit/experiment/test_dry_run.py`
- [X] T034 [P] [US3] Write failing test: given a report where evaluation has the highest `estimated_total_ms`, the evaluation `PhaseEstimate` has `is_bottleneck=True` in `tests/unit/experiment/test_dry_run.py`

### Implementation for User Story 3

- [ ] ~~T035~~ [US3] _Merged into T023_ — equal-percentage edge case handling is now part of T023
- [ ] ~~T036~~ [US3] _Merged into T025_ — `★` bottleneck indicator rendering is now part of T025

**Checkpoint**: Bottleneck is correctly flagged in both structured data and ASCII output. All T033–T034 tests pass.

---

## Phase 6: User Story 5 — Meta-Evolution Cost Estimation (Priority: P2)

**Goal**: When `config.is_meta_evolution` is True, estimate inner run cost and multiply by `outer_generations × trials_per_config × population_size` for total meta-evolution cost. Report the multiplication breakdown.

**Independent Test**: Provide a config with `config.meta` enabled and verify the report shows the outer × inner multiplication structure.

### Tests for User Story 5

- [X] T037 [P] [US5] Write failing test: `dry_run()` with meta-evolution enabled returns a `DryRunReport` with `meta_estimate` not None and `meta_estimate.total_estimated_ms > 0` in `tests/unit/experiment/test_dry_run.py`
- [X] T038 [P] [US5] Write failing test: `meta_estimate.total_inner_runs == outer_generations * trials_per_config * population_size` in `tests/unit/experiment/test_dry_run.py`
- [X] T039 [P] [US5] Write failing test: `DryRunReport.summary()` includes a meta-evolution section with inner run cost and multiplication breakdown when meta-evolution is enabled in `tests/unit/experiment/test_dry_run.py`

### Implementation for User Story 5

- [X] T040 [US5] Implement `_estimate_meta_evolution(config, evaluator, seed, timeout_per_phase)` that estimates inner run cost by applying core benchmarking to the inner config, then computes `MetaEstimate` with outer multiplication in `evolve/experiment/dry_run.py`
- [X] T041 [US5] Integrate meta-evolution detection into `dry_run()`: check `config.is_meta_evolution`, call `_estimate_meta_evolution()`, and set `DryRunReport.meta_estimate` accordingly in `evolve/experiment/dry_run.py`
- [X] T042 [US5] Extend `DryRunReport.summary()` to include a meta-evolution breakdown section below the phase table when `meta_estimate` is not None in `evolve/experiment/dry_run.py`

**Checkpoint**: Meta-evolution configs produce a report with inner run cost × outer multiplier. `"meta_evolution"` appears in `active_subsystems`. All T037–T039 tests pass.

---

## Phase 7: User Story 1 Extended — Optional Subsystem Benchmarking (Priority: P1)

**Goal**: Extend the core benchmarking to cover ERP, NSGA-II ranking, decoder, and tracking overhead when enabled in config. Populate `active_subsystems` and `caveats`.

**Independent Test**: Provide configs with each optional subsystem enabled and verify the corresponding phases appear in the breakdown.

### Tests for US1 Extended

- [X] T043 [P] [US1] Write failing test: `dry_run()` with ERP enabled includes `erp_intent` and `erp_matchability` phases in `phase_estimates` in `tests/unit/experiment/test_dry_run.py`
- [X] T044 [P] [US1] Write failing test: `dry_run()` with multiobjective config includes a `ranking` phase in `phase_estimates` in `tests/unit/experiment/test_dry_run.py`
- [X] T045 [P] [US1] Write failing test: `dry_run()` with a decoder configured includes a `decoding` phase in `phase_estimates` in `tests/unit/experiment/test_dry_run.py`
- [X] T046 [P] [US1] Write failing test: `dry_run()` with tracking enabled includes a `tracking` phase in `phase_estimates` in `tests/unit/experiment/test_dry_run.py`
- [X] T047 [P] [US1] Write failing test: `DryRunReport.active_subsystems` lists all enabled subsystems (e.g., `("erp", "multiobjective", "decoder", "tracking")`) in `tests/unit/experiment/test_dry_run.py`
- [X] T048 [P] [US1] Write failing test: `DryRunReport.caveats` includes ERP recovery caveat when ERP is enabled and remote tracking caveat when remote MLflow is configured in `tests/unit/experiment/test_dry_run.py`

### Implementation for US1 Extended

- [X] T049 [US1] Implement `_benchmark_erp(config, sample_population, timeout)` that benchmarks one intent evaluation call and one matchability check (including genome distance), returning `PhaseEstimate` objects for `erp_intent` and `erp_matchability` in `evolve/experiment/dry_run.py`
- [X] T050 [US1] Implement `_benchmark_ranking(config, sample_population, timeout)` that benchmarks `fast_non_dominated_sort()` and `crowding_distance()` — must create a population-sized set of random multi-objective fitnesses (not reuse the small sample from T019) — returning a `PhaseEstimate` for `ranking` in `evolve/experiment/dry_run.py`
- [X] T051 [US1] Implement `_benchmark_decoder(config, sample_genome, timeout)` that benchmarks one decode operation, returning a `PhaseEstimate` for `decoding` in `evolve/experiment/dry_run.py`
- [X] T052 [US1] Implement `_benchmark_tracking(config, timeout)` that benchmarks one `log_metrics()` call for local MLflow backends, returning a `PhaseEstimate` for `tracking` in `evolve/experiment/dry_run.py`
- [X] T053 [US1] Implement `_detect_active_subsystems(config)` that returns a tuple of active subsystem names by checking `config.erp`, `config.multiobjective`, `config.merge`, `config.decoder`, `config.tracking`, `config.meta` in `evolve/experiment/dry_run.py`
- [X] T054 [US1] Implement `_collect_caveats(config)` that builds a list of caveat strings: ERP recovery caveat if ERP enabled, remote tracking caveat if tracking is remote, early-stop caveat if stopping criteria configured, point-estimate caveat always in `evolve/experiment/dry_run.py`
- [X] T055 [US1] Integrate all optional subsystem benchmarks into `dry_run()`: conditionally call ERP/ranking/decoder/tracking benchmarkers, merge results into `phase_estimates`, set `active_subsystems` and `caveats` in `evolve/experiment/dry_run.py`

**Checkpoint**: Configs with any combination of ERP, NSGA-II, decoder, merge, and tracking produce correct phase breakdowns. All T043–T048 tests pass.

---

## Phase 8: User Story 4 — Memory and Scale Projections (Priority: P3)

**Goal**: Estimate peak memory usage based on genome size, population size, and history accumulation. Populate `MemoryEstimate` in the report.

**Independent Test**: Run dry-run with a known genome type and population size, verify memory estimate is reasonable.

### Tests for User Story 4

- [X] T056 [P] [US4] Write failing test: `DryRunReport.memory.genome_bytes > 0` for a vector genome config in `tests/unit/experiment/test_dry_run.py`
- [X] T057 [P] [US4] Write failing test: `DryRunReport.memory.total_bytes == population_bytes + history_bytes` in `tests/unit/experiment/test_dry_run.py`
- [X] T058 [P] [US4] Write failing test: `DryRunReport.summary()` includes a memory line showing population and history estimates in `tests/unit/experiment/test_dry_run.py`

### Implementation for User Story 4

- [X] T059 [US4] Implement `_estimate_memory(config, sample_genome)` that measures genome byte size via `sys.getsizeof` (or `.nbytes` for numpy arrays), computes `population_bytes = (genome_bytes + individual_overhead) × population_size × 2` and `history_bytes = metrics_dict_estimate × max_generations`, returning a `MemoryEstimate` in `evolve/experiment/dry_run.py`
- [X] T060 [US4] Integrate `_estimate_memory()` into `dry_run()` so `DryRunReport.memory` is populated in `evolve/experiment/dry_run.py`
- [X] T061 [US4] Ensure `DryRunReport.summary()` includes the memory line (`~XX MB population, ~YY MB history | Total: ~ZZ MB`) in `evolve/experiment/dry_run.py`

**Checkpoint**: Memory estimates are populated and displayed. All T056–T058 tests pass.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, documentation, and validation

- [X] T062 [P] Add `dry_run` and `DryRunReport` to `evolve/experiment/__init__.py` exports (if `__init__.py` exists and re-exports)
- [X] T063 [P] Add `MetaEstimate` to `evolve/experiment/__init__.py` exports (if applicable)
- [X] T064 Run `DryRunReport.summary()` output against the quickstart.md examples to verify format consistency
- [X] T065 Run full test suite (`pytest tests/unit/experiment/test_dry_run.py -v`) and confirm all tests pass
- [X] T066 Run type checker (`mypy evolve/experiment/dry_run.py --strict`) and fix any type errors
- [X] T067 Verify `dry_run()` completes in <10 seconds for a standard config (SC-001 validation)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 MVP (Phase 3)**: Depends on Phase 2 — core functionality
- **US2 (Phase 4)**: Depends on Phase 2 — can run in parallel with Phase 3
- **US3 (Phase 5)**: Depends on Phase 3 (needs `_compute_percentages_and_bottleneck` and `summary()`)
- **US5 (Phase 6)**: Depends on Phase 3 (reuses core benchmarking logic)
- **US1 Extended (Phase 7)**: Depends on Phase 3 (extends `dry_run()` with optional subsystems)
- **US4 (Phase 8)**: Depends on Phase 2 — can run in parallel with Phase 3
- **Polish (Phase 9)**: Depends on all previous phases

### User Story Dependencies

- **US1 (P1)**: After Phase 2 — no dependencies on other stories
- **US2 (P2)**: After Phase 2 — independent of US1 (resource detection is standalone)
- **US3 (P2)**: After US1 — needs percentage/bottleneck logic and summary()
- **US5 (P2)**: After US1 — reuses core benchmarking for inner run estimation
- **US1 Extended (P1)**: After US1 — extends the core `dry_run()` function
- **US4 (P3)**: After Phase 2 — independent of US1 (memory estimation is standalone)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Helpers before orchestrators
- Core logic before integration into `dry_run()`
- Story complete before moving to next priority

### Parallel Opportunities

- Phase 2: T003+T004, T005+T006, T007+T008, T009+T010 can be paired (dataclass + test), and pairs marked [P] can run in parallel
- Phase 3: T013–T017 (all tests) can run in parallel before implementation begins
- Phase 4: T026–T027 (tests) can run in parallel; T028–T030 (detection helpers) can run in parallel
- Phase 7: T043–T048 (all tests) can run in parallel; T049–T054 (subsystem benchmarkers) can run in parallel
- Phase 8: T056–T058 (tests) can run in parallel
- US2 (Phase 4) and US4 (Phase 8) can run in parallel with US1 (Phase 3)

---

## Parallel Example: User Story 1

```bash
# Launch all tests first (all [P]):
Task: T013 — dry_run returns DryRunReport with core phases
Task: T014 — merge phase included when enabled
Task: T015 — percentages sum to 100%, one bottleneck
Task: T016 — ValueError on invalid config
Task: T017 — structural constants match config

# Then implement sequentially (dependency chain):
Task: T018 — _validate_config
Task: T019 — _create_sample_population
Task: T020 — _derive_structural_constants
Task: T021 — _benchmark_phase
Task: T022 — _benchmark_core_phases (depends on T019, T020, T021)
Task: T023 — _compute_percentages_and_bottleneck
Task: T024 — dry_run() orchestrator (depends on T018–T023)
Task: T025 — DryRunReport.summary()
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T002)
2. Complete Phase 2: Foundational dataclasses (T003–T012)
3. Complete Phase 3: User Story 1 core benchmarking (T013–T025)
4. **STOP and VALIDATE**: `dry_run(config)` works for standard configs, returns ASCII table

### Incremental Delivery

1. Setup + Foundational → All dataclasses importable
2. Add US1 → Core cost estimate works (MVP!)
3. Add US2 → Hardware detection populates `resources`
4. Add US3 → Bottleneck flagged in table with ★
5. Add US5 → Meta-evolution multiplier shown
6. Add US1 Extended → ERP/NSGA-II/decoder/tracking phases appear
7. Add US4 → Memory estimates added
8. Polish → Exports, types, perf validation

### Notes

- [P] tasks = different files or independent functions, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
