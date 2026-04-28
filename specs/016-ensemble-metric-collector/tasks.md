# Tasks: Ensemble Metric Collector & Reference Guide

**Input**: Design documents from `specs/016-ensemble-metric-collector/`  
**Branch**: `016-ensemble-metric-collector`  
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)  
**Total tasks**: 21 | **User stories**: 3 (P1, P2, P3)

---

## Task Format Reference

- **[P]** — Parallelisable: works on a different file from currently in-progress tasks with no pending dependencies
- **[US1/US2/US3]** — User story this task belongs to
- All implementation tasks include exact file paths

---

## Phase 1: Setup

**Purpose**: Confirm the test baseline is green on the feature branch before any source changes.

- [X] T001 Run `pytest tests/unit/experiment/collectors/ -x -q` on branch `016-ensemble-metric-collector` and confirm all existing collector tests pass with zero failures

**Checkpoint**: Baseline confirmed — safe to begin implementation.

---

## Phase 2: Foundational (Blocking Prerequisite)

**Purpose**: Add the new `MetricCategory.ENSEMBLE` enum value. Everything else depends on this string value existing for engine dispatch and type-safety.

⚠️ **No user story work can begin until T002 is complete.**

- [X] T002 Add `ENSEMBLE = "ensemble"` to the `MetricCategory` enum in `evolve/config/tracking.py` and update the class docstring to document the new value (mirrors the pattern of existing entries such as `SYMBIOGENESIS = "symbiogenesis"`)

**Checkpoint**: Foundation ready — `MetricCategory.ENSEMBLE` exists; all three user story phases can now proceed.

---

## Phase 3: User Story 1 — Enable Ensemble Observability (Priority: P1) 🎯 MVP

**Goal**: A researcher adds `MetricCategory.ENSEMBLE` (or the string `"ensemble"`) to their config and all applicable ensemble metrics appear automatically in every generation's logged data, without any changes to selection, reproduction, or fitness computation.

**Independent Test**: Construct a `CollectionContext` with a synthetic population and call `EnsembleMetricCollector().collect(context)` directly — no engine needed. Verify the returned dict contains the four always-present keys with finite float values in their documented ranges.

### Tests for User Story 1

> **Write these FIRST. Confirm they fail before writing any implementation.**

- [X] T003 [P] [US1] Create `tests/unit/experiment/collectors/test_ensemble.py` with mock fixtures (`MockFitness` dual-path, `MockIndividual`, `MockPopulation`, `make_context()` helper) following the pattern in `tests/unit/experiment/collectors/test_erp.py`; add test classes covering all US1 acceptance scenarios and edge cases:
  - `TestGiniCoefficient`: uniform population → `0.0`; single monopoly individual → `(N-1)/N`; zero-total fitness → `0.0`; ground-truth three-element case
  - `TestParticipationRatio`: uniform → `float(N)`; monopoly → `1.0`; zero-total → `float(N)`
  - `TestTopKConcentration`: monopoly → `1.0`; uniform → k/100; zero-total → `0.0`; top-k count clamped to ≥ 1 when k% rounds to 0
  - `TestExpertTurnover`: fully stable elite → `0.0`; fully replaced elite → `1.0`; empty `previous_elites` list → `1.0`; `previous_elites=None` → key absent from result
  - `TestEdgeCases`: empty population → `{}`; all individuals have `None` fitness → `{}`; negative fitness values are shifted before computation; single individual

### Implementation for User Story 1

- [X] T004 [US1] Create `evolve/experiment/collectors/ensemble.py` with the `EnsembleMetricCollector` `@dataclass` skeleton: `top_k_percent: float = 10.0`, `elite_size: int | None = None`, `__post_init__` validation (`ValueError` when `top_k_percent` not in `(0.0, 100.0]`; `ValueError` when `elite_size is not None and elite_size < 1`), `reset(self) -> None` no-op, stub `collect(self, context: CollectionContext) -> dict[str, float]` returning `{}`; include module-level `import numpy as np` and `# NO ML FRAMEWORK IMPORTS ALLOWED` header

- [X] T005 [US1] Implement `_extract_fitnesses(self, context: CollectionContext) -> np.ndarray | None` private helper in `evolve/experiment/collectors/ensemble.py`: iterate `context.population`, extract scalar via `hasattr(ind.fitness, "values")` dual-path (`float(ind.fitness.values[0])` or `float(ind.fitness.value)`), skip `None`-fitness individuals with a single `_logger.debug()` per skipped individual, collect into `np.ndarray`; return `None` when array is empty; apply negative-fitness shift (`fitnesses = fitnesses - np.min(fitnesses)`) when `np.min(fitnesses) < 0` and emit one `_logger.debug()` per generation where shifting is applied

- [X] T006 [US1] Implement `ensemble/gini_coefficient` in `EnsembleMetricCollector.collect()` in `evolve/experiment/collectors/ensemble.py` using the O(N log N) Lorenz cumulative-sum formula: sort shifted fitness array, compute `cumsum`; `gini = (N + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / N`; return `0.0` as the degenerate case when `np.sum(fitnesses) == 0`; call `_extract_fitnesses()` and short-circuit to `{}` when it returns `None`

- [X] T007 [US1] Implement `ensemble/participation_ratio` in `EnsembleMetricCollector.collect()` in `evolve/experiment/collectors/ensemble.py`: `pr = np.sum(fitnesses)**2 / np.sum(fitnesses**2)`; degenerate case: return `float(len(fitnesses))` when `np.sum(fitnesses**2) == 0`

- [X] T008 [US1] Implement `ensemble/top_k_concentration` in `EnsembleMetricCollector.collect()` in `evolve/experiment/collectors/ensemble.py`: `k_count = max(1, math.ceil(self.top_k_percent / 100 * N))`; use `np.partition(fitnesses, -k_count)[-k_count:]` to avoid full sort; `concentration = np.sum(top_k) / np.sum(fitnesses)`; degenerate case: return `0.0` when `np.sum(fitnesses) == 0`

- [X] T009 [US1] Implement `ensemble/expert_turnover` in `EnsembleMetricCollector.collect()` in `evolve/experiment/collectors/ensemble.py`: guard with `if context.previous_elites is None: return` for this metric only; compute `elite_count = self.elite_size if self.elite_size is not None else max(1, math.ceil(self.top_k_percent / 100 * N))`; derive current elite by sorting population by scalar fitness descending and taking the top `elite_count` individuals; compute `current_ids = {id(ind) for ind in current_elite}`; `prev_ids = {id(ind) for ind in context.previous_elites}`; `turnover = len(current_ids - prev_ids) / len(current_ids)`; degenerate case: return `1.0` when `context.previous_elites == []` (all current elites are new)

- [X] T010 [P] [US1] Register `EnsembleMetricCollector` in `evolve/experiment/collectors/__init__.py`: add `from evolve.experiment.collectors.ensemble import EnsembleMetricCollector` import and add `"EnsembleMetricCollector"` to `__all__`; update module docstring to list the new collector

- [X] T011 [US1] Add `EnsembleMetricCollector` dispatch to `evolve/core/engine.py`:
  - In `__init__`: add top-level import of `EnsembleMetricCollector` at the module level alongside `ERPMetricCollector`, `SpeciationMetricCollector`, and other collectors (not a lazy guard-scoped import); add `self._ensemble_collector: Any = None` and `self._prev_ensemble_elites: list[Any] | None = None`; set `self._ensemble_collector = EnsembleMetricCollector()` when `"ensemble" in config.metric_categories`
  - In `_compute_metrics()`: add a block `if "ensemble" in categories and self._ensemble_collector is not None:` that constructs `CollectionContext(population=population, generation=self._generation, previous_elites=self._prev_ensemble_elites)`, calls `self._ensemble_collector.collect(context)`, merges result into `metrics`, then updates `self._prev_ensemble_elites` by sorting the population by scalar fitness descending and taking the top `max(1, math.ceil(self._ensemble_collector.top_k_percent / 100 * len(population)))` individuals as the new elite set to track

**Checkpoint**: User Story 1 is fully functional. `EnsembleMetricCollector().collect(context)` returns correct metric values for all US1 acceptance scenarios. All T003 tests pass.

---

## Phase 4: User Story 2 — Metric Collector Reference Guide (Priority: P2)

**Goal**: A developer or researcher can open `docs/guides/metric-collectors-reference.md`, find any collector by name, and read its formula, plain-language intuition, evolutionary interpretation, enabling mechanism, and degenerate-case behavior — without consulting source code.

**Independent Test**: File exists at `docs/guides/metric-collectors-reference.md`; contains one section per collector class in `evolve/experiment/collectors/` (existing + new); each section includes formula, intuition sentence, interpretation sentence, enabling mechanism, and degenerate-case behavior per FR-015.

### Implementation for User Story 2

- [X] T012 [US2] Create `docs/guides/metric-collectors-reference.md` with: introduction paragraph, summary table listing all collector class names with their enabling mechanisms and primary metric keys, and section headers for each collector (the eight existing ones + `EnsembleMetricCollector`); add a `docs/guides/` directory if it does not already exist

- [X] T013 [US2] Write full documentation sections in `docs/guides/metric-collectors-reference.md` for four MetricCategory-gated collectors: `DerivedAnalyticsCollector` (`MetricCategory.DERIVED`; metrics: `selection_pressure`, `fitness_improvement_velocity`, `population_entropy`), `ERPMetricCollector` (`MetricCategory.ERP`, auto-enabled when ERP reproduction active; metrics: `mating_success_rate`, `attempted_matings`, per-protocol rates), `FitnessMetadataCollector` (`MetricCategory.METADATA`; metrics: extracted `Fitness.metadata` key-value pairs), `IslandsMetricCollector` (auto-enabled when island model active; metrics: per-island best/mean fitness, migration counts); each section uses the standard table: metric key | formula | intuition | evolutionary interpretation | degenerate-case behavior; the degenerate-case behavior column is **required per FR-015** — document what the metric returns (and why) when the denominator is zero or input is otherwise undefined, for every row

- [X] T014 [US2] Write full documentation sections in `docs/guides/metric-collectors-reference.md` for four remaining existing collectors: `MergeMetricCollector` (manual instantiation via `MetricCategory.SYMBIOGENESIS` engine guard; metrics: merge success rate, symbiont fate counts), `MultiObjectiveMetricCollector` (`MetricCategory.MULTIOBJECTIVE`, auto-enabled when MO active; metrics: `hypervolume`, `front_size`, crowding distance), `NEATMetricCollector` (manual instantiation; metrics: topology complexity, gene innovation counts), `SpeciationMetricCollector` (`MetricCategory.SPECIATION`; metrics: `species_count`, birth/extinction rates, stagnation); same table format as T013

- [X] T015 [US2] Write the `EnsembleMetricCollector` section in `docs/guides/metric-collectors-reference.md` covering all five metrics (`ensemble/gini_coefficient`, `ensemble/participation_ratio`, `ensemble/top_k_concentration`, `ensemble/expert_turnover`, `ensemble/specialization_index`); each metric row in the table MUST include a degenerate-case note per FR-015 — specifically: `participation_ratio` returns `float(population_size)` when total fitness is zero (all contribute equally in the uniform-zero limit); `specialization_index` returns `0.0` when total fitness variance is zero (no specialization in converged population)

**Checkpoint**: User Story 2 is complete. Reference guide covers all collectors with formulas, intuition, interpretation, enabling mechanisms, and degenerate-case notes.

---

## Phase 5: User Story 3 — Specialization Index Guard (Priority: P3)

**Goal**: `ensemble/specialization_index` appears in the result dict only when `context.species_info` is not `None`, its value correctly reflects how much of the fitness variance is explained by species membership (η²), and it returns `0.0` gracefully when total fitness variance is zero.

**Independent Test**: Two `CollectionContext` instances — one with `species_info` mapping two species with highly divergent mean fitnesses, one with `species_info=None` — confirm presence/absence of the key. A third context with all-identical fitness confirms the `0.0` degenerate case. A fourth with identical fitness distributions across species confirms `≈0.0`.

### Tests for User Story 3

- [X] T016 [P] [US3] Add `TestSpecializationIndex` test class to `tests/unit/experiment/collectors/test_ensemble.py`: two species with maximally divergent fitness → result close to `1.0`; two species with identical fitness distributions → result close to `0.0`; `species_info=None` → key absent; all-identical population fitness (SS_total == 0) → `0.0`; single individual with `species_info` → returns `0.0` (degenerate)

### Implementation for User Story 3

- [X] T017 [US3] Implement `ensemble/specialization_index` in `EnsembleMetricCollector.collect()` in `evolve/experiment/collectors/ensemble.py`: guard with `if context.species_info is None:` to omit this metric; extract UNSHIFTED scalar fitnesses (variance is shift-invariant; re-extract from population directly); compute `grand_mean = np.mean(fitnesses_raw)` and `SS_total = np.sum((fitnesses_raw - grand_mean)**2)`; degenerate case: add `"ensemble/specialization_index": 0.0` to result and return early when `SS_total == 0`; compute `SS_between = sum(n_s * (species_mean_s - grand_mean)**2 for each species s in context.species_info.items())`; add `"ensemble/specialization_index": float(SS_between / SS_total)` to result; species_info values are indices into `context.population`

**Checkpoint**: User Story 3 complete. Specialization Index guard is correct; all T016 tests pass.

---

## Phase 6: Polish & Cross-cutting Concerns

**Purpose**: Type safety, full regression pass, and config serialisation verification.

- [X] T018 [P] Run `mypy evolve/experiment/collectors/ensemble.py --strict` (or equivalent project mypy config) and fix any type annotation errors so the file is fully mypy-clean

- [X] T019 [P] Run `pytest tests/unit/experiment/collectors/test_ensemble.py -v` and confirm all tests pass, covering SC-006 (branch coverage for all spec edge cases), SC-007 (Gini ground-truth values), and SC-008 (Expert Turnover identity ground-truth values)

- [X] T020 [P] Run `pytest tests/unit/config/test_tracking.py -v` to confirm `MetricCategory.ENSEMBLE` serialises and deserialises correctly, `has_category()` returns `True` when the category is present, and no existing tracking tests regress

- [X] T020b [US1] Write an integration test in `tests/integration/test_ensemble_engine.py` asserting that constructing `EvolutionConfig(metric_categories=frozenset({"core", "ensemble"}))` and running for one generation produces `"ensemble/gini_coefficient"` as a key in `result.history[0]` — verifying SC-003 (enabling the category causes metrics to appear in logged run data end-to-end without any additional configuration)

---

## Dependencies (Story Completion Order)

```
T001 (baseline)
  └─► T002 (MetricCategory.ENSEMBLE) — BLOCKS all stories
        ├─► Phase 3 (US1): T003→T004→T005→T006/T007/T008→T009→T010→T011
        ├─► Phase 4 (US2): T012→T013→T014→T015  [independent of Phase 3]
        └─► Phase 5 (US3): T016→T017             [independent of Phase 4]
              └─► Phase 6: T018, T019, T020, T020b
```

US2 (reference guide) and US3 (specialization index) are **independent of each other** after T002. US2 can be worked in parallel with US1/US3 since it is pure documentation.

---

## Parallel Execution Examples

**Fastest sequential path (single agent)**:
T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016 → T017 → T018 → T019 → T020 → T020b

**Two-agent split**:
- Agent A: T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T016 → T017
- Agent B: (waits for T002) → T012 → T013 → T014 → T015

---

## Implementation Strategy

**MVP scope (User Story 1 only)**:
T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011

This delivers a fully functional `EnsembleMetricCollector` with four always-present metrics and `expert_turnover`. No reference guide and no `specialization_index` yet, but the feature is usable and all P1 acceptance criteria are met.

**Full delivery**: add T012–T015 (reference guide) and T016–T017 (specialization index), then T018–T020 (polish).

---

## Format Validation

All 21 tasks follow the checklist format:
- ✅ Every task starts with `- [ ]`
- ✅ Every task has a sequential ID (T001–T020, T020b)
- ✅ `[P]` marker present only where a different file is being modified with no pending file-level dependencies
- ✅ `[US1]`/`[US2]`/`[US3]` labels on all user story phase tasks
- ✅ Exact file path included in every task description
