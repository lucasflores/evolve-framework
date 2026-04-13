# Feature Specification: Feature & Cleanup Backlog

**Feature Branch**: `009-feature-cleanup-backlog`  
**Created**: 2026-04-13  
**Status**: Draft  
**Input**: User description: "Feature & Cleanup Backlog — bugs, features, and cleanup across population statistics, engine callbacks, population dynamics metrics, callback ordering, meta-evolution MLflow tracking, and UnifiedConfig extension"

## Clarifications

### Session 2026-04-13

- Q: Should PopulationStatistics store a minimize flag or rename fields to min_fitness/max_fitness? → A: Store `minimize` flag on PopulationStatistics — keep `best_fitness`/`worst_fitness` field names, add `minimize: bool` field.
- Q: Which callback ordering mechanism? → A: Numeric priority levels (e.g., `priority: int`, lower runs first; TrackingCallback defaults to high value like 1000).
- Q: How should representation-aware distance functions be resolved? → A: Protocol method on Genome — each genome type implements `distance(self, other) -> float`.
- Q: How should meta-evolution MLflow hierarchy be structured? → A: Nested runs for hierarchy via `mlflow.start_run(nested=True)`, plus tags on inner runs for easy filtering.
- Q: How should datasets be represented in UnifiedConfig? → A: Typed fields — `training_data` and `validation_data` with a simple wrapper type (name + reference/path + optional context string).

## User Scenarios & Testing

### User Story 1 — Correct Best/Worst Fitness Reporting (Priority: P1)

A researcher runs a maximization experiment (`minimize=False`) and checks `population.statistics.best_fitness` to see the strongest individual. Currently `_compute_statistics()` hardcodes `np.argmin`, so the reported "best" is actually the *worst* individual. The same inverted value propagates through `engine._compute_metrics()` into TrackingCallback, corrupting the MLflow `best_fitness` metric. After this fix, `best_fitness` always reflects the optimization direction.

**Why this priority**: Incorrect fitness reporting undermines every experiment that uses maximization. It silently corrupts tracking data, making published results unreliable.

**Independent Test**: Create a population with known fitness values, compute statistics with both `minimize=True` and `minimize=False`, and verify `best_fitness`/`worst_fitness` match the expected individuals. Verify the metrics dict from `_compute_metrics()` also reflects the correct direction.

**Acceptance Scenarios**:

1. **Given** a population with fitnesses [1.0, 5.0, 3.0] and `minimize=True`, **When** statistics are computed, **Then** `best_fitness` is 1.0 and `worst_fitness` is 5.0.
2. **Given** a population with fitnesses [1.0, 5.0, 3.0] and `minimize=False`, **When** statistics are computed, **Then** `best_fitness` is 5.0 and `worst_fitness` is 1.0.
3. **Given** the engine runs with `minimize=False`, **When** `_compute_metrics()` is called, **Then** the `best_fitness` key in the metrics dict equals the maximum fitness value.
4. **Given** TrackingCallback receives metrics, **When** it logs to MLflow, **Then** the logged `best_fitness` matches the correct optimization direction.

---

### User Story 2 — Callbacks Persist Through `engine.run()` (Priority: P1)

A researcher registers callbacks via `create_engine(..., callbacks=[my_callback])` and later calls `engine.run(population)` without re-passing callbacks. Currently `run()` overwrites `self._callbacks` with its `callbacks` parameter (defaulting to `[]`), silently discarding the registered callbacks. After this fix, callbacks from engine creation are preserved and merged with any additional callbacks passed to `run()`.

**Why this priority**: Silent callback loss is a data-integrity bug. Users lose tracking, checkpointing, or custom logic without any error or warning.

**Independent Test**: Create an engine with a HistoryCallback, call `run()` without passing callbacks, and verify the callback's `on_generation_end` was invoked.

**Acceptance Scenarios**:

1. **Given** an engine created with `callbacks=[history_cb]`, **When** `run(population)` is called with no `callbacks` argument, **Then** `history_cb.on_generation_end()` is called each generation.
2. **Given** an engine created with `callbacks=[cb_a]`, **When** `run(population, callbacks=[cb_b])` is called, **Then** both `cb_a` and `cb_b` receive all lifecycle events.
3. **Given** an engine with no creation-time callbacks, **When** `run(population, callbacks=[cb_a])` is called, **Then** `cb_a` receives all lifecycle events (backward-compatible).

---

### User Story 3 — `PopulationStatistics` Is Minimize-Aware (Priority: P1)

A consumer of `PopulationStatistics` (callback, downstream analysis) receives `best_fitness` and `worst_fitness` but has no way to know whether these refer to min or max. After cleanup, a `minimize: bool` field is added to `PopulationStatistics` so consumers can unambiguously interpret the values. The field names `best_fitness`/`worst_fitness` are preserved (no breaking rename), and their values are computed according to the stored `minimize` flag.

**Why this priority**: This is the data-model fix that makes bug #1 and feature #5 coherent. Without it, consumers cannot safely interpret the statistics.

**Independent Test**: Construct `PopulationStatistics` for a maximization problem and verify the `minimize` field is stored and `best_fitness`/`worst_fitness` reflect the correct direction.

**Acceptance Scenarios**:

1. **Given** a `PopulationStatistics` object, **When** a consumer inspects `stats.minimize`, **Then** the value is `True` or `False` matching the optimization direction, making best/worst unambiguous.
2. **Given** the engine computes statistics, **When** the `minimize` flag is `True` or `False`, **Then** `best_fitness` always refers to the individual with the best objective value for the given direction.
3. **Given** a `PopulationStatistics` created with `minimize=False`, **When** `best_fitness` is read, **Then** it equals the maximum fitness value in the population.

---

### User Story 4 — Correct `best_fitness`/`worst_fitness` in Metrics Dict (Priority: P1)

The `_compute_metrics()` method currently reads `stats.best_fitness` without considering `minimize`. After fixing bug #1 and cleanup #6, the metrics dict must propagate the corrected values. The `best_fitness` and `worst_fitness` keys in the metrics dict—which form the public API that callbacks and history consumers rely on—must reflect the optimization direction.

**Why this priority**: The metrics dict is the primary interface for callbacks. It must be correct for all downstream consumers including TrackingCallback, HistoryCallback, and user code.

**Independent Test**: Run the engine with `minimize=False`, capture the metrics dict, and verify `best_fitness` equals the maximum fitness in the population.

**Acceptance Scenarios**:

1. **Given** a minimization run, **When** the engine emits metrics, **Then** `best_fitness` equals the minimum fitness value.
2. **Given** a maximization run, **When** the engine emits metrics, **Then** `best_fitness` equals the maximum fitness value.
3. **Given** any run, **When** a callback receives the metrics dict via `on_generation_end`, **Then** it can trust `best_fitness` and `worst_fitness` reflect the correct optimization direction.

---

### User Story 5 — Native Population Dynamics Metrics (Priority: P2)

A researcher wants to diagnose *why* evolution stalls. Currently only `best_fitness`, `mean_fitness`, and `std_fitness` are tracked. The engine should natively compute richer metrics that answer: Is the population diverse or collapsed? Is the search still moving? Is evolution exploring or exploiting?

**Fitness distribution metrics**: median, quartiles (Q1/Q3), min/max (respecting minimize), fitness range, and count of unique fitness values (especially informative for discrete landscapes).

**Genome diversity metrics**: Mean per-gene standard deviation, mean distance from centroid, sampled pairwise distances. These must be representation-aware: L2 distance for vectors, edit distance for sequences, Hamming for binary, etc. Each genome type implements a `distance(self, other) -> float` protocol method. The engine calls this protocol method for all diversity computations, ensuring generalization across genome representations without external dispatch.

**Search movement metrics**: Centroid drift (distance between consecutive generation centroids), cosine similarity between consecutive best genomes, and a boolean "best changed" flag.

These metrics are computed in the engine and included in the metrics dict alongside existing fitness stats.

**Why this priority**: Rich diagnostics are the highest-value feature. Without them, users resort to ad-hoc analysis scripts that duplicate logic and miss cross-generation signals.

**Independent Test**: Run evolution for several generations with a known fitness landscape, capture the metrics dict, and verify each new metric key is present with expected types and reasonable values.

**Acceptance Scenarios**:

1. **Given** a population after generation N, **When** metrics are computed, **Then** the metrics dict includes `median_fitness`, `q1_fitness`, `q3_fitness`, `min_fitness`, `max_fitness`, `fitness_range`, and `unique_fitness_count`.
2. **Given** a vector population, **When** diversity metrics are computed, **Then** the metrics dict includes `mean_gene_std`, `mean_distance_from_centroid`, and `mean_pairwise_distance` using L2 distance.
3. **Given** a sequence population, **When** diversity metrics are computed, **Then** diversity metrics use edit distance (or equivalent representation-appropriate metric).
4. **Given** two consecutive generations, **When** movement metrics are computed, **Then** the metrics dict includes `centroid_drift`, `best_genome_similarity`, and `best_changed`.
5. **Given** a population with many duplicate fitness values, **When** `unique_fitness_count` is computed, **Then** it correctly counts the distinct fitness values.

---

### User Story 6 — Callback Priority / Ordering Guarantees (Priority: P2)

A researcher writes a custom callback that injects computed metrics (e.g., a diversity score) into the `metrics` dict. This callback must run *before* TrackingCallback so the injected metrics get logged to MLflow. Currently, ordering depends on list position and TrackingCallback is inserted by the factory before user callbacks. There is no way to express execution order.

After this feature, callbacks support numeric priority levels (`priority: int`, lower values run first). TrackingCallback defaults to a high priority value (e.g., 1000) so user callbacks for metric injection at default priority (0) naturally run before tracking. Callbacks with equal priority run in registration order (stable sort).

**Why this priority**: Without ordering guarantees, metric-injecting callbacks are fragile and position-dependent.

**Independent Test**: Register two callbacks where one injects a metric and the other logs metrics. Verify the injecting callback runs first regardless of registration order.

**Acceptance Scenarios**:

1. **Given** a callback with a higher priority (or earlier phase) than TrackingCallback, **When** `on_generation_end` is invoked, **Then** the high-priority callback runs before TrackingCallback.
2. **Given** two callbacks with the same priority, **When** they are invoked, **Then** they run in registration order (stable sort).
3. **Given** a callback registered at engine creation and another passed to `run()`, **When** both have priorities, **Then** ordering respects priorities regardless of registration source.

---

### User Story 7 — Meta-Evolution MLflow Tracking (Priority: P2)

A researcher runs meta-evolution (outer loop optimizing hyperparameters, inner loop running full evolution) and wants to see all activity in the MLflow UI. Currently the outer loop runs silently and inner `engine.run()` calls receive no callbacks, so the entire meta-evolution process is invisible to tracking.

After this feature, meta-evolution natively logs to MLflow using nested runs: a parent run for the outer loop and child runs (via `mlflow.start_run(nested=True)`) for each inner evolution trial. Inner runs also carry summary tags (`meta_generation`, `meta_parent_run_id`, `config_hash`) to enable programmatic filtering and querying independent of hierarchy.

**Why this priority**: Meta-evolution without observability is effectively unusable for serious research. Users cannot analyze why certain hyperparameter configurations succeed or fail.

**Independent Test**: Run a small meta-evolution (2 outer generations, 3 candidates each, 5 inner generations) with MLflow tracking enabled. Verify the MLflow UI shows the hierarchical structure with both outer and inner metrics logged.

**Acceptance Scenarios**:

1. **Given** a meta-evolution run with tracking enabled, **When** the outer loop executes, **Then** a parent MLflow run is created that logs outer-loop metrics (generation, best config fitness, parameter distributions).
2. **Given** an inner evolution trial, **When** it executes within meta-evolution, **Then** it appears as a nested/child MLflow run under the parent with full inner metrics (fitness progression, timing).
3. **Given** a completed meta-evolution, **When** viewing the MLflow UI, **Then** the hierarchical structure is navigable — the parent run summarizes the meta-evolution and child runs show individual trials.
4. **Given** a meta-evolution result, **When** the run completes, **Then** the best configuration is logged as an artifact on the parent run.

---

### User Story 8 — UnifiedConfig Extension: Datasets and Tags in MLflow (Priority: P3)

A researcher specifies training and validation datasets in `UnifiedConfig` and wants them logged to MLflow's native `Datasets` field (currently empty). Additionally, `UnifiedConfig.tags` creates a "tags" parameter but does not populate MLflow's native `Tags` field.

After this feature, `UnifiedConfig` has typed fields `training_data` and `validation_data` using a simple wrapper type carrying a name, data reference/path, and optional context string. TrackingCallback logs these to MLflow's native `Datasets` field using the `mlflow.log_input()` API. Tags populate MLflow's native `Tags` field via `mlflow.set_tags()` in addition to the existing parameter logging for backward compatibility.

**Why this priority**: Uses existing MLflow capabilities that are currently left empty. Improves experiment organization and discoverability.

**Independent Test**: Create a `UnifiedConfig` with datasets and tags, run evolution with tracking, and verify MLflow's native Datasets and Tags fields are populated.

**Acceptance Scenarios**:

1. **Given** a `UnifiedConfig` with training data specified, **When** TrackingCallback logs to MLflow, **Then** the dataset appears in MLflow's native `Datasets` field.
2. **Given** a `UnifiedConfig` with validation/test data specified, **When** TrackingCallback logs to MLflow, **Then** those datasets also appear in the native `Datasets` field.
3. **Given** a `UnifiedConfig` with `tags={"experiment_type": "ablation"}`, **When** TrackingCallback logs to MLflow, **Then** MLflow's native run `Tags` field contains those tags AND they remain accessible as parameters for backward compatibility.
4. **Given** a `UnifiedConfig` without datasets, **When** TrackingCallback logs, **Then** no error occurs and the `Datasets` field is simply empty.

---

### Edge Cases

- What happens when all individuals have identical fitness? (`unique_fitness_count` = 1, `std_fitness` = 0, diversity metrics still computable)
- What happens when the population has only one individual? (Statistics degenerate gracefully — no pairwise distances, centroid is the individual itself)
- What happens when a genome type has no natural distance metric? (A default fallback or clear error message is provided)
- What happens when meta-evolution inner runs fail? (Failures are tracked in MLflow with error status, not silently swallowed)
- What happens when callbacks mutate the metrics dict in `on_generation_end`? (Later callbacks see the mutated dict — this is the intended behavior for metric injection)
- What happens when both creation-time and run-time callbacks exist with conflicting priorities? (Priority ordering is deterministic regardless of source)
- What happens when `evaluation_data` is provided but `log_datasets` is False? (Datasets are not logged — config controls behavior)

## Requirements

### Functional Requirements

- **FR-001**: `Population._compute_statistics()` MUST respect the `minimize` flag when determining `best_fitness` and `worst_fitness`.
- **FR-002**: `PopulationStatistics` MUST store a `minimize: bool` field so consumers can unambiguously interpret `best_fitness`/`worst_fitness`. Field names are preserved (no rename).
- **FR-003**: `engine.run()` MUST preserve callbacks registered at engine creation time. Additional callbacks passed to `run()` MUST be merged with, not replace, creation-time callbacks.
- **FR-004**: `engine._compute_metrics()` MUST respect the `minimize` flag when labeling `best_fitness` and `worst_fitness` in the metrics dict.
- **FR-005**: The engine MUST compute fitness distribution metrics: median, Q1, Q3, min, max (direction-aware), fitness range, and unique fitness count.
- **FR-006**: The engine MUST compute genome diversity metrics: mean per-gene standard deviation, mean distance from centroid, and sampled mean pairwise distance.
- **FR-007**: Genome diversity metrics MUST be representation-aware via a `distance(self, other) -> float` protocol method on `Genome`. Each genome type implements this method with an appropriate distance function (L2 for vectors, edit distance for sequences, Hamming for binary, etc.).
- **FR-008**: The engine MUST compute search movement metrics: centroid drift, best-genome similarity (cosine similarity between consecutive best genomes), and a boolean `best_changed` flag.
- **FR-009**: All new metrics MUST be included in the metrics dict alongside existing fitness stats.
- **FR-010**: Callbacks MUST support numeric priority levels (`priority: int`, lower runs first). TrackingCallback MUST default to a high priority value (e.g., 1000). Callbacks with equal priority MUST run in registration order (stable sort).
- **FR-011**: Meta-evolution MUST log to MLflow using nested runs: a parent run for the outer loop with child runs for inner trials via `mlflow.start_run(nested=True)`. Inner runs MUST carry summary tags (`meta_generation`, `meta_parent_run_id`, `config_hash`) for programmatic filtering.
- **FR-012**: Inner evolution trials within meta-evolution MUST receive tracking callbacks so their metrics are visible in MLflow.
- **FR-013**: `UnifiedConfig` MUST have typed `training_data` and `validation_data` fields using a wrapper type carrying name, data reference/path, and optional context string.
- **FR-014**: TrackingCallback MUST log datasets to MLflow's native `Datasets` field when datasets are provided in the config.
- **FR-015**: TrackingCallback MUST log `UnifiedConfig.tags` to MLflow's native run `Tags` field in addition to the existing parameter logging.
- **FR-016**: All changes MUST maintain backward compatibility — existing code that does not use new features must continue to work without modification.
- **FR-017**: Population dynamics metrics computation MUST be gated by TrackingConfig categories so users can opt in/out of expensive computations.

### Key Entities

- **PopulationStatistics**: Aggregated metrics for a population, extended with minimize-awareness and richer fitness distribution data.
- **Metrics Dict**: The `dict[str, Any]` returned by `_compute_metrics()` — the public API for all callbacks and history consumers.
- **Callback Priority**: A numeric `priority: int` field on callbacks (lower values run first; default 0 for user callbacks, 1000 for TrackingCallback).
- **MetaEvolutionResult**: Extended with tracking metadata for the MLflow run hierarchy.
- **UnifiedConfig**: Extended with dataset fields and improved tag handling.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All fitness-related statistics and metrics report the correct optimization direction for both minimization and maximization problems — verifiable by unit tests comparing expected vs. actual values.
- **SC-002**: Callbacks registered at engine creation persist through any number of `run()` calls without re-registration — verifiable by lifecycle event counts.
- **SC-003**: Population dynamics metrics (fitness distribution, genome diversity, search movement) are available in the metrics dict for every generation — verifiable by key presence and type checks after a multi-generation run.
- **SC-004**: Genome diversity metrics produce valid results for at least vector and sequence genome types — verifiable by running evolution with each genome type and checking metric values.
- **SC-005**: Callback ordering is deterministic and respects declared priorities — verifiable by checking execution order in a test with multiple callbacks of different priorities.
- **SC-006**: Meta-evolution runs produce a navigable hierarchy in MLflow with parent and child runs — verifiable by querying MLflow for nested run structure after a meta-evolution run.
- **SC-007**: MLflow's native Datasets and Tags fields are populated when the corresponding data is provided in UnifiedConfig — verifiable by querying MLflow run metadata after a tracked run.
- **SC-008**: All existing tests continue to pass without modification — verifiable by running the full test suite before and after changes.

## Assumptions

- The framework's immutable/frozen dataclass pattern will be maintained — statistics and config objects remain frozen.
- MLflow's nested run API (`mlflow.start_run(nested=True)`) is available and suitable for the meta-evolution hierarchy.
- MLflow's `mlflow.log_input()` API is available for native dataset logging.
- The existing `TrackingConfig.categories` mechanism (with `MetricCategory` enum) is the right place to gate new metric computations.
- Representation-aware distance computation will use a protocol/dispatch pattern consistent with the existing `Genome` protocol, not require modifying every genome class.
- The `minimize` flag on `EvolutionConfig` is the authoritative source of truth for optimization direction.
- Backward compatibility means: existing code that doesn't use new features must produce identical behavior (same metrics, same callback invocations, same MLflow output).
