# Feature Specification: Comprehensive MLflow Metrics Tracking

**Feature Branch**: `006-mlflow-metrics-tracking`  
**Created**: March 17, 2026  
**Status**: Draft  
**Input**: User description: "Extend the evolve framework's experiment tracking to capture all observable metrics through MLflow integration, bridging the gap between UnifiedConfig/create_engine() and the existing ExperimentRunner tracking infrastructure."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Declarative Tracking with UnifiedConfig (Priority: P1)

A researcher using `create_engine()` wants to enable MLflow tracking without switching to `ExperimentRunner`. They add a `tracking` field to their `UnifiedConfig` and the engine automatically logs all metrics to MLflow during evolution.

**Why this priority**: This is the core use case—users currently cannot get any tracking when using the simpler `create_engine()` path. Enabling declarative tracking removes the biggest observability gap.

**Independent Test**: Can be fully tested by creating a `UnifiedConfig` with `tracking=TrackingConfig(...)`, running evolution, and verifying that an MLflow run exists with logged metrics.

**Acceptance Scenarios**:

1. **Given** a UnifiedConfig with `tracking=TrackingConfig(enabled=True, backend="mlflow")`, **When** `create_engine()` is called and evolution runs, **Then** an MLflow run is created with experiment name, parameters logged, and per-generation metrics recorded.
2. **Given** a UnifiedConfig with `tracking=None` (default), **When** evolution runs, **Then** no MLflow run is created and no tracking overhead occurs.
3. **Given** a UnifiedConfig with tracking enabled but MLflow not installed, **When** the engine is created, **Then** a clear error message indicates MLflow must be installed.

---

### User Story 2 - Full Population Statistics per Generation (Priority: P1)

A data scientist debugging a stuck evolution needs visibility into population health beyond best/mean fitness. They want to see diversity, worst fitness, species count, and other population-level statistics logged every generation.

**Why this priority**: Without comprehensive population metrics, users cannot diagnose common issues like premature convergence, diversity loss, or selection pressure problems. This is fundamental observability.

**Independent Test**: Can be tested by running evolution with enhanced metrics enabled and verifying that each generation log contains diversity_score, worst_fitness, fitness_range, and (when speciation is enabled) species_count.

**Acceptance Scenarios**:

1. **Given** tracking with enhanced population metrics enabled, **When** a generation completes, **Then** the tracker receives best_fitness, mean_fitness, std_fitness, worst_fitness, fitness_range, median_fitness, and diversity_score.
2. **Given** an evolution run with speciation enabled, **When** metrics are computed, **Then** species_count and average_species_size are included in recorded metrics.
3. **Given** an elitist strategy, **When** the next generation starts, **Then** elite_turnover_rate (fraction of new elites vs previous generation) is logged.

---

### User Story 3 - Timing Instrumentation (Priority: P2)

A performance engineer optimizing a large-scale evolution needs to identify bottlenecks. They want to see time breakdowns for evaluation, selection, crossover, and mutation phases logged per generation.

**Why this priority**: Performance profiling is critical for scaling, but secondary to basic observability. Users can work without timing data but benefit significantly when optimizing.

**Independent Test**: Can be tested by enabling timing metrics and verifying that generation_time_ms, evaluation_time_ms, selection_time_ms, and variation_time_ms are logged for each generation.

**Acceptance Scenarios**:

1. **Given** tracking with timing enabled, **When** a generation completes, **Then** total generation_time_ms is logged.
2. **Given** timing breakdown enabled, **When** evolution runs, **Then** evaluation_time_ms, selection_time_ms, crossover_time_ms, and mutation_time_ms are logged separately.
3. **Given** a parallel backend, **When** timing is enabled, **Then** wall_clock_time and cpu_time are distinguished.

---

### User Story 4 - ERP Mating Statistics (Priority: P2)

A user running Evolvable Reproduction Protocols wants visibility into mating dynamics—success rates, protocol effectiveness, and reproductive isolation patterns.

**Why this priority**: ERP is a major framework feature, but its internal metrics (successful_matings, attempted_matings) are computed but never exposed to tracking. Users cannot debug ERP behavior without this.

**Independent Test**: Can be tested by running ERP evolution with tracking enabled and verifying mating_success_rate, attempted_matings, and successful_matings appear in logged metrics.

**Acceptance Scenarios**:

1. **Given** an ERP evolution with tracking enabled, **When** a generation completes, **Then** mating_success_rate (successful/attempted) is logged.
2. **Given** ERP with multiple reproduction protocols, **When** metrics are collected, **Then** per-protocol success rates are available.
3. **Given** mating_success_rate drops to zero, **When** the next generation runs, **Then** an event or warning is logged indicating reproductive collapse.

---

### User Story 5 - Multi-Objective Metrics (Priority: P2)

A multi-objective optimization user wants to track Pareto front quality over generations, including hypervolume, IGD, spread, and front size.

**Why this priority**: Multi-objective metrics exist in the codebase but are never logged. Users doing NSGA-II style optimization cannot assess convergence without manually computing these metrics.

**Independent Test**: Can be tested by running a multi-objective optimization with tracking and verifying hypervolume, pareto_front_size, and spread metrics are logged.

**Acceptance Scenarios**:

1. **Given** a multi-objective config with tracking enabled, **When** a generation completes, **Then** hypervolume (if 2-3 objectives), pareto_front_size, and crowding_diversity are logged.
2. **Given** a reference point is configured, **When** hypervolume is computed, **Then** it uses the configured reference point.
3. **Given** more than 3 objectives, **When** hypervolume would be too expensive, **Then** approximate or alternative indicators (pareto_front_size, spread) are logged instead.

---

### User Story 6 - Fitness Metadata Extraction (Priority: P3)

A reinforcement learning user's fitness function returns rich metadata (episode rewards, steps, collisions, etc.) embedded in Fitness.metadata. They want this domain-specific data automatically extracted and logged.

**Why this priority**: This enables domain-specific observability without custom callbacks. Important but requires users to structure their evaluators correctly, so lower priority than core metrics.

**Independent Test**: Can be tested by creating an evaluator that populates Fitness.metadata and verifying those fields appear in logged metrics (prefixed appropriately).

**Acceptance Scenarios**:

1. **Given** individuals with Fitness.metadata containing {"episode_reward": 450, "steps": 200}, **When** metrics are collected, **Then** meta_episode_reward_best, meta_episode_reward_mean, meta_steps_best, meta_steps_mean are logged.
2. **Given** mixed metadata (some individuals have fields others don't), **When** aggregation occurs, **Then** only fields present in >50% of individuals are aggregated.
3. **Given** nested metadata structures, **When** extraction occurs, **Then** top-level numeric fields are extracted with flattened keys.

---

### User Story 7 - Derived Analytics Metrics (Priority: P3)

An experienced practitioner wants computed analytics that require combining multiple raw metrics: selection pressure, improvement velocity, population entropy, and convergence indicators.

**Why this priority**: Derived metrics provide actionable insights but require core metrics to work first. Useful for advanced users diagnosing subtle issues.

**Independent Test**: Can be tested by running evolution and verifying selection_pressure, fitness_improvement_velocity, and population_entropy appear in logged metrics.

**Acceptance Scenarios**:

1. **Given** tracking with derived metrics enabled, **When** a generation completes, **Then** selection_pressure (best/mean fitness ratio) is logged.
2. **Given** at least 2 generations have passed, **When** metrics are computed, **Then** fitness_improvement_velocity (rate of best fitness change) is logged.
3. **Given** a population with fitness values, **When** entropy is computed, **Then** population_entropy (based on fitness distribution bins) is logged.

---

### Edge Cases

- What happens when MLflow server is unreachable during a run?
  - Graceful degradation: log warning and continue evolution without tracking.
- How does the system handle runs that are interrupted mid-generation?
  - Partial metrics are logged; run is marked as failed/interrupted in MLflow.
- What happens when tracking is enabled but the backend library isn't installed?
  - Clear ImportError at engine creation time, not mid-run. Example: `ImportError: MLflow >= 2.0 required for tracking. Install with: pip install mlflow`
- How are extremely large populations (>10,000) handled for diversity metrics?
  - Sampling-based diversity computation with `diversity_sample_size=1000` (configurable) to cap overhead at O(sample²) instead of O(n²).
- What happens when Fitness.metadata contains non-numeric values?
  - Non-numeric fields are skipped during aggregation with debug log.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Tracking Integration

- **FR-001**: System MUST provide a `TrackingConfig` dataclass that specifies tracking backend, experiment name, and metric collection options.
- **FR-002**: `UnifiedConfig` MUST accept an optional `tracking: TrackingConfig` field that enables declarative tracking configuration.
- **FR-003**: `create_engine()` MUST automatically wire tracking callbacks when `config.tracking` is present.
- **FR-004**: System MUST preserve the existing `ExperimentRunner` + `ExperimentConfig` path as a fully functional alternative.
- **FR-005**: When tracking is enabled but the backend library is not installed, system MUST raise a clear ImportError at engine creation time.
- **FR-028**: When MLflow server becomes unreachable during evolution, system MUST buffer metrics in memory and attempt periodic reconnection with eventual consistency, allowing evolution to continue uninterrupted.
- **FR-029**: System MUST require MLflow 2.0 or higher, leveraging modern APIs including batch `mlflow.log_metrics()` and system metrics capabilities.

#### Extended Population Metrics

- **FR-006**: `compute_generation_metrics()` MUST be extended to compute worst_fitness, fitness_range, median_fitness, and quartiles (25th/75th percentiles) when enhanced metrics are enabled.
- **FR-007**: System MUST compute diversity_score per generation using configurable distance function (default: euclidean for vector genomes).
- **FR-008**: When speciation is enabled, system MUST include species_count, average_species_size, species_sizes, and largest_species_fitness in generation metrics.
- **FR-009**: System MUST compute elite_turnover_rate as fraction of elites that are new compared to previous generation.

#### Timing Instrumentation

- **FR-010**: `EvolutionEngine._step()` MUST record total generation_time_ms when timing is enabled.
- **FR-011**: System MUST optionally provide fine-grained timing breakdown: evaluation_time_ms, selection_time_ms, crossover_time_ms, mutation_time_ms.
- **FR-012**: Timing overhead MUST be less than 1% of total generation time for populations under 1000 individuals.

#### Specialized Metric Collectors

- **FR-013**: System MUST provide an ERP metric collector that captures mating_success_rate, attempted_matings, successful_matings, and per-protocol success rates.
- **FR-014**: System MUST provide a multi-objective metric collector that captures hypervolume (for 2-3 objectives), pareto_front_size, spread, and crowding_diversity.
- **FR-015**: System MUST provide a speciation metric collector that captures species_births, species_extinctions, stagnation_counts, and species dynamics per generation.
- **FR-016**: System MUST provide an islands metric collector (for island model parallelism) that captures inter_island_variance, intra_island_variance, and migration_events. *(Acceptance: Given island parallelism enabled with tracking, when a migration event occurs, then migration_events counter increments and inter_island_variance is logged.)*
- **FR-017**: System MUST provide a NEAT metric collector that captures average_node_count, average_connection_count, and topology_innovations. *(Acceptance: Given NEAT genome population with tracking, when evolution runs, then average_node_count and average_connection_count are logged per generation.)*

#### Fitness Metadata Extraction

- **FR-018**: System MUST automatically extract numeric fields from Fitness.metadata when metadata logging is enabled.
- **FR-019**: Extracted metadata fields MUST be aggregated (best, mean, std) across the population and logged with a distinguishing prefix (meta_*).
- **FR-020**: Metadata extraction MUST handle missing fields gracefully by only aggregating fields present in the majority of individuals.

#### Derived Analytics

- **FR-021**: System MUST compute selection_pressure as the ratio of best fitness to mean fitness.
- **FR-022**: System MUST compute fitness_improvement_velocity as the rate of change in best fitness over a configurable window (default: 5 generations).
- **FR-023**: System MUST compute population_entropy based on fitness distribution binning.

<!-- FR-024 removed: elite_turnover_rate already covered by FR-009 under Extended Population Metrics -->

#### Configuration & Opt-in

- **FR-025**: All extended metrics (timing, ERP, multi-objective, speciation, islands, NEAT, metadata, derived) MUST be opt-in via TrackingConfig flags.
- **FR-026**: Default TrackingConfig MUST log only core metrics (fitness statistics) to ensure tracking overhead remains under 5% of total evolution time (per SC-003).
- **FR-027**: TrackingConfig MUST be JSON-serializable for reproducibility and experiment versioning.

### Key Entities

- **TrackingConfig**: Configuration dataclass specifying backend (mlflow, wandb), experiment name, run name, metric categories to enable, and backend-specific options.
- **MetricCollector**: Protocol for specialized metric collectors (ERP, multi-objective, speciation, etc.) with `collect(population, context) -> dict[str, float]` interface.
- **EnhancedGenerationMetrics**: Extended metrics dictionary structure returned by expanded `compute_generation_metrics()`.
- **TimingContext**: Context manager for measuring phase durations within `_step()`.

### Assumptions

- MLflow 2.0+ is required; no compatibility layer for MLflow 1.x will be provided.
- Diversity metrics use sampling for populations larger than 1000 individuals to maintain performance.
- Hypervolume computation is only performed for 2-3 objective problems due to computational cost.
- Fitness.metadata is expected to contain flat (non-nested) numeric fields for automatic extraction.
- The existing `MLflowTracker` class will be extended rather than replaced.

## Clarifications

### Session 2026-03-17

- Q: When MLflow is unreachable during evolution (network failure, server down), what degradation behavior should occur? → A: Buffer metrics in memory and attempt periodic reconnection with eventual consistency
- Q: What MLflow version should be required for compatibility? → A: Require MLflow 2.0+ and use modern APIs (mlflow.log_metrics batch, system metrics)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can enable full MLflow tracking for a `create_engine()` workflow with 3 or fewer lines of configuration added to their existing UnifiedConfig.
- **SC-002**: At least 15 distinct metrics categories are available (core fitness, diversity, timing, ERP, multi-objective, speciation, islands, NEAT, metadata aggregates, derived analytics).
- **SC-003**: Tracking overhead (time added by metric collection and logging) is less than 5% of total evolution time for standard populations (100-1000 individuals).
- **SC-004**: All metrics logged during evolution can be visualized in MLflow UI without additional post-processing.
- **SC-005**: 100% of metrics currently computed but not logged (ERP mating stats, multi-objective hypervolume, speciation counts) are captured when their respective collectors are enabled.
- **SC-006**: Users can reproduce exact experiment results by loading a saved UnifiedConfig with TrackingConfig from a previous MLflow run.
- **SC-007**: Documentation includes a tutorial demonstrating end-to-end tracking workflow with MLflow UI screenshots.
