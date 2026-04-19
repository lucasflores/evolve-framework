# Feature Specification: Dry-Run Statistics Tool

**Feature Branch**: `014-dry-run-statistics`  
**Created**: 2026-04-18  
**Status**: Draft  
**Input**: User description: "evolutionary run dry-run statistics tool. Input: UnifiedConfig and the available Computational Resources (this can be autodetermined). Output: how long the run will take with a granular break down therein (tells us which parts of the execution are most computationally expensive) and any other useful statistics."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quick Cost Estimate Before a Long Run (Priority: P1)

A researcher has configured an evolutionary experiment via `UnifiedConfig` and wants to understand how long the full run will take before committing compute resources. They invoke the dry-run tool, which micro-benchmarks a single invocation of each atomic operation (one evaluation, one crossover, one mutation, one selection pass) and multiplies by the structural constants derived from the config (population size × generation count). The tool also detects and benchmarks any advanced modes enabled in the config — ERP intent/matchability checks, NSGA-II non-dominated sorting, genome decoding, and metric collectors — so the estimate reflects the full configured pipeline. The researcher receives a granular breakdown showing estimated wall-clock time per phase and total run duration, enabling them to decide whether to proceed, adjust parameters, or switch backends.

**Why this priority**: This is the core value proposition — without a cost estimate, none of the other stories matter.

**Independent Test**: Can be fully tested by providing a `UnifiedConfig` with known parameters and verifying that the tool returns a timing breakdown with per-phase estimates and a total duration estimate.

**Acceptance Scenarios**:

1. **Given** a valid `UnifiedConfig` with `population_size=200`, `max_generations=500`, and a sequential backend, **When** the dry-run tool is invoked, **Then** the tool returns a report containing estimated total wall-clock time, per-phase breakdown (evaluation, selection, variation), per-generation average time, and the structural constants used for extrapolation.
2. **Given** a valid `UnifiedConfig` with merge enabled, **When** the dry-run tool is invoked, **Then** the merge phase appears in the breakdown with its own estimated cost.
3. **Given** a valid `UnifiedConfig`, **When** the dry-run tool completes, **Then** the per-operation micro-benchmark timings and the multipliers (population size, offspring count, generations) are reported alongside the estimates (see FR-010).
4. **Given** a valid `UnifiedConfig` with ERP enabled, **When** the dry-run tool is invoked, **Then** ERP intent evaluation and matchability checking appear as separate phases in the breakdown.
5. **Given** a valid `UnifiedConfig` with multiobjective settings, **When** the dry-run tool is invoked, **Then** non-dominated sorting and crowding distance computation appear as a "ranking" phase in the breakdown.
6. **Given** a valid `UnifiedConfig` with a decoder configured, **When** the dry-run tool is invoked, **Then** the decoding cost appears as a separate `"decoding"` phase in the breakdown.

---

### User Story 2 - Auto-Detect Computational Resources (Priority: P2)

A researcher wants the tool to automatically detect available compute resources — CPU count, available memory, GPU presence and type — so that the cost estimate reflects the actual hardware the run will execute on. The researcher does not need to manually specify hardware details; the tool inspects the environment and factors resource availability into its projections (e.g., parallel backend scales with CPU count, accelerated backends use GPU).

**Why this priority**: Accurate estimates require knowing the hardware. Auto-detection removes a manual step and reduces estimation error.

**Independent Test**: Can be tested by invoking the resource detection on a known machine and verifying that CPU count, memory, and GPU availability are correctly reported.

**Acceptance Scenarios**:

1. **Given** a machine with 8 CPU cores and 32 GB RAM, **When** the dry-run tool auto-detects resources, **Then** the report includes detected CPU count, available memory, and GPU status (present/absent).
2. **Given** a machine with a CUDA-capable GPU, **When** the dry-run tool is invoked with a config using an accelerated backend, **Then** the GPU model and memory are included in the resource report and factored into timing estimates.
3. **Given** a containerized environment with CPU limits, **When** the dry-run tool auto-detects resources, **Then** it respects container CPU limits rather than reporting host CPU count.

---

### User Story 3 - Identify Computational Bottlenecks (Priority: P2)

A researcher wants to understand which phase of the evolutionary pipeline dominates runtime so they can optimize the most impactful component. The dry-run report highlights the most expensive phase and provides relative percentages, making it immediately clear where optimization effort should be directed.

**Why this priority**: Knowing total time is useful, but knowing *where* the time is spent is what drives actionable optimization decisions.

**Independent Test**: Can be tested by running the dry-run on a config where evaluation is known to dominate, and verifying the report correctly identifies evaluation as the bottleneck with the highest percentage share.

**Acceptance Scenarios**:

1. **Given** a completed dry-run, **When** the report is generated, **Then** each phase includes both absolute estimated time and percentage of total time.
2. **Given** a config with an expensive evaluation function and fast operators, **When** the dry-run completes, **Then** the report identifies evaluation as the dominant phase (highest percentage).
3. **Given** a completed dry-run, **When** the report is generated, **Then** the phase with the highest percentage share is explicitly flagged as the bottleneck.

---

### User Story 4 - Memory and Scale Projections (Priority: P3)

A researcher wants to know estimated peak memory usage and data volumes for the full run so they can verify the machine has sufficient resources. The dry-run tool estimates memory based on genome size, population size, and history accumulation.

**Why this priority**: Prevents out-of-memory failures mid-run, which are costly in long experiments.

**Independent Test**: Can be tested by running the dry-run with a known genome type and population size, and verifying the memory estimate is within a reasonable range of measured usage.

**Acceptance Scenarios**:

1. **Given** a `UnifiedConfig` with `population_size=1000` and a vector genome of dimension 100, **When** the dry-run completes, **Then** the report includes estimated peak memory usage for the population.
2. **Given** a `UnifiedConfig` with MLflow tracking enabled, **When** the dry-run completes, **Then** the report includes an estimate of the number of metrics logged and approximate storage footprint.

---

### User Story 5 - Meta-Evolution Cost Estimation (Priority: P2)

A researcher has configured a meta-evolution experiment where an outer loop evolves `UnifiedConfig` objects, and each outer-loop evaluation runs a complete inner evolutionary run. The dry-run tool detects `config.is_meta_evolution` and estimates the total cost as `outer_generations × trials_per_config × inner_run_cost`, clearly showing the multiplicative nature of meta-evolution. This prevents researchers from accidentally launching a meta-evolution run that would take days when they expected hours.

**Why this priority**: Meta-evolution is the single largest hidden cost multiplier — a misconfigured meta-evolution run can be 1000× more expensive than expected. Surfacing this is critical for preventing wasted compute.

**Independent Test**: Can be tested by providing a `UnifiedConfig` with `config.meta` enabled and verifying the report shows the outer × inner multiplication structure.

**Acceptance Scenarios**:

1. **Given** a `UnifiedConfig` with meta-evolution enabled (`meta.outer_generations=10`, `meta.trials_per_config=3`, `population_size=20`, `max_generations=100`), **When** the dry-run tool is invoked, **Then** the report shows estimated inner run cost, the outer loop multiplier, and total estimated time as `inner_cost × 10 × 3 × 20` (outer_generations × trials_per_config × outer_population_size).
2. **Given** a meta-evolution config, **When** the report is generated, **Then** it clearly warns that meta-evolution multiplies the base cost and shows the multiplication breakdown.

---

### Edge Cases

- What happens when the evaluation function is stochastic and timings vary significantly across individuals? Since only one invocation is timed, the estimate is a point estimate; the report should note this limitation.
- What happens when the configured stopping criteria would terminate the run early (e.g., fitness threshold reached)? The report should note that the actual run may terminate earlier than `max_generations`.
- What happens when no evaluator is configured or the config is invalid? The tool should validate the config and return a clear error before attempting any benchmarking.
- What happens when a single operation micro-benchmark times out (e.g., an evaluation that hangs)? The tool should abort that benchmark, report the timeout, and still produce estimates for the remaining phases.
- What happens when meta-evolution is enabled but the inner config is incomplete? The tool should validate both outer and inner configs and report errors for each.
- What happens when ERP recovery triggers conditionally during a real run but not during benchmarking? The report should note that ERP recovery overhead is not included in the estimate and may add cost if mating success rate drops.
- What happens when MLflow tracking is configured with a remote server? The tool should note that per-generation tracking overhead depends on network latency and is not benchmarked.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a standalone function that accepts a `UnifiedConfig` as input and returns a `DryRunReport`, without executing a full evolutionary run.
- **FR-002**: System MUST micro-benchmark each atomic operation using the actual configured backend. For evaluation, this means invoking the configured evaluator's batch interface with a minimal batch (e.g., 2× worker count for parallel, one vectorized batch for GPU) to capture real parallelism and hardware behavior. For operators (crossover, mutation, selection, merge), a single invocation is benchmarked.
- **FR-003**: System MUST measure per-phase wall-clock time for: population initialization, evaluation (including decoding if a decoder is configured), selection, variation (crossover + mutation), merge (if enabled), ERP overhead (intent + matchability, if ERP enabled), and ranking (non-dominated sort + crowding distance, if multiobjective enabled).
- **FR-004**: System MUST compute estimated full-run time by multiplying per-operation micro-benchmark timings by the structural constants from the config: evaluation cost (derived from real backend batch timing, including decode) × number of batches × generations, variation cost × offspring_count × generations, selection cost × parents_needed × generations, merge cost × (population_size × merge_rate) × generations (if enabled), ERP overhead × population_size × generations (if ERP enabled), and ranking cost × generations (if multiobjective enabled).
- **FR-005**: System MUST auto-detect available computational resources: CPU core count (respecting container limits), available system memory, and GPU presence/type/memory.
- **FR-006**: System MUST report per-phase percentage of total estimated time and flag the phase with the highest share as the bottleneck.
- **FR-007**: System MUST estimate peak memory usage based on genome representation, population size, and history accumulation.
- **FR-008**: System MUST validate the `UnifiedConfig` before sampling and return actionable errors if the configuration is incomplete or invalid.
- **FR-009**: System MUST enforce a configurable timeout on each micro-benchmark invocation to prevent the dry-run itself from becoming expensive (e.g., a single evaluation that never returns).
- **FR-010**: System MUST report the per-operation micro-benchmark timings and the structural multipliers used for extrapolation alongside the final estimates, so users can assess estimate reliability.
- **FR-011**: System MUST return results as a structured `DryRunReport` data object for programmatic consumption. The report MUST also provide a human-readable summary as a formatted ASCII table showing each phase, its estimated time, percentage of total, and a bottleneck indicator on the dominant phase.
- **FR-012**: When `config.is_meta_evolution` is True, the system MUST estimate the cost of a single inner evolutionary run, then multiply by `outer_generations × trials_per_config × outer_population_size` to produce the total meta-evolution cost estimate (each outer generation evaluates the full outer population, each evaluation requiring `trials_per_config` inner runs). The report MUST show both the inner run estimate and the outer multiplication breakdown.
- **FR-013**: When `config.is_erp_enabled` is True, the system MUST benchmark ERP intent evaluation and matchability checking as additional phases, estimating per-generation cost as intent cost × candidate parents + matchability cost × attempted matings.
- **FR-014**: When `config.is_multiobjective` is True, the system MUST benchmark NSGA-II non-dominated sorting and crowding distance computation as a "ranking" phase, with cost scaling as O(M × N²) where M = number of objectives and N = population size.
- **FR-015**: When `config.decoder` is configured, the system MUST benchmark the decode operation and report it as a separate `"decoding"` phase in the breakdown (one decode per individual per evaluation).
- **FR-016**: When `config.tracking.enabled` is True, the system MUST note the per-generation tracking overhead in the report. For local backends, this can be benchmarked; for remote backends, a caveat about network latency MUST be included.
- **FR-017**: The system MUST report which optional subsystems are active (ERP, multiobjective, meta-evolution, decoder, merge, tracking) so the user understands which cost components are included in the estimate.

### Key Entities

- **DryRunReport**: The output data structure containing timing breakdown, resource detection results, memory estimates, bottleneck identification, and sample metadata.
- **ComputeResources**: Detected hardware capabilities — CPU count, memory, GPU info — used to contextualize estimates.
- **PhaseEstimate**: Per-phase timing data including measured sample time, extrapolated full-run time, and percentage of total.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can obtain a cost estimate for any valid `UnifiedConfig` in under 10 seconds for typical configurations, since only one invocation of each operation is benchmarked regardless of population size.
- **SC-002**: Time estimates are within 2× of actual run duration for 80% of configurations when the same hardware and evaluation function are used. *(Post-release validation target — verified empirically via benchmark suite, not an acceptance test.)*
- **SC-003**: The bottleneck phase identified by the tool matches the actual most-expensive phase (by wall-clock time) in 90% of real runs. *(Post-release validation target — verified empirically via benchmark suite, not an acceptance test.)*
- **SC-004**: Resource auto-detection correctly identifies CPU count, GPU presence, and available memory on Linux and macOS environments.
- **SC-005**: Users can make informed decisions about backend selection, population sizing, or parameter tuning based on the dry-run report before committing to a full run.

## Clarifications

### Session 2026-04-18

- Q: How should researchers invoke the dry-run statistics tool? → A: Standalone function `dry_run(config) → DryRunReport`
- Q: What estimation approach should the tool use? → A: Pure micro-benchmark — time exactly one invocation of each atomic operation (evaluation, crossover, mutation, selection), then multiply by structural constants from config (population size, offspring count, generation count)
- Q: How should the tool handle parallel/accelerated backend scaling? → A: Benchmark through the real configured backend — invoke the evaluator's batch interface with a minimal batch to capture actual parallelism/GPU behavior, then extrapolate
- Q: What format should the human-readable summary use? → A: Formatted string with an ASCII table showing phase, estimated time, percentage of total, and a bottleneck flag on the dominant phase

## Assumptions

- The cost of each atomic operation (evaluation, crossover, mutation, selection) is approximately uniform across individuals and across generations — i.e., the per-invocation micro-benchmark is representative of per-invocation cost at scale.
- The configured evaluator backend (sequential, parallel, JAX, PyTorch) is available and functional at dry-run time, so the tool can invoke it with a minimal batch to measure real backend behavior.
- The existing timing utilities in `evolve/utils/timing.py` provide sufficient precision for micro-benchmarking individual operation calls.
- Genome factory functions are available through the registry for the configured `genome_type`, enabling creation of sample genomes for micro-benchmarking without a full engine setup.
- The tool runs on the same machine (or equivalent hardware) where the full evolutionary run will execute; cross-machine estimation is out of scope.
- The dry-run tool is a pre-run utility, not an online profiler — it does not monitor or modify an in-progress evolution run.
- Container-aware CPU detection is handled via existing patterns in the parallel backend (`os.sched_getaffinity` or `/proc` inspection).
- ERP recovery overhead is conditional (triggers only when mating success rate drops below threshold) and cannot be reliably estimated from a micro-benchmark; it is excluded from the estimate with a caveat.
- NSGA-II non-dominated sort cost scales as O(M × N²); the benchmark uses the actual population size to produce a representative timing.
- Meta-evolution inner run cost is estimated by recursively applying the dry-run to the inner config; the outer loop cost is the inner estimate × outer_generations × trials_per_config × outer_population_size.
- MLflow tracking with a remote server introduces network latency that cannot be benchmarked locally; the report notes this as an unbenchmarkable overhead when remote tracking is configured.
