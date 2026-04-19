# Data Model: Dry-Run Statistics Tool

**Feature**: 014-dry-run-statistics  
**Date**: 2026-04-18

## Entities

### PhaseEstimate

Per-phase timing data for a single evolutionary phase.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Phase identifier: `"initialization"`, `"evaluation"`, `"decoding"`, `"selection"`, `"variation"`, `"merge"`, `"erp_intent"`, `"erp_matchability"`, `"ranking"`, `"tracking"` |
| measured_time_ms | float | Wall-clock time measured from micro-benchmark (single invocation or minimal batch) |
| operations_per_generation | int | Number of times this operation is called per generation (structural constant) |
| estimated_total_ms | float | Extrapolated total time: `measured_time_ms × operations_per_generation × generations` (or `× 1` for init) |
| percentage | float | Percentage of total estimated run time (0.0–100.0) |
| is_bottleneck | bool | True if this phase has the highest percentage |

**Validation rules**:
- `measured_time_ms >= 0.0`
- `operations_per_generation >= 0`
- `estimated_total_ms >= 0.0`
- `0.0 <= percentage <= 100.0`
- Exactly one phase has `is_bottleneck = True` per report

**Relationships**: Aggregated in `DryRunReport.phase_estimates` (one per active phase). Only phases that are enabled by the config are included — e.g., `"merge"` only appears if `merge_rate > 0`, `"erp_intent"` only if ERP is enabled, `"ranking"` only if multiobjective is configured, `"decoding"` only if a decoder is set.

---

### ComputeResources

Detected hardware capabilities of the execution environment.

| Field | Type | Description |
|-------|------|-------------|
| cpu_count | int | Number of available CPUs (container-aware) |
| total_memory_bytes | int ∣ None | Total system memory in bytes; None if undetectable |
| gpu_available | bool | Whether a GPU is present and usable |
| gpu_name | str ∣ None | GPU model name (e.g., "NVIDIA A100"); None if no GPU |
| gpu_memory_bytes | int ∣ None | GPU memory in bytes; None if no GPU or undetectable |
| backend_name | str | Configured backend name: `"sequential"`, `"parallel"`, `"jax"`, `"torch"` |
| backend_workers | int ∣ None | Number of parallel workers (parallel backend only); None otherwise |

**Validation rules**:
- `cpu_count >= 1`
- If `gpu_available is False`, then `gpu_name` and `gpu_memory_bytes` must be None

**Relationships**: Embedded in `DryRunReport.resources`.

---

### MemoryEstimate

Estimated memory usage for the full run.

| Field | Type | Description |
|-------|------|-------------|
| genome_bytes | int | Measured byte size of a single genome |
| individual_overhead_bytes | int | Estimated per-individual overhead (metadata, UUID, fitness) |
| population_bytes | int | Estimated memory for full population: `(genome_bytes + individual_overhead_bytes) × population_size × 2` |
| history_bytes | int | Estimated memory for metrics history: `estimated_metrics_dict_size × max_generations` |
| total_bytes | int | `population_bytes + history_bytes` |

**Validation rules**:
- All fields `>= 0`
- `total_bytes == population_bytes + history_bytes`

**Relationships**: Embedded in `DryRunReport.memory`.

---

### DryRunReport

Top-level output containing the complete dry-run analysis.

| Field | Type | Description |
|-------|------|-------------|
| config_hash | str | Hash of the input `UnifiedConfig` for traceability |
| phase_estimates | tuple[PhaseEstimate, ...] | Per-phase timing breakdown (ordered by percentage descending) |
| total_estimated_ms | float | Sum of all `estimated_total_ms` across phases |
| estimated_generations | int | `max_generations` from config (upper bound; stopping criteria may terminate earlier) |
| resources | ComputeResources | Detected compute environment |
| memory | MemoryEstimate | Memory usage projection |
| seed_used | int | Seed used for sample genome creation |
| early_stop_possible | bool | True if config has stopping criteria that could terminate before `max_generations` |
| active_subsystems | tuple[str, ...] | Names of optional subsystems active in the config (e.g., `("erp", "multiobjective", "merge", "decoder", "tracking")`) |
| meta_estimate | MetaEstimate ∣ None | Meta-evolution cost breakdown; None if meta-evolution is not enabled |
| caveats | tuple[str, ...] | Human-readable caveats about the estimate (e.g., "ERP recovery overhead not included", "Remote MLflow tracking latency not benchmarked") |

**State transitions**: None — this is an immutable output produced once.

**Relationships**:
- Contains `tuple[PhaseEstimate, ...]` (1:many)
- Contains `ComputeResources` (1:1)
- Contains `MemoryEstimate` (1:1)
- Contains `MetaEstimate` (0:1, only if meta-evolution enabled)

**Behavior**:
- `summary() → str`: Returns formatted ASCII table with timing breakdown, resource info, and memory estimate.
- `__str__()`: Delegates to `summary()`.

---

### MetaEstimate

Cost breakdown for meta-evolution, showing how the outer loop multiplies the inner run cost.

| Field | Type | Description |
|-------|------|-------------|
| inner_run_estimate_ms | float | Estimated cost of a single inner evolutionary run (sum of all inner phases × inner generations) |
| outer_generations | int | Number of outer-loop generations |
| trials_per_config | int | Number of inner runs per configuration candidate |
| total_inner_runs | int | `outer_generations × trials_per_config × population_size` (worst case, no cache hits) |
| total_estimated_ms | float | `inner_run_estimate_ms × total_inner_runs` |

**Validation rules**:
- `inner_run_estimate_ms >= 0.0`
- `outer_generations >= 1`
- `trials_per_config >= 1`
- `total_inner_runs == outer_generations × trials_per_config × population_size`

**Relationships**: Embedded in `DryRunReport.meta_estimate` (present only when meta-evolution is configured).

## Entity Relationship Summary

```
DryRunReport
├── phase_estimates: tuple[PhaseEstimate, ...]  (1:N, N = 4–10 depending on active subsystems)
├── resources: ComputeResources           (1:1)
├── memory: MemoryEstimate                (1:1)
├── meta_estimate: MetaEstimate           (0:1, if meta-evolution enabled)
└── caveats: tuple[str, ...]               (0:N)
```
