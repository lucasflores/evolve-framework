# Public API Contract: Dry-Run Statistics Tool

**Feature**: 014-dry-run-statistics  
**Date**: 2026-04-18  
**Module**: `evolve.experiment.dry_run`

## Exported Symbols

### Function: `dry_run`

```python
def dry_run(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable[[Any], float] | None = None,
    seed: int | None = None,
    timeout_per_phase: float = 30.0,
) -> DryRunReport:
    """
    Estimate the cost of an evolutionary run without executing it.

    Micro-benchmarks a single invocation of each atomic operation
    (evaluation, crossover, mutation, selection, merge) using the
    real configured backend, then multiplies by structural constants
    from the config to project total run time.

    Automatically detects and benchmarks optional subsystems enabled
    in the config: ERP (intent/matchability), NSGA-II (non-dominated
    sort/crowding), genome decoders, and tracking overhead.

    For meta-evolution configs, estimates inner run cost and multiplies
    by outer_generations × trials_per_config.

    Args:
        config: Complete experiment configuration. All operators,
            genome type, and evaluator must be resolvable via registries.
        evaluator: Optional explicit evaluator. If None, resolved from
            config.evaluator via the evaluator registry.
        seed: Random seed for sample genome creation. If None, uses
            config.seed or generates one.
        timeout_per_phase: Maximum seconds allowed for each micro-benchmark.
            If a phase exceeds this, it is aborted and reported as timed out.

    Returns:
        DryRunReport with per-phase timing breakdown, resource detection,
        memory estimate, meta-evolution breakdown (if applicable),
        active subsystem list, and formatted summary.

    Raises:
        ValueError: If config is invalid or missing required fields
            (e.g., no evaluator provided and none in config).
        KeyError: If an operator or genome type is not in the registry.
    """
```

**Signature contract**:
- Input: `UnifiedConfig` (required), plus optional overrides matching `create_engine` patterns
- Output: `DryRunReport` (frozen dataclass)
- Side effects: None (no files written, no state mutated, no tracking calls)
- Thread safety: Safe to call from any thread (no shared mutable state)

---

### Dataclass: `DryRunReport`

```python
@dataclass(frozen=True)
class DryRunReport:
    config_hash: str
    phase_estimates: tuple[PhaseEstimate, ...]
    total_estimated_ms: float
    estimated_generations: int
    resources: ComputeResources
    memory: MemoryEstimate
    seed_used: int
    early_stop_possible: bool
    active_subsystems: tuple[str, ...]
    meta_estimate: MetaEstimate | None
    caveats: tuple[str, ...]

    def summary(self) -> str:
        """Return formatted ASCII table summary."""

    def __str__(self) -> str:
        """Delegate to summary()."""
```

### Dataclass: `PhaseEstimate`

```python
@dataclass(frozen=True)
class PhaseEstimate:
    name: str
    measured_time_ms: float
    operations_per_generation: int
    estimated_total_ms: float
    percentage: float
    is_bottleneck: bool
```

### Dataclass: `ComputeResources`

```python
@dataclass(frozen=True)
class ComputeResources:
    cpu_count: int
    total_memory_bytes: int | None
    gpu_available: bool
    gpu_name: str | None
    gpu_memory_bytes: int | None
    backend_name: str
    backend_workers: int | None
```

### Dataclass: `MemoryEstimate`

```python
@dataclass(frozen=True)
class MemoryEstimate:
    genome_bytes: int
    individual_overhead_bytes: int
    population_bytes: int
    history_bytes: int
    total_bytes: int
```

### Dataclass: `MetaEstimate`

```python
@dataclass(frozen=True)
class MetaEstimate:
    inner_run_estimate_ms: float
    outer_generations: int
    trials_per_config: int
    total_inner_runs: int
    total_estimated_ms: float
```

## Versioning

- All exported types are frozen dataclasses (immutable)
- Adding new optional fields to dataclasses is a non-breaking change
- Removing or renaming fields is a breaking change
- The `dry_run()` function signature follows the same optional-parameter extension pattern as `create_engine()`

## Import Path

```python
from evolve.experiment.dry_run import dry_run, DryRunReport, MetaEstimate
```
