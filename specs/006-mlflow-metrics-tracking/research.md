# Research: MLflow Metrics Tracking Integration

**Feature**: 006-mlflow-metrics-tracking  
**Date**: March 17, 2026  
**Status**: Complete

## Table of Contents

1. [MLflow 2.0 API Best Practices](#1-mlflow-20-api-best-practices)
2. [Graceful Degradation Patterns](#2-graceful-degradation-patterns)
3. [Metric Aggregation Strategies](#3-metric-aggregation-strategies)
4. [Performance Considerations](#4-performance-considerations)
5. [Existing Codebase Integration Points](#5-existing-codebase-integration-points)

---

## 1. MLflow 2.0 API Best Practices

### Decision: Use MLflow 2.0+ batch logging APIs

**Rationale**: MLflow 2.0 introduced significant improvements for high-frequency metric logging:

- **Batch `log_metrics()`**: Single API call for multiple metrics reduces network overhead
- **System metrics**: Built-in CPU/memory tracking (opt-in)
- **Async logging**: Background thread for non-blocking metric submission

**Implementation Pattern**:

```python
# Preferred: Batch logging (MLflow 2.0+)
metrics = {
    "best_fitness": 10.5,
    "mean_fitness": 5.2,
    "diversity_score": 0.85,
    "generation_time_ms": 150.3,
}
mlflow.log_metrics(metrics, step=generation)

# Avoid: Individual log_metric calls
# mlflow.log_metric("best_fitness", 10.5, step=generation)  # N network calls
```

**Alternatives Considered**:

1. **Individual `log_metric()` calls**: Rejected due to N network calls per generation
2. **Custom buffering layer**: MLflow 2.0 already provides this internally
3. **MLflow 1.x compatibility**: Rejected per FR-029; unnecessary maintenance burden

### Decision: Use explicit run management

**Rationale**: Explicit `start_run()`/`end_run()` provides cleaner resource management than context managers for long-running evolutions.

```python
class MLflowTracker:
    def start_run(self, config: TrackingConfig) -> None:
        mlflow.start_run(run_name=config.run_name)
        mlflow.log_params(config.to_params_dict())
        
    def end_run(self) -> None:
        mlflow.end_run()
```

---

## 2. Graceful Degradation Patterns

### Decision: In-memory buffering with periodic flush

**Rationale**: Evolution runs should never fail due to tracking infrastructure issues (FR-028).

**Implementation Pattern**:

```python
@dataclass
class ResilientMLflowTracker:
    """Tracker with graceful degradation for unreachable servers."""
    
    _buffer: list[dict[str, Any]] = field(default_factory=list)
    _buffer_size_limit: int = 1000
    _last_flush_attempt: float = 0.0
    _flush_interval: float = 30.0  # seconds
    _connected: bool = True
    
    def log_generation(self, generation: int, metrics: dict[str, float]) -> None:
        """Log metrics, buffering if server unreachable."""
        self._buffer.append({"step": generation, "metrics": metrics})
        
        if self._should_flush():
            self._try_flush()
    
    def _try_flush(self) -> None:
        """Attempt to flush buffer to MLflow server."""
        try:
            for entry in self._buffer:
                mlflow.log_metrics(entry["metrics"], step=entry["step"])
            self._buffer.clear()
            self._connected = True
        except Exception as e:
            logger.warning(f"MLflow server unreachable: {e}. Metrics buffered.")
            self._connected = False
    
    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return (
            len(self._buffer) >= self._buffer_size_limit or
            time.time() - self._last_flush_attempt >= self._flush_interval
        )
```

**Error Handling Hierarchy**:

1. **Network timeout**: Buffer metrics, retry on next generation
2. **Authentication failure**: Log error once, continue with buffering
3. **Server error (5xx)**: Exponential backoff retry
4. **Buffer overflow**: Log warning, drop oldest metrics (circular buffer)

**Alternatives Considered**:

1. **Fail-fast**: Rejected - evolution more valuable than tracking
2. **Write-through to local file**: Adds complexity; MLflow has local artifact store
3. **Discard on failure**: Loses observability data; buffering preferred

---

## 3. Metric Aggregation Strategies

### Decision: Hierarchical metric categories with opt-in granularity

**Rationale**: Different users need different detail levels. Core metrics (fitness stats) should always be available with minimal overhead. Enhanced metrics are opt-in per FR-025/FR-026.

**Metric Categories**:

| Category | Metrics | Default | Cost |
|----------|---------|---------|------|
| **Core** | best/mean/std_fitness | Always | O(n) |
| **Extended Population** | worst, median, quartiles, range | Opt-in | O(n log n) |
| **Diversity** | diversity_score, population_entropy | Opt-in | O(n²) or O(n·k) sampled |
| **Timing** | generation_time_ms, phase breakdowns | Opt-in | Negligible |
| **Speciation** | species_count, sizes, dynamics | Opt-in (auto when speciation enabled) | O(s) |
| **Multi-Objective** | hypervolume, front_size, spread | Opt-in (auto when MO enabled) | O(n log n) |
| **ERP** | mating_success_rate, protocol stats | Opt-in (auto when ERP enabled) | O(n) |
| **Metadata** | Fitness.metadata extraction | Opt-in | O(n·m) |
| **Derived** | selection_pressure, velocity, entropy | Opt-in | O(1) from other metrics |

### Decision: Aggregation functions for population-level metrics

**Rationale**: Fitness values need multiple aggregations to understand distribution.

```python
def aggregate_fitness_values(values: np.ndarray) -> dict[str, float]:
    """Compute comprehensive fitness statistics."""
    return {
        "best_fitness": float(np.max(values)),
        "worst_fitness": float(np.min(values)),
        "mean_fitness": float(np.mean(values)),
        "median_fitness": float(np.median(values)),
        "std_fitness": float(np.std(values)),
        "fitness_range": float(np.ptp(values)),
        "q25_fitness": float(np.percentile(values, 25)),
        "q75_fitness": float(np.percentile(values, 75)),
    }
```

### Decision: Metadata extraction with majority-field policy

**Rationale**: Fitness.metadata may have inconsistent keys across individuals. Only aggregate fields present in >50% of population to avoid sparse metrics.

```python
def extract_metadata_metrics(
    individuals: Sequence[Individual],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Extract numeric metadata fields present in majority of individuals."""
    metadata_lists: dict[str, list[float]] = defaultdict(list)
    
    for ind in individuals:
        if ind.fitness and ind.fitness.metadata:
            for key, value in ind.fitness.metadata.items():
                if isinstance(value, (int, float)):
                    metadata_lists[key].append(float(value))
    
    result = {}
    min_count = len(individuals) * threshold
    
    for key, values in metadata_lists.items():
        if len(values) >= min_count:
            result[f"meta_{key}_best"] = max(values)
            result[f"meta_{key}_mean"] = float(np.mean(values))
            result[f"meta_{key}_std"] = float(np.std(values))
    
    return result
```

---

## 4. Performance Considerations

### Decision: Sampling-based diversity for large populations

**Rationale**: Pairwise diversity is O(n²). For populations >1,000, use sampling to maintain <1% overhead (FR-012).

**Implementation**:

```python
def compute_diversity_score(
    genomes: Sequence[G],
    distance_fn: DistanceFunction[G],
    max_samples: int = 1000,
    rng: Random | None = None,
) -> float:
    """Compute diversity with optional sampling for large populations."""
    n = len(genomes)
    
    if n <= max_samples:
        # Full pairwise computation
        distances = [
            distance_fn(genomes[i], genomes[j])
            for i in range(n) for j in range(i + 1, n)
        ]
    else:
        # Sampling-based estimation
        rng = rng or Random()
        sample_indices = rng.sample(range(n), max_samples)
        sample = [genomes[i] for i in sample_indices]
        distances = [
            distance_fn(sample[i], sample[j])
            for i in range(len(sample)) for j in range(i + 1, len(sample))
        ]
    
    return float(np.mean(distances)) if distances else 0.0
```

**Performance Budget**:

| Population Size | Diversity Method | Pairs Computed | Time @ 10μs/pair |
|-----------------|------------------|----------------|------------------|
| 100 | Full | 4,950 | 50ms |
| 1,000 | Full | 499,500 | 5s |
| 1,000 | Sampled (k=1000) | 499,500 | 5s |
| 10,000 | Sampled (k=1000) | 499,500 | 5s |
| 100,000 | Sampled (k=1000) | 499,500 | 5s |

### Decision: TimingContext with minimal overhead

**Rationale**: Phase timing should add negligible overhead (<1% for populations <1,000).

```python
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timing_context(label: str, results: dict[str, float]) -> Iterator[None]:
    """Context manager for timing code blocks."""
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        results[f"{label}_time_ms"] = elapsed_ms
```

**Overhead Analysis**:

- `perf_counter()` calls: ~100ns each
- Dictionary assignment: ~50ns
- **Total per context**: ~250ns (0.00025ms)
- **5 phase timings per generation**: ~1.25μs

For a 100ms generation, timing overhead is 0.00125% — well under 1% target.

---

## 5. Existing Codebase Integration Points

### UnifiedConfig Extension

**Current State**: `UnifiedConfig` has optional nested configs (`stopping`, `callbacks`, `erp`, `multiobjective`, `meta`).

**Integration Point**: Add `tracking: TrackingConfig | None = None` field following existing pattern.

```python
# evolve/config/unified.py (lines ~180-190)
@dataclass(frozen=True)
class UnifiedConfig:
    # ... existing fields ...
    
    stopping: StoppingConfig | None = None
    callbacks: CallbackConfig | None = None
    erp: ERPSettings | None = None
    multiobjective: MultiObjectiveConfig | None = None
    meta: MetaEvolutionConfig | None = None
    tracking: TrackingConfig | None = None  # NEW
```

### create_engine() Wiring

**Current State**: `create_engine()` in `evolve/factory/engine.py` builds callbacks from `config.callbacks`.

**Integration Point**: Extend callback building to include tracking callback when `config.tracking` is present.

```python
# evolve/factory/engine.py
def _build_callbacks(config: UnifiedConfig) -> list[Callback]:
    callbacks = []
    # ... existing callback building ...
    
    # Add tracking callback if configured
    if config.tracking:
        from evolve.experiment.tracking import create_tracking_callback
        callbacks.append(create_tracking_callback(config.tracking))
    
    return callbacks
```

### Existing Metrics Integration

**Current State**: `compute_generation_metrics()` in `evolve/experiment/metrics.py` computes basic fitness stats.

**Integration Point**: Extend with optional enhanced metrics.

```python
# evolve/experiment/metrics.py
def compute_generation_metrics(
    fitness_values: list[float],
    diversity: float | None = None,
    *,
    # NEW optional parameters
    enhanced: bool = False,
    population: Population | None = None,
    collectors: list[MetricCollector] | None = None,
) -> dict[str, float]:
    """Compute generation metrics with optional enhancements."""
    # ... existing implementation ...
```

### MLflowTracker Enhancement

**Current State**: `MLflowTracker` in `evolve/experiment/tracking/mlflow_tracker.py` has basic logging.

**Integration Point**: Add batch logging, buffering, and reconnection logic.

---

## Summary of Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | MLflow 2.0+ batch logging | Reduces network overhead, modern API |
| 2 | In-memory buffering for resilience | Evolution continues when server unreachable |
| 3 | Hierarchical metric categories | Users opt into detail level they need |
| 4 | Majority-field metadata extraction | Handles inconsistent Fitness.metadata |
| 5 | Sampling diversity at k=1000 | O(1) complexity for large populations |
| 6 | perf_counter timing | <1μs overhead per phase |
| 7 | TrackingConfig in UnifiedConfig | Matches existing nested config pattern |
