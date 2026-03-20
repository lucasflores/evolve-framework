# Data Model: MLflow Metrics Tracking

**Feature**: 006-mlflow-metrics-tracking  
**Date**: March 17, 2026  
**Status**: Complete

## Overview

This document defines the data structures for comprehensive MLflow metrics tracking integration into the evolve framework.

---

## 1. TrackingConfig

The central configuration dataclass for declarative tracking setup.

### Schema

```python
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

class MetricCategory(Enum):
    """Categories of metrics that can be enabled."""
    CORE = "core"                    # Always enabled: best/mean/std fitness
    EXTENDED_POPULATION = "extended_population"  # worst, median, quartiles
    DIVERSITY = "diversity"          # diversity_score, population_entropy
    TIMING = "timing"                # generation_time_ms, phase breakdowns
    SPECIATION = "speciation"        # species_count, dynamics (auto when speciation)
    MULTIOBJECTIVE = "multiobjective"  # hypervolume, front_size (auto when MO)
    ERP = "erp"                      # mating_success_rate (auto when ERP)
    METADATA = "metadata"            # Fitness.metadata extraction
    DERIVED = "derived"              # selection_pressure, velocity, entropy


@dataclass(frozen=True)
class TrackingConfig:
    """
    Configuration for experiment tracking (FR-001).
    
    JSON-serializable configuration specifying tracking backend,
    experiment naming, and metric collection options.
    
    Attributes:
        enabled: Master switch for tracking (default: True when present)
        backend: Tracking backend ("mlflow", "wandb", "local", "null")
        experiment_name: MLflow experiment name
        run_name: Optional run name (auto-generated if None)
        tracking_uri: MLflow tracking server URI (None = local)
        
        # Metric category flags (FR-025)
        categories: Set of enabled metric categories
        
        # Advanced options
        log_interval: Log every N generations (default: 1)
        buffer_size: Max metrics to buffer when disconnected
        flush_interval: Seconds between flush attempts
        
        # Metadata extraction options
        metadata_threshold: Min fraction of individuals for field extraction
        metadata_prefix: Prefix for extracted metadata fields
    """
    
    # Core settings
    enabled: bool = True
    backend: Literal["mlflow", "wandb", "local", "null"] = "mlflow"
    experiment_name: str = "evolve"
    run_name: str | None = None
    tracking_uri: str | None = None
    
    # Metric categories (FR-025, FR-026)
    categories: frozenset[MetricCategory] = field(
        default_factory=lambda: frozenset({MetricCategory.CORE})
    )
    
    # Logging options
    log_interval: int = 1
    buffer_size: int = 1000
    flush_interval: float = 30.0
    
    # Metadata extraction (FR-020)
    metadata_threshold: float = 0.5
    metadata_prefix: str = "meta_"
    
    # Timing options (FR-012)
    timing_breakdown: bool = False  # Fine-grained phase timing
    
    # Diversity options
    diversity_sample_size: int = 1000  # Max samples for diversity computation
    
    # Multi-objective options
    hypervolume_reference: tuple[float, ...] | None = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        if self.metadata_threshold < 0 or self.metadata_threshold > 1:
            raise ValueError("metadata_threshold must be in [0, 1]")
        if self.diversity_sample_size < 10:
            raise ValueError("diversity_sample_size must be >= 10")
    
    def has_category(self, category: MetricCategory) -> bool:
        """Check if a metric category is enabled."""
        return category in self.categories
    
    def with_category(self, *categories: MetricCategory) -> "TrackingConfig":
        """Return new config with additional categories enabled."""
        from dataclasses import replace
        new_categories = self.categories | frozenset(categories)
        return replace(self, categories=new_categories)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary (FR-027)."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_uri": self.tracking_uri,
            "categories": [c.value for c in self.categories],
            "log_interval": self.log_interval,
            "buffer_size": self.buffer_size,
            "flush_interval": self.flush_interval,
            "metadata_threshold": self.metadata_threshold,
            "metadata_prefix": self.metadata_prefix,
            "timing_breakdown": self.timing_breakdown,
            "diversity_sample_size": self.diversity_sample_size,
            "hypervolume_reference": self.hypervolume_reference,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingConfig":
        """Create from dictionary."""
        data = data.copy()
        if "categories" in data:
            data["categories"] = frozenset(
                MetricCategory(c) for c in data["categories"]
            )
        return cls(**data)
```

### Factory Methods

```python
@classmethod
def minimal(cls) -> "TrackingConfig":
    """Core metrics only, minimal overhead."""
    return cls(categories=frozenset({MetricCategory.CORE}))

@classmethod
def standard(cls, experiment_name: str = "evolve") -> "TrackingConfig":
    """Standard metrics for typical experiments."""
    return cls(
        experiment_name=experiment_name,
        categories=frozenset({
            MetricCategory.CORE,
            MetricCategory.EXTENDED_POPULATION,
            MetricCategory.TIMING,
        }),
    )

@classmethod
def comprehensive(cls, experiment_name: str = "evolve") -> "TrackingConfig":
    """All metrics enabled for detailed analysis."""
    return cls(
        experiment_name=experiment_name,
        categories=frozenset(MetricCategory),
    )
```

---

## 2. MetricCollector Protocol

Protocol for specialized metric collectors (FR-013 through FR-017).

### Schema

```python
from typing import Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.core.types import Individual

G = TypeVar("G")


class MetricCollector(Protocol[G]):
    """
    Protocol for specialized metric collectors.
    
    Each collector computes domain-specific metrics from population state.
    Collectors are composable and opt-in via TrackingConfig.categories.
    
    Example:
        >>> collector = ERPMetricCollector()
        >>> metrics = collector.collect(population, context)
        >>> # {"mating_success_rate": 0.75, "attempted_matings": 100, ...}
    """
    
    @property
    def category(self) -> MetricCategory:
        """The metric category this collector provides."""
        ...
    
    def collect(
        self,
        population: "Population[G]",
        context: "CollectionContext",
    ) -> dict[str, float]:
        """
        Collect metrics from current population state.
        
        Args:
            population: Current generation population
            context: Additional context (generation, config, timing, etc.)
            
        Returns:
            Dictionary of metric_name -> value
            
        Note:
            Metric names should be descriptive and include collector prefix
            to avoid collisions (e.g., "erp_mating_success_rate").
        """
        ...
    
    def reset(self) -> None:
        """
        Reset collector state between runs.
        
        Called when a new evolution run starts.
        """
        ...


@dataclass
class CollectionContext:
    """
    Context passed to metric collectors.
    
    Provides access to generation number, configuration,
    timing data, and previous metrics for derived computations.
    """
    
    generation: int
    config: "UnifiedConfig"
    tracking_config: TrackingConfig
    timing: dict[str, float]  # Phase timing results
    previous_metrics: dict[str, float] | None  # Previous generation
    rng: Random  # For sampling
    
    # Optional domain-specific context
    species: list["Species"] | None = None
    pareto_front: list["Individual"] | None = None
    mating_stats: "MatingStats" | None = None
    elite_ids: set[str] | None = None
    previous_elite_ids: set[str] | None = None
```

---

## 3. EnhancedGenerationMetrics

Extended metrics dictionary structure (FR-006 through FR-024).

### Schema

```python
from typing import TypedDict, NotRequired

class CoreMetrics(TypedDict):
    """Core fitness statistics (always computed)."""
    best_fitness: float
    mean_fitness: float
    std_fitness: float


class ExtendedPopulationMetrics(TypedDict, total=False):
    """Extended population statistics (FR-006)."""
    worst_fitness: float
    median_fitness: float
    fitness_range: float  # max - min
    q25_fitness: float
    q75_fitness: float


class DiversityMetrics(TypedDict, total=False):
    """Diversity measurements (FR-007)."""
    diversity_score: float  # Mean pairwise distance
    population_entropy: float  # Fitness distribution entropy


class TimingMetrics(TypedDict, total=False):
    """Timing instrumentation (FR-010, FR-011)."""
    generation_time_ms: float
    evaluation_time_ms: float
    selection_time_ms: float
    crossover_time_ms: float
    mutation_time_ms: float
    wall_clock_time_ms: float
    cpu_time_ms: float


class SpeciationMetrics(TypedDict, total=False):
    """Speciation statistics (FR-008, FR-015)."""
    species_count: int
    average_species_size: float
    largest_species_fitness: float
    species_births: int
    species_extinctions: int
    stagnation_count: int


class MultiObjectiveMetrics(TypedDict, total=False):
    """Multi-objective performance (FR-014)."""
    hypervolume: float
    pareto_front_size: int
    spread: float
    crowding_diversity: float


class ERPMetrics(TypedDict, total=False):
    """Evolvable Reproduction Protocol metrics (FR-013)."""
    mating_success_rate: float
    attempted_matings: int
    successful_matings: int
    # Per-protocol: protocol_{name}_success_rate


class DerivedMetrics(TypedDict, total=False):
    """Derived analytics (FR-021 through FR-024)."""
    selection_pressure: float  # best / mean
    fitness_improvement_velocity: float  # rate of change
    elite_turnover_rate: float  # fraction of new elites


class EnhancedGenerationMetrics(
    CoreMetrics,
    ExtendedPopulationMetrics,
    DiversityMetrics,
    TimingMetrics,
    SpeciationMetrics,
    MultiObjectiveMetrics,
    ERPMetrics,
    DerivedMetrics,
):
    """
    Complete generation metrics dictionary.
    
    Combines all metric categories. Only categories enabled in
    TrackingConfig will be populated.
    
    Example:
        >>> metrics = compute_enhanced_metrics(population, context)
        >>> metrics["best_fitness"]  # Always present
        10.5
        >>> metrics.get("hypervolume")  # Only if MO enabled
        None
    """
    generation: int  # Always included


# Type alias for flexibility
GenerationMetricsDict = dict[str, float | int]
```

---

## 4. Specialized Collectors

### ERPMetricCollector

```python
@dataclass
class ERPMetricCollector(MetricCollector[G]):
    """
    Collector for Evolvable Reproduction Protocol metrics (FR-013).
    
    Captures mating dynamics, protocol effectiveness, and
    reproductive isolation patterns.
    """
    
    @property
    def category(self) -> MetricCategory:
        return MetricCategory.ERP
    
    def collect(
        self,
        population: Population[G],
        context: CollectionContext,
    ) -> dict[str, float]:
        if context.mating_stats is None:
            return {}
        
        stats = context.mating_stats
        metrics = {
            "erp_mating_success_rate": (
                stats.successful / stats.attempted
                if stats.attempted > 0 else 0.0
            ),
            "erp_attempted_matings": float(stats.attempted),
            "erp_successful_matings": float(stats.successful),
        }
        
        # Per-protocol success rates
        for protocol, pstats in stats.by_protocol.items():
            metrics[f"erp_protocol_{protocol}_success_rate"] = (
                pstats.successful / pstats.attempted
                if pstats.attempted > 0 else 0.0
            )
        
        return metrics
    
    def reset(self) -> None:
        pass  # No state
```

### MultiObjectiveMetricCollector

```python
@dataclass
class MultiObjectiveMetricCollector(MetricCollector[G]):
    """
    Collector for multi-objective metrics (FR-014).
    
    Computes hypervolume, Pareto front size, spread, and
    crowding diversity for 2-3 objective problems.
    """
    
    @property
    def category(self) -> MetricCategory:
        return MetricCategory.MULTIOBJECTIVE
    
    def collect(
        self,
        population: Population[G],
        context: CollectionContext,
    ) -> dict[str, float]:
        from evolve.multiobjective import (
            pareto_front,
            hypervolume_2d,
            crowding_distance,
        )
        
        if context.pareto_front is None:
            # Compute Pareto front if not provided
            front = pareto_front(population.individuals)
        else:
            front = context.pareto_front
        
        metrics: dict[str, float] = {
            "mo_pareto_front_size": float(len(front)),
        }
        
        if len(front) > 0 and front[0].fitness:
            n_objectives = len(front[0].fitness.objectives)
            
            if n_objectives <= 3:
                # Compute hypervolume for 2-3 objectives
                ref = context.tracking_config.hypervolume_reference
                if ref is None:
                    # Auto-compute reference point from worst values
                    ref = self._compute_reference(front)
                
                if n_objectives == 2:
                    points = np.array([
                        ind.fitness.objectives for ind in front
                    ])
                    metrics["mo_hypervolume"] = hypervolume_2d(
                        points, np.array(ref)
                    )
            
            # Crowding diversity
            if len(front) >= 2:
                cd = crowding_distance(front)
                metrics["mo_crowding_diversity"] = float(np.mean(cd))
        
        return metrics
```

### DerivedAnalyticsCollector

```python
@dataclass
class DerivedAnalyticsCollector(MetricCollector[G]):
    """
    Collector for derived analytics (FR-021 through FR-024).
    
    Computes selection pressure, improvement velocity, and
    elite turnover from core metrics.
    """
    
    _history: list[float] = field(default_factory=list)
    _window_size: int = 10
    
    @property
    def category(self) -> MetricCategory:
        return MetricCategory.DERIVED
    
    def collect(
        self,
        population: Population[G],
        context: CollectionContext,
    ) -> dict[str, float]:
        # Get current fitness stats from previous collectors
        prev = context.previous_metrics or {}
        
        fitness_values = [
            ind.fitness.value for ind in population.individuals
            if ind.fitness is not None
        ]
        
        if not fitness_values:
            return {}
        
        best = max(fitness_values)
        mean = float(np.mean(fitness_values))
        
        metrics: dict[str, float] = {}
        
        # Selection pressure (FR-021)
        if mean > 0:
            metrics["derived_selection_pressure"] = best / mean
        
        # Track best fitness for velocity
        self._history.append(best)
        
        # Fitness improvement velocity (FR-022)
        if len(self._history) >= 2:
            recent = self._history[-self._window_size:]
            if len(recent) >= 2:
                velocity = (recent[-1] - recent[0]) / len(recent)
                metrics["derived_fitness_velocity"] = velocity
        
        # Population entropy (FR-023)
        if len(fitness_values) > 1:
            # Bin fitness values and compute entropy
            hist, _ = np.histogram(fitness_values, bins=20)
            probs = hist / len(fitness_values)
            probs = probs[probs > 0]  # Remove zeros
            entropy = -np.sum(probs * np.log(probs))
            metrics["derived_population_entropy"] = float(entropy)
        
        # Elite turnover rate (FR-024)
        if context.elite_ids and context.previous_elite_ids:
            new_elites = context.elite_ids - context.previous_elite_ids
            turnover = len(new_elites) / len(context.elite_ids)
            metrics["derived_elite_turnover_rate"] = turnover
        
        return metrics
    
    def reset(self) -> None:
        self._history.clear()
```

---

## 5. TimingContext

Context manager for phase timing (FR-010, FR-011).

```python
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

@contextmanager
def timing_context(
    label: str,
    results: dict[str, float],
) -> Iterator[None]:
    """
    Context manager for timing code blocks.
    
    Records elapsed time in milliseconds to results dict.
    
    Example:
        >>> timing = {}
        >>> with timing_context("evaluation", timing):
        ...     evaluate_population(pop)
        >>> timing["evaluation_time_ms"]
        45.2
    """
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        results[f"{label}_time_ms"] = elapsed_ms


@dataclass
class GenerationTimer:
    """
    Accumulates timing for a single generation.
    
    Provides both total generation time and optional
    fine-grained phase breakdown.
    """
    
    _start: float | None = None
    _results: dict[str, float] = field(default_factory=dict)
    _breakdown_enabled: bool = True
    
    def start(self) -> None:
        """Start generation timer."""
        self._start = perf_counter()
        self._results.clear()
    
    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a phase within the generation."""
        if self._breakdown_enabled:
            with timing_context(name, self._results):
                yield
        else:
            yield
    
    def finish(self) -> dict[str, float]:
        """Finish timing and return results."""
        if self._start is not None:
            self._results["generation_time_ms"] = (
                (perf_counter() - self._start) * 1000
            )
        return self._results.copy()
```

---

## 6. Relationships

```
UnifiedConfig
    └── tracking: TrackingConfig | None
            ├── categories: frozenset[MetricCategory]
            └── [backend settings]

create_engine()
    └── TrackingCallback
            ├── tracker: MetricTracker (MLflowTracker, LocalTracker, etc.)
            └── collectors: list[MetricCollector]
                    ├── ExtendedPopulationCollector
                    ├── DiversityCollector
                    ├── ERPMetricCollector
                    ├── MultiObjectiveMetricCollector
                    ├── SpeciationMetricCollector
                    ├── NEATMetricCollector
                    ├── MetadataCollector
                    └── DerivedAnalyticsCollector

EvolutionEngine._step()
    └── GenerationTimer
            └── timing_context("phase", results)

compute_enhanced_metrics()
    ├── Core metrics (always)
    └── Collector metrics (opt-in per category)
```

---

## 7. Migration Notes

### From ExperimentRunner

Existing `ExperimentRunner` + `ExperimentConfig` path remains fully functional (FR-004). Users can migrate incrementally:

```python
# Old approach (still works)
runner = ExperimentRunner(config, engine, population, tracker=MLflowTracker())
result = runner.run()

# New approach with UnifiedConfig
config = UnifiedConfig(
    ...,
    tracking=TrackingConfig.standard("my_experiment"),
)
engine = create_engine(config, evaluator)
result = engine.run(population)  # Tracking automatic
```

### JSON Schema Compatibility

`TrackingConfig.to_dict()` produces JSON-serializable output compatible with `UnifiedConfig.to_json()`:

```json
{
  "name": "my_experiment",
  "population_size": 100,
  "tracking": {
    "enabled": true,
    "backend": "mlflow",
    "experiment_name": "evolve",
    "categories": ["core", "timing", "extended_population"],
    "log_interval": 1
  }
}
```
