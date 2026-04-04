"""
Tracking Configuration.

Provides TrackingConfig dataclass for declarative experiment tracking setup
and MetricCategory enum for controlling metric collection granularity.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


class MetricCategory(Enum):
    """
    Categories of metrics that can be enabled for tracking.

    Metrics are organized into opt-in categories to control
    overhead and log volume. CORE is always enabled by default.

    Categories:
        CORE: Always enabled - best/mean/std fitness
        EXTENDED_POPULATION: worst, median, quartiles, range
        DIVERSITY: diversity_score, population_entropy
        TIMING: generation_time_ms, phase breakdowns
        SPECIATION: species_count, dynamics (auto when speciation enabled)
        MULTIOBJECTIVE: hypervolume, front_size (auto when MO enabled)
        ERP: mating_success_rate (auto when ERP enabled)
        METADATA: Fitness.metadata extraction
        DERIVED: selection_pressure, velocity, entropy
    """

    CORE = "core"
    EXTENDED_POPULATION = "extended_population"
    DIVERSITY = "diversity"
    TIMING = "timing"
    SPECIATION = "speciation"
    MULTIOBJECTIVE = "multiobjective"
    ERP = "erp"
    METADATA = "metadata"
    DERIVED = "derived"


@dataclass(frozen=True)
class TrackingConfig:
    """
    Configuration for experiment tracking (FR-001).

    JSON-serializable configuration specifying tracking backend,
    experiment naming, and metric collection options.

    This dataclass is frozen (immutable) for hashability and thread-safety.
    Use `with_category()` to create modified copies.

    Attributes:
        enabled: Master switch for tracking (default: True when present)
        backend: Tracking backend ("mlflow", "wandb", "local", "null")
        experiment_name: MLflow experiment name
        run_name: Optional run name (auto-generated if None)
        tracking_uri: MLflow tracking server URI (None = local ./mlruns/)
        categories: Set of enabled metric categories
        log_interval: Log every N generations (default: 1)
        buffer_size: Max metrics to buffer when disconnected
        flush_interval: Seconds between flush attempts
        metadata_threshold: Min fraction of individuals for field extraction
        metadata_prefix: Prefix for extracted metadata fields
        timing_breakdown: Enable fine-grained phase timing
        diversity_sample_size: Max samples for diversity computation
        hypervolume_reference: Reference point for hypervolume computation
        velocity_window: Generations for computing improvement velocity

    Example:
        >>> config = TrackingConfig(
        ...     experiment_name="sphere_optimization",
        ...     categories=frozenset({MetricCategory.CORE, MetricCategory.TIMING}),
        ... )
        >>> # Add a category
        >>> config = config.with_category(MetricCategory.DIVERSITY)
    """

    # -------------------------------------------------------------------------
    # Core settings
    # -------------------------------------------------------------------------

    enabled: bool = True
    """Master switch for tracking."""

    backend: Literal["mlflow", "wandb", "local", "null"] = "mlflow"
    """Tracking backend."""

    experiment_name: str = "evolve"
    """Experiment name for grouping runs."""

    run_name: str | None = None
    """Optional run name (auto-generated if None)."""

    tracking_uri: str | None = None
    """Tracking server URI (None = local ./mlruns/)."""

    # -------------------------------------------------------------------------
    # Metric categories (FR-025, FR-026)
    # -------------------------------------------------------------------------

    categories: frozenset[MetricCategory] = field(
        default_factory=lambda: frozenset({MetricCategory.CORE})
    )
    """Enabled metric categories. Default: CORE only for minimal overhead."""

    # -------------------------------------------------------------------------
    # Logging options
    # -------------------------------------------------------------------------

    log_interval: int = 1
    """Log every N generations."""

    buffer_size: int = 1000
    """Max metrics to buffer when server unreachable."""

    flush_interval: float = 30.0
    """Seconds between flush attempts when buffering."""

    # -------------------------------------------------------------------------
    # Metadata extraction (FR-018-020)
    # -------------------------------------------------------------------------

    metadata_threshold: float = 0.5
    """Min fraction of individuals with a field for it to be aggregated."""

    metadata_prefix: str = "meta_"
    """Prefix for extracted metadata fields."""

    # -------------------------------------------------------------------------
    # Timing options (FR-010-012)
    # -------------------------------------------------------------------------

    timing_breakdown: bool = False
    """Enable fine-grained phase timing (evaluation, selection, variation)."""

    # -------------------------------------------------------------------------
    # Diversity options
    # -------------------------------------------------------------------------

    diversity_sample_size: int = 1000
    """Max samples for diversity computation (performance requirement)."""

    # -------------------------------------------------------------------------
    # Multi-objective options
    # -------------------------------------------------------------------------

    hypervolume_reference: tuple[float, ...] | None = None
    """Reference point for hypervolume computation."""

    # -------------------------------------------------------------------------
    # Derived analytics options (FR-022)
    # -------------------------------------------------------------------------

    velocity_window: int = 5
    """Number of generations for computing fitness improvement velocity."""

    # -------------------------------------------------------------------------
    # System & advanced logging (FR-030+)
    # -------------------------------------------------------------------------

    system_metrics: bool = False
    """Enable system metrics logging (CPU, memory, GPU usage)."""

    log_datasets: bool = False
    """Enable dataset logging (logs initial population as MLflow dataset)."""

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        if not 0.0 <= self.metadata_threshold <= 1.0:
            raise ValueError("metadata_threshold must be in [0, 1]")
        if self.diversity_sample_size < 10:
            raise ValueError("diversity_sample_size must be >= 10")
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be > 0")
        if self.velocity_window < 1:
            raise ValueError("velocity_window must be >= 1")

    # -------------------------------------------------------------------------
    # Category helpers
    # -------------------------------------------------------------------------

    def has_category(self, category: MetricCategory) -> bool:
        """Check if a metric category is enabled."""
        return category in self.categories

    def with_category(self, *categories: MetricCategory) -> TrackingConfig:
        """
        Return new config with additional categories enabled.

        Args:
            *categories: Categories to add.

        Returns:
            New TrackingConfig with categories added.
        """
        from dataclasses import replace

        new_categories = self.categories | frozenset(categories)
        return replace(self, categories=new_categories)

    def without_category(self, *categories: MetricCategory) -> TrackingConfig:
        """
        Return new config with categories disabled.

        Args:
            *categories: Categories to remove.

        Returns:
            New TrackingConfig with categories removed.
        """
        from dataclasses import replace

        new_categories = self.categories - frozenset(categories)
        return replace(self, categories=new_categories)

    # -------------------------------------------------------------------------
    # Serialization (FR-027)
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_uri": self.tracking_uri,
            "categories": sorted(c.value for c in self.categories),
            "log_interval": self.log_interval,
            "buffer_size": self.buffer_size,
            "flush_interval": self.flush_interval,
            "metadata_threshold": self.metadata_threshold,
            "metadata_prefix": self.metadata_prefix,
            "timing_breakdown": self.timing_breakdown,
            "diversity_sample_size": self.diversity_sample_size,
            "hypervolume_reference": (
                list(self.hypervolume_reference) if self.hypervolume_reference else None
            ),
            "velocity_window": self.velocity_window,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingConfig:
        """
        Create from dictionary.

        Args:
            data: Dictionary from JSON deserialization.

        Returns:
            TrackingConfig instance.
        """
        data = data.copy()

        # Convert categories list to frozenset of enums
        if "categories" in data:
            data["categories"] = frozenset(MetricCategory(c) for c in data["categories"])

        # Convert hypervolume_reference list to tuple
        if data.get("hypervolume_reference"):
            data["hypervolume_reference"] = tuple(data["hypervolume_reference"])

        return cls(**data)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def minimal(cls) -> TrackingConfig:
        """
        Create config with core metrics only.

        Suitable for production runs where overhead must be minimal.
        Logs only best/mean/std fitness.

        Returns:
            TrackingConfig with CORE category only.
        """
        return cls(categories=frozenset({MetricCategory.CORE}))

    @classmethod
    def standard(cls, experiment_name: str = "evolve") -> TrackingConfig:
        """
        Create config with standard metrics for typical experiments.

        Includes core fitness, extended population stats, and timing.
        Good balance of observability and overhead.

        Args:
            experiment_name: Experiment name for grouping runs.

        Returns:
            TrackingConfig with CORE, EXTENDED_POPULATION, TIMING.
        """
        return cls(
            experiment_name=experiment_name,
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.EXTENDED_POPULATION,
                    MetricCategory.TIMING,
                }
            ),
        )

    @classmethod
    def comprehensive(cls, experiment_name: str = "evolve") -> TrackingConfig:
        """
        Create config with all metrics enabled.

        Suitable for detailed analysis and debugging.
        Higher overhead but maximum observability.

        Args:
            experiment_name: Experiment name for grouping runs.

        Returns:
            TrackingConfig with all categories enabled.
        """
        return cls(
            experiment_name=experiment_name,
            categories=frozenset(MetricCategory),
            system_metrics=True,
            log_datasets=True,
        )


__all__ = [
    "MetricCategory",
    "TrackingConfig",
]
