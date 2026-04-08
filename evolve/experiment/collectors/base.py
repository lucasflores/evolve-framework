"""
Base types for metric collectors.

Provides the MetricCollector protocol and CollectionContext dataclass
that all specialized collectors implement.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.core.types import Individual


G = TypeVar("G")  # Genome type


@dataclass
class MatingStats:
    """
    Statistics from ERP mating operations.

    Collected by ERP engines during reproduction phase
    and passed to collectors via CollectionContext.

    Attributes:
        attempted_matings: Total mating attempts this generation.
        successful_matings: Successful matings producing offspring.
        protocol_attempts: Attempts per protocol name.
        protocol_successes: Successes per protocol name.
    """

    attempted_matings: int = 0
    successful_matings: int = 0
    protocol_attempts: dict[str, int] = field(default_factory=dict)
    protocol_successes: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall mating success rate."""
        if self.attempted_matings == 0:
            return 0.0
        return self.successful_matings / self.attempted_matings

    def protocol_success_rate(self, protocol: str) -> float:
        """Calculate success rate for a specific protocol."""
        attempts = self.protocol_attempts.get(protocol, 0)
        if attempts == 0:
            return 0.0
        successes = self.protocol_successes.get(protocol, 0)
        return successes / attempts


@dataclass
class CollectionContext:
    """
    Context passed to metric collectors during collection.

    Provides access to population state and optional domain-specific
    data (speciation info, mating stats, etc.) needed for metric computation.

    Attributes:
        generation: Current generation number.
        population: Current population of individuals.
        previous_elites: Elite individuals from previous generation (for turnover).
        species_info: Species assignments when speciation enabled.
        mating_stats: ERP mating statistics when ERP enabled.
        pareto_front: Current Pareto front when multi-objective enabled.
        island_populations: Per-island populations when island model enabled.
        elapsed_time_ms: Elapsed wall-clock time in milliseconds.
        rng_seed: Seed for any stochastic computations (determinism).
        extra: Additional domain-specific data.
    """

    generation: int
    population: Population[Any]

    # Optional context for specialized collectors
    previous_elites: list[Individual[Any]] | None = None
    species_info: dict[int, list[int]] | None = None  # species_id -> individual indices
    mating_stats: MatingStats | None = None
    pareto_front: list[Individual[Any]] | None = None
    island_populations: list[Population[Any]] | None = None
    elapsed_time_ms: float | None = None
    rng_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def population_size(self) -> int:
        """Get current population size."""
        return len(self.population)

    def has_speciation(self) -> bool:
        """Check if speciation info is available."""
        return self.species_info is not None

    def has_erp(self) -> bool:
        """Check if ERP mating stats are available."""
        return self.mating_stats is not None

    def has_pareto_front(self) -> bool:
        """Check if Pareto front is available."""
        return self.pareto_front is not None

    def has_islands(self) -> bool:
        """Check if island populations are available."""
        return self.island_populations is not None


class MetricCollector(Protocol[G]):  # type: ignore[misc]
    """
    Protocol for specialized metric collectors.

    Each collector computes domain-specific metrics from population state.
    Collectors are composable and opt-in via TrackingConfig.categories.

    Type Parameters:
        G: Genome type (covariant).

    Example:
        >>> from evolve.experiment.collectors.erp import ERPMetricCollector
        >>>
        >>> collector = ERPMetricCollector()
        >>> context = CollectionContext(
        ...     generation=10,
        ...     population=population,
        ...     mating_stats=mating_stats,
        ... )
        >>> metrics = collector.collect(context)
        >>> # metrics = {"mating_success_rate": 0.85, "attempted_matings": 100, ...}
    """

    def collect(self, context: CollectionContext) -> dict[str, float]:
        """
        Collect metrics from current population state.

        Args:
            context: Collection context with population and optional metadata.

        Returns:
            Dictionary of metric names to float values.
            Empty dict if no metrics can be computed from the given context.
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state between runs.

        Called when starting a new evolution run to clear history
        (e.g., for velocity computation that spans generations).
        """
        ...


__all__ = [
    "CollectionContext",
    "MatingStats",
    "MetricCollector",
]
