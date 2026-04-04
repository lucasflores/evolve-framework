"""
Islands Metric Collector.

Collects metrics from island model parallel evolution:
- inter_island_variance: Fitness variance between islands
- intra_island_variance: Average fitness variance within islands
- migration_events: Count of migration events

Implements FR-016 from the tracking specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.experiment.collectors.base import CollectionContext

if TYPE_CHECKING:
    from evolve.core.population import Population


_logger = logging.getLogger(__name__)


@dataclass
class IslandsMetricCollector:
    """
    Collector for island model parallelism metrics.

    Tracks fitness distribution across and within islands to monitor
    diversity and migration effectiveness.

    Attributes:
        track_migration: Whether to track migration event counts.
        warn_on_no_islands: Log warning when no island data available.

    Example:
        >>> from evolve.experiment.collectors.islands import IslandsMetricCollector
        >>>
        >>> collector = IslandsMetricCollector()
        >>> context = CollectionContext(
        ...     generation=10,
        ...     population=population,
        ...     island_populations=[island1, island2, island3],
        ... )
        >>> metrics = collector.collect(context)
        >>> metrics.get("inter_island_variance")
        0.25
    """

    track_migration: bool = True
    warn_on_no_islands: bool = True

    # Track warning state
    _warned_no_islands: bool = False

    # Track migration events
    _migration_count: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._warned_no_islands = False
        self._migration_count = 0

    def collect(self, context: CollectionContext) -> dict[str, Any]:
        """
        Collect island model metrics from context.

        Args:
            context: Collection context with island_populations.

        Returns:
            Dictionary of island metrics.
        """
        metrics: dict[str, Any] = {}

        if not context.has_islands():
            self._check_no_islands()
            return metrics

        island_populations = context.island_populations
        if not island_populations or len(island_populations) == 0:
            return metrics

        # Compute island means for inter-island variance
        island_means = []
        island_variances = []

        for island in island_populations:
            island_fitness = self._get_fitness_values(island)
            if len(island_fitness) > 0:
                island_means.append(np.mean(island_fitness))
                island_variances.append(np.var(island_fitness))

        if len(island_means) >= 2:
            # Inter-island variance: variance of island means
            metrics["inter_island_variance"] = float(np.var(island_means))

            # Intra-island variance: mean of island variances
            metrics["intra_island_variance"] = float(np.mean(island_variances))

            # Island count
            metrics["island_count"] = len(island_populations)

        # Migration events if tracked
        if self.track_migration:
            migration_count = self._get_migration_count(context)
            if migration_count is not None:
                metrics["migration_events"] = migration_count

        return metrics

    def reset(self) -> None:
        """Reset internal state between runs."""
        self._warned_no_islands = False
        self._migration_count = 0

    def record_migration(self, count: int = 1) -> None:
        """
        Record migration events.

        Call this when migrations occur to track the count.

        Args:
            count: Number of migration events to record.
        """
        self._migration_count += count

    def _get_fitness_values(self, population: Population[Any]) -> list[float]:
        """Extract fitness values from a population."""
        values = []
        for ind in population.individuals:
            if ind.fitness is not None:
                if hasattr(ind.fitness, "values"):
                    values.append(float(ind.fitness.values[0]))
                elif hasattr(ind.fitness, "value"):
                    values.append(float(ind.fitness.value))
        return values

    def _get_migration_count(self, context: CollectionContext) -> int | None:
        """
        Get migration count from context or internal state.

        Args:
            context: Collection context.

        Returns:
            Migration event count, or None.
        """
        # Check context extra for migration count
        if "migration_events" in context.extra:
            return int(context.extra["migration_events"])

        # Use internal tracked count
        if self._migration_count > 0:
            count = self._migration_count
            # Reset after reading
            self._migration_count = 0
            return count

        return None

    def _check_no_islands(self) -> None:
        """Log warning if no islands data available."""
        if self.warn_on_no_islands and not self._warned_no_islands:
            _logger.debug("No island populations in context - islands metrics not available")
            self._warned_no_islands = True
