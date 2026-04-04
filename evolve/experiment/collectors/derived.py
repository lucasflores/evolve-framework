"""
Derived Analytics Collector.

Computes analytical metrics that combine multiple raw metrics:
- selection_pressure: Ratio of best to mean fitness
- fitness_improvement_velocity: Change rate over configurable window
- population_entropy: Diversity measure using fitness histogram

Implements FR-021, FR-022, FR-023 from the tracking specification.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.experiment.collectors.base import CollectionContext

if TYPE_CHECKING:
    pass


_logger = logging.getLogger(__name__)


@dataclass
class DerivedAnalyticsCollector:
    """
    Collector for computed analytics metrics.

    Derives higher-level insights from raw population metrics including
    selection pressure, fitness velocity, and population entropy.

    Attributes:
        velocity_window: Number of generations for velocity computation.
        entropy_bins: Number of bins for population entropy histogram.
        enable_selection_pressure: Compute selection_pressure metric.
        enable_velocity: Compute fitness_improvement_velocity.
        enable_entropy: Compute population_entropy.

    Example:
        >>> from evolve.experiment.collectors.derived import DerivedAnalyticsCollector
        >>>
        >>> collector = DerivedAnalyticsCollector(velocity_window=5)
        >>> context = CollectionContext(generation=10, population=population)
        >>> metrics = collector.collect(context)
        >>> metrics.get("selection_pressure")
        1.25
    """

    velocity_window: int = 5
    entropy_bins: int = 20
    enable_selection_pressure: bool = True
    enable_velocity: bool = True
    enable_entropy: bool = True

    # History state for velocity computation
    _best_fitness_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _mean_fitness_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _generation_history: deque[int] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        """Initialize history with correct maxlen."""
        # Ensure history deques have appropriate maxlen
        if not isinstance(self._best_fitness_history, deque):
            object.__setattr__(
                self, "_best_fitness_history", deque(maxlen=max(self.velocity_window * 2, 50))
            )
        if not isinstance(self._mean_fitness_history, deque):
            object.__setattr__(
                self, "_mean_fitness_history", deque(maxlen=max(self.velocity_window * 2, 50))
            )
        if not isinstance(self._generation_history, deque):
            object.__setattr__(
                self, "_generation_history", deque(maxlen=max(self.velocity_window * 2, 50))
            )

    def collect(self, context: CollectionContext) -> dict[str, Any]:
        """
        Collect derived analytics metrics from context.

        Args:
            context: Collection context with population.

        Returns:
            Dictionary of derived analytics metrics.
        """
        metrics: dict[str, Any] = {}

        # Get fitness statistics
        stats = context.population.statistics

        best_fitness = self._extract_best_fitness(stats, context)
        mean_fitness = self._extract_mean_fitness(stats, context)

        # Update history
        if best_fitness is not None:
            self._best_fitness_history.append(best_fitness)
        if mean_fitness is not None:
            self._mean_fitness_history.append(mean_fitness)
        self._generation_history.append(context.generation)

        # Compute selection pressure
        if self.enable_selection_pressure:
            sp = self._compute_selection_pressure(best_fitness, mean_fitness)
            if sp is not None:
                metrics["selection_pressure"] = sp

        # Compute fitness improvement velocity
        if self.enable_velocity:
            velocity = self._compute_velocity()
            if velocity is not None:
                metrics["fitness_improvement_velocity"] = velocity

        # Compute population entropy
        if self.enable_entropy:
            entropy = self._compute_entropy(context)
            if entropy is not None:
                metrics["population_entropy"] = entropy

        return metrics

    def reset(self) -> None:
        """Reset history state between runs."""
        self._best_fitness_history.clear()
        self._mean_fitness_history.clear()
        self._generation_history.clear()

    def _extract_best_fitness(
        self,
        stats: Any,
        context: CollectionContext,
    ) -> float | None:
        """Extract best fitness value from statistics."""
        if hasattr(stats, "best_fitness") and stats.best_fitness is not None:
            if hasattr(stats.best_fitness, "values"):
                return float(stats.best_fitness.values[0])
            elif hasattr(stats.best_fitness, "value"):
                return float(stats.best_fitness.value)
            return float(stats.best_fitness)

        # Fallback: get from population best
        try:
            best_list = context.population.best(1)
            if best_list and best_list[0].fitness is not None:
                fitness = best_list[0].fitness
                if hasattr(fitness, "values"):
                    return float(fitness.values[0])
                # Numeric fitness values (single-objective)
                return float(fitness)  # type: ignore[arg-type]
        except (AttributeError, TypeError, IndexError):
            pass

        return None

    def _extract_mean_fitness(
        self,
        stats: Any,
        context: CollectionContext,
    ) -> float | None:
        """Extract mean fitness value from statistics."""
        if hasattr(stats, "mean_fitness") and stats.mean_fitness is not None:
            if hasattr(stats.mean_fitness, "values"):
                return float(stats.mean_fitness.values[0])
            elif hasattr(stats.mean_fitness, "value"):
                return float(stats.mean_fitness.value)
            return float(stats.mean_fitness)

        # Fallback: compute from population
        fitness_values = []
        for ind in context.population.individuals:
            if ind.fitness is not None:
                if hasattr(ind.fitness, "values"):
                    fitness_values.append(float(ind.fitness.values[0]))
                elif hasattr(ind.fitness, "value"):
                    fitness_values.append(float(ind.fitness.value))

        if fitness_values:
            return float(np.mean(fitness_values))

        return None

    def _compute_selection_pressure(
        self,
        best_fitness: float | None,
        mean_fitness: float | None,
    ) -> float | None:
        """
        Compute selection pressure as best/mean ratio.

        Selection pressure indicates how much better the best individual
        is compared to the average. Higher values indicate stronger selection.

        Args:
            best_fitness: Best individual's fitness.
            mean_fitness: Mean population fitness.

        Returns:
            Selection pressure ratio, or None if not computable.
        """
        if best_fitness is None or mean_fitness is None:
            return None

        if mean_fitness == 0:
            # Avoid division by zero
            if best_fitness == 0:
                return 1.0
            return float("inf") if best_fitness > 0 else float("-inf")

        return best_fitness / mean_fitness

    def _compute_velocity(self) -> float | None:
        """
        Compute fitness improvement velocity over window.

        Velocity is the average rate of fitness improvement over the
        last `velocity_window` generations.

        Returns:
            Fitness improvement velocity, or None if insufficient history.
        """
        if len(self._best_fitness_history) < 2:
            return None

        # Get window of recent history
        window = min(self.velocity_window, len(self._best_fitness_history))
        recent_best = list(self._best_fitness_history)[-window:]

        if len(recent_best) < 2:
            return None

        # Compute velocity as linear regression slope
        x = np.arange(len(recent_best))
        y = np.array(recent_best)

        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def _compute_entropy(self, context: CollectionContext) -> float | None:
        """
        Compute population entropy using fitness histogram.

        Entropy measures the diversity of fitness values in the population.
        Higher entropy indicates more diverse fitness distribution.

        Args:
            context: Collection context.

        Returns:
            Population entropy in bits, or None if not computable.
        """
        # Extract fitness values
        fitness_values = []
        for ind in context.population.individuals:
            if ind.fitness is not None:
                if hasattr(ind.fitness, "values"):
                    fitness_values.append(float(ind.fitness.values[0]))
                elif hasattr(ind.fitness, "value"):
                    fitness_values.append(float(ind.fitness.value))

        if len(fitness_values) < 2:
            return None

        arr = np.array(fitness_values)

        # Handle constant fitness
        if np.all(arr == arr[0]):
            return 0.0

        # Create histogram
        try:
            hist, _ = np.histogram(arr, bins=self.entropy_bins)
        except Exception:
            return None

        # Compute probabilities
        probs = hist / len(arr)

        # Remove zero probabilities
        probs = probs[probs > 0]

        if len(probs) == 0:
            return 0.0

        # Compute entropy: -sum(p * log2(p))
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log2(self.entropy_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)
