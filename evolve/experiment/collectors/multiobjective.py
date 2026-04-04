"""
Multi-Objective Metric Collector.

Collects Pareto front quality metrics for multi-objective optimization:
- pareto_front_size: Number of individuals on the Pareto front
- hypervolume: Volume dominated by the front (2-3 objectives)
- crowding_diversity: Mean crowding distance on the front
- spread: Distribution uniformity along the front

Implements FR-014 from the tracking specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.experiment.collectors.base import CollectionContext

if TYPE_CHECKING:
    from evolve.core.types import Individual
    from evolve.multiobjective.fitness import MultiObjectiveFitness


_logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveMetricCollector:
    """
    Collector for multi-objective optimization metrics.

    Computes Pareto front quality indicators including hypervolume,
    spread, and crowding diversity. Supports 2-3 objectives with
    exact computation and falls back to approximate indicators for
    higher-dimensional problems.

    Attributes:
        reference_point: Reference point for hypervolume computation.
            If None, uses nadir point estimate from population.
        enable_spread: Whether to compute spread metric (<2 objectives only).
        enable_crowding: Whether to compute crowding diversity.
        warn_on_empty_front: Log warning when Pareto front is empty.

    Example:
        >>> from evolve.experiment.collectors.multiobjective import MultiObjectiveMetricCollector
        >>> from evolve.experiment.collectors.base import CollectionContext
        >>>
        >>> collector = MultiObjectiveMetricCollector()
        >>> context = CollectionContext(
        ...     generation=10,
        ...     population=population,
        ...     pareto_front=front_individuals,
        ... )
        >>> metrics = collector.collect(context)
        >>> metrics["pareto_front_size"]
        25
    """

    reference_point: np.ndarray | list[float] | None = None
    enable_spread: bool = True
    enable_crowding: bool = True
    warn_on_empty_front: bool = True

    # Track if we've already warned (to avoid log spam)
    _warned_empty_front: bool = False
    _warned_high_dim: bool = False

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._warned_empty_front = False
        self._warned_high_dim = False

    def collect(self, context: CollectionContext) -> dict[str, Any]:
        """
        Collect multi-objective metrics from context.

        Args:
            context: Collection context with population and optional Pareto front.

        Returns:
            Dictionary of multi-objective metrics:
            - pareto_front_size: Number of individuals on the front
            - hypervolume: Dominated volume (if reference point available)
            - crowding_diversity: Mean crowding distance
            - spread: Distribution uniformity
        """
        metrics: dict[str, Any] = {}

        # Get Pareto front from context, or compute from population
        front = self._get_pareto_front(context)

        if front is None or len(front) == 0:
            self._check_empty_front()
            metrics["pareto_front_size"] = 0
            return metrics

        # Always compute front size
        metrics["pareto_front_size"] = len(front)

        # Get objective values as numpy array
        objectives = self._extract_objectives(front)

        if objectives is None or len(objectives) == 0:
            return metrics

        n_objectives = objectives.shape[1]

        # Compute hypervolume (exact for 2D, approximate for 3D+)
        hv = self._compute_hypervolume(objectives, context)
        if hv is not None:
            metrics["hypervolume"] = hv

        # Compute spread (only for 2D fronts)
        if self.enable_spread and n_objectives == 2:
            spread_val = self._compute_spread(objectives)
            if spread_val is not None:
                metrics["spread"] = spread_val

        # Compute crowding diversity
        if self.enable_crowding:
            crowding = self._compute_crowding_diversity(front)
            if crowding is not None:
                metrics["crowding_diversity"] = crowding

        return metrics

    def reset(self) -> None:
        """Reset internal state between runs."""
        self._warned_empty_front = False
        self._warned_high_dim = False

    def _get_pareto_front(self, context: CollectionContext) -> list[Individual[Any]] | None:
        """
        Get Pareto front from context or compute from population.

        Args:
            context: Collection context.

        Returns:
            List of individuals on the Pareto front, or None.
        """
        if context.has_pareto_front():
            return context.pareto_front

        # Try to compute from population if it has MO fitness
        from evolve.multiobjective.dominance import pareto_front
        from evolve.multiobjective.fitness import MultiObjectiveFitness

        individuals = context.population.individuals
        if not individuals:
            return None

        # Check if first individual has MO fitness
        first = individuals[0]
        if first.fitness is None:
            return None
        if not isinstance(first.fitness, MultiObjectiveFitness):
            return None

        # Extract fitnesses and compute front
        fitnesses = []
        for ind in individuals:
            if ind.fitness is not None and isinstance(ind.fitness, MultiObjectiveFitness):
                fitnesses.append(ind.fitness)
            else:
                return None  # Mixed fitness types not supported

        front_indices = pareto_front(fitnesses)
        return [individuals[i] for i in front_indices]

    def _extract_objectives(self, front: list[Individual[Any]]) -> np.ndarray | None:
        """
        Extract objective values from Pareto front individuals.

        Args:
            front: List of individuals on the Pareto front.

        Returns:
            2D array of shape (n_individuals, n_objectives), or None.
        """
        from evolve.multiobjective.fitness import MultiObjectiveFitness

        objectives_list = []
        for ind in front:
            if ind.fitness is None:
                continue
            if isinstance(ind.fitness, MultiObjectiveFitness):
                objectives_list.append(ind.fitness.objectives)
            elif hasattr(ind.fitness, "values"):
                # Tuple-based MO fitness
                objectives_list.append(np.array(ind.fitness.values))
            else:
                continue

        if not objectives_list:
            return None

        return np.array(objectives_list)

    def _compute_hypervolume(
        self,
        objectives: np.ndarray,
        context: CollectionContext,
    ) -> float | None:
        """
        Compute hypervolume indicator.

        Args:
            objectives: Objective values, shape (n, m).
            context: Collection context for reference point.

        Returns:
            Hypervolume value, or None if not computable.
        """
        n_objectives = objectives.shape[1]

        # Get reference point
        ref = self._get_reference_point(objectives, context)
        if ref is None:
            return None

        if n_objectives == 2:
            from evolve.multiobjective.metrics import hypervolume_2d

            try:
                return hypervolume_2d(objectives, ref)
            except Exception as e:
                _logger.debug(f"Hypervolume computation failed: {e}")
                return None

        elif n_objectives == 3:
            # Use approximate hypervolume for 3D
            return self._approximate_hypervolume_3d(objectives, ref)

        else:
            # For >3 objectives, use approximate indicator
            if not self._warned_high_dim:
                _logger.info(f"Using approximate hypervolume for {n_objectives} objectives")
                self._warned_high_dim = True
            return self._approximate_hypervolume_nd(objectives, ref)

    def _get_reference_point(
        self,
        objectives: np.ndarray,
        context: CollectionContext,
    ) -> np.ndarray | None:
        """
        Get reference point for hypervolume computation.

        Args:
            objectives: Objective values.
            context: Collection context (may have config reference).

        Returns:
            Reference point array, or None.
        """
        # Use configured reference point if available
        if self.reference_point is not None:
            ref: np.ndarray = np.atleast_1d(np.array(self.reference_point))
            return ref

        # Check context extra for hypervolume_reference config
        if "hypervolume_reference" in context.extra:
            ref = np.atleast_1d(np.array(context.extra["hypervolume_reference"]))
            return ref

        # Estimate nadir point from current front
        # Use worst value per objective minus a small margin
        nadir: np.ndarray = np.min(objectives, axis=0)  # Assuming maximization
        margin = 0.1 * np.abs(nadir - np.max(objectives, axis=0))
        margin = np.where(margin == 0, 0.1, margin)  # Avoid zero margin

        result: np.ndarray = nadir - margin
        return result

    def _approximate_hypervolume_3d(
        self,
        objectives: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Approximate hypervolume for 3D problems.

        Uses a simple Monte Carlo approach.

        Args:
            objectives: Objective values, shape (n, 3).
            reference: Reference point.

        Returns:
            Approximate hypervolume.
        """
        # Simple Monte Carlo approximation
        n_samples = 10000
        n_points = len(objectives)

        if n_points == 0:
            return 0.0

        # Find bounding box (reference to ideal)
        ideal = np.max(objectives, axis=0)

        # Check if ideal > reference on all dimensions
        if not np.all(ideal > reference):
            return 0.0

        # Sample random points in bounding box
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        samples = rng.uniform(reference, ideal, size=(n_samples, 3))

        # Count dominated samples
        dominated = 0
        for sample in samples:
            for point in objectives:
                if np.all(point >= sample):
                    dominated += 1
                    break

        # Estimate volume
        box_volume = np.prod(ideal - reference)
        return float(box_volume * dominated / n_samples)

    def _approximate_hypervolume_nd(
        self,
        objectives: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Approximate hypervolume for n-dimensional problems.

        Uses Monte Carlo sampling - slower but works for any dimension.

        Args:
            objectives: Objective values, shape (n, m).
            reference: Reference point.

        Returns:
            Approximate hypervolume.
        """
        n_samples = 50000  # More samples for higher dimensions
        n_points, n_dim = objectives.shape

        if n_points == 0:
            return 0.0

        # Find bounding box
        ideal = np.max(objectives, axis=0)

        if not np.all(ideal > reference):
            return 0.0

        # Sample random points
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        samples = rng.uniform(reference, ideal, size=(n_samples, n_dim))

        # Vectorized dominance check
        # A sample is dominated if any point dominates it
        dominated_mask = np.zeros(n_samples, dtype=bool)
        for point in objectives:
            dominated_mask |= np.all(samples <= point, axis=1)

        # Estimate volume
        box_volume = np.prod(ideal - reference)
        return float(box_volume * np.sum(dominated_mask) / n_samples)

    def _compute_spread(self, objectives: np.ndarray) -> float | None:
        """
        Compute spread metric for 2D front.

        Args:
            objectives: Objective values, shape (n, 2).

        Returns:
            Spread value in [0, 1], or None if not computable.
        """
        if objectives.shape[1] != 2:
            return None

        if len(objectives) < 2:
            return 1.0  # No spread with < 2 points

        from evolve.multiobjective.metrics import spread

        try:
            return spread(objectives)
        except Exception as e:
            _logger.debug(f"Spread computation failed: {e}")
            return None

    def _compute_crowding_diversity(
        self,
        front: list[Individual[Any]],
    ) -> float | None:
        """
        Compute mean crowding distance as diversity metric.

        Args:
            front: Pareto front individuals.

        Returns:
            Mean finite crowding distance, or None.
        """
        from evolve.multiobjective.crowding import crowding_distance
        from evolve.multiobjective.fitness import MultiObjectiveFitness

        # Extract MO fitnesses
        fitnesses: list[MultiObjectiveFitness] = []
        for ind in front:
            if ind.fitness is not None and isinstance(ind.fitness, MultiObjectiveFitness):
                fitnesses.append(ind.fitness)

        if len(fitnesses) < 3:
            # Not enough points for meaningful crowding
            return None

        # Compute crowding distances
        indices = list(range(len(fitnesses)))
        try:
            distances = crowding_distance(fitnesses, indices)
        except Exception as e:
            _logger.debug(f"Crowding distance computation failed: {e}")
            return None

        # Compute mean of finite distances
        finite_distances = [d for d in distances.values() if np.isfinite(d)]

        if not finite_distances:
            return None

        return float(np.mean(finite_distances))

    def _check_empty_front(self) -> None:
        """Log warning if Pareto front is empty (once per run)."""
        if self.warn_on_empty_front and not self._warned_empty_front:
            _logger.warning(
                "Pareto front is empty - no dominated individuals found. "
                "This may indicate evaluation issues or single-objective fitness."
            )
            self._warned_empty_front = True
