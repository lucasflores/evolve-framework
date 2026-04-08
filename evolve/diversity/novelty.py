"""
Novelty search and quality-diversity (MAP-Elites) support.

Provides mechanisms for behavioral diversity:
- NoveltyArchive: Archive of novel behaviors
- BehaviorCharacterization: Protocol for extracting behavior descriptors
- QDArchive: Quality-Diversity archive (MAP-Elites)

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Generic, Protocol, TypeVar

import numpy as np

from evolve.core.types import Fitness, Individual

G = TypeVar("G")


# ============================================================================
# Behavior Characterization
# ============================================================================


class BehaviorCharacterization(Protocol[G]):
    """
    Protocol for extracting behavior descriptors from individuals.

    Used for novelty search and behavioral diversity.
    The behavior descriptor should capture the phenotypic
    behavior of an individual, not just its genotype.
    """

    def characterize(self, individual: Individual[G]) -> np.ndarray:
        """
        Extract behavior vector from individual.

        Args:
            individual: Evaluated individual

        Returns:
            Behavior descriptor (fixed-length vector)
        """
        ...


@dataclass
class FitnessBehavior(Generic[G]):
    """
    Simple behavior characterization using fitness values.

    Uses the fitness values directly as the behavior descriptor.
    Useful when behavior is implicitly captured by fitness.
    """

    def characterize(self, individual: Individual[G]) -> np.ndarray:
        """Extract fitness values as behavior."""
        if individual.fitness is None:
            return np.zeros(1)
        return np.array(individual.fitness.values)


@dataclass
class GenomeBehavior(Generic[G]):
    """
    Behavior characterization using genome directly.

    Uses the genome as the behavior descriptor.
    Useful for direct genotype-behavior mapping.
    """

    def characterize(self, individual: Individual[G]) -> np.ndarray:
        """Extract genome as behavior."""
        genome = individual.genome
        if hasattr(genome, "genes"):
            return np.asarray(genome.genes)
        elif isinstance(genome, np.ndarray):
            return genome
        else:
            raise ValueError(f"Cannot characterize genome of type {type(genome)}")


# ============================================================================
# Novelty Archive
# ============================================================================


@dataclass
class NoveltyArchive(Generic[G]):
    """
    Archive of novel behaviors for novelty search.

    Behaviors are added if sufficiently novel compared
    to the current archive and population. Novelty is
    measured as the average distance to k nearest neighbors.

    Attributes:
        behaviors: List of archived behavior vectors
        max_size: Maximum archive size (oldest removed when exceeded)
        k_neighbors: Number of neighbors for novelty calculation
        novelty_threshold: Minimum novelty to be added to archive

    Example:
        >>> archive = NoveltyArchive(k_neighbors=15, novelty_threshold=0.1)
        >>> novelty = archive.novelty(behavior, population_behaviors)
        >>> archive.maybe_add(behavior, novelty)
    """

    behaviors: list[np.ndarray] = field(default_factory=list)
    max_size: int = 1000
    k_neighbors: int = 15
    novelty_threshold: float = 0.1

    # Statistics
    _total_added: int = field(default=0, repr=False)
    _total_evaluated: int = field(default=0, repr=False)

    def novelty(
        self,
        behavior: np.ndarray,
        population_behaviors: Sequence[np.ndarray] | None = None,
    ) -> float:
        """
        Calculate novelty of a behavior.

        Novelty = average distance to k nearest neighbors
        in archive + population.

        Args:
            behavior: Behavior vector to evaluate
            population_behaviors: Optional current population behaviors

        Returns:
            Novelty score (higher = more novel)
        """
        self._total_evaluated += 1

        # Combine archive and population behaviors
        if population_behaviors is not None:
            all_behaviors = list(self.behaviors) + list(population_behaviors)
        else:
            all_behaviors = list(self.behaviors)

        if not all_behaviors:
            return float("inf")

        # Calculate distances to all behaviors
        distances = [float(np.linalg.norm(behavior - b)) for b in all_behaviors]

        # Sort and take k nearest
        distances.sort()
        k = min(self.k_neighbors, len(distances))

        return sum(distances[:k]) / k

    def maybe_add(
        self,
        behavior: np.ndarray,
        novelty_score: float | None = None,
        population_behaviors: Sequence[np.ndarray] | None = None,
    ) -> bool:
        """
        Add behavior to archive if sufficiently novel.

        Args:
            behavior: Behavior vector to potentially add
            novelty_score: Pre-computed novelty (computed if None)
            population_behaviors: Population for novelty calculation

        Returns:
            True if behavior was added to archive
        """
        if novelty_score is None:
            novelty_score = self.novelty(behavior, population_behaviors)

        if novelty_score >= self.novelty_threshold:
            self.behaviors.append(behavior.copy())
            self._total_added += 1

            # Prune if over capacity (remove oldest)
            if len(self.behaviors) > self.max_size:
                self.behaviors = self.behaviors[-self.max_size :]

            return True

        return False

    def add_batch(
        self,
        behaviors: Sequence[np.ndarray],
        population_behaviors: Sequence[np.ndarray] | None = None,
    ) -> int:
        """
        Add multiple behaviors, returning count added.

        Args:
            behaviors: Behavior vectors to evaluate
            population_behaviors: Current population behaviors

        Returns:
            Number of behaviors added to archive
        """
        added = 0
        for behavior in behaviors:
            if self.maybe_add(behavior, population_behaviors=population_behaviors):
                added += 1
        return added

    def get_novelty_scores(
        self,
        behaviors: Sequence[np.ndarray],
        population_behaviors: Sequence[np.ndarray] | None = None,
    ) -> list[float]:
        """
        Calculate novelty scores for multiple behaviors.

        Args:
            behaviors: Behavior vectors to evaluate
            population_behaviors: Current population behaviors

        Returns:
            List of novelty scores
        """
        return [self.novelty(b, population_behaviors) for b in behaviors]

    @property
    def size(self) -> int:
        """Current archive size."""
        return len(self.behaviors)

    @property
    def add_rate(self) -> float:
        """Fraction of evaluated behaviors that were added."""
        if self._total_evaluated == 0:
            return 0.0
        return self._total_added / self._total_evaluated

    def clear(self) -> None:
        """Clear the archive."""
        self.behaviors = []
        self._total_added = 0
        self._total_evaluated = 0


# ============================================================================
# Quality-Diversity Archive (MAP-Elites)
# ============================================================================


@dataclass
class QDArchive(Generic[G]):
    """
    Quality-Diversity archive for MAP-Elites.

    Maintains a grid of cells, each containing the best
    individual found for that behavior niche. This enables
    simultaneous optimization of quality (fitness) and
    diversity (behavior coverage).

    Attributes:
        dimensions: Grid size per behavior dimension (e.g., (10, 10) for 2D)
        bounds: Behavior space bounds (low, high)
        archive: Mapping from cell indices to individuals

    Example:
        >>> archive = QDArchive(
        ...     dimensions=(10, 10),
        ...     bounds=(np.array([0, 0]), np.array([1, 1]))
        ... )
        >>> added = archive.try_add(individual, behavior)
        >>> print(f"Coverage: {archive.coverage:.2%}")
    """

    dimensions: tuple[int, ...]
    bounds: tuple[np.ndarray, np.ndarray]
    archive: dict[tuple[int, ...], Individual[G]] = field(default_factory=dict)

    # Statistics
    _total_attempts: int = field(default=0, repr=False)
    _total_added: int = field(default=0, repr=False)
    _total_improved: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate bounds."""
        if len(self.bounds[0]) != len(self.dimensions):
            raise ValueError(
                f"Bounds dimensions ({len(self.bounds[0])}) must match "
                f"grid dimensions ({len(self.dimensions)})"
            )

    def get_cell(self, behavior: np.ndarray) -> tuple[int, ...]:
        """
        Map behavior to grid cell indices.

        Args:
            behavior: Behavior vector

        Returns:
            Tuple of cell indices
        """
        # Normalize to [0, 1]
        low, high = self.bounds
        normalized = (behavior - low) / (high - low + 1e-10)

        # Clip to valid range
        normalized = np.clip(normalized, 0, 0.9999)

        # Map to grid indices
        indices = tuple(min(int(n * d), d - 1) for n, d in zip(normalized, self.dimensions))

        return indices

    def try_add(
        self,
        individual: Individual[G],
        behavior: np.ndarray,
        minimize: bool = False,
    ) -> bool:
        """
        Add individual if it improves its cell.

        Args:
            individual: Individual to potentially add
            behavior: Behavior vector
            minimize: If True, lower fitness is better

        Returns:
            True if individual was added or improved cell
        """
        self._total_attempts += 1

        cell = self.get_cell(behavior)
        current = self.archive.get(cell)

        if current is None:
            # Empty cell - add directly
            self.archive[cell] = individual
            self._total_added += 1
            return True

        # Compare fitness
        if individual.fitness is None:
            return False
        if current.fitness is None:
            self.archive[cell] = individual
            self._total_improved += 1
            return True

        new_fit = individual.fitness.values[0]
        old_fit = current.fitness.values[0]

        improved = new_fit < old_fit if minimize else new_fit > old_fit

        if improved:
            self.archive[cell] = individual
            self._total_improved += 1
            return True

        return False

    def add_batch(
        self,
        individuals: Sequence[Individual[G]],
        behaviors: Sequence[np.ndarray],
        minimize: bool = False,
    ) -> int:
        """
        Try to add multiple individuals.

        Args:
            individuals: Individuals to add
            behaviors: Corresponding behavior vectors
            minimize: If True, lower fitness is better

        Returns:
            Number of successful additions
        """
        added = 0
        for ind, beh in zip(individuals, behaviors):
            if self.try_add(ind, beh, minimize):
                added += 1
        return added

    @property
    def coverage(self) -> float:
        """Fraction of cells occupied."""
        total_cells = 1
        for d in self.dimensions:
            total_cells *= d
        return len(self.archive) / total_cells

    @property
    def size(self) -> int:
        """Number of occupied cells."""
        return len(self.archive)

    @property
    def total_cells(self) -> int:
        """Total number of cells in grid."""
        total = 1
        for d in self.dimensions:
            total *= d
        return total

    @property
    def best_fitness(self) -> float | None:
        """Best fitness in archive."""
        if not self.archive:
            return None

        fitnesses = [
            ind.fitness.values[0] for ind in self.archive.values() if ind.fitness is not None
        ]

        return max(fitnesses) if fitnesses else None

    @property
    def mean_fitness(self) -> float | None:
        """Mean fitness in archive."""
        if not self.archive:
            return None

        fitnesses = [
            ind.fitness.values[0] for ind in self.archive.values() if ind.fitness is not None
        ]

        return sum(fitnesses) / len(fitnesses) if fitnesses else None

    def sample(self, n: int, rng: Random) -> list[Individual[G]]:
        """
        Sample n individuals uniformly from archive.

        Args:
            n: Number of individuals to sample
            rng: Random number generator

        Returns:
            List of sampled individuals
        """
        if not self.archive:
            return []

        keys = list(self.archive.keys())
        n = min(n, len(keys))
        selected_keys = rng.sample(keys, n)

        return [self.archive[k] for k in selected_keys]

    def get_elites(self, n: int, minimize: bool = False) -> list[Individual[G]]:
        """
        Get n best individuals from archive.

        Args:
            n: Number of individuals to get
            minimize: If True, get lowest fitness

        Returns:
            List of best individuals
        """
        if not self.archive:
            return []

        individuals = [ind for ind in self.archive.values() if ind.fitness is not None]

        individuals.sort(
            key=lambda ind: ind.fitness.values[0] if ind.fitness else float("inf"),
            reverse=not minimize,
        )

        return individuals[:n]

    def get_all_individuals(self) -> list[Individual[G]]:
        """Get all individuals in archive."""
        return list(self.archive.values())

    def get_all_behaviors(self) -> list[np.ndarray]:
        """
        Get behavior vectors for all occupied cells.

        Returns cell center points, not original behaviors.
        """
        behaviors = []
        low, high = self.bounds

        for cell in self.archive:
            # Convert cell indices to behavior space
            behavior = np.array(
                [
                    low[i] + (cell[i] + 0.5) * (high[i] - low[i]) / self.dimensions[i]
                    for i in range(len(cell))
                ]
            )
            behaviors.append(behavior)

        return behaviors

    def clear(self) -> None:
        """Clear the archive."""
        self.archive = {}
        self._total_attempts = 0
        self._total_added = 0
        self._total_improved = 0


# ============================================================================
# Novelty-based Fitness
# ============================================================================


def novelty_fitness(
    individuals: Sequence[Individual[G]],
    characterization: BehaviorCharacterization[G],
    archive: NoveltyArchive[G],
    weight_novelty: float = 1.0,
    weight_fitness: float = 0.0,
) -> list[Fitness]:
    """
    Calculate combined novelty and fitness scores.

    Can be used for pure novelty search (weight_fitness=0)
    or combined novelty + fitness.

    Args:
        individuals: Population to score
        characterization: Behavior extraction function
        archive: Novelty archive
        weight_novelty: Weight for novelty component
        weight_fitness: Weight for fitness component

    Returns:
        List of combined Fitness objects
    """
    # Extract behaviors
    behaviors = [characterization.characterize(ind) for ind in individuals]

    # Calculate novelty scores
    novelty_scores = archive.get_novelty_scores(behaviors, behaviors)

    # Combine with fitness if needed
    results = []

    for i, ind in enumerate(individuals):
        novelty = novelty_scores[i] * weight_novelty

        if weight_fitness > 0 and ind.fitness is not None:
            fitness_val = ind.fitness.values[0] * weight_fitness
        else:
            fitness_val = 0.0

        combined = novelty + fitness_val

        results.append(
            Fitness(
                values=np.array([combined]),
                metadata={
                    "novelty": novelty_scores[i],
                    "raw_fitness": ind.fitness.values[0] if ind.fitness else None,
                },
            )
        )

    return results
