"""
Population container - Ordered collection of individuals with statistics.

This module provides the Population class that manages collections of
individuals and computes aggregate statistics.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from evolve.core.types import Fitness, Individual

G = TypeVar("G")


@dataclass(frozen=True)
class PopulationStatistics:
    """
    Computed population metrics.

    Attributes:
        size: Number of individuals
        best_fitness: Best fitness in population
        worst_fitness: Worst fitness in population
        mean_fitness: Mean fitness across population
        std_fitness: Standard deviation of fitness
        diversity: Genotypic/phenotypic diversity measure (0-1)
        species_count: Number of distinct species
        front_sizes: Individuals per Pareto front (multi-objective only)
        evaluated_count: Number of individuals with fitness computed
    """

    size: int
    best_fitness: Fitness | None
    worst_fitness: Fitness | None
    mean_fitness: Fitness | None
    std_fitness: float | None
    diversity: float = 0.0
    species_count: int = 0
    front_sizes: list[int] | None = None
    evaluated_count: int = 0


class Population(Generic[G]):
    """
    Ordered collection of individuals with statistics.

    Population is the main container for candidate solutions.
    It provides:
    - Iteration and indexing over individuals
    - Cached statistics computation
    - Selection of best individuals
    - Immutable update pattern (returns new populations)

    Example:
        >>> individuals = [Individual(genome=g) for g in genomes]
        >>> pop = Population(individuals)
        >>> len(pop)
        50
        >>> pop.best(5)  # Returns 5 best individuals
        >>> pop.statistics.mean_fitness
    """

    def __init__(
        self,
        individuals: Sequence[Individual[G]],
        generation: int = 0,
    ) -> None:
        """
        Create population from individuals.

        Args:
            individuals: Sequence of individuals (must not be empty)
            generation: Current generation number

        Raises:
            ValueError: If individuals is empty
        """
        if not individuals:
            raise ValueError("Population cannot be empty")

        self._individuals: tuple[Individual[G], ...] = tuple(individuals)
        self._generation = generation
        self._statistics: PopulationStatistics | None = None

    @property
    def individuals(self) -> Sequence[Individual[G]]:
        """Immutable view of individuals."""
        return self._individuals

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def statistics(self) -> PopulationStatistics:
        """
        Computed statistics (cached, recomputed on mutation).

        Returns:
            PopulationStatistics with aggregate metrics
        """
        if self._statistics is None:
            self._statistics = self._compute_statistics()
        return self._statistics

    def _compute_statistics(self) -> PopulationStatistics:
        """Compute aggregate statistics for the population."""
        size = len(self._individuals)

        # Count evaluated individuals
        evaluated = [ind for ind in self._individuals if ind.fitness is not None]
        evaluated_count = len(evaluated)

        if evaluated_count == 0:
            return PopulationStatistics(
                size=size,
                best_fitness=None,
                worst_fitness=None,
                mean_fitness=None,
                std_fitness=None,
                evaluated_count=0,
            )

        # Get fitness values (assume minimization for best/worst)
        fitness_values = [ind.fitness for ind in evaluated if ind.fitness is not None]

        # For single-objective, compute simple statistics
        if fitness_values and fitness_values[0].n_objectives == 1:
            values = np.array([f.values[0] for f in fitness_values])
            best_idx = int(np.argmin(values))
            worst_idx = int(np.argmax(values))
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))

            best_fitness = fitness_values[best_idx]
            worst_fitness = fitness_values[worst_idx]
            mean_fitness = Fitness.scalar(mean_val)
        else:
            # Multi-objective: best is first Pareto front representative
            # For now, just return first individual's fitness as "best"
            best_fitness = fitness_values[0]
            worst_fitness = fitness_values[-1]
            mean_values = np.mean([f.values for f in fitness_values], axis=0)
            mean_fitness = Fitness(values=mean_values)
            std_val = None

        # Count species
        species_ids = {
            ind.metadata.species_id
            for ind in self._individuals
            if ind.metadata.species_id is not None
        }
        species_count = len(species_ids)

        return PopulationStatistics(
            size=size,
            best_fitness=best_fitness,
            worst_fitness=worst_fitness,
            mean_fitness=mean_fitness,
            std_fitness=std_val,
            species_count=species_count,
            evaluated_count=evaluated_count,
        )

    def __len__(self) -> int:
        """Number of individuals."""
        return len(self._individuals)

    def __iter__(self) -> Iterator[Individual[G]]:
        """Iterate over individuals."""
        return iter(self._individuals)

    def __getitem__(self, idx: int) -> Individual[G]:
        """Access individual by index."""
        return self._individuals[idx]

    def best(self, n: int = 1, minimize: bool = True) -> Sequence[Individual[G]]:
        """
        Return n best individuals by fitness.

        Args:
            n: Number of individuals to return
            minimize: If True, lower fitness is better

        Returns:
            Sequence of n best individuals

        Raises:
            ValueError: If n > population size or no evaluated individuals
        """
        if n > len(self._individuals):
            raise ValueError(
                f"Cannot select {n} individuals from population of {len(self._individuals)}"
            )

        # Filter to evaluated individuals
        evaluated = [ind for ind in self._individuals if ind.fitness is not None]

        if not evaluated:
            raise ValueError("No evaluated individuals in population")

        # Sort by fitness (single-objective assumed)
        if evaluated[0].fitness is not None and evaluated[0].fitness.n_objectives == 1:
            sorted_individuals = sorted(
                evaluated,
                key=lambda ind: float(ind.fitness.values[0]) if ind.fitness else float("inf"),
                reverse=not minimize,
            )
        else:
            # Multi-objective: return first n (should use Pareto ranking)
            sorted_individuals = evaluated

        return sorted_individuals[:n]

    def with_individuals(
        self,
        individuals: Sequence[Individual[G]],
        generation: int | None = None,
    ) -> Population[G]:
        """
        Return new population with updated individuals.

        Args:
            individuals: New individual sequence
            generation: New generation number (increments if None)

        Returns:
            New Population instance
        """
        new_gen = generation if generation is not None else self._generation + 1
        return Population(individuals=individuals, generation=new_gen)

    def increment_ages(self) -> Population[G]:
        """Return new population with all individual ages incremented."""
        aged = [ind.increment_age() for ind in self._individuals]
        return self.with_individuals(aged, self._generation)

    def filter_evaluated(self) -> Population[G]:
        """Return new population containing only evaluated individuals."""
        evaluated = [ind for ind in self._individuals if ind.fitness is not None]
        if not evaluated:
            raise ValueError("No evaluated individuals to filter")
        return Population(individuals=evaluated, generation=self._generation)

    def sample(
        self,
        n: int,
        rng: Random,  # type: ignore[name-defined]
        replace: bool = False,
    ) -> Sequence[Individual[G]]:
        """
        Random sample of individuals.

        Args:
            n: Number to sample
            rng: Random number generator
            replace: If True, sample with replacement

        Returns:
            Sequence of sampled individuals
        """
        from random import Random

        if not isinstance(rng, Random):
            raise TypeError("rng must be a Random instance")

        if replace:
            return [rng.choice(self._individuals) for _ in range(n)]
        else:
            if n > len(self._individuals):
                raise ValueError(
                    f"Cannot sample {n} without replacement from {len(self._individuals)}"
                )
            return rng.sample(list(self._individuals), n)
