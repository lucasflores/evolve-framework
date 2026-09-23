"""
Multi-Objective Selection Operators.

Implements NSGA-II selection and crowded tournament selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Generic, TypeVar

import numpy as np

from evolve.core.types import Individual
from evolve.multiobjective.crowding import crowding_distance
from evolve.multiobjective.fitness import MultiObjectiveFitness
from evolve.multiobjective.ranking import fast_non_dominated_sort

G = TypeVar("G")


@dataclass
class NSGA2Selector(Generic[G]):
    """
    NSGA-II selection using rank and crowding distance.

    Selection preference (in order):
    1. Lower rank (closer to Pareto front)
    2. Higher crowding distance (more diverse)

    This is the environmental selection operator that maintains
    diversity while converging toward the Pareto front.

    Attributes:
        directions: Direction of each raw objective value, ``"maximize"`` or
            ``"minimize"`` (e.g. from ``ObjectiveSpec.direction``). ``None``
            means all maximize, the ``MultiObjectiveFitness`` convention.

    Example:
        >>> selector = NSGA2Selector(directions=("maximize", "minimize"))
        >>> selected = selector.select(population, n_select=50, rng=rng)
    """

    directions: tuple[str, ...] | None = None

    def to_maximization(self, values: np.ndarray) -> np.ndarray:
        """
        Map raw objective values into the all-maximize space NSGA-II ranks in.

        This is the one place objective directions are applied; fitness
        objects keep their raw values for reporting.

        Raises:
            ValueError: If the number of values does not match ``directions``.
        """
        if self.directions is None:
            return values
        if len(values) != len(self.directions):
            raise ValueError(
                f"Fitness has {len(values)} objective values but "
                f"{len(self.directions)} objective directions are declared"
            )
        return np.where(np.asarray(self.directions) == "minimize", -values, values)

    def select(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        rng: Random,  # noqa: ARG002
    ) -> list[Individual[G]]:
        """
        Select individuals based on rank and crowding.

        Algorithm:
        1. Non-dominated sort into fronts
        2. Add complete fronts until next front would exceed n_select
        3. For last front: sort by crowding distance, take best

        Args:
            population: Population with multi-objective fitnesses
            n_select: Number of individuals to select
            rng: Random number generator (unused but kept for interface)

        Returns:
            Selected individuals
        """
        if n_select >= len(population):
            return list(population)

        fitnesses = self.ranking_fitnesses(population)

        # Non-dominated sorting
        fronts = fast_non_dominated_sort(fitnesses)

        # Build selected list front by front
        selected: list[Individual[G]] = []

        for front in fronts:
            if len(selected) + len(front) <= n_select:
                # Add entire front
                selected.extend(population[i] for i in front)
            else:
                # Need to select subset using crowding distance
                distances = crowding_distance(fitnesses, front)

                # Sort by crowding distance (descending - higher is better)
                sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)

                # Take remaining slots
                remaining = n_select - len(selected)
                selected.extend(population[i] for i in sorted_front[:remaining])
                break

        return selected

    def ranking_fitnesses(
        self,
        population: Sequence[Individual[G]],
    ) -> list[MultiObjectiveFitness]:
        """
        Fitnesses as NSGA-II ranks them, one per individual (same order).

        Objectives go through ``to_maximization``. A core ``Fitness`` keeps its
        ``constraints`` as ``constraint_violations`` (both use > 0 = violated),
        so constrained domination applies: feasible individuals always rank
        ahead of infeasible ones.

        Raises:
            TypeError: If an individual is unevaluated or has another fitness type.
        """
        from evolve.core.types import Fitness

        fitnesses: list[MultiObjectiveFitness] = []
        for ind in population:
            fitness = ind.fitness
            if isinstance(fitness, MultiObjectiveFitness):
                values, constraints = fitness.objectives, fitness.constraint_violations
            elif isinstance(fitness, Fitness):
                values, constraints = fitness.values, fitness.constraints
            else:
                raise TypeError(f"Expected MultiObjectiveFitness or Fitness, got {type(fitness)}")
            fitnesses.append(
                MultiObjectiveFitness(
                    objectives=self.to_maximization(values),
                    constraint_violations=constraints,
                )
            )
        return fitnesses

    def get_ranking_info(
        self,
        population: Sequence[Individual[G]],
    ) -> tuple[dict[int, int], dict[int, float]]:
        """
        Get ranking and crowding information for a population.

        Useful for debugging or visualization.

        Args:
            population: Population to analyze

        Returns:
            Tuple of (ranks dict, crowding distances dict)
        """
        fitnesses = self.ranking_fitnesses(population)

        fronts = fast_non_dominated_sort(fitnesses)

        ranks: dict[int, int] = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        all_distances: dict[int, float] = {}
        for front in fronts:
            distances = crowding_distance(fitnesses, front)
            all_distances.update(distances)

        return ranks, all_distances


@dataclass
class CrowdedTournamentSelection(Generic[G]):
    """
    Binary tournament using crowded comparison operator.

    Used for mating selection in NSGA-II. Winner is the individual with:
    1. Better (lower) rank, OR
    2. Same rank but higher crowding distance

    This differs from NSGA2Selector which is for environmental selection.

    Attributes:
        tournament_size: Number of candidates per tournament (default: 2)

    Example:
        >>> selector = CrowdedTournamentSelection(tournament_size=2)
        >>> parents = selector.select(
        ...     population, n_select=50, ranks=ranks, crowding=crowding, rng=rng
        ... )
    """

    tournament_size: int = 2

    def select(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        ranks: dict[int, int],
        crowding: dict[int, float],
        rng: Random,
    ) -> list[Individual[G]]:
        """
        Select individuals via crowded tournaments.

        Args:
            population: Population to select from
            n_select: Number of individuals to select
            ranks: Dict mapping index to Pareto rank
            crowding: Dict mapping index to crowding distance
            rng: Random number generator

        Returns:
            Selected individuals
        """
        pop_size = len(population)
        selected: list[Individual[G]] = []

        for _ in range(n_select):
            # Random tournament
            if self.tournament_size >= pop_size:
                candidates = list(range(pop_size))
            else:
                candidates = rng.sample(range(pop_size), self.tournament_size)

            # Crowded comparison to find winner
            best = candidates[0]
            for c in candidates[1:]:
                if self._crowded_compare(best, c, ranks, crowding) < 0:
                    best = c

            selected.append(population[best])

        return selected

    def _crowded_compare(
        self,
        i: int,
        j: int,
        ranks: dict[int, int],
        crowding: dict[int, float],
    ) -> int:
        """
        Crowded comparison operator.

        Returns:
            > 0 if i is better (should be selected over j)
            < 0 if j is better
            0 if equal
        """
        # Lower rank is better
        rank_i = ranks.get(i, float("inf"))
        rank_j = ranks.get(j, float("inf"))

        if rank_i < rank_j:
            return 1  # i is better
        if rank_i > rank_j:
            return -1  # j is better

        # Same rank: prefer higher crowding distance (more diverse)
        crowd_i = crowding.get(i, 0.0)
        crowd_j = crowding.get(j, 0.0)

        if crowd_i > crowd_j:
            return 1  # i is better
        if crowd_i < crowd_j:
            return -1  # j is better

        return 0  # Equal

    def select_with_precomputed(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        rng: Random,
    ) -> list[Individual[G]]:
        """
        Select individuals, computing ranks and crowding internally.

        Convenience method that doesn't require pre-computed ranks/crowding.

        Args:
            population: Population to select from
            n_select: Number to select
            rng: Random number generator

        Returns:
            Selected individuals
        """
        selector = NSGA2Selector[G]()
        ranks, crowding = selector.get_ranking_info(population)
        return self.select(population, n_select, ranks, crowding, rng)
