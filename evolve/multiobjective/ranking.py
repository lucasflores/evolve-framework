"""
NSGA-II Non-Dominated Sorting.

Implements the fast non-dominated sorting algorithm from NSGA-II
for efficient Pareto ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evolve.multiobjective.dominance import dominates
from evolve.multiobjective.fitness import MultiObjectiveFitness


@dataclass
class RankedIndividual:
    """
    Individual with ranking information for selection.

    Attributes:
        index: Original index in population
        rank: Pareto rank (0 = first front, Pareto-optimal)
        crowding_distance: Crowding distance (higher = more isolated)
    """

    index: int
    rank: int
    crowding_distance: float = 0.0

    def __lt__(self, other: RankedIndividual) -> bool:
        """
        Comparison for sorting: prefer lower rank, then higher crowding.
        """
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance


def fast_non_dominated_sort(
    fitnesses: Sequence[MultiObjectiveFitness],
) -> list[list[int]]:
    """
    NSGA-II fast non-dominated sorting.

    Assigns all individuals to ranked fronts:
    - Front 0 is the Pareto front (non-dominated)
    - Front 1 is dominated only by front 0
    - Front k is dominated only by fronts 0..k-1

    Algorithm:
    1. For each solution, compute domination count and dominated set
    2. Solutions with domination count 0 form the first front
    3. For each solution in current front, decrement domination count
       of solutions it dominates; those reaching 0 form next front
    4. Repeat until all solutions assigned

    Complexity: O(M * N^2) where M = objectives, N = population size

    Args:
        fitnesses: Sequence of multi-objective fitnesses

    Returns:
        List of fronts, each containing indices of individuals.
        fronts[0] is the Pareto front, fronts[1] is the second front, etc.

    Example:
        >>> fitnesses = [
        ...     MultiObjectiveFitness(np.array([3.0, 1.0])),
        ...     MultiObjectiveFitness(np.array([2.0, 2.0])),
        ...     MultiObjectiveFitness(np.array([1.0, 3.0])),
        ...     MultiObjectiveFitness(np.array([1.5, 1.5])),
        ... ]
        >>> fronts = fast_non_dominated_sort(fitnesses)
        >>> fronts[0]  # Pareto front
        [0, 1, 2]
        >>> fronts[1]  # Second front (dominated by front 0)
        [3]
    """
    n = len(fitnesses)
    if n == 0:
        return []

    # domination_count[i] = number of solutions that dominate solution i
    domination_count: list[int] = [0] * n

    # dominated_set[i] = set of solution indices that solution i dominates
    dominated_set: list[set[int]] = [set() for _ in range(n)]

    # First front (Pareto front)
    first_front: list[int] = []

    # Compute domination relationships
    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            if dominates(fitnesses[p], fitnesses[q]):
                # p dominates q
                dominated_set[p].add(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                # q dominates p
                domination_count[p] += 1

        # If p is non-dominated, it's in the first front
        if domination_count[p] == 0:
            first_front.append(p)

    # Handle edge case: if no non-dominated solutions, all have same fitness
    if not first_front:
        # All solutions are equivalent - return single front with all
        return [list(range(n))]

    fronts: list[list[int]] = [first_front]

    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts):
        next_front: list[int] = []

        for p in fronts[current_front]:
            # For each solution p in current front
            for q in dominated_set[p]:
                # Decrement domination count of solutions dominated by p
                domination_count[q] -= 1

                if domination_count[q] == 0:
                    # q is now non-dominated in remaining solutions
                    next_front.append(q)

        current_front += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def assign_ranks(
    fitnesses: Sequence[MultiObjectiveFitness],
) -> list[RankedIndividual]:
    """
    Assign Pareto ranks to all individuals.

    Convenience function that wraps fast_non_dominated_sort
    and returns RankedIndividual objects.

    Args:
        fitnesses: Sequence of multi-objective fitnesses

    Returns:
        List of RankedIndividual objects (one per fitness)
    """
    fronts = fast_non_dominated_sort(fitnesses)

    ranked: list[RankedIndividual] = [
        RankedIndividual(index=i, rank=-1) for i in range(len(fitnesses))
    ]

    for rank, front in enumerate(fronts):
        for idx in front:
            ranked[idx] = RankedIndividual(index=idx, rank=rank)

    return ranked
