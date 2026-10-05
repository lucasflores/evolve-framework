"""
Niching and fitness sharing for diversity preservation.

Provides mechanisms to maintain population diversity by
adjusting fitness based on crowding in the search space.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from evolve.core.types import Individual, fitness_sort_key

G = TypeVar("G")


def explicit_fitness_sharing(
    individuals: Sequence[Individual[G]],
    distance_fn: Callable[[G, G], float],
    sigma_share: float,
    alpha: float = 1.0,
) -> list[float]:
    """
    Calculate shared fitness for each individual.

    Reduces fitness of individuals in crowded regions
    to promote diversity. Each individual's fitness is
    divided by its niche count (sum of sharing values
    with all other individuals).

    shared_fitness[i] = raw_fitness[i] / niche_count[i]

    The sharing function is:
    sh(d) = 1 - (d / sigma_share)^alpha  if d < sigma_share
    sh(d) = 0                            if d >= sigma_share

    Args:
        individuals: Population with fitness values
        distance_fn: Function to compute genome distance
        sigma_share: Niche radius - individuals within this
                    distance share fitness
        alpha: Shape parameter for sharing function (default: 1.0)
               Higher values make sharing more localized

    Returns:
        List of shared fitness values in same order as input

    Example:
        >>> shared = explicit_fitness_sharing(
        ...     population, euclidean_distance, sigma_share=1.0
        ... )
    """
    n = len(individuals)

    if n == 0:
        return []

    # Compute niche counts
    niche_counts = [0.0] * n

    for i in range(n):
        for j in range(n):
            dist = distance_fn(
                individuals[i].genome,
                individuals[j].genome,
            )
            if dist < sigma_share:
                # Triangular sharing function
                sharing = 1.0 - (dist / sigma_share) ** alpha
                niche_counts[i] += sharing

    # Compute shared fitness
    shared_fitness = []
    for i, ind in enumerate(individuals):
        raw = ind.fitness.values[0] if ind.fitness is not None else 0.0

        # Divide by niche count (minimum 1 to avoid division by zero)
        shared = raw / max(niche_counts[i], 1.0)
        shared_fitness.append(shared)

    return shared_fitness


def crowding_distance(
    individuals: Sequence[Individual[G]],
    n_objectives: int = 1,
) -> list[float]:
    """
    Calculate crowding distance for multi-objective optimization.

    Crowding distance measures how close an individual is to
    its neighbors in objective space. Used in NSGA-II for
    tie-breaking within Pareto fronts.

    Args:
        individuals: Population with fitness values
        n_objectives: Number of objectives

    Returns:
        List of crowding distances
    """
    n = len(individuals)

    if n == 0:
        return []

    if n <= 2:
        return [float("inf")] * n

    # Initialize distances
    distances = [0.0] * n

    # Get fitness values
    fitness_values: list[np.ndarray] = []
    for ind in individuals:
        if ind.fitness is not None:
            fitness_values.append(np.array(ind.fitness.values))
        else:
            fitness_values.append(np.array([0.0] * n_objectives))

    # For each objective
    for m in range(n_objectives):
        # Sort by objective m
        sorted_indices = sorted(
            range(n),
            key=lambda i, _m=m: (  # type: ignore[misc]
                float(fitness_values[i][_m]) if _m < len(fitness_values[i]) else 0.0
            ),
        )

        # Boundary individuals get infinite distance
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        # Get range
        f_max = (
            fitness_values[sorted_indices[-1]][m]
            if m < len(fitness_values[sorted_indices[-1]])
            else 0.0
        )
        f_min = (
            fitness_values[sorted_indices[0]][m]
            if m < len(fitness_values[sorted_indices[0]])
            else 0.0
        )
        f_range = f_max - f_min

        if f_range == 0:
            continue

        # Interior individuals
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]

            f_prev = fitness_values[prev_idx][m] if m < len(fitness_values[prev_idx]) else 0.0
            f_next = fitness_values[next_idx][m] if m < len(fitness_values[next_idx]) else 0.0

            distances[idx] += (f_next - f_prev) / f_range

    return distances


@dataclass(frozen=True)
class Clearing:
    """
    Clearing in survival: the settings ``clear()`` runs with each generation.

    Attributes:
        distance: Distance between two genomes; 0 for copies.
        closeness: Largest distance that still counts as a copy (``<=``).
        copies: Survivors per group before any held-back candidate; None
            holds nothing back, so the groups are only measured and logged.
    """

    distance: Callable[[Any, Any], float]
    closeness: float = 0.0
    copies: int | None = None

    def __post_init__(self) -> None:
        """Refuse a cap below 1 or a negative closeness."""
        if self.copies is not None and self.copies < 1:
            raise ValueError(f"copies must be at least 1 (or None), got {self.copies}")
        if self.closeness < 0:
            raise ValueError(f"closeness must be non-negative, got {self.closeness}")


def clear(
    genomes: Sequence[G],
    order: Sequence[int],
    distance: Callable[[G, G], float],
    closeness: float,
    copies: int | None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Clearing (Petrowski 1996): group near-copies, keep a few of each group.

    Candidates are taken best first, in ``order``, which the caller ranks
    (Pareto rank then crowding, or feasibility then value). The best
    candidate not yet in a group leads a new one, and every ungrouped
    candidate within ``closeness`` of the leader joins it: ``distance <=
    closeness``, so a closeness of 0 groups exact copies. Membership is
    distance to the leader, never chained through another member. The
    first ``copies`` of each group, in order, win; the rest are held back.

    Args:
        genomes: The candidates' genomes.
        order: Every index into ``genomes`` once, best first.
        distance: Distance between two genomes; 0 for copies.
        closeness: Largest distance that still counts as a copy.
        copies: Winners per group, at least 1; None holds nothing back, so
            the groups are only measured.

    Returns:
        (winners, held_back, sizes): indices into ``genomes``, each list in
        ``order``, and each group's size in the order the groups formed.

    Raises:
        ValueError: On a cap below 1, a negative closeness, or an order that
            is not every index once.
    """
    if copies is not None and copies < 1:
        raise ValueError(f"copies must be at least 1 (or None), got {copies}")
    if closeness < 0:
        raise ValueError(f"closeness must be non-negative, got {closeness}")
    if sorted(order) != list(range(len(genomes))):
        raise ValueError("order must list every candidate's index exactly once")

    # ponytail: compares each candidate with every group leader, O(n x groups)
    # distance calls; vectorise the distance if populations reach the thousands
    grouped = [False] * len(genomes)
    winners: list[int] = []
    held_back: list[int] = []
    sizes: list[int] = []
    for leader in order:
        if grouped[leader]:
            continue
        group = [leader]
        grouped[leader] = True
        for other in order:
            if not grouped[other] and distance(genomes[leader], genomes[other]) <= closeness:
                group.append(other)
                grouped[other] = True
        sizes.append(len(group))
        keep = len(group) if copies is None else copies
        winners += group[:keep]
        held_back += group[keep:]
    return winners, held_back, sizes


def deterministic_crowding_pairing(
    parents: Sequence[Individual[G]],
    offspring: Sequence[Individual[G]],
    distance_fn: Callable[[G, G], float],
    minimize: bool = False,
) -> list[Individual[G]]:
    """
    Deterministic crowding for speciation-free niching.

    Each offspring competes against the nearest parent.
    The winner (feasibility first, then fitness; the parent on ties)
    survives to the next generation.

    Args:
        parents: Parent population
        offspring: Offspring population (same size as parents)
        distance_fn: Function to compute genome distance
        minimize: If True, lower fitness is better (default False)

    Returns:
        Surviving individuals
    """
    if len(parents) != len(offspring):
        raise ValueError("Parents and offspring must have same size")

    survivors = []

    # Pair each offspring with nearest parent
    used_parents: set[int] = set()

    for child in offspring:
        # Find nearest unused parent
        min_dist = float("inf")
        nearest_idx = -1

        for i, parent in enumerate(parents):
            if i in used_parents:
                continue

            dist = distance_fn(child.genome, parent.genome)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            used_parents.add(nearest_idx)
            parent = parents[nearest_idx]

            # Compare fitness (feasibility first) - winner survives
            if fitness_sort_key(child.fitness, minimize) < fitness_sort_key(
                parent.fitness, minimize
            ):
                survivors.append(child)
            else:
                survivors.append(parent)
        else:
            survivors.append(child)

    return survivors
