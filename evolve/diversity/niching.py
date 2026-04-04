"""
Niching and fitness sharing for diversity preservation.

Provides mechanisms to maintain population diversity by
adjusting fitness based on crowding in the search space.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from evolve.core.types import Individual

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
        if ind.fitness is not None:
            raw = ind.fitness.values[0]
        else:
            raw = 0.0

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
    fitness_values = []
    for ind in individuals:
        if ind.fitness is not None:
            fitness_values.append(ind.fitness.values)
        else:
            fitness_values.append(tuple([0.0] * n_objectives))

    # For each objective
    for m in range(n_objectives):
        # Sort by objective m
        sorted_indices = sorted(
            range(n),
            key=lambda i: fitness_values[i][m] if m < len(fitness_values[i]) else 0.0,
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


def clearing(
    individuals: Sequence[Individual[G]],
    distance_fn: Callable[[G, G], float],
    sigma_clear: float,
    kappa: int = 1,
) -> list[float]:
    """
    Clearing procedure for niching.

    Within each niche (defined by sigma_clear), only the best
    kappa individuals keep their fitness. Others get zero.
    This creates strong separation between niches.

    Args:
        individuals: Population with fitness values
        distance_fn: Function to compute genome distance
        sigma_clear: Clearing radius
        kappa: Number of winners per niche

    Returns:
        List of cleared fitness values
    """
    n = len(individuals)

    if n == 0:
        return []

    # Get raw fitness values
    raw_fitness = []
    for ind in individuals:
        if ind.fitness is not None:
            raw_fitness.append(ind.fitness.values[0])
        else:
            raw_fitness.append(0.0)

    # Sort by fitness (descending)
    sorted_indices = sorted(range(n), key=lambda i: raw_fitness[i], reverse=True)

    # Track which individuals are cleared
    cleared = [False] * n
    cleared_fitness = [0.0] * n

    for idx in sorted_indices:
        if cleared[idx]:
            continue

        # This individual is a niche winner
        cleared_fitness[idx] = raw_fitness[idx]

        # Count winners in this niche
        niche_winners = 1

        # Clear nearby individuals
        for other_idx in sorted_indices:
            if other_idx == idx or cleared[other_idx]:
                continue

            dist = distance_fn(
                individuals[idx].genome,
                individuals[other_idx].genome,
            )

            if dist < sigma_clear:
                if niche_winners < kappa:
                    # Still room for winners
                    cleared_fitness[other_idx] = raw_fitness[other_idx]
                    niche_winners += 1
                else:
                    # Clear this individual
                    cleared_fitness[other_idx] = 0.0

                cleared[other_idx] = True

        cleared[idx] = True

    return cleared_fitness


def deterministic_crowding_pairing(
    parents: Sequence[Individual[G]],
    offspring: Sequence[Individual[G]],
    distance_fn: Callable[[G, G], float],
) -> list[Individual[G]]:
    """
    Deterministic crowding for speciation-free niching.

    Each offspring competes against the nearest parent.
    The winner survives to the next generation.

    Args:
        parents: Parent population
        offspring: Offspring population (same size as parents)
        distance_fn: Function to compute genome distance

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

            # Compare fitness - winner survives
            child_fit = child.fitness.values[0] if child.fitness else float("-inf")
            parent_fit = parent.fitness.values[0] if parent.fitness else float("-inf")

            if child_fit > parent_fit:
                survivors.append(child)
            else:
                survivors.append(parent)
        else:
            survivors.append(child)

    return survivors
