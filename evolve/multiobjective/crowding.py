"""
Crowding Distance calculation.

Implements crowding distance from NSGA-II for diversity preservation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from evolve.multiobjective.fitness import MultiObjectiveFitness


def crowding_distance(
    fitnesses: Sequence[MultiObjectiveFitness],
    front_indices: list[int],
) -> dict[int, float]:
    """
    Calculate crowding distance for individuals in a front.
    
    Crowding distance measures how close an individual is to its
    neighbors in objective space. Higher distance = more isolated = more diverse.
    
    Algorithm (per objective):
    1. Sort individuals by objective value
    2. Assign infinite distance to boundary solutions
    3. For interior solutions: distance += (obj[i+1] - obj[i-1]) / (max - min)
    
    Boundary solutions (best/worst on any objective) get infinite distance
    to ensure they're always preserved.
    
    Args:
        fitnesses: All fitnesses in population
        front_indices: Indices of individuals in this front
        
    Returns:
        Dict mapping index to crowding distance
        
    Example:
        >>> fitnesses = [
        ...     MultiObjectiveFitness(np.array([1.0, 4.0])),
        ...     MultiObjectiveFitness(np.array([2.0, 3.0])),
        ...     MultiObjectiveFitness(np.array([3.0, 2.0])),
        ...     MultiObjectiveFitness(np.array([4.0, 1.0])),
        ... ]
        >>> distances = crowding_distance(fitnesses, [0, 1, 2, 3])
        >>> distances[0]  # Boundary: infinite
        inf
        >>> distances[1]  # Interior: finite
        2.0  # (3-1)/3 + (4-2)/3 = 2/3 + 2/3 = 1.33 per objective
    """
    n_front = len(front_indices)
    
    # Edge cases
    if n_front == 0:
        return {}
    
    if n_front <= 2:
        # All boundary solutions
        return {i: float('inf') for i in front_indices}
    
    # Initialize distances
    distances: dict[int, float] = {i: 0.0 for i in front_indices}
    
    # Get number of objectives from first fitness
    n_objectives = fitnesses[front_indices[0]].n_objectives
    
    for m in range(n_objectives):
        # Sort front by objective m
        sorted_indices = sorted(
            front_indices,
            key=lambda i: fitnesses[i].objectives[m]
        )
        
        # Boundary solutions get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Calculate objective range
        obj_min = fitnesses[sorted_indices[0]].objectives[m]
        obj_max = fitnesses[sorted_indices[-1]].objectives[m]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            # All same value on this objective - skip
            continue
        
        # Interior solutions
        for k in range(1, n_front - 1):
            idx = sorted_indices[k]
            
            # Skip if already infinite
            if distances[idx] == float('inf'):
                continue
            
            prev_obj = fitnesses[sorted_indices[k - 1]].objectives[m]
            next_obj = fitnesses[sorted_indices[k + 1]].objectives[m]
            
            # Add normalized distance contribution
            distances[idx] += (next_obj - prev_obj) / obj_range
    
    return distances


def crowding_distance_assignment(
    fitnesses: Sequence[MultiObjectiveFitness],
    fronts: list[list[int]],
) -> dict[int, float]:
    """
    Calculate crowding distance for all individuals across all fronts.
    
    Convenience function that applies crowding_distance to each front.
    
    Args:
        fitnesses: All fitnesses
        fronts: List of fronts from fast_non_dominated_sort
        
    Returns:
        Dict mapping every index to its crowding distance
    """
    all_distances: dict[int, float] = {}
    
    for front in fronts:
        front_distances = crowding_distance(fitnesses, front)
        all_distances.update(front_distances)
    
    return all_distances
