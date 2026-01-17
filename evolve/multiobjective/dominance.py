"""
Pareto Dominance utilities.

Implements Pareto dominance relation and Pareto front extraction
with support for constrained optimization.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from evolve.multiobjective.fitness import MultiObjectiveFitness


def dominates(
    a: MultiObjectiveFitness,
    b: MultiObjectiveFitness,
    *,
    strict: bool = True,
) -> bool:
    """
    Check if fitness 'a' Pareto-dominates fitness 'b'.
    
    Uses constrained domination (feasibility-first) when constraints present:
    1. Feasible solutions always dominate infeasible ones
    2. Among infeasible solutions, less total violation dominates
    3. Among feasible solutions, Pareto dominance applies
    
    For Pareto dominance (maximization):
    - a dominates b if: a >= b for ALL objectives AND a > b for at least one
    
    Args:
        a: First fitness (potential dominator)
        b: Second fitness (potential dominated)
        strict: If True (default), require at least one strictly better objective.
                If False, weak dominance (a >= b for all objectives).
    
    Returns:
        True if 'a' dominates 'b'
    
    Example:
        >>> f1 = MultiObjectiveFitness(np.array([3.0, 2.0]))
        >>> f2 = MultiObjectiveFitness(np.array([2.0, 1.0]))
        >>> dominates(f1, f2)  # True: f1 >= f2 on all, > on at least one
        True
        >>> dominates(f2, f1)  # False
        False
    """
    # Constraint handling (feasibility-first)
    a_feasible = a.is_feasible
    b_feasible = b.is_feasible
    
    if a_feasible and not b_feasible:
        # Feasible dominates infeasible
        return True
    
    if not a_feasible and b_feasible:
        # Infeasible cannot dominate feasible
        return False
    
    if not a_feasible and not b_feasible:
        # Both infeasible: compare total constraint violation
        return a.total_constraint_violation < b.total_constraint_violation
    
    # Both feasible: Pareto dominance on objectives
    # a dominates b if: a >= b for all objectives AND (if strict) a > b for at least one
    
    # Check if a is at least as good as b on all objectives
    at_least_equal = np.all(a.objectives >= b.objectives)
    if not at_least_equal:
        return False
    
    if strict:
        # Need at least one strictly better objective
        strictly_better = np.any(a.objectives > b.objectives)
        return bool(strictly_better)
    
    return True


def weakly_dominates(
    a: MultiObjectiveFitness,
    b: MultiObjectiveFitness,
) -> bool:
    """
    Check weak Pareto dominance: a >= b on all objectives.
    
    Unlike strict dominance, doesn't require strictly better on any objective.
    
    Args:
        a: First fitness
        b: Second fitness
        
    Returns:
        True if a weakly dominates b
    """
    return dominates(a, b, strict=False)


def is_non_dominated(
    fitness: MultiObjectiveFitness,
    others: Sequence[MultiObjectiveFitness],
) -> bool:
    """
    Check if a fitness is non-dominated by any in a set.
    
    Args:
        fitness: Fitness to check
        others: Set of other fitnesses
        
    Returns:
        True if no fitness in 'others' dominates 'fitness'
    """
    for other in others:
        if dominates(other, fitness):
            return False
    return True


def pareto_front(
    fitnesses: Sequence[MultiObjectiveFitness],
) -> list[int]:
    """
    Get indices of non-dominated solutions (Pareto front).
    
    The Pareto front consists of all solutions that are not
    dominated by any other solution in the set.
    
    Complexity: O(M * N^2) where M = objectives, N = population size
    
    Args:
        fitnesses: Sequence of multi-objective fitnesses
        
    Returns:
        List of indices of solutions on the Pareto front
        
    Example:
        >>> fitnesses = [
        ...     MultiObjectiveFitness(np.array([3.0, 1.0])),  # idx 0
        ...     MultiObjectiveFitness(np.array([2.0, 2.0])),  # idx 1
        ...     MultiObjectiveFitness(np.array([1.0, 3.0])),  # idx 2
        ...     MultiObjectiveFitness(np.array([1.5, 1.5])),  # idx 3 - dominated
        ... ]
        >>> pareto_front(fitnesses)
        [0, 1, 2]  # idx 3 dominated by idx 1
    """
    n = len(fitnesses)
    if n == 0:
        return []
    
    is_dominated = [False] * n
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(fitnesses[j], fitnesses[i]):
                is_dominated[i] = True
                break
    
    return [i for i in range(n) if not is_dominated[i]]


def pareto_rank(
    fitnesses: Sequence[MultiObjectiveFitness],
) -> dict[int, int]:
    """
    Assign Pareto rank to each fitness.
    
    Rank 0 = Pareto front (non-dominated)
    Rank 1 = dominated only by rank 0
    etc.
    
    This is a simpler version of fast_non_dominated_sort that
    returns ranks directly.
    
    Args:
        fitnesses: Sequence of fitnesses
        
    Returns:
        Dict mapping index to rank
    """
    from evolve.multiobjective.ranking import fast_non_dominated_sort
    
    fronts = fast_non_dominated_sort(fitnesses)
    ranks = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    return ranks
