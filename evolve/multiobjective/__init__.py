"""
Multi-Objective Optimization Module.

This module provides Pareto dominance, NSGA-II ranking, crowding distance,
and selection operators for multi-objective optimization.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from evolve.multiobjective.fitness import MultiObjectiveFitness
from evolve.multiobjective.dominance import dominates, pareto_front
from evolve.multiobjective.ranking import fast_non_dominated_sort, RankedIndividual
from evolve.multiobjective.crowding import crowding_distance
from evolve.multiobjective.selection import (
    NSGA2Selector,
    CrowdedTournamentSelection,
)
from evolve.multiobjective.metrics import (
    hypervolume_2d,
    hypervolume_contribution,
)

__all__ = [
    # Fitness
    "MultiObjectiveFitness",
    # Dominance
    "dominates",
    "pareto_front",
    # Ranking
    "fast_non_dominated_sort",
    "RankedIndividual",
    # Crowding
    "crowding_distance",
    # Selection
    "NSGA2Selector",
    "CrowdedTournamentSelection",
    # Metrics
    "hypervolume_2d",
    "hypervolume_contribution",
]
