"""
Tests for minimize-aware PopulationStatistics.

Covers T007 (minimize-aware stats), T012 (stats.minimize field).
"""

from __future__ import annotations

import numpy as np

from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.representation.vector import VectorGenome


def _make_individuals(fitness_values: list[float]) -> list[Individual[VectorGenome]]:
    """Create individuals with known scalar fitness values."""
    individuals = []
    for i, fv in enumerate(fitness_values):
        genome = VectorGenome(genes=np.array([float(i)]))
        ind = Individual(genome=genome, fitness=Fitness.scalar(fv))
        individuals.append(ind)
    return individuals


class TestMinimizeAwareStatistics:
    """T007: best_fitness/worst_fitness respect minimize flag."""

    def test_best_is_min_when_minimize_true(self):
        """With minimize=True, best_fitness should be the minimum."""
        inds = _make_individuals([5.0, 1.0, 3.0])
        pop = Population(individuals=inds, minimize=True)
        stats = pop.statistics

        assert stats.best_fitness is not None
        assert float(stats.best_fitness.values[0]) == 1.0

    def test_worst_is_max_when_minimize_true(self):
        """With minimize=True, worst_fitness should be the maximum."""
        inds = _make_individuals([5.0, 1.0, 3.0])
        pop = Population(individuals=inds, minimize=True)
        stats = pop.statistics

        assert stats.worst_fitness is not None
        assert float(stats.worst_fitness.values[0]) == 5.0

    def test_best_is_max_when_minimize_false(self):
        """With minimize=False, best_fitness should be the maximum."""
        inds = _make_individuals([5.0, 1.0, 3.0])
        pop = Population(individuals=inds, minimize=False)
        stats = pop.statistics

        assert stats.best_fitness is not None
        assert float(stats.best_fitness.values[0]) == 5.0

    def test_worst_is_min_when_minimize_false(self):
        """With minimize=False, worst_fitness should be the minimum."""
        inds = _make_individuals([5.0, 1.0, 3.0])
        pop = Population(individuals=inds, minimize=False)
        stats = pop.statistics

        assert stats.worst_fitness is not None
        assert float(stats.worst_fitness.values[0]) == 1.0

    def test_multi_objective_fallback(self):
        """Multi-objective populations fall back to first-element heuristic."""
        genomes = [VectorGenome(genes=np.array([float(i)])) for i in range(3)]
        individuals = [
            Individual(genome=genomes[0], fitness=Fitness(values=np.array([1.0, 2.0]))),
            Individual(genome=genomes[1], fitness=Fitness(values=np.array([3.0, 1.0]))),
            Individual(genome=genomes[2], fitness=Fitness(values=np.array([2.0, 3.0]))),
        ]
        pop = Population(individuals=individuals, minimize=True)
        stats = pop.statistics

        # Multi-objective: best/worst still set (fallback behavior)
        assert stats.best_fitness is not None
        assert stats.worst_fitness is not None

    def test_empty_fitness_population(self):
        """Population with no evaluated individuals returns None stats."""
        genomes = [VectorGenome(genes=np.array([float(i)])) for i in range(3)]
        individuals = [Individual(genome=g) for g in genomes]  # No fitness set
        pop = Population(individuals=individuals)
        stats = pop.statistics

        assert stats.best_fitness is None
        assert stats.worst_fitness is None
        assert stats.mean_fitness is None


class TestStatisticsMinimizeField:
    """T012: PopulationStatistics.minimize field is accessible and correct."""

    def test_minimize_true_stored(self):
        """stats.minimize reflects minimize=True."""
        inds = _make_individuals([1.0, 2.0])
        pop = Population(individuals=inds, minimize=True)
        assert pop.statistics.minimize is True

    def test_minimize_false_stored(self):
        """stats.minimize reflects minimize=False."""
        inds = _make_individuals([1.0, 2.0])
        pop = Population(individuals=inds, minimize=False)
        assert pop.statistics.minimize is False

    def test_default_minimize_is_true(self):
        """Default Population (no minimize arg) produces stats.minimize=True."""
        inds = _make_individuals([1.0, 2.0])
        pop = Population(individuals=inds)
        assert pop.statistics.minimize is True
