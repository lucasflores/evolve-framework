"""
Feasibility-first (Deb's rules) ranking of single-objective fitness.

Feasible beats infeasible; between infeasible, lower total violation
sum(max(0, c)) wins; between feasible, the objective value decides.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.core.callbacks import HallOfFameCallback
from evolve.core.operators.selection import RankSelection, RouletteSelection, TournamentSelection
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.representation.vector import VectorGenome


def _ind(value: float, constraints: list[float] | None = None) -> Individual[VectorGenome]:
    return Individual(
        genome=VectorGenome(genes=np.array([value])),
        fitness=Fitness(
            values=np.array([value]),
            constraints=None if constraints is None else np.array(constraints),
        ),
    )


def _population(minimize: bool) -> tuple[Population[VectorGenome], list[Individual[VectorGenome]]]:
    """Raw values rank the infeasible individuals best; Deb's rules rank them last.

    Returns the population and its individuals in Deb order (best first).
    """
    sign = 1.0 if minimize else -1.0
    feasible = _ind(sign * 5.0, [-1.0, 0.0])
    slightly_infeasible = _ind(sign * 0.0, [0.5, -3.0])
    very_infeasible = _ind(sign * -10.0, [1.0, 1.0])
    individuals = [very_infeasible, slightly_infeasible, feasible]
    return Population(individuals, minimize=minimize), [
        feasible,
        slightly_infeasible,
        very_infeasible,
    ]


@pytest.mark.parametrize("minimize", [True, False])
class TestFeasibilityFirst:
    def test_population_best_orders_by_deb_rules(self, minimize: bool) -> None:
        population, deb_order = _population(minimize)

        assert list(population.best(3, minimize=minimize)) == deb_order

    def test_statistics_best_is_feasible(self, minimize: bool) -> None:
        population, deb_order = _population(minimize)

        stats = population.statistics

        assert stats.best_fitness is deb_order[0].fitness
        assert stats.worst_fitness is deb_order[-1].fitness

    def test_tournament_picks_feasible(self, minimize: bool) -> None:
        population, deb_order = _population(minimize)
        selection = TournamentSelection(tournament_size=3, minimize=minimize)

        selected = selection.select(population, 10, Random(0))

        assert all(ind is deb_order[0] for ind in selected)

    def test_rank_selection_ranks_most_violating_last(self, minimize: bool) -> None:
        population, deb_order = _population(minimize)
        # selection_pressure=2.0 gives the worst rank zero probability
        selection = RankSelection(selection_pressure=2.0, minimize=minimize)

        selected = selection.select(population, 200, Random(0))

        assert deb_order[-1] not in selected

    def test_hall_of_fame_keeps_feasible_first(self, minimize: bool) -> None:
        population, deb_order = _population(minimize)
        hall = HallOfFameCallback(max_size=3)

        hall.on_generation_end(0, population, {})

        assert hall.archive == deb_order


def test_multiobjective_best_puts_feasible_first() -> None:
    """Vector fitness has no scalar order, but feasibility still ranks first (stable)."""
    genome = VectorGenome(genes=np.zeros(1))
    infeasible = Individual(genome=genome, fitness=Fitness(np.array([9.0, 9.0]), np.array([1.0])))
    feasible_a = Individual(genome=genome, fitness=Fitness(np.array([1.0, 2.0])))
    feasible_b = Individual(genome=genome, fitness=Fitness(np.array([2.0, 1.0]), np.array([0.0])))

    best = Population([infeasible, feasible_a, feasible_b]).best(3)

    assert list(best) == [feasible_a, feasible_b, infeasible]


def test_roulette_ignores_constraints() -> None:
    """Documented decision: roulette is proportional to raw values only."""
    constrained, _ = _population(minimize=True)
    unconstrained = Population([_ind(float(ind.fitness.values[0])) for ind in constrained])

    picked = RouletteSelection(minimize=True).select(constrained, 50, Random(3))
    reference = RouletteSelection(minimize=True).select(unconstrained, 50, Random(3))

    assert [ind.fitness.values[0] for ind in picked] == [ind.fitness.values[0] for ind in reference]
