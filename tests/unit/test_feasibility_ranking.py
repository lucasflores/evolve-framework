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


def test_vector_fitness_best_ranks_first_value_feasibility_first() -> None:
    """Without a ranker (single-objective mode) vector fitness ranks on values[0]."""
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


def test_sort_key_is_ascending_in_both_directions() -> None:
    """Best first with plain sorted()/min(); no reverse= or max() needed."""
    from evolve.core.types import fitness_sort_key

    for minimize in (True, False):
        population, deb_order = _population(minimize)
        shuffled = list(reversed(deb_order))

        ranked = sorted(shuffled, key=lambda ind: fitness_sort_key(ind.fitness, minimize))

        assert ranked == deb_order
        assert (
            min(shuffled, key=lambda ind: fitness_sort_key(ind.fitness, minimize)) is deb_order[0]
        )
    assert fitness_sort_key(None, True) == fitness_sort_key(None, False) == (2, 0.0, 0.0)


def test_one_violation_formula() -> None:
    """Fitness, MultiObjectiveFitness and dominance agree on total violation."""
    from evolve.core.types import total_violation
    from evolve.multiobjective.fitness import MultiObjectiveFitness

    constraints = np.array([0.5, -2.0, 1.25])
    core = Fitness(values=np.array([1.0]), constraints=constraints)
    mo = MultiObjectiveFitness(objectives=np.array([1.0, 2.0]), constraint_violations=constraints)
    worse = Fitness(values=np.array([1.0]), constraints=np.array([2.0]))

    assert total_violation(constraints) == 1.75
    assert core.total_constraint_violation == mo.total_constraint_violation == 1.75
    assert core.dominates(worse) and not worse.dominates(core)
    assert total_violation(None) == 0.0


def test_ranker_orders_best_by_front_then_crowding() -> None:
    """With an NSGA-II ranker, best(n) follows Pareto fronts, then crowding distance."""
    from evolve.multiobjective import NSGA2Selector

    genome = VectorGenome(genes=np.zeros(1))
    points = [(1.0, 1.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0), (2.1, 2.1)]
    individuals = [Individual(genome=genome, fitness=Fitness(np.array(p))) for p in points]
    population = Population(individuals, ranker=NSGA2Selector())

    best = population.best(5)

    # Front 0 = (3,1), (1,3) boundaries (infinite crowding), then (2.1,2.1); (2,2), (1,1) after
    assert best[:2] == [individuals[1], individuals[3]]
    assert best[2] is individuals[4]
    assert best[3:] == [individuals[2], individuals[0]]
    assert population.ranking == NSGA2Selector().get_ranking_info(individuals)


def test_derived_populations_keep_the_ranker() -> None:
    from evolve.multiobjective import NSGA2Selector

    ranker = NSGA2Selector()
    population = Population([_ind(1.0), _ind(2.0)], ranker=ranker)

    assert population.with_individuals(list(population)).ranker is ranker
    assert population.filter_evaluated().ranker is ranker
    assert population.increment_ages().ranker is ranker
