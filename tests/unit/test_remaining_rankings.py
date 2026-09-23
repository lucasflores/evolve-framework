"""
Every component that picks a "best" or "worst" ranks feasibility-first.

Deb's rules on values[0] via fitness_sort_key / Population.best: feasible
beats infeasible, then lower total violation, then the value.
"""

from __future__ import annotations

import json
from random import Random
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.representation.vector import VectorGenome


def _ind(value: float, constraints: list[float] | None = None) -> Individual[VectorGenome]:
    return Individual(
        genome=VectorGenome(genes=np.array([value])),
        fitness=Fitness(
            values=np.array([value]),
            constraints=None if constraints is None else np.array(constraints),
        ),
    )


def _deb_ordered(minimize: bool) -> list[Individual[VectorGenome]]:
    """Best first by Deb's rules; raw values alone would rank them in reverse."""
    sign = 1.0 if minimize else -1.0
    return [
        _ind(sign * 5.0, [-1.0]),  # feasible, worst raw value
        _ind(sign * 0.0, [0.5]),  # slightly infeasible
        _ind(sign * -10.0, [2.0]),  # very infeasible, best raw value
    ]


class SumWithFloor:
    """Minimize sum(genes) subject to genes[0] >= floor."""

    capabilities = EvaluatorCapabilities(n_objectives=1, n_constraints=1)

    def __init__(self, floor: float = 0.5) -> None:
        self.floor = floor

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        out = []
        for ind in individuals:
            g = np.asarray(ind.genome.genes)
            constraint = np.array([self.floor - g[0]])
            out.append(Fitness(values=np.array([g.sum()]), constraints=constraint))
        return out


@pytest.mark.parametrize("minimize", [True, False])
def test_erp_fitness_ranks(minimize: bool) -> None:
    from evolve.config import UnifiedConfig
    from evolve.config.erp import ERPSettings
    from evolve.factory import create_engine

    config = UnifiedConfig(
        population_size=3,
        selection="tournament",
        crossover="sbx",
        mutation="gaussian",
        genome_type="vector",
        genome_params={"dimensions": 1, "bounds": (-1.0, 1.0)},
        minimize=minimize,
        selection_params={"minimize": minimize},
        erp=ERPSettings(),
    )
    engine = create_engine(config, evaluator=lambda _g: 0.0)
    deb = _deb_ordered(minimize)

    ranks = engine._compute_fitness_ranks(Population(list(reversed(deb)), minimize=minimize))

    assert [ranks[ind.id] for ind in deb] == [0, 1, 2]


class TestIslands:
    def test_constrained_island_run_reports_feasible_best(self) -> None:
        from evolve.core.operators import BlendCrossover, GaussianMutation, TournamentSelection
        from evolve.diversity.islands.engine import IslandConfig, IslandEvolutionEngine

        engine = IslandEvolutionEngine(
            config=IslandConfig(
                n_islands=2, population_per_island=10, max_generations=8, migration_interval=2
            ),
            evaluator=SumWithFloor(),
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.5, sigma=0.3),
            seed=1,
        )

        result = engine.run(
            lambda rng: VectorGenome(genes=np.array([rng.uniform(-2, 2) for _ in range(3)]))
        )

        assert result.best.fitness is not None
        assert result.best.fitness.is_feasible
        stats = engine._history[-1]
        assert stats["global_best"] == float(result.best.fitness.values[0])

    @pytest.mark.parametrize("minimize", [True, False])
    def test_migration_emigrates_best_and_replaces_worst(self, minimize: bool) -> None:
        from evolve.diversity.islands.island import Island
        from evolve.diversity.islands.migration import (
            BestMigration,
            MigrationController,
            TournamentMigration,
        )

        best, middle, worst = _deb_ordered(minimize)
        island = Island(id=0, population=[worst, best, middle])

        (emigrant,) = BestMigration(minimize=minimize).select_emigrants(island, 1, Random(0))
        (winner,) = TournamentMigration(tournament_size=3, minimize=minimize).select_emigrants(
            island, 1, Random(0)
        )
        immigrant = _ind(1.0)
        MigrationController(policy=BestMigration(), minimize=minimize)._replace_worst(
            island, [immigrant]
        )

        assert emigrant.id == best.id
        assert winner.id == best.id
        assert worst not in island.population
        assert immigrant in island.population

    def test_island_engine_passes_its_direction_to_migration(self) -> None:
        from evolve.core.operators import BlendCrossover, GaussianMutation, TournamentSelection
        from evolve.diversity.islands.engine import IslandConfig, IslandEvolutionEngine

        engine = IslandEvolutionEngine(
            config=IslandConfig(minimize=True),
            evaluator=SumWithFloor(),
            selection=TournamentSelection(),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(),
        )

        assert engine.migration_controller.minimize is True
        assert engine.migration_controller.policy.minimize is True


@pytest.mark.parametrize("minimize", [True, False])
def test_species_best_member_and_stagnation(minimize: bool) -> None:
    from evolve.diversity.speciation import Species

    best, middle, worst = _deb_ordered(minimize)
    species = Species(id=0, representative=worst, members=[worst, middle])

    species.update_stagnation(minimize=minimize)  # only infeasible members
    assert species.stagnation_counter == 1
    species.members.append(best)
    species.update_stagnation(minimize=minimize)  # feasibility arrives

    assert species.get_best_member(minimize=minimize) is best
    assert species.stagnation_counter == 0
    assert species.best_fitness_ever == float(best.fitness.values[0])


def test_species_best_fitness_is_feasible_first() -> None:
    from evolve.diversity.speciation import Species

    best, middle, worst = _deb_ordered(minimize=False)
    species = Species(id=0, representative=worst, members=[worst, middle, best])

    assert species.best_fitness == float(best.fitness.values[0])


@pytest.mark.parametrize("minimize", [True, False])
def test_novelty_archive(minimize: bool) -> None:
    from evolve.diversity.novelty import QDArchive

    best, middle, worst = _deb_ordered(minimize)
    archive = QDArchive(dimensions=(2,), bounds=(np.array([0.0]), np.array([1.0])))

    assert archive.try_add(worst, np.array([0.1]), minimize)
    assert archive.try_add(best, np.array([0.1]), minimize)  # feasible replaces
    assert not archive.try_add(middle, np.array([0.1]), minimize)
    assert archive.try_add(middle, np.array([0.9]), minimize)

    assert archive.get_elites(2, minimize) == [best, middle]


def test_novelty_best_fitness_is_feasible_first() -> None:
    from evolve.diversity.novelty import QDArchive

    best, _, worst = _deb_ordered(minimize=False)
    archive = QDArchive(dimensions=(2,), bounds=(np.array([0.0]), np.array([1.0])))
    archive.try_add(worst, np.array([0.1]))
    archive.try_add(best, np.array([0.9]))

    assert archive.best_fitness == float(best.fitness.values[0])


@pytest.mark.parametrize("minimize", [True, False])
def test_deterministic_crowding_prefers_feasible(minimize: bool) -> None:
    from evolve.diversity.niching import deterministic_crowding_pairing

    best, _, worst = _deb_ordered(minimize)

    survivors = deterministic_crowding_pairing(
        [best], [worst], lambda _a, _b: 0.0, minimize=minimize
    )
    kept_parent = deterministic_crowding_pairing(
        [worst], [best], lambda _a, _b: 0.0, minimize=minimize
    )

    assert survivors == [best]
    assert kept_parent == [best]


def test_ensemble_elite_history_is_feasibility_first() -> None:
    from evolve.core.engine import EvolutionConfig, EvolutionEngine, create_initial_population
    from evolve.core.operators import BlendCrossover, GaussianMutation, TournamentSelection
    from evolve.utils.random import create_rng

    engine = EvolutionEngine(
        config=EvolutionConfig(
            population_size=10,
            max_generations=3,
            metric_categories=frozenset({"core", "ensemble"}),
        ),
        evaluator=SumWithFloor(floor=1.9),
        selection=TournamentSelection(),
        crossover=BlendCrossover(),
        mutation=GaussianMutation(),
        seed=3,
    )
    bounds = (np.full(3, -2.0), np.full(3, 2.0))
    population = create_initial_population(
        lambda r: VectorGenome.random(3, bounds, r), 10, create_rng(3)
    )

    result = engine.run(population)

    raw_lowest = min(result.population, key=lambda ind: float(ind.fitness.values[0]))
    assert not raw_lowest.fitness.is_feasible  # the case raw-value ranking gets wrong
    assert engine._prev_ensemble_elites == list(result.population.best(1, minimize=True))


def test_tracking_best_solution_artifact_is_feasible_first() -> None:
    from evolve.config.tracking import TrackingConfig
    from evolve.experiment.tracking.callback import TrackingCallback

    logged: list[dict[str, Any]] = []

    def capture(path: str, _name: str) -> None:
        with open(path) as f:
            logged.append(json.load(f))

    callback = TrackingCallback(config=TrackingConfig(backend="null"))
    callback._tracker = MagicMock(log_artifact=capture)
    best, middle, worst = _deb_ordered(minimize=True)

    callback._log_best_solution(Population([worst, middle, best], minimize=True))

    assert len(logged) == 1
    assert logged[0]["fitness"] == 5.0
    assert logged[0]["genes"] == [5.0]


def test_clearing_niche_winner_is_feasible_first() -> None:
    from evolve.diversity.niching import clearing

    best, _, worst = _deb_ordered(minimize=False)

    cleared = clearing([worst, best], lambda _a, _b: 0.0, sigma_clear=1.0, kappa=1)

    assert cleared == [0.0, float(best.fitness.values[0])]
