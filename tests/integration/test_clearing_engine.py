"""Clearing in the engine's multi-objective survival."""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
import pytest

from evolve.config.multiobjective import ConstraintSpec, MultiObjectiveConfig, ObjectiveSpec
from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.operators.crossover import SimulatedBinaryCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.diversity.niching import Clearing
from evolve.diversity.speciation import GenomeDistance
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.multiobjective.selection import CrowdedTournamentSelection
from evolve.representation.vector import VectorGenome
from tests.integration.test_fixed_seed_runs import TwoObjectives

OBJECTIVES = MultiObjectiveConfig(
    objectives=(
        ObjectiveSpec(name="spread", direction="minimize"),
        ObjectiveSpec(name="lead", direction="maximize"),
    )
)


class Unchanged:
    """A mutation that leaves the genome as it is: children copy their parents."""

    def mutate(self, genome: VectorGenome, rng: Random) -> VectorGenome:
        return genome.copy()


def _engine(
    clearing: Clearing | None,
    *,
    size: int = 8,
    generations: int = 1,
    mutation: Any = None,
    crossover_rate: float = 0.0,
    evaluator: Any = None,
    multiobjective: MultiObjectiveConfig | None = OBJECTIVES,
    seed: int = 4,
) -> EvolutionEngine:
    return EvolutionEngine(
        config=EvolutionConfig(
            population_size=size, max_generations=generations, crossover_rate=crossover_rate
        ),
        evaluator=evaluator or TwoObjectives(),
        selection=CrowdedTournamentSelection() if multiobjective else TournamentSelection(),
        crossover=SimulatedBinaryCrossover(),
        mutation=mutation or Unchanged(),
        seed=seed,
        multiobjective=multiobjective,
        clearing=clearing,
    )


def _population(rows: list[list[float]]) -> Population[VectorGenome]:
    bounds = (np.zeros(len(rows[0])), np.ones(len(rows[0])))
    return Population(
        [Individual(genome=VectorGenome(genes=np.array(r), bounds=bounds)) for r in rows]
    )


def _rows(population: Population[VectorGenome]) -> list[tuple[float, ...]]:
    return [tuple(ind.genome.genes.tolist()) for ind in population.individuals]


A, B, C, D = [0.3, 0.3], [0.9, 0.1], [0.1, 0.9], [0.6, 0.6]
COPIES = Clearing(distance=GenomeDistance(), closeness=0.0, copies=1)


class TestSurvival:
    def test_every_distinct_candidate_survives_before_any_copy(self) -> None:
        # Children copy their parents, so the pool holds many copies of A
        start = _population([A, A, A, A, A, B, C, D])
        result = _engine(COPIES).run(start)
        rows = _rows(result.population)
        assert {tuple(r) for r in (A, B, C, D)} <= set(rows)
        # Without clearing, survival keeps whatever ranks best, copies included
        plain = _rows(_engine(None).run(start).population)
        assert not {tuple(r) for r in (A, B, C, D)} <= set(plain)

    def test_copies_fill_places_only_when_the_groups_cannot(self) -> None:
        start = _population([A, A, A, A, A, B, C, D])
        rows = _rows(_engine(COPIES).run(start).population)
        assert len(rows) == 8
        assert len(set(rows)) == 4  # four groups; the other four places are copies

    def test_with_enough_groups_no_copy_survives(self) -> None:
        engine = _engine(COPIES, size=4)
        pool = _evaluated(engine, [A, A, B, C, D, A])
        survivors, ranking = engine._survive(pool, 4, engine._nsga2)
        rows = [tuple(i.genome.genes.tolist()) for i in survivors]
        assert sorted(rows) == sorted(tuple(r) for r in (A, B, C, D))
        assert set(ranking[0]) == set(range(4))

    def test_within_a_group_the_better_ranked_candidate_wins(self) -> None:
        # Near-copies: the feasible one wins though the infeasible one
        # dominates it on the objectives, because ranking puts feasible first
        evaluator = Limit()
        engine = _engine(
            Clearing(distance=GenomeDistance(), closeness=0.05, copies=1),
            size=3,
            evaluator=evaluator,
            multiobjective=MultiObjectiveConfig(
                objectives=OBJECTIVES.objectives, constraints=(ConstraintSpec(name="limit"),)
            ),
        )
        infeasible, feasible = [0.3, 0.29], [0.3, 0.31]
        pool = _evaluated(engine, [infeasible, feasible, [0.9, 0.4], C])
        survivors, _ = engine._survive(pool, 3, engine._nsga2)
        rows = [i.genome.genes.tolist() for i in survivors]
        assert feasible in rows
        assert infeasible not in rows

    def test_places_left_over_are_filled_from_the_held_back_only(self) -> None:
        engine = _engine(COPIES, size=3)
        pool = _evaluated(engine, [A, A, A, B])
        survivors, _ = engine._survive(pool, 3, engine._nsga2)
        # Two groups win two places; the third goes to another copy of A, never
        # to a candidate already through
        assert len({ind.id for ind in survivors}) == 3
        assert sorted(tuple(i.genome.genes.tolist()) for i in survivors) == sorted(
            [tuple(A), tuple(A), tuple(B)]
        )

    def test_fillers_rank_behind_every_winner_for_mating(self) -> None:
        engine = _engine(COPIES, size=3)
        pool = _evaluated(engine, [A, A, A, B])
        survivors, (ranks, crowding) = engine._survive(pool, 3, engine._nsga2)
        # Two winners first, then the held-back copy that filled the last place
        assert set(ranks) == set(crowding) == {0, 1, 2}
        assert ranks[2] > max(ranks[0], ranks[1])

    def test_within_a_group_the_more_isolated_candidate_wins(self) -> None:
        # Four non-dominated points; the two near-copies X1, X2 share a rank,
        # and X1 sits further from its neighbours on the front
        engine = _engine(Clearing(distance=GenomeDistance(), closeness=0.05, copies=1), size=3)
        rows = {"X1": [0.5, 0.5], "X2": [0.51, 0.5], "Y": [0.0, 0.0], "Z": [1.0, 1.0]}
        objectives = {"X1": (5.0, 5.5), "X2": (5.2, 5.6), "Y": (0.0, 0.0), "Z": (10.0, 10.0)}
        pool = [
            Individual(genome=VectorGenome(genes=np.array(rows[k]))).with_fitness(
                Fitness(values=np.array(objectives[k]))
            )
            for k in rows
        ]
        survivors, _ = engine._survive(pool, 3, engine._nsga2)
        kept = {k for k in rows for ind in survivors if ind.genome.genes.tolist() == rows[k]}
        assert kept == {"X1", "Y", "Z"}

    def test_measuring_only_changes_nothing(self) -> None:
        start = _population([[0.1 * i, 1 - 0.1 * i] for i in range(4)] + [[0.5, 0.5]] * 4)
        options = {"generations": 5, "mutation": GaussianMutation(), "crossover_rate": 0.9}
        plain = _engine(None, **options).run(start)
        measured = _engine(
            Clearing(distance=GenomeDistance(), closeness=0.0, copies=None), **options
        ).run(start)
        assert _rows(measured.population) == _rows(plain.population)
        assert all(h["clearing_held_back"] == 0 for h in measured.history)


class TestMetrics:
    def test_each_generation_logs_its_groups(self) -> None:
        start = _population([A, A, A, A, A, B, C, D])
        result = _engine(COPIES, generations=3).run(start)
        assert len(result.history) == 3
        first = result.history[0]
        # Pool of 16, parents and their copying children: four groups, the
        # largest holding at least A's five parent copies; one kept per group
        assert first["clearing_groups"] == 4
        assert first["clearing_largest"] >= 5
        assert first["clearing_held_back"] == 12

    def test_no_clearing_logs_nothing(self) -> None:
        result = _engine(None).run(_population([A, B, C, D, A, B, C, D]))
        assert not any(k.startswith("clearing_") for k in result.history[0])


class TestRefusal:
    def test_single_objective_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="multi-objective"):
            _engine(COPIES, multiobjective=None, evaluator=_Sum())


class Limit(TwoObjectives):
    """TwoObjectives with one constraint: gene 1 at least 0.3."""

    capabilities = EvaluatorCapabilities(n_objectives=2, n_constraints=1)

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        return [
            Fitness(values=f.values, constraints=np.array([0.3 - ind.genome.genes[1]]))
            for f, ind in zip(super().evaluate(individuals, seed), individuals)
        ]


class _Sum:
    capabilities = EvaluatorCapabilities(n_objectives=1)

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        return [Fitness(values=np.array([ind.genome.genes.sum()])) for ind in individuals]


def _evaluated(engine: EvolutionEngine, rows: list[list[float]]) -> list[Individual]:
    population = _population(rows)
    fitness = engine.evaluator.evaluate(population.individuals)
    return [ind.with_fitness(f) for ind, f in zip(population.individuals, fitness)]
