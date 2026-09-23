"""
Multi-objective runs declared through UnifiedConfig and built by create_engine().

Every test here goes through the declarative path only:
``UnifiedConfig(...).with_multiobjective(...)`` -> ``create_engine`` ->
``create_initial_population`` -> ``engine.run``.
"""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
import pytest

from evolve.config import ObjectiveSpec, UnifiedConfig
from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.factory import create_engine, create_initial_population
from evolve.multiobjective import NSGA2Selector


class SumAndFirstGene:
    """Objective ``a`` = sum(genes), objective ``b`` = genes[0].

    With ``limit`` set, one constraint ``genes[1] - limit <= 0`` is attached.
    Every evaluated individual is logged (with its fitness) in evaluation order.
    """

    capabilities = EvaluatorCapabilities(n_objectives=2, n_constraints=1)

    def __init__(self, limit: float | None = None) -> None:
        self.limit = limit
        self.evaluated: list[Individual[Any]] = []

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        out = []
        for ind in individuals:
            g = np.asarray(ind.genome.genes)
            constraints = None if self.limit is None else np.array([g[1] - self.limit])
            fitness = Fitness(values=np.array([g.sum(), g[0]]), constraints=constraints)
            self.evaluated.append(ind.with_fitness(fitness))
            out.append(fitness)
        return out


def _config(
    directions: tuple[str, str] = ("maximize", "maximize"),
    **overrides: Any,
) -> UnifiedConfig:
    params: dict[str, Any] = {
        "name": "mo",
        "population_size": 20,
        "max_generations": 5,
        "selection": "tournament",
        "crossover": "blend",
        "mutation": "gaussian",
        "genome_type": "vector",
        "genome_params": {"dimensions": 3, "bounds": (0.0, 1.0)},
        "seed": 1,
    }
    params.update(overrides)
    return UnifiedConfig(**params).with_multiobjective(
        objectives=tuple(
            ObjectiveSpec(name=name, direction=d)  # type: ignore[arg-type]
            for name, d in zip("ab", directions, strict=True)
        ),
    )


@pytest.mark.integration
class TestMultiObjectiveEngine:
    """The engine create_engine() builds for a multi-objective config."""

    def test_runs_through_create_engine(self) -> None:
        """A multi-objective config runs end to end instead of crashing in selection."""
        cfg = _config(max_generations=3)
        engine = create_engine(cfg, evaluator=SumAndFirstGene())

        result = engine.run(create_initial_population(cfg))

        assert result.generations == 3
        assert len(result.population) == cfg.population_size

    def test_survivors_are_nsga2_selection_of_parents_plus_offspring(self) -> None:
        """Each generation keeps NSGA2Selector's pick from parents + a full brood."""
        cfg = _config(max_generations=4)
        evaluator = SumAndFirstGene()
        recorder = _StepRecorder(evaluator)
        engine = create_engine(cfg, evaluator=evaluator)

        engine.run(create_initial_population(cfg), callbacks=[recorder])

        assert len(recorder.steps) == 4
        for parents, offspring, survivors in recorder.steps:
            assert len(offspring) == cfg.population_size
            expected = NSGA2Selector().select(parents + offspring, cfg.population_size, Random(0))
            assert [ind.id for ind in survivors] == [ind.id for ind in expected]

    def test_best_is_on_first_front(self) -> None:
        """result.best is a non-dominated member of the final population."""
        cfg = _config(max_generations=3)
        result = create_engine(cfg, evaluator=SumAndFirstGene()).run(create_initial_population(cfg))

        ranks, _ = NSGA2Selector().get_ranking_info(result.population.individuals)
        best_index = [ind.id for ind in result.population].index(result.best.id)
        assert ranks[best_index] == 0


class _StepRecorder:
    """Callback recording (parents, evaluated offspring, survivors) per generation."""

    def __init__(self, evaluator: SumAndFirstGene) -> None:
        self.evaluator = evaluator
        self.steps: list[tuple[list[Individual[Any]], ...]] = []
        self._parents: list[Individual[Any]] = []
        self._log_start = 0

    def on_generation_start(self, generation: int, population: Any) -> None:
        self._parents = list(population)
        self._log_start = len(self.evaluator.evaluated)

    def on_generation_end(self, generation: int, population: Any, metrics: Any) -> None:
        offspring = self.evaluator.evaluated[self._log_start :]
        self.steps.append((self._parents, offspring, list(population)))
