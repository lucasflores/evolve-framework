"""
Multi-objective runs declared through UnifiedConfig and built by create_engine().

Every test here goes through the declarative path only:
``UnifiedConfig(...).with_multiobjective(...)`` -> ``create_engine`` ->
``create_initial_population`` -> ``engine.run``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from evolve.config import ObjectiveSpec, UnifiedConfig
from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.factory import create_engine, create_initial_population


class SumAndFirstGene:
    """Objective ``a`` = sum(genes), objective ``b`` = genes[0].

    With ``limit`` set, one constraint ``genes[1] - limit <= 0`` is attached.
    Every evaluated individual is logged (with its fitness) in evaluation order.
    """

    capabilities = EvaluatorCapabilities(n_objectives=2, n_constraints=1)

    def __init__(self, limit: float | None = None) -> None:
        self.limit = limit
        self.evaluated: list[Individual[Any]] = []

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:  # noqa: ARG002
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
