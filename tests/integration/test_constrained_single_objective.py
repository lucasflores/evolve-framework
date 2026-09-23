"""
Single-objective runs through UnifiedConfig + create_engine(), with and
without constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from evolve.config import StoppingConfig, UnifiedConfig
from evolve.core.types import Fitness
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.factory import create_engine, create_initial_population


class SumWithFloor:
    """Minimize sum(genes) subject to genes[0] >= 0.5 (constraint 0.5 - genes[0] <= 0)."""

    capabilities = EvaluatorCapabilities(n_objectives=1, n_constraints=1)

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        out = []
        for ind in individuals:
            g = np.asarray(ind.genome.genes)
            out.append(Fitness(values=np.array([g.sum()]), constraints=np.array([0.5 - g[0]])))
        return out


def _config(**overrides: Any) -> UnifiedConfig:
    params: dict[str, Any] = {
        "name": "so",
        "population_size": 20,
        "max_generations": 8,
        "elitism": 2,
        "selection": "tournament",
        "crossover": "blend",
        "mutation": "gaussian",
        "genome_type": "vector",
        "genome_params": {"dimensions": 4, "bounds": (-2.0, 2.0)},
        "seed": 123,
    }
    params.update(overrides)
    return UnifiedConfig(**params)


@pytest.mark.integration
class TestConstrainedSingleObjective:
    def test_best_is_feasible(self) -> None:
        """Minimizing the sum pulls genes[0] below its floor; best must stay feasible."""
        cfg = _config(max_generations=15)

        result = create_engine(cfg, evaluator=SumWithFloor()).run(create_initial_population(cfg))

        assert result.best.fitness is not None
        assert result.best.fitness.is_feasible


class SumAndSecond:
    """Two values, but a single-objective config: rank on values[0] = sum(genes)."""

    capabilities = EvaluatorCapabilities(n_objectives=2, n_constraints=1)

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        out = []
        for ind in individuals:
            g = np.asarray(ind.genome.genes)
            out.append(
                Fitness(values=np.array([g.sum(), g[1]]), constraints=np.array([0.5 - g[0]]))
            )
        return out


@pytest.mark.integration
class TestVectorFitnessSingleObjective:
    """llm_judge rubrics, batch evaluators: several values, ranked on the first."""

    def test_history_and_stopping_rank_on_first_value(self) -> None:
        cfg = _config(max_generations=30, stopping=StoppingConfig(fitness_threshold=-4.5))

        result = create_engine(cfg, evaluator=SumAndSecond()).run(create_initial_population(cfg))

        assert "best_fitness" in result.history[0]
        assert result.best.fitness is not None
        assert result.best.fitness.is_feasible
        assert result.history[-1]["best_fitness"] == float(result.best.fitness.values[0])
        assert result.stop_reason.startswith("Fitness")


@pytest.mark.integration
class TestUnconstrainedRegression:
    """Pinned results from main before feasibility-first ranking.

    Unconstrained runs must not change. The tolerance only absorbs last-ulp
    libm differences across CI platforms; any change in selection shows up
    far above it.
    """

    @pytest.mark.parametrize(
        ("selection", "minimize", "best", "last_mean"),
        [
            ("tournament", True, 0.010572214302964119, 0.17619207498360298),
            ("rank", True, 0.4496181037594442, 0.9057238536359398),
            ("tournament", False, 15.40464104400667, 14.503368873241659),
            ("roulette", True, 0.0282603696801689, 0.3002451820312997),
        ],
    )
    def test_sphere_run_is_unchanged(
        self, selection: str, minimize: bool, best: float, last_mean: float
    ) -> None:
        cfg = _config(
            selection=selection,
            selection_params={"minimize": minimize},
            minimize=minimize,
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )

        result = create_engine(cfg).run(create_initial_population(cfg))

        assert result.best.fitness is not None
        assert float(result.best.fitness.values[0]) == pytest.approx(best, rel=1e-9)
        assert result.history[-1]["best_fitness"] == pytest.approx(best, rel=1e-9)
        assert result.history[-1]["mean_fitness"] == pytest.approx(last_mean, rel=1e-9)
