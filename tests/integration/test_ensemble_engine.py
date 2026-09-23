"""
Integration tests for EnsembleMetricCollector wired into EvolutionEngine.

Validates SC-003 from the ensemble metric spec:
  When `metric_categories` includes "ensemble", the engine must collect
  ensemble metrics at every generation and record them in result.history.
"""

from __future__ import annotations

import numpy as np
import pytest

from evolve.core.engine import EvolutionConfig, EvolutionEngine, create_initial_population
from evolve.core.operators.crossover import BlendCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome
from evolve.utils.random import create_rng

_N_DIMS = 3
_BOUNDS = (np.full(_N_DIMS, -5.0), np.full(_N_DIMS, 5.0))
_POP_SIZE = 10
_N_GENERATIONS = 3


def _sphere(genome: np.ndarray) -> float:
    return -float(np.sum(genome**2))


def _make_engine_and_population(
    metric_categories: frozenset[str],
    seed: int = 0,
) -> tuple[EvolutionEngine, object]:
    config = EvolutionConfig(
        population_size=_POP_SIZE,
        max_generations=_N_GENERATIONS,
        metric_categories=metric_categories,
    )
    engine = EvolutionEngine(
        config=config,
        evaluator=FunctionEvaluator(_sphere),
        selection=TournamentSelection(tournament_size=3),
        crossover=BlendCrossover(alpha=0.5),
        mutation=GaussianMutation(mutation_rate=0.3, sigma=0.2),
        seed=seed,
    )
    rng = create_rng(seed)
    pop = create_initial_population(
        genome_factory=lambda r: VectorGenome.random(_N_DIMS, _BOUNDS, r),
        population_size=_POP_SIZE,
        rng=rng,
    )
    return engine, pop


@pytest.mark.integration
class TestEnsembleEngineIntegration:
    """SC-003: ensemble metrics appear in result.history when category is enabled."""

    def test_ensemble_keys_present_in_every_history_entry(self) -> None:
        """All three always-present ensemble metric keys must appear in every generation."""
        engine, pop = _make_engine_and_population(frozenset({"core", "ensemble"}))
        result = engine.run(pop)

        assert len(result.history) == _N_GENERATIONS, (
            f"Expected {_N_GENERATIONS} history entries, got {len(result.history)}"
        )

        always_present = {
            "ensemble/gini_coefficient",
            "ensemble/participation_ratio",
            "ensemble/top_k_concentration",
        }
        for gen_idx, gen_metrics in enumerate(result.history):
            for key in always_present:
                assert key in gen_metrics, (
                    f"Key '{key}' missing from history[{gen_idx}]; "
                    f"available keys: {sorted(gen_metrics)}"
                )

    def test_ensemble_gini_in_valid_range(self) -> None:
        """Gini coefficient must be in [0, 1] for every generation."""
        engine, pop = _make_engine_and_population(frozenset({"core", "ensemble"}))
        result = engine.run(pop)

        for gen_idx, gen_metrics in enumerate(result.history):
            gini = gen_metrics["ensemble/gini_coefficient"]
            assert 0.0 <= gini <= 1.0, (
                f"gini_coefficient={gini!r} out of [0,1] at generation {gen_idx}"
            )

    def test_ensemble_participation_ratio_in_valid_range(self) -> None:
        """Participation ratio must be in [1, population_size] for every generation."""
        engine, pop = _make_engine_and_population(frozenset({"core", "ensemble"}))
        result = engine.run(pop)

        for gen_idx, gen_metrics in enumerate(result.history):
            pr = gen_metrics["ensemble/participation_ratio"]
            assert 1.0 <= pr <= float(_POP_SIZE), (
                f"participation_ratio={pr!r} out of [1, {_POP_SIZE}] at generation {gen_idx}"
            )

    def test_expert_turnover_appears_from_generation_1(self) -> None:
        """expert_turnover must be present from generation 1 onward (prev elites known)."""
        engine, pop = _make_engine_and_population(frozenset({"core", "ensemble"}))
        result = engine.run(pop)

        # Generation 0: no previous elites yet — key may be absent
        # Generation 1+: previous elites are known — key must be present
        for gen_idx in range(1, len(result.history)):
            assert "ensemble/expert_turnover" in result.history[gen_idx], (
                f"expert_turnover missing from history[{gen_idx}]"
            )
            turnover = result.history[gen_idx]["ensemble/expert_turnover"]
            assert 0.0 <= turnover <= 1.0, (
                f"expert_turnover={turnover!r} out of [0,1] at generation {gen_idx}"
            )

    def test_ensemble_absent_when_category_not_enabled(self) -> None:
        """No ensemble metrics must appear when 'ensemble' is not in metric_categories."""
        engine, pop = _make_engine_and_population(frozenset({"core"}))
        result = engine.run(pop)

        for gen_idx, gen_metrics in enumerate(result.history):
            ensemble_keys = [k for k in gen_metrics if k.startswith("ensemble/")]
            assert ensemble_keys == [], (
                f"Unexpected ensemble keys at generation {gen_idx}: {ensemble_keys}"
            )


@pytest.mark.integration
def test_ensemble_disabled_with_warning_in_multiobjective_mode() -> None:
    """Scalar fitness-mass metrics have no meaning for vector fitness."""
    from evolve.config import ObjectiveSpec, UnifiedConfig
    from evolve.config.tracking import MetricCategory, TrackingConfig
    from evolve.core.types import Fitness
    from evolve.evaluation.evaluator import EvaluatorCapabilities
    from evolve.factory import create_engine
    from evolve.factory import create_initial_population as initial_population

    class _TwoObjectives:
        capabilities = EvaluatorCapabilities(n_objectives=2)

        def evaluate(self, individuals, seed=None):
            return [Fitness(values=np.asarray(ind.genome.genes[:2])) for ind in individuals]

    config = UnifiedConfig(
        population_size=10,
        max_generations=2,
        selection="tournament",
        crossover="blend",
        mutation="gaussian",
        genome_type="vector",
        genome_params={"dimensions": 3, "bounds": (0.0, 1.0)},
        tracking=TrackingConfig(
            backend="null", categories=frozenset({MetricCategory.CORE, MetricCategory.ENSEMBLE})
        ),
    ).with_multiobjective(objectives=(ObjectiveSpec(name="a"), ObjectiveSpec(name="b")))

    with pytest.warns(UserWarning, match="ensemble metrics are disabled"):
        engine = create_engine(config, evaluator=_TwoObjectives())
    result = engine.run(initial_population(config))

    assert not any(key.startswith("ensemble/") for key in result.history[-1])
