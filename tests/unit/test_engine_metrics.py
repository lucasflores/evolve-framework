"""
Tests for minimize-aware metrics dict and extended population dynamics metrics.

Covers T014 (correct best_fitness/worst_fitness for both minimize modes),
T017 (fitness distribution), T022 (genome diversity), T024 (search movement).
"""

from __future__ import annotations

import numpy as np
import pytest

from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.representation.vector import VectorGenome

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_engine_callbacks)
# ---------------------------------------------------------------------------


class _DummyEvaluator:
    """Returns fitness = sum of genes."""

    def evaluate(self, individuals, seed=None):
        return [Fitness.scalar(float(np.sum(ind.genome.genes))) for ind in individuals]


class _DummySelection:
    def select(self, population, n, rng):
        inds = list(population.individuals)
        return [inds[i % len(inds)] for i in range(n)]


class _DummyCrossover:
    def crossover(self, g1, g2, rng):
        return g1.copy(), g2.copy()


class _DummyMutation:
    def mutate(self, g, rng):
        return g.copy()


def _make_population(
    fitness_values: list[float], minimize: bool = True
) -> Population[VectorGenome]:
    individuals = [
        Individual(
            genome=VectorGenome(genes=np.array([fv])),
            fitness=Fitness.scalar(fv),
        )
        for fv in fitness_values
    ]
    return Population(individuals=individuals, minimize=minimize)


def _make_engine(
    minimize: bool = True, metric_categories: frozenset[str] | None = None
) -> EvolutionEngine[VectorGenome]:
    cats = metric_categories or frozenset({"core"})
    config = EvolutionConfig(
        population_size=5,
        max_generations=2,
        elitism=1,
        minimize=minimize,
        metric_categories=cats,
    )
    return EvolutionEngine(
        config=config,
        evaluator=_DummyEvaluator(),
        selection=_DummySelection(),
        crossover=_DummyCrossover(),
        mutation=_DummyMutation(),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMinimizeAwareMetrics:
    """T014: _compute_metrics() best/worst reflect minimize flag."""

    def test_minimize_true_best_is_min(self):
        """With minimize=True, best_fitness = min, worst_fitness = max."""
        engine = _make_engine(minimize=True)
        pop = _make_population([1.0, 5.0, 3.0], minimize=True)

        # Access the internal _compute_metrics
        engine._timer.reset()
        engine._timer.start_generation()
        engine._timer.end_generation()
        metrics = engine._compute_metrics(pop)

        assert metrics["best_fitness"] == 1.0
        assert metrics["worst_fitness"] == 5.0

    def test_minimize_false_best_is_max(self):
        """With minimize=False, best_fitness = max, worst_fitness = min."""
        engine = _make_engine(minimize=False)
        pop = _make_population([1.0, 5.0, 3.0], minimize=False)

        engine._timer.reset()
        engine._timer.start_generation()
        engine._timer.end_generation()
        metrics = engine._compute_metrics(pop)

        assert metrics["best_fitness"] == 5.0
        assert metrics["worst_fitness"] == 1.0

    def test_worst_fitness_present_in_metrics(self):
        """worst_fitness is now always emitted in metrics dict."""
        engine = _make_engine(minimize=True)
        pop = _make_population([2.0, 4.0], minimize=True)

        engine._timer.reset()
        engine._timer.start_generation()
        engine._timer.end_generation()
        metrics = engine._compute_metrics(pop)

        assert "worst_fitness" in metrics

    def test_end_to_end_minimize_false(self):
        """Full run with minimize=False reports correct best in result."""
        engine = _make_engine(minimize=False)
        pop = _make_population([1.0, 2.0, 3.0, 4.0, 5.0], minimize=False)
        result = engine.run(initial_population=pop)

        # Best individual should have the highest fitness
        assert result.best.fitness is not None
        assert float(result.best.fitness.values[0]) == max(
            float(ind.fitness.values[0])
            for ind in result.population.individuals
            if ind.fitness is not None
        )


# ---------------------------------------------------------------------------
# T017: Fitness Distribution Metrics
# ---------------------------------------------------------------------------


def _init_timer(engine: EvolutionEngine) -> None:
    """Initialize timer so _compute_metrics doesn't fail."""
    engine._timer.reset()
    engine._timer.start_generation()
    engine._timer.end_generation()


class TestFitnessDistributionMetrics:
    """T017: median, Q1, Q3, min, max, fitness_range, unique_fitness_count."""

    def test_distribution_metrics_present(self):
        engine = _make_engine(metric_categories=frozenset({"core", "extended_population"}))
        pop = _make_population([1.0, 2.0, 3.0, 4.0, 5.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert "median_fitness" in metrics
        assert "q1_fitness" in metrics
        assert "q3_fitness" in metrics
        assert "min_fitness" in metrics
        assert "max_fitness" in metrics
        assert "fitness_range" in metrics
        assert "unique_fitness_count" in metrics

    def test_distribution_values_correct(self):
        engine = _make_engine(metric_categories=frozenset({"core", "extended_population"}))
        pop = _make_population([1.0, 2.0, 3.0, 4.0, 5.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert metrics["median_fitness"] == pytest.approx(3.0)
        assert metrics["min_fitness"] == pytest.approx(1.0)
        assert metrics["max_fitness"] == pytest.approx(5.0)
        assert metrics["fitness_range"] == pytest.approx(4.0)
        assert metrics["unique_fitness_count"] == 5

    def test_distribution_metrics_absent_without_category(self):
        engine = _make_engine(metric_categories=frozenset({"core"}))
        pop = _make_population([1.0, 2.0, 3.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert "median_fitness" not in metrics
        assert "q1_fitness" not in metrics


# ---------------------------------------------------------------------------
# T022: Genome Diversity Metrics
# ---------------------------------------------------------------------------


class TestGenomeDiversityMetrics:
    """T022: mean_gene_std, mean_distance_from_centroid, mean_pairwise_distance."""

    def test_diversity_metrics_present(self):
        engine = _make_engine(metric_categories=frozenset({"core", "diversity"}))
        pop = _make_population([1.0, 2.0, 3.0, 4.0, 5.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert "mean_gene_std" in metrics
        assert "mean_distance_from_centroid" in metrics
        assert "mean_pairwise_distance" in metrics

    def test_mean_gene_std_correct(self):
        """Identical genomes → std = 0."""
        individuals = [
            Individual(
                genome=VectorGenome(genes=np.array([1.0, 2.0])),
                fitness=Fitness.scalar(float(i)),
            )
            for i in range(5)
        ]
        pop = Population(individuals=individuals, minimize=True)
        engine = _make_engine(metric_categories=frozenset({"core", "diversity"}))
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert metrics["mean_gene_std"] == pytest.approx(0.0)

    def test_diversity_absent_without_category(self):
        engine = _make_engine(metric_categories=frozenset({"core"}))
        pop = _make_population([1.0, 2.0, 3.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert "mean_gene_std" not in metrics
        assert "mean_pairwise_distance" not in metrics


# ---------------------------------------------------------------------------
# T024: Search Movement Metrics
# ---------------------------------------------------------------------------


class TestSearchMovementMetrics:
    """T024: centroid_drift, best_genome_similarity, best_changed."""

    def test_best_changed_true_on_first_generation(self):
        engine = _make_engine(metric_categories=frozenset({"core", "diversity"}))
        pop = _make_population([1.0, 2.0, 3.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert metrics["best_changed"] is True

    def test_centroid_drift_present_after_two_calls(self):
        engine = _make_engine(metric_categories=frozenset({"core", "diversity"}))
        pop1 = _make_population([1.0, 2.0, 3.0])
        pop2 = _make_population([4.0, 5.0, 6.0])
        _init_timer(engine)

        engine._compute_metrics(pop1)  # sets _prev_centroid
        metrics = engine._compute_metrics(pop2)

        assert "centroid_drift" in metrics
        assert metrics["centroid_drift"] > 0

    def test_no_centroid_drift_on_first_call(self):
        engine = _make_engine(metric_categories=frozenset({"core", "diversity"}))
        pop = _make_population([1.0, 2.0, 3.0])
        _init_timer(engine)
        metrics = engine._compute_metrics(pop)

        assert "centroid_drift" not in metrics
