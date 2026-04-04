"""
Unit tests for experiment metrics.

Tests compute_generation_metrics, compute_diversity_score, and
compute_elite_turnover_rate.
"""

import numpy as np

from evolve.experiment.metrics import (
    compute_diversity_score,
    compute_elite_turnover_rate,
    compute_generation_metrics,
)


class TestComputeGenerationMetrics:
    """Tests for compute_generation_metrics function."""

    def test_core_metrics_always_present(self) -> None:
        """Core metrics are always returned."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(fitness_values)

        assert "best_fitness" in metrics
        assert "min_fitness" in metrics
        assert "mean_fitness" in metrics
        assert "std_fitness" in metrics
        assert "max_fitness" in metrics

    def test_best_fitness_is_max(self) -> None:
        """Best fitness is the maximum value."""
        fitness_values = [1.0, 5.0, 3.0]
        metrics = compute_generation_metrics(fitness_values)

        assert metrics["best_fitness"] == 5.0
        assert metrics["max_fitness"] == 5.0

    def test_min_fitness_is_min(self) -> None:
        """Min fitness is the minimum value."""
        fitness_values = [1.0, 5.0, 3.0]
        metrics = compute_generation_metrics(fitness_values)

        assert metrics["min_fitness"] == 1.0

    def test_mean_fitness_correct(self) -> None:
        """Mean fitness is computed correctly."""
        fitness_values = [2.0, 4.0, 6.0]
        metrics = compute_generation_metrics(fitness_values)

        assert metrics["mean_fitness"] == 4.0

    def test_std_fitness_correct(self) -> None:
        """Std fitness is computed correctly."""
        fitness_values = [2.0, 4.0, 6.0]
        metrics = compute_generation_metrics(fitness_values)

        expected_std = np.std([2.0, 4.0, 6.0])
        assert abs(metrics["std_fitness"] - expected_std) < 1e-10

    def test_diversity_included_when_provided(self) -> None:
        """Diversity is included when provided."""
        fitness_values = [1.0, 2.0, 3.0]
        metrics = compute_generation_metrics(fitness_values, diversity=0.75)

        assert metrics["diversity"] == 0.75

    def test_diversity_not_included_when_none(self) -> None:
        """Diversity is not included when None."""
        fitness_values = [1.0, 2.0, 3.0]
        metrics = compute_generation_metrics(fitness_values, diversity=None)

        assert "diversity" not in metrics


class TestExtendedMetrics:
    """Tests for extended population statistics (FR-006)."""

    def test_extended_metrics_not_included_by_default(self) -> None:
        """Extended metrics are not included when extended=False."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(fitness_values, extended=False)

        assert "worst_fitness" not in metrics
        assert "median_fitness" not in metrics
        assert "fitness_q1" not in metrics
        assert "fitness_q3" not in metrics
        assert "fitness_range" not in metrics

    def test_extended_metrics_included_when_enabled(self) -> None:
        """Extended metrics are included when extended=True."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(fitness_values, extended=True)

        assert "worst_fitness" in metrics
        assert "median_fitness" in metrics
        assert "fitness_q1" in metrics
        assert "fitness_q3" in metrics
        assert "fitness_range" in metrics

    def test_worst_fitness_maximization(self) -> None:
        """Worst fitness is min when maximizing (default)."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(
            fitness_values,
            extended=True,
            minimize=False,
        )

        assert metrics["worst_fitness"] == 1.0

    def test_worst_fitness_minimization(self) -> None:
        """Worst fitness is max when minimizing."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(
            fitness_values,
            extended=True,
            minimize=True,
        )

        assert metrics["worst_fitness"] == 5.0

    def test_median_fitness_correct(self) -> None:
        """Median fitness is computed correctly."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_generation_metrics(fitness_values, extended=True)

        assert metrics["median_fitness"] == 3.0

    def test_median_fitness_even_count(self) -> None:
        """Median fitness works for even count."""
        fitness_values = [1.0, 2.0, 3.0, 4.0]
        metrics = compute_generation_metrics(fitness_values, extended=True)

        assert metrics["median_fitness"] == 2.5

    def test_quartiles_correct(self) -> None:
        """Quartiles are computed correctly."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        metrics = compute_generation_metrics(fitness_values, extended=True)

        # NumPy percentile with default interpolation
        assert metrics["fitness_q1"] == np.percentile(fitness_values, 25)
        assert metrics["fitness_q3"] == np.percentile(fitness_values, 75)

    def test_fitness_range_correct(self) -> None:
        """Fitness range is max - min."""
        fitness_values = [2.0, 5.0, 8.0]
        metrics = compute_generation_metrics(fitness_values, extended=True)

        assert metrics["fitness_range"] == 6.0  # 8.0 - 2.0


class TestComputeDiversityScore:
    """Tests for compute_diversity_score function (FR-007)."""

    def test_diversity_zero_for_single_genome(self) -> None:
        """Diversity is 0 for single genome."""
        genomes = [np.array([1.0, 2.0, 3.0])]
        score = compute_diversity_score(genomes)

        assert score == 0.0

    def test_diversity_zero_for_empty_population(self) -> None:
        """Diversity is 0 for empty population."""
        genomes = []
        score = compute_diversity_score(genomes)

        assert score == 0.0

    def test_diversity_positive_for_different_genomes(self) -> None:
        """Diversity is positive for different genomes."""
        genomes = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        score = compute_diversity_score(genomes)

        assert score > 0.0

    def test_diversity_zero_for_identical_genomes(self) -> None:
        """Diversity is 0 for identical genomes."""
        genomes = [
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        ]
        score = compute_diversity_score(genomes)

        assert score == 0.0

    def test_diversity_uses_euclidean_by_default(self) -> None:
        """Default distance is Euclidean."""
        genomes = [
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0]),  # Distance = 5.0
        ]
        score = compute_diversity_score(genomes)

        assert abs(score - 5.0) < 1e-10

    def test_diversity_custom_distance_function(self) -> None:
        """Custom distance function is used."""
        genomes = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]

        # Manhattan distance
        def manhattan(a, b):
            return np.sum(np.abs(a - b))

        score = compute_diversity_score(genomes, distance_fn=manhattan)

        # |1-3| + |2-4| = 4
        assert abs(score - 4.0) < 1e-10

    def test_diversity_sampling_reduces_population(self) -> None:
        """Sampling is used for large populations."""
        # Create large population
        rng = np.random.default_rng(42)
        genomes = [rng.random(5) for _ in range(100)]

        # With sampling
        score_sampled = compute_diversity_score(
            genomes,
            sample_size=10,
            rng=rng,
        )

        # Should still compute without error
        assert score_sampled >= 0.0

    def test_diversity_deterministic_with_rng(self) -> None:
        """Sampling is deterministic with RNG."""
        genomes = [np.random.default_rng(i).random(3) for i in range(50)]

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        score1 = compute_diversity_score(genomes, sample_size=10, rng=rng1)
        score2 = compute_diversity_score(genomes, sample_size=10, rng=rng2)

        assert score1 == score2


class TestComputeEliteTurnoverRate:
    """Tests for compute_elite_turnover_rate function (FR-009)."""

    def test_turnover_zero_no_change(self) -> None:
        """Turnover is 0 when elites don't change."""
        previous = {1, 2, 3, 4, 5}
        current = {1, 2, 3, 4, 5}

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 0.0

    def test_turnover_one_complete_replacement(self) -> None:
        """Turnover is 1 when all elites replaced."""
        previous = {1, 2, 3, 4, 5}
        current = {6, 7, 8, 9, 10}

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 1.0

    def test_turnover_partial_replacement(self) -> None:
        """Turnover is correct for partial replacement."""
        previous = {1, 2, 3, 4, 5}
        current = {1, 2, 6, 7, 8}  # 3 new

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 0.6  # 3/5

    def test_turnover_empty_previous(self) -> None:
        """Turnover is 0 with empty previous."""
        previous = set()
        current = {1, 2, 3}

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 0.0

    def test_turnover_empty_current(self) -> None:
        """Turnover is 0 with empty current."""
        previous = {1, 2, 3}
        current = set()

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 0.0

    def test_turnover_different_sizes(self) -> None:
        """Turnover handles different elite sizes."""
        previous = {1, 2, 3}
        current = {1, 4, 5, 6, 7}  # 4 new out of 5

        rate = compute_elite_turnover_rate(previous, current)

        assert rate == 0.8  # 4/5
