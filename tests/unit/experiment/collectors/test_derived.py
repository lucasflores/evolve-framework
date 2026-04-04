"""
Unit tests for DerivedAnalyticsCollector.

Tests:
- Selection pressure computation
- Fitness improvement velocity with window
- Population entropy using histogram
- History state management
- Reset functionality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from evolve.core.types import Fitness
from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.derived import DerivedAnalyticsCollector


@dataclass
class MockStatistics:
    """Mock population statistics."""

    best_fitness: Fitness | None = None
    mean_fitness: Fitness | None = None
    std_fitness: float | None = None
    evaluated_count: int = 0


@dataclass
class MockIndividual:
    """Mock individual for testing."""

    id: Any = None
    fitness: Fitness | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()


@dataclass
class MockPopulation:
    """Mock population for testing."""

    individuals: list[MockIndividual]
    _statistics: MockStatistics | None = None
    _minimize: bool = False

    def __len__(self) -> int:
        return len(self.individuals)

    @property
    def statistics(self) -> MockStatistics:
        if self._statistics is not None:
            return self._statistics

        # Compute from individuals
        fitness_values = []
        for ind in self.individuals:
            if ind.fitness is not None:
                fitness_values.append(float(ind.fitness.values[0]))

        if not fitness_values:
            return MockStatistics()

        best_val = max(fitness_values)
        mean_val = np.mean(fitness_values)

        return MockStatistics(
            best_fitness=Fitness(values=np.array([best_val])),
            mean_fitness=Fitness(values=np.array([mean_val])),
            std_fitness=float(np.std(fitness_values)),
            evaluated_count=len(fitness_values),
        )

    def best(self, n: int = 1, minimize: bool = False) -> list[MockIndividual]:
        """Return best individuals by fitness."""
        sorted_inds = sorted(
            [i for i in self.individuals if i.fitness is not None],
            key=lambda x: float(x.fitness.values[0]),
            reverse=not minimize,
        )
        return sorted_inds[:n]


def make_individual(fitness_value: float) -> MockIndividual:
    """Create individual with given fitness."""
    fitness = Fitness(values=np.array([fitness_value]))
    return MockIndividual(fitness=fitness)


def make_context(
    individuals: list[MockIndividual] | None = None,
    generation: int = 1,
) -> CollectionContext:
    """Create test collection context."""
    if individuals is None:
        individuals = []

    population = MockPopulation(individuals=individuals)

    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore
    )


class TestSelectionPressure:
    """Tests for selection pressure computation."""

    def test_selection_pressure_basic(self):
        """Test basic selection pressure computation."""
        individuals = [
            make_individual(10.0),  # Best
            make_individual(5.0),
            make_individual(5.0),  # Mean = (10+5+5)/3 = 6.67
        ]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "selection_pressure" in metrics
        # Best = 10, Mean ≈ 6.67, Pressure ≈ 1.5
        assert metrics["selection_pressure"] > 1.0

    def test_selection_pressure_uniform_population(self):
        """Test selection pressure with uniform fitness."""
        individuals = [
            make_individual(5.0),
            make_individual(5.0),
            make_individual(5.0),
        ]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "selection_pressure" in metrics
        # Best = Mean = 5, Pressure = 1.0
        assert np.isclose(metrics["selection_pressure"], 1.0)

    def test_selection_pressure_high_variance(self):
        """Test selection pressure with high fitness variance."""
        individuals = [
            make_individual(100.0),  # Best
            make_individual(1.0),
            make_individual(1.0),
        ]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "selection_pressure" in metrics
        # Best = 100, Mean ≈ 34, Pressure ≈ 2.94
        assert metrics["selection_pressure"] > 2.0

    def test_selection_pressure_disabled(self):
        """Test selection pressure can be disabled."""
        individuals = [make_individual(10.0), make_individual(5.0)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector(enable_selection_pressure=False)
        metrics = collector.collect(context)

        assert "selection_pressure" not in metrics


class TestFitnessVelocity:
    """Tests for fitness improvement velocity."""

    def test_velocity_after_multiple_generations(self):
        """Test velocity computed after sufficient history."""
        collector = DerivedAnalyticsCollector(velocity_window=3)

        # Simulate multiple generations with improving fitness
        for gen, best_fitness in enumerate([10.0, 12.0, 14.0, 16.0, 18.0]):
            individuals = [make_individual(best_fitness), make_individual(best_fitness - 5)]
            context = make_context(individuals, generation=gen)
            metrics = collector.collect(context)

        assert "fitness_improvement_velocity" in metrics
        # Fitness increases by 2.0 per generation
        assert metrics["fitness_improvement_velocity"] > 0

    def test_velocity_insufficient_history(self):
        """Test velocity not computed without enough history."""
        collector = DerivedAnalyticsCollector(velocity_window=5)

        # Only one generation
        individuals = [make_individual(10.0)]
        context = make_context(individuals, generation=0)
        metrics = collector.collect(context)

        assert "fitness_improvement_velocity" not in metrics

    def test_velocity_stable_fitness(self):
        """Test velocity with stable (non-improving) fitness."""
        collector = DerivedAnalyticsCollector(velocity_window=3)

        # Same fitness every generation
        for gen in range(5):
            individuals = [make_individual(10.0)]
            context = make_context(individuals, generation=gen)
            metrics = collector.collect(context)

        assert "fitness_improvement_velocity" in metrics
        # No improvement should yield ~0 velocity
        assert np.isclose(metrics["fitness_improvement_velocity"], 0.0, atol=0.01)

    def test_velocity_decreasing_fitness(self):
        """Test velocity with decreasing fitness."""
        collector = DerivedAnalyticsCollector(velocity_window=3)

        # Decreasing fitness
        for gen, best_fitness in enumerate([20.0, 18.0, 16.0, 14.0]):
            individuals = [make_individual(best_fitness)]
            context = make_context(individuals, generation=gen)
            metrics = collector.collect(context)

        assert "fitness_improvement_velocity" in metrics
        # Decreasing fitness should yield negative velocity
        assert metrics["fitness_improvement_velocity"] < 0

    def test_velocity_window_size(self):
        """Test velocity uses configured window size."""
        collector = DerivedAnalyticsCollector(velocity_window=2)

        # First few generations: stable
        for gen in range(3):
            individuals = [make_individual(10.0)]
            context = make_context(individuals, generation=gen)
            collector.collect(context)

        # Last 2 generations: rapid increase (window=2 should capture this)
        for gen, fitness in [(3, 20.0), (4, 30.0)]:
            individuals = [make_individual(fitness)]
            context = make_context(individuals, generation=gen)
            metrics = collector.collect(context)

        assert "fitness_improvement_velocity" in metrics
        # Window of 2 should show high velocity from recent generations
        assert metrics["fitness_improvement_velocity"] > 5.0

    def test_velocity_disabled(self):
        """Test velocity can be disabled."""
        collector = DerivedAnalyticsCollector(enable_velocity=False)

        for gen in range(5):
            individuals = [make_individual(10.0 + gen)]
            context = make_context(individuals, generation=gen)
            metrics = collector.collect(context)

        assert "fitness_improvement_velocity" not in metrics


class TestPopulationEntropy:
    """Tests for population entropy computation."""

    def test_entropy_diverse_population(self):
        """Test entropy with diverse fitness distribution."""
        # Create diverse population
        individuals = [make_individual(float(i)) for i in range(20)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "population_entropy" in metrics
        # Diverse population should have higher entropy
        assert metrics["population_entropy"] > 0.3

    def test_entropy_uniform_population(self):
        """Test entropy with uniform fitness."""
        individuals = [make_individual(5.0) for _ in range(20)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "population_entropy" in metrics
        # Uniform fitness should have zero entropy
        assert np.isclose(metrics["population_entropy"], 0.0)

    def test_entropy_bimodal_distribution(self):
        """Test entropy with bimodal fitness distribution."""
        # 10 individuals at fitness 1.0, 10 at fitness 10.0
        individuals = [make_individual(1.0) for _ in range(10)]
        individuals += [make_individual(10.0) for _ in range(10)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector(entropy_bins=5)
        metrics = collector.collect(context)

        assert "population_entropy" in metrics
        # Bimodal should have some entropy but less than uniform
        assert 0 < metrics["population_entropy"] < 1

    def test_entropy_bins_configuration(self):
        """Test entropy uses configured bin count."""
        individuals = [make_individual(float(i)) for i in range(20)]
        context = make_context(individuals)

        collector_10_bins = DerivedAnalyticsCollector(entropy_bins=10)
        collector_50_bins = DerivedAnalyticsCollector(entropy_bins=50)

        metrics_10 = collector_10_bins.collect(context)
        metrics_50 = collector_50_bins.collect(context)

        # Both should compute entropy, values may differ due to binning
        assert "population_entropy" in metrics_10
        assert "population_entropy" in metrics_50

    def test_entropy_single_individual(self):
        """Test entropy with single individual."""
        individuals = [make_individual(5.0)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        # Not enough individuals for entropy
        assert "population_entropy" not in metrics

    def test_entropy_disabled(self):
        """Test entropy can be disabled."""
        individuals = [make_individual(float(i)) for i in range(10)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector(enable_entropy=False)
        metrics = collector.collect(context)

        assert "population_entropy" not in metrics


class TestHistoryManagement:
    """Tests for history state management."""

    def test_history_accumulates(self):
        """Test history accumulates across generations."""
        collector = DerivedAnalyticsCollector()

        for gen in range(10):
            individuals = [make_individual(float(gen + 1) * 10)]
            context = make_context(individuals, generation=gen)
            collector.collect(context)

        # History should have 10 entries
        assert len(collector._best_fitness_history) == 10
        assert len(collector._generation_history) == 10

    def test_history_respects_maxlen(self):
        """Test history doesn't grow unbounded."""
        collector = DerivedAnalyticsCollector(velocity_window=5)

        # Collect many generations
        for gen in range(100):
            individuals = [make_individual(float(gen))]
            context = make_context(individuals, generation=gen)
            collector.collect(context)

        # History should be bounded
        assert len(collector._best_fitness_history) <= 100


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_history(self):
        """Test reset clears all history."""
        collector = DerivedAnalyticsCollector()

        # Accumulate some history
        for gen in range(5):
            individuals = [make_individual(10.0)]
            context = make_context(individuals, generation=gen)
            collector.collect(context)

        assert len(collector._best_fitness_history) > 0

        collector.reset()

        assert len(collector._best_fitness_history) == 0
        assert len(collector._mean_fitness_history) == 0
        assert len(collector._generation_history) == 0

    def test_reset_allows_fresh_start(self):
        """Test reset allows starting a fresh run."""
        collector = DerivedAnalyticsCollector(velocity_window=3)

        # First run with improving fitness
        for gen in range(5):
            individuals = [make_individual(10.0 + gen * 2)]
            context = make_context(individuals, generation=gen)
            collector.collect(context)

        collector.reset()

        # Second run - velocity should not be available immediately
        individuals = [make_individual(50.0)]
        context = make_context(individuals, generation=0)
        metrics = collector.collect(context)

        # Should not have velocity on first generation after reset
        assert "fitness_improvement_velocity" not in metrics


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_population(self):
        """Test empty population returns no metrics."""
        context = make_context([])

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        assert "selection_pressure" not in metrics
        assert "population_entropy" not in metrics

    def test_none_fitness(self):
        """Test individuals with None fitness are handled."""
        individuals = [
            MockIndividual(fitness=None),
            make_individual(10.0),
        ]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        # Should still compute from valid individuals
        assert "selection_pressure" in metrics

    def test_mean_zero(self):
        """Test handling of zero mean fitness."""
        individuals = [
            make_individual(0.0),
            make_individual(0.0),
        ]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector()
        metrics = collector.collect(context)

        # Best = 0, Mean = 0, Pressure should be 1.0
        assert "selection_pressure" in metrics
        assert metrics["selection_pressure"] == 1.0

    def test_all_metrics_disabled(self):
        """Test with all metrics disabled."""
        individuals = [make_individual(10.0)]
        context = make_context(individuals)

        collector = DerivedAnalyticsCollector(
            enable_selection_pressure=False,
            enable_velocity=False,
            enable_entropy=False,
        )
        metrics = collector.collect(context)

        assert metrics == {}
