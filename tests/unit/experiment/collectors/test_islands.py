"""
Unit tests for IslandsMetricCollector.

Tests:
- Inter-island variance computation
- Intra-island variance computation
- Migration event tracking
- Reset functionality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from evolve.core.types import Fitness
from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.islands import IslandsMetricCollector


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

    def __len__(self) -> int:
        return len(self.individuals)


def make_individual(fitness_value: float) -> MockIndividual:
    """Create individual with given fitness."""
    fitness = Fitness(values=np.array([fitness_value]))
    return MockIndividual(fitness=fitness)


def make_island(fitness_values: list[float]) -> MockPopulation:
    """Create an island population with given fitness values."""
    individuals = [make_individual(f) for f in fitness_values]
    return MockPopulation(individuals=individuals)


def make_context(
    islands: list[MockPopulation] | None = None,
    generation: int = 1,
    extra: dict[str, Any] | None = None,
) -> CollectionContext:
    """Create test collection context with islands."""
    # Main population is combination of all islands
    all_individuals = []
    if islands:
        for island in islands:
            all_individuals.extend(island.individuals)

    population = MockPopulation(individuals=all_individuals)

    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore
        island_populations=islands,  # type: ignore
        extra=extra or {},
    )


class TestInterIslandVariance:
    """Tests for inter-island variance computation."""

    def test_variance_between_different_islands(self):
        """Test variance computed between islands with different means."""
        islands = [
            make_island([10.0, 11.0, 12.0]),  # Mean ≈ 11
            make_island([20.0, 21.0, 22.0]),  # Mean ≈ 21
            make_island([30.0, 31.0, 32.0]),  # Mean ≈ 31
        ]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert "inter_island_variance" in metrics
        # Variance of [11, 21, 31] should be significant
        assert metrics["inter_island_variance"] > 50.0

    def test_variance_zero_for_identical_islands(self):
        """Test variance is zero when all islands have same mean."""
        islands = [
            make_island([10.0, 10.0, 10.0]),
            make_island([10.0, 10.0, 10.0]),
            make_island([10.0, 10.0, 10.0]),
        ]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert "inter_island_variance" in metrics
        assert np.isclose(metrics["inter_island_variance"], 0.0)

    def test_island_count_reported(self):
        """Test island count is reported."""
        islands = [make_island([1.0, 2.0]) for _ in range(5)]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert metrics["island_count"] == 5


class TestIntraIslandVariance:
    """Tests for intra-island variance computation."""

    def test_variance_within_diverse_islands(self):
        """Test average variance within diverse islands."""
        islands = [
            make_island([1.0, 10.0, 20.0]),  # High variance
            make_island([5.0, 15.0, 25.0]),  # High variance
        ]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert "intra_island_variance" in metrics
        assert metrics["intra_island_variance"] > 30.0

    def test_variance_zero_for_uniform_islands(self):
        """Test variance is zero when all islands are uniform."""
        islands = [
            make_island([10.0, 10.0, 10.0]),
            make_island([20.0, 20.0, 20.0]),
        ]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert "intra_island_variance" in metrics
        assert np.isclose(metrics["intra_island_variance"], 0.0)


class TestMigrationEvents:
    """Tests for migration event tracking."""

    def test_migration_from_context_extra(self):
        """Test migration count from context extra."""
        islands = [make_island([1.0, 2.0]), make_island([3.0, 4.0])]
        context = make_context(islands, extra={"migration_events": 5})

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert metrics["migration_events"] == 5

    def test_migration_from_record_method(self):
        """Test migration count from record_migration method."""
        islands = [make_island([1.0, 2.0]), make_island([3.0, 4.0])]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        collector.record_migration(3)
        collector.record_migration(2)
        metrics = collector.collect(context)

        assert metrics["migration_events"] == 5

    def test_migration_resets_after_read(self):
        """Test migration count resets after being read."""
        islands = [make_island([1.0, 2.0]), make_island([3.0, 4.0])]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        collector.record_migration(3)

        # First collection should show 3
        metrics1 = collector.collect(context)
        assert metrics1["migration_events"] == 3

        # Second collection should not have migration_events (reset)
        metrics2 = collector.collect(context)
        assert "migration_events" not in metrics2

    def test_migration_disabled(self):
        """Test migration tracking can be disabled."""
        islands = [make_island([1.0, 2.0]), make_island([3.0, 4.0])]
        context = make_context(islands, extra={"migration_events": 5})

        collector = IslandsMetricCollector(track_migration=False)
        metrics = collector.collect(context)

        assert "migration_events" not in metrics


class TestNoIslandsHandling:
    """Tests for handling missing island data."""

    def test_no_islands_returns_empty(self):
        """Test empty metrics when no islands available."""
        context = make_context(islands=None)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert metrics == {}

    def test_empty_islands_list(self):
        """Test empty metrics when islands list is empty."""
        context = make_context(islands=[])

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        assert metrics == {}

    def test_single_island_insufficient(self):
        """Test metrics not computed with single island."""
        islands = [make_island([1.0, 2.0, 3.0])]
        context = make_context(islands)

        collector = IslandsMetricCollector()
        metrics = collector.collect(context)

        # Need at least 2 islands for variance
        assert "inter_island_variance" not in metrics


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_migration_count(self):
        """Test reset clears migration count."""
        collector = IslandsMetricCollector()
        collector.record_migration(10)

        assert collector._migration_count == 10

        collector.reset()

        assert collector._migration_count == 0

    def test_reset_clears_warning_state(self):
        """Test reset clears warning state."""
        context = make_context(islands=None)

        collector = IslandsMetricCollector()
        collector.collect(context)  # Triggers warning

        assert collector._warned_no_islands is True

        collector.reset()

        assert collector._warned_no_islands is False
