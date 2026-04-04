"""
Comprehensive integration tests for MLflow metrics tracking.

Tests all user stories from the 006-mlflow-metrics-tracking spec:
- US1: Declarative tracking with UnifiedConfig
- US2: Full population statistics
- US3: Timing instrumentation
- US4: ERP mating statistics (via collectors)
- US5: Multi-objective metrics (via collectors)
- US6: Fitness metadata extraction (via collectors)
- US7: Derived analytics (via collectors)

These tests use mock/null backends to avoid MLflow dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from evolve.config.multiobjective import ObjectiveSpec
from evolve.config.tracking import MetricCategory, TrackingConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.types import Fitness
from evolve.experiment.collectors import (
    DerivedAnalyticsCollector,
    ERPMetricCollector,
    FitnessMetadataCollector,
    IslandsMetricCollector,
    MultiObjectiveMetricCollector,
    NEATMetricCollector,
    SpeciationMetricCollector,
)
from evolve.experiment.collectors.base import CollectionContext, MatingStats

# -----------------------------------------------------------------------------
# Mock Types
# -----------------------------------------------------------------------------


@dataclass
class MockStatistics:
    """Mock statistics for populations."""

    best_fitness: Fitness | None = None
    mean_fitness: Any = None
    std_fitness: Any = None


@dataclass
class MockIndividual:
    """Mock individual for testing."""

    id: Any = None
    fitness: Fitness | None = None
    genome: Any = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()


@dataclass
class MockPopulation:
    """Mock population for testing."""

    individuals: list[MockIndividual]
    _statistics: MockStatistics | None = None

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, idx: int) -> MockIndividual:
        return self.individuals[idx]

    @property
    def statistics(self) -> MockStatistics:
        """Get population statistics."""
        if self._statistics is not None:
            return self._statistics

        if not self.individuals:
            return MockStatistics()

        fitness_values = [
            ind.fitness.values[0] for ind in self.individuals if ind.fitness is not None
        ]
        if fitness_values:
            best = Fitness(values=np.array([max(fitness_values)]))
            return MockStatistics(
                best_fitness=best,
                mean_fitness=np.mean(fitness_values),
                std_fitness=np.std(fitness_values),
            )
        return MockStatistics()

    def best(self, n: int = 1) -> list[MockIndividual]:
        """Get best individuals by fitness."""
        sorted_inds = sorted(
            [i for i in self.individuals if i.fitness],
            key=lambda x: x.fitness.values[0],
            reverse=True,
        )
        return sorted_inds[:n]


def make_individual(fit: float, metadata: dict[str, Any] | None = None) -> MockIndividual:
    """Create individual with given fitness."""
    fitness = Fitness(values=np.array([fit]), metadata=metadata or {})
    return MockIndividual(fitness=fitness)


def make_population(fitnesses: list[float]) -> MockPopulation:
    """Create population with given fitness values."""
    return MockPopulation([make_individual(f) for f in fitnesses])


def make_context(
    population: MockPopulation,
    generation: int = 1,
    **extra: Any,
) -> CollectionContext:
    """Create collection context."""
    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore
        extra=extra,
    )


# -----------------------------------------------------------------------------
# US1: Declarative Tracking with UnifiedConfig
# -----------------------------------------------------------------------------


class TestUserStory1DeclarativeTracking:
    """US1: Enable MLflow tracking through UnifiedConfig."""

    def test_unified_config_accepts_tracking_config(self):
        """UnifiedConfig accepts TrackingConfig field."""
        tracking = TrackingConfig(
            backend="null",
            experiment_name="test_experiment",
        )

        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=tracking,
        )

        assert config.tracking is not None
        assert config.tracking.experiment_name == "test_experiment"

    def test_tracking_config_serialization(self):
        """TrackingConfig round-trips through to_dict/from_dict."""
        tracking = TrackingConfig(
            experiment_name="serialization_test",
            categories=frozenset({MetricCategory.CORE, MetricCategory.TIMING}),
            log_interval=5,
        )

        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=tracking,
        )

        config_dict = config.to_dict()
        restored = UnifiedConfig.from_dict(config_dict)

        assert restored.tracking is not None
        assert restored.tracking.experiment_name == "serialization_test"
        assert restored.tracking.log_interval == 5

    def test_no_overhead_when_tracking_none(self):
        """No tracking overhead when tracking=None (FR-004)."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=None,  # Explicitly no tracking
        )

        assert config.tracking is None


# -----------------------------------------------------------------------------
# US2: Full Population Statistics
# -----------------------------------------------------------------------------


class TestUserStory2PopulationStatistics:
    """US2: Comprehensive population health metrics."""

    def test_speciation_collector_metrics(self):
        """SpeciationMetricCollector provides species dynamics."""
        collector = SpeciationMetricCollector()

        # Create context with species info
        # species_info is a dict: species_id -> list of individual indices
        population = make_population([1.0, 2.0, 3.0, 4.0, 5.0])
        context = CollectionContext(
            generation=10,
            population=population,  # type: ignore
            species_info={
                1: [0, 1],  # Species 1 has individuals at index 0 and 1
                2: [2, 3, 4],  # Species 2 has individuals 2, 3, 4
            },
        )

        metrics = collector.collect(context)

        # Note: metrics use plain names, not spec_ prefix
        assert "species_count" in metrics
        assert metrics["species_count"] == 2


# -----------------------------------------------------------------------------
# US3: Timing Instrumentation
# -----------------------------------------------------------------------------


class TestUserStory3TimingInstrumentation:
    """US3: Per-phase timing breakdown."""

    def test_timing_config_option_exists(self):
        """TrackingConfig has timing_breakdown option."""
        tracking = TrackingConfig(
            experiment_name="timing_test",
            timing_breakdown=True,
        )

        assert tracking.timing_breakdown is True

    def test_timing_category_available(self):
        """TIMING category can be enabled."""
        tracking = TrackingConfig(
            experiment_name="timing_test",
            categories=frozenset({MetricCategory.CORE, MetricCategory.TIMING}),
        )

        assert tracking.has_category(MetricCategory.TIMING)


# -----------------------------------------------------------------------------
# US4: ERP Mating Statistics
# -----------------------------------------------------------------------------


class TestUserStory4ERPMatingStats:
    """US4: ERP mating dynamics (FR-013)."""

    def test_erp_collector_mating_success(self):
        """ERPMetricCollector computes mating success rate."""
        collector = ERPMetricCollector()

        population = make_population([1.0, 2.0, 3.0])

        # Create MatingStats dataclass
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=85,
            protocol_attempts={"default": 50, "adaptive": 50},
            protocol_successes={"default": 45, "adaptive": 40},
        )

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        # Note: metric keys don't have erp_ prefix
        assert "mating_success_rate" in metrics
        assert metrics["mating_success_rate"] == 0.85
        assert metrics["attempted_matings"] == 100
        assert metrics["successful_matings"] == 85

    def test_erp_auto_enabled_with_erp_config(self):
        """ERP category auto-enabled when using with_erp() (FR-025)."""
        base = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(enabled=True),
        )

        config = base.with_erp()

        assert config.tracking is not None
        assert config.tracking.has_category(MetricCategory.ERP)


# -----------------------------------------------------------------------------
# US5: Multi-Objective Metrics
# -----------------------------------------------------------------------------


class TestUserStory5MultiObjectiveMetrics:
    """US5: Pareto front quality metrics (FR-014)."""

    def test_multiobjective_collector_hypervolume(self):
        """MultiObjectiveMetricCollector computes hypervolume."""
        collector = MultiObjectiveMetricCollector(
            reference_point=(10.0, 10.0),
        )

        # Create Pareto front
        from evolve.multiobjective.fitness import MultiObjectiveFitness

        pareto_individuals = []
        for obj_values in [(1.0, 9.0), (5.0, 5.0), (9.0, 1.0)]:
            # MultiObjectiveFitness uses 'objectives' not 'values'
            fitness = MultiObjectiveFitness(objectives=np.array(obj_values))
            pareto_individuals.append(MockIndividual(fitness=fitness, id=uuid4()))

        population = MockPopulation(individuals=pareto_individuals)
        context = CollectionContext(
            generation=10,
            population=population,  # type: ignore
            pareto_front=pareto_individuals,
        )

        metrics = collector.collect(context)

        # Note: metric keys don't have mo_ prefix
        assert "pareto_front_size" in metrics
        assert metrics["pareto_front_size"] == 3
        assert "hypervolume" in metrics

    def test_multiobjective_auto_enabled(self):
        """MULTIOBJECTIVE category auto-enabled with with_multiobjective()."""
        base = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(enabled=True),
        )

        config = base.with_multiobjective(
            objectives=(
                ObjectiveSpec(name="obj1", direction="minimize"),
                ObjectiveSpec(name="obj2", direction="maximize"),
            ),
        )

        assert config.tracking is not None
        assert config.tracking.has_category(MetricCategory.MULTIOBJECTIVE)


# -----------------------------------------------------------------------------
# US6: Fitness Metadata Extraction
# -----------------------------------------------------------------------------


class TestUserStory6FitnessMetadata:
    """US6: Automatic metadata extraction (FR-018-020)."""

    def test_metadata_collector_extracts_numeric_fields(self):
        """FitnessMetadataCollector extracts numeric metadata."""
        collector = FitnessMetadataCollector(
            prefix="meta_",
            aggregations=("best", "mean"),
        )

        # Create population with metadata
        individuals = [
            make_individual(10.0, {"episode_reward": 100, "steps": 500}),
            make_individual(8.0, {"episode_reward": 80, "steps": 450}),
            make_individual(9.0, {"episode_reward": 90, "steps": 480}),
        ]
        population = MockPopulation(individuals)
        context = make_context(population)

        metrics = collector.collect(context)

        assert "meta_episode_reward_best" in metrics
        assert "meta_episode_reward_mean" in metrics
        assert "meta_steps_best" in metrics

    def test_metadata_threshold_respected(self):
        """Only fields present in majority are aggregated (FR-020)."""
        collector = FitnessMetadataCollector(
            threshold=0.6,  # 60% must have field
        )

        # Only 1 of 3 has "rare_field"
        individuals = [
            make_individual(10.0, {"common": 100, "rare_field": 1}),
            make_individual(8.0, {"common": 80}),
            make_individual(9.0, {"common": 90}),
        ]
        population = MockPopulation(individuals)
        context = make_context(population)

        metrics = collector.collect(context)

        # common should be present (100%), rare_field should not (33%)
        assert any("common" in key for key in metrics)
        assert not any("rare_field" in key for key in metrics)


# -----------------------------------------------------------------------------
# US7: Derived Analytics
# -----------------------------------------------------------------------------


class TestUserStory7DerivedAnalytics:
    """US7: Computed analytical metrics (FR-021-023)."""

    def test_derived_collector_selection_pressure(self):
        """DerivedAnalyticsCollector computes selection_pressure."""
        collector = DerivedAnalyticsCollector()

        # Selection pressure = best / mean
        # Best = 10, Mean = 5, so pressure = 2.0
        population = make_population([10.0, 5.0, 5.0, 3.0, 2.0])
        context = make_context(population)

        metrics = collector.collect(context)

        assert "selection_pressure" in metrics
        assert metrics["selection_pressure"] == pytest.approx(2.0, rel=0.1)

    def test_derived_collector_population_entropy(self):
        """DerivedAnalyticsCollector computes population_entropy."""
        collector = DerivedAnalyticsCollector()

        # Uniform population should have higher entropy
        population = make_population([1.0, 2.0, 3.0, 4.0, 5.0])
        context = make_context(population)

        metrics = collector.collect(context)

        assert "population_entropy" in metrics
        assert metrics["population_entropy"] > 0

    def test_derived_collector_improvement_velocity(self):
        """DerivedAnalyticsCollector tracks velocity over generations."""
        collector = DerivedAnalyticsCollector(velocity_window=3)

        # Simulate improving fitness over generations
        for gen, best_fit in enumerate([10.0, 12.0, 15.0, 20.0], start=1):
            population = make_population([best_fit, best_fit - 2, best_fit - 3])
            context = make_context(population, generation=gen)
            metrics = collector.collect(context)

        # After 4 generations, should have velocity
        assert "fitness_improvement_velocity" in metrics


# -----------------------------------------------------------------------------
# Additional Collectors (Phase 10)
# -----------------------------------------------------------------------------


class TestIslandsCollector:
    """Tests for IslandsMetricCollector (FR-016)."""

    def test_islands_inter_variance(self):
        """Computes variance between island means."""
        collector = IslandsMetricCollector()

        # Create islands with different mean fitness
        island1 = MockPopulation([make_individual(10.0), make_individual(11.0)])
        island2 = MockPopulation([make_individual(20.0), make_individual(21.0)])

        # All individuals in main population
        all_inds = island1.individuals + island2.individuals
        population = MockPopulation(all_inds)

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            island_populations=[island1, island2],  # type: ignore
        )

        metrics = collector.collect(context)

        assert "inter_island_variance" in metrics
        assert metrics["inter_island_variance"] > 0  # Islands have different means


class TestNEATCollector:
    """Tests for NEATMetricCollector (FR-017)."""

    def test_neat_topology_metrics(self):
        """Computes average node and connection counts."""
        collector = NEATMetricCollector(track_innovations=False)

        # Create mock graph genomes
        @dataclass
        class MockGraphGenome:
            nodes: list[int]
            connections: list[tuple[int, int]]

        # Individuals with different topology sizes
        ind1 = MockIndividual(
            genome=MockGraphGenome(nodes=[1, 2, 3], connections=[(1, 2), (2, 3)]),
            fitness=Fitness(values=np.array([1.0])),
        )
        ind2 = MockIndividual(
            genome=MockGraphGenome(nodes=[1, 2, 3, 4], connections=[(1, 2), (2, 3), (3, 4)]),
            fitness=Fitness(values=np.array([2.0])),
        )

        population = MockPopulation([ind1, ind2])
        context = make_context(population)

        metrics = collector.collect(context)

        assert "average_node_count" in metrics
        assert metrics["average_node_count"] == 3.5  # (3+4)/2
        assert "average_connection_count" in metrics
        assert metrics["average_connection_count"] == 2.5  # (2+3)/2


# -----------------------------------------------------------------------------
# Configuration Validation
# -----------------------------------------------------------------------------


class TestTrackingConfigValidation:
    """Tests for TrackingConfig validation (FR-026)."""

    def test_default_config_core_only(self):
        """Default TrackingConfig has only CORE category (minimal overhead)."""
        tracking = TrackingConfig()

        assert MetricCategory.CORE in tracking.categories
        assert len(tracking.categories) == 1

    def test_minimal_factory(self):
        """TrackingConfig.minimal() returns core-only config."""
        tracking = TrackingConfig.minimal()

        assert tracking.categories == frozenset({MetricCategory.CORE})

    def test_comprehensive_factory(self):
        """TrackingConfig.comprehensive() enables all categories."""
        tracking = TrackingConfig.comprehensive("full_analysis")

        assert len(tracking.categories) > 5
        assert MetricCategory.CORE in tracking.categories
        assert MetricCategory.DIVERSITY in tracking.categories
        assert MetricCategory.TIMING in tracking.categories


# -----------------------------------------------------------------------------
# Collector Reset
# -----------------------------------------------------------------------------


class TestCollectorReset:
    """Tests for collector reset between runs."""

    def test_derived_collector_reset_clears_history(self):
        """DerivedAnalyticsCollector.reset() clears velocity history."""
        collector = DerivedAnalyticsCollector()

        # Build up history
        for gen in range(5):
            population = make_population([gen + 1])
            context = make_context(population, generation=gen)
            collector.collect(context)

        assert len(collector._best_fitness_history) > 0

        collector.reset()

        assert len(collector._best_fitness_history) == 0

    def test_erp_collector_reset_clears_warning(self):
        """ERPMetricCollector.reset() clears warning flags."""
        collector = ERPMetricCollector()

        # Trigger zero-rate warning
        population = make_population([1.0])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=0,
        )
        context = CollectionContext(
            generation=1,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )
        collector.collect(context)

        assert collector._zero_success_warned is True

        collector.reset()

        assert collector._zero_success_warned is False
