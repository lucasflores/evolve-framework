"""
Unit tests for MultiObjectiveMetricCollector.

Tests:
- Pareto front size computation
- Hypervolume calculation (2D and 3D)
- Spread metric for 2D fronts
- Crowding diversity
- Empty front handling
- Reference point configuration
- High-dimensional fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.multiobjective import MultiObjectiveMetricCollector
from evolve.multiobjective.fitness import MultiObjectiveFitness


@dataclass
class MockIndividual:
    """Mock individual for testing."""
    id: Any = None
    fitness: Any = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()


@dataclass
class MockPopulation:
    """Mock population for testing."""
    individuals: list[MockIndividual]
    
    def __len__(self) -> int:
        return len(self.individuals)


def make_mo_individual(objectives: list[float]) -> MockIndividual:
    """Create individual with multi-objective fitness."""
    fitness = MultiObjectiveFitness(np.array(objectives))
    return MockIndividual(fitness=fitness)


def make_context(
    individuals: list[MockIndividual] | None = None,
    pareto_front: list[MockIndividual] | None = None,
    generation: int = 1,
    extra: dict[str, Any] | None = None,
) -> CollectionContext:
    """Create test collection context."""
    if individuals is None:
        individuals = []
    
    population = MockPopulation(individuals=individuals)
    
    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore
        pareto_front=pareto_front,  # type: ignore
        extra=extra or {},
    )


class TestParetoFrontSize:
    """Tests for Pareto front size metric."""
    
    def test_front_size_from_context(self):
        """Test front size when context provides pareto_front."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert metrics["pareto_front_size"] == 3
    
    def test_front_size_computed_from_population(self):
        """Test front size computed from population when no pareto_front in context."""
        # Create population where all are non-dominated (all on front)
        individuals = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(individuals=individuals, pareto_front=None)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert metrics["pareto_front_size"] == 3
    
    def test_front_size_with_dominated_individuals(self):
        """Test front size correctly excludes dominated individuals."""
        # Only first 2 are on front, third is dominated
        individuals = [
            make_mo_individual([3.0, 3.0]),  # Dominates all
            make_mo_individual([2.0, 2.0]),  # Dominated by first
            make_mo_individual([1.0, 1.0]),  # Dominated by both
        ]
        context = make_context(individuals=individuals, pareto_front=None)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        # Only the first is truly non-dominated
        assert metrics["pareto_front_size"] == 1
    
    def test_empty_front_returns_zero(self):
        """Test empty front returns size 0."""
        context = make_context(pareto_front=[])
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert metrics["pareto_front_size"] == 0


class TestHypervolume:
    """Tests for hypervolume computation."""
    
    def test_hypervolume_2d_simple(self):
        """Test 2D hypervolume with simple front."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(
            reference_point=[0.0, 0.0]
        )
        metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert metrics["hypervolume"] > 0
        # Expected HV = 6.0 (3*1 + 2*1 + 1*1 via sweepline algorithm)
        assert np.isclose(metrics["hypervolume"], 6.0, atol=0.1)
    
    def test_hypervolume_with_configured_reference(self):
        """Test hypervolume uses configured reference point."""
        front = [
            make_mo_individual([4.0, 2.0]),
            make_mo_individual([3.0, 3.0]),
            make_mo_individual([2.0, 4.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(
            reference_point=[1.0, 1.0]
        )
        metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert metrics["hypervolume"] > 0
    
    def test_hypervolume_reference_from_context_extra(self):
        """Test hypervolume uses reference from context.extra."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
        ]
        context = make_context(
            pareto_front=front,
            extra={"hypervolume_reference": [0.0, 0.0]}
        )
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert metrics["hypervolume"] > 0
    
    def test_hypervolume_3d_approximate(self):
        """Test 3D hypervolume uses approximation."""
        front = [
            make_mo_individual([3.0, 1.0, 2.0]),
            make_mo_individual([2.0, 2.0, 2.0]),
            make_mo_individual([1.0, 3.0, 2.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(
            reference_point=[0.0, 0.0, 0.0]
        )
        metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert metrics["hypervolume"] > 0
    
    def test_hypervolume_auto_reference_estimation(self):
        """Test hypervolume estimates reference when not provided."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()  # No reference set
        metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert metrics["hypervolume"] > 0


class TestSpread:
    """Tests for spread metric."""
    
    def test_spread_2d_uniform(self):
        """Test spread with uniformly distributed front."""
        # Equally spaced points
        front = [
            make_mo_individual([4.0, 1.0]),
            make_mo_individual([3.0, 2.0]),
            make_mo_individual([2.0, 3.0]),
            make_mo_individual([1.0, 4.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert "spread" in metrics
        # Uniform distribution should have low spread (close to 0)
        assert metrics["spread"] < 0.5
    
    def test_spread_2d_clustered(self):
        """Test spread with clustered front."""
        # Clustered points
        front = [
            make_mo_individual([4.0, 1.0]),
            make_mo_individual([4.01, 1.01]),  # Clustered with first
            make_mo_individual([1.0, 4.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert "spread" in metrics
        # Clustered should have higher spread
    
    def test_spread_disabled(self):
        """Test spread not computed when disabled."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(enable_spread=False)
        metrics = collector.collect(context)
        
        assert "spread" not in metrics
    
    def test_spread_not_computed_for_3d(self):
        """Test spread not computed for >2 objectives."""
        front = [
            make_mo_individual([3.0, 1.0, 2.0]),
            make_mo_individual([2.0, 2.0, 2.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(enable_spread=True)
        metrics = collector.collect(context)
        
        # Spread only for 2D
        assert "spread" not in metrics


class TestCrowdingDiversity:
    """Tests for crowding diversity metric."""
    
    def test_crowding_diversity_computed(self):
        """Test crowding diversity is computed for front."""
        # Need at least 3 points for meaningful crowding
        front = [
            make_mo_individual([4.0, 1.0]),
            make_mo_individual([3.0, 2.0]),
            make_mo_individual([2.0, 3.0]),
            make_mo_individual([1.0, 4.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        assert "crowding_diversity" in metrics
        assert metrics["crowding_diversity"] > 0
    
    def test_crowding_diversity_disabled(self):
        """Test crowding not computed when disabled."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([2.0, 2.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(enable_crowding=False)
        metrics = collector.collect(context)
        
        assert "crowding_diversity" not in metrics
    
    def test_crowding_requires_minimum_points(self):
        """Test crowding not computed with <3 points."""
        front = [
            make_mo_individual([3.0, 1.0]),
            make_mo_individual([1.0, 3.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        # Not enough points
        assert "crowding_diversity" not in metrics


class TestEmptyFrontWarning:
    """Tests for empty front warning behavior."""
    
    def test_warning_on_empty_front(self, caplog):
        """Test warning logged when front is empty."""
        context = make_context(pareto_front=[])
        
        collector = MultiObjectiveMetricCollector(warn_on_empty_front=True)
        
        with caplog.at_level(logging.WARNING):
            collector.collect(context)
        
        assert "Pareto front is empty" in caplog.text
    
    def test_warning_only_once(self, caplog):
        """Test warning only logged once per run."""
        context = make_context(pareto_front=[])
        
        collector = MultiObjectiveMetricCollector(warn_on_empty_front=True)
        
        with caplog.at_level(logging.WARNING):
            collector.collect(context)
            collector.collect(context)
            collector.collect(context)
        
        # Should only appear once
        assert caplog.text.count("Pareto front is empty") == 1
    
    def test_warning_disabled(self, caplog):
        """Test no warning when disabled."""
        context = make_context(pareto_front=[])
        
        collector = MultiObjectiveMetricCollector(warn_on_empty_front=False)
        
        with caplog.at_level(logging.WARNING):
            collector.collect(context)
        
        assert "Pareto front is empty" not in caplog.text
    
    def test_reset_clears_warning_flag(self, caplog):
        """Test reset allows warning again."""
        context = make_context(pareto_front=[])
        
        collector = MultiObjectiveMetricCollector(warn_on_empty_front=True)
        
        with caplog.at_level(logging.WARNING):
            collector.collect(context)
            caplog.clear()
            
            collector.reset()
            collector.collect(context)
        
        # Should warn again after reset
        assert "Pareto front is empty" in caplog.text


class TestHighDimensionalFallback:
    """Tests for high-dimensional objective handling."""
    
    def test_4d_uses_approximate(self, caplog):
        """Test 4+ objectives uses approximate hypervolume."""
        front = [
            make_mo_individual([3.0, 1.0, 2.0, 1.5]),
            make_mo_individual([2.0, 2.0, 2.0, 2.0]),
            make_mo_individual([1.0, 3.0, 2.0, 2.5]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(
            reference_point=[0.0, 0.0, 0.0, 0.0]
        )
        
        with caplog.at_level(logging.INFO):
            metrics = collector.collect(context)
        
        assert "hypervolume" in metrics
        assert "approximate hypervolume for 4 objectives" in caplog.text
    
    def test_high_dim_warning_only_once(self, caplog):
        """Test high-dimension info logged only once."""
        front = [
            make_mo_individual([3.0, 1.0, 2.0, 1.5]),
            make_mo_individual([2.0, 2.0, 2.0, 2.0]),
        ]
        context = make_context(pareto_front=front)
        
        collector = MultiObjectiveMetricCollector(
            reference_point=[0.0, 0.0, 0.0, 0.0]
        )
        
        with caplog.at_level(logging.INFO):
            collector.collect(context)
            collector.collect(context)
        
        assert caplog.text.count("approximate hypervolume") == 1


class TestReset:
    """Tests for collector reset functionality."""
    
    def test_reset_clears_state(self):
        """Test reset clears all internal state."""
        collector = MultiObjectiveMetricCollector()
        
        # Trigger internal state changes
        context = make_context(pareto_front=[])
        collector.collect(context)
        
        assert collector._warned_empty_front is True
        
        collector.reset()
        
        assert collector._warned_empty_front is False
        assert collector._warned_high_dim is False


class TestMixedFitness:
    """Tests for handling mixed or invalid fitness types."""
    
    def test_non_mo_fitness_returns_empty(self):
        """Test non-MO fitness population returns no metrics."""
        # Create individuals with scalar fitness
        ind1 = MockIndividual(fitness=MagicMock(value=1.0))
        ind2 = MockIndividual(fitness=MagicMock(value=2.0))
        
        context = make_context(individuals=[ind1, ind2], pareto_front=None)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        # Should just return front_size=0 for empty front
        assert metrics.get("pareto_front_size", 0) == 0
    
    def test_none_fitness_handled(self):
        """Test individuals with None fitness are handled."""
        ind1 = MockIndividual(fitness=None)
        ind2 = make_mo_individual([2.0, 2.0])
        
        context = make_context(individuals=[ind1, ind2], pareto_front=None)
        
        collector = MultiObjectiveMetricCollector()
        metrics = collector.collect(context)
        
        # Should handle gracefully
        assert "pareto_front_size" in metrics
