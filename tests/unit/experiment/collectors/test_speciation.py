"""
Unit tests for SpeciationMetricCollector.

Tests species dynamics tracking and metrics computation.
"""

import numpy as np
import pytest
from dataclasses import dataclass, field
from unittest.mock import Mock
from typing import Any

from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.speciation import SpeciationMetricCollector


@dataclass
class MockFitness:
    """Mock Fitness class for testing."""
    _value: float
    
    def __getitem__(self, idx: int) -> float:
        return self._value


@dataclass
class MockIndividual:
    """Mock Individual class for testing."""
    id: int
    fitness: MockFitness | None = None


class MockPopulation:
    """Mock Population class for testing."""
    
    def __init__(self, individuals: list[MockIndividual]):
        self._individuals = individuals
    
    def __getitem__(self, idx: int) -> MockIndividual:
        return self._individuals[idx]
    
    def __len__(self) -> int:
        return len(self._individuals)


class TestSpeciationMetricCollectorBasic:
    """Basic tests for SpeciationMetricCollector."""
    
    def test_returns_empty_when_no_species_info(self) -> None:
        """Returns empty dict when species_info is None."""
        collector = SpeciationMetricCollector()
        
        population = MockPopulation([])
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info=None,
        )
        
        metrics = collector.collect(context)
        
        assert metrics == {}
    
    def test_species_count_correct(self) -> None:
        """Species count reflects number of species."""
        collector = SpeciationMetricCollector(track_dynamics=False)
        
        individuals = [
            MockIndividual(i, MockFitness(float(i)))
            for i in range(6)
        ]
        population = MockPopulation(individuals)
        
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1, 2], 1: [3, 4], 2: [5]},
        )
        
        metrics = collector.collect(context)
        
        assert metrics["species_count"] == 3.0
    
    def test_average_species_size_correct(self) -> None:
        """Average species size is computed correctly."""
        collector = SpeciationMetricCollector(track_dynamics=False)
        
        individuals = [
            MockIndividual(i, MockFitness(float(i)))
            for i in range(6)
        ]
        population = MockPopulation(individuals)
        
        # 3 individuals + 2 individuals + 1 individual = 6 total, 3 species
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1, 2], 1: [3, 4], 2: [5]},
        )
        
        metrics = collector.collect(context)
        
        assert metrics["average_species_size"] == 2.0  # 6/3
    
    def test_largest_species_fitness_correct(self) -> None:
        """Largest species fitness is best fitness in largest species."""
        collector = SpeciationMetricCollector(track_dynamics=False)
        
        individuals = [
            MockIndividual(0, MockFitness(1.0)),
            MockIndividual(1, MockFitness(5.0)),  # Best in largest species
            MockIndividual(2, MockFitness(3.0)),
            MockIndividual(3, MockFitness(10.0)),  # Higher but in smaller species
        ]
        population = MockPopulation(individuals)
        
        # Species 0 is largest (3 members)
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1, 2], 1: [3]},
        )
        
        metrics = collector.collect(context)
        
        assert metrics["largest_species_fitness"] == 5.0


class TestSpeciationDynamics:
    """Tests for species dynamics tracking."""
    
    def test_species_births_detected(self) -> None:
        """New species are detected as births."""
        collector = SpeciationMetricCollector(track_dynamics=True)
        
        individuals = [MockIndividual(i, MockFitness(1.0)) for i in range(5)]
        population = MockPopulation(individuals)
        
        # Generation 0: species 0, 1
        context1 = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1], 1: [2, 3, 4]},
        )
        collector.collect(context1)
        
        # Generation 1: species 0, 1, 2 (new species 2)
        context2 = CollectionContext(
            generation=1,
            population=population,  # type: ignore
            species_info={0: [0], 1: [1, 2], 2: [3, 4]},
        )
        metrics = collector.collect(context2)
        
        assert metrics["species_births"] == 1.0
    
    def test_species_extinctions_detected(self) -> None:
        """Extinct species are detected."""
        collector = SpeciationMetricCollector(track_dynamics=True)
        
        individuals = [MockIndividual(i, MockFitness(1.0)) for i in range(5)]
        population = MockPopulation(individuals)
        
        # Generation 0: species 0, 1, 2
        context1 = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0], 1: [1, 2], 2: [3, 4]},
        )
        collector.collect(context1)
        
        # Generation 1: species 0, 2 (species 1 extinct)
        context2 = CollectionContext(
            generation=1,
            population=population,  # type: ignore
            species_info={0: [0, 1], 2: [2, 3, 4]},
        )
        metrics = collector.collect(context2)
        
        assert metrics["species_extinctions"] == 1.0
    
    def test_stagnation_count_increments(self) -> None:
        """Stagnation count increases when fitness doesn't improve."""
        collector = SpeciationMetricCollector(track_dynamics=True)
        
        # Generation 0
        individuals_gen0 = [
            MockIndividual(0, MockFitness(5.0)),
            MockIndividual(1, MockFitness(5.0)),
        ]
        population_gen0 = MockPopulation(individuals_gen0)
        
        context1 = CollectionContext(
            generation=0,
            population=population_gen0,  # type: ignore
            species_info={0: [0, 1]},
        )
        collector.collect(context1)
        
        # Generation 1: same or lower fitness
        individuals_gen1 = [
            MockIndividual(0, MockFitness(4.0)),  # Lower
            MockIndividual(1, MockFitness(5.0)),  # Same
        ]
        population_gen1 = MockPopulation(individuals_gen1)
        
        context2 = CollectionContext(
            generation=1,
            population=population_gen1,  # type: ignore
            species_info={0: [0, 1]},
        )
        metrics = collector.collect(context2)
        
        assert metrics["stagnation_count"] == 1.0
    
    def test_stagnation_resets_on_improvement(self) -> None:
        """Stagnation resets when fitness improves."""
        collector = SpeciationMetricCollector(track_dynamics=True)
        
        # Generation 0
        individuals_gen0 = [MockIndividual(0, MockFitness(5.0))]
        population_gen0 = MockPopulation(individuals_gen0)
        
        context1 = CollectionContext(
            generation=0,
            population=population_gen0,  # type: ignore
            species_info={0: [0]},
        )
        collector.collect(context1)
        
        # Generation 1: no improvement
        individuals_gen1 = [MockIndividual(0, MockFitness(4.0))]
        population_gen1 = MockPopulation(individuals_gen1)
        
        context2 = CollectionContext(
            generation=1,
            population=population_gen1,  # type: ignore
            species_info={0: [0]},
        )
        collector.collect(context2)
        
        # Generation 2: improvement
        individuals_gen2 = [MockIndividual(0, MockFitness(6.0))]
        population_gen2 = MockPopulation(individuals_gen2)
        
        context3 = CollectionContext(
            generation=2,
            population=population_gen2,  # type: ignore
            species_info={0: [0]},
        )
        metrics = collector.collect(context3)
        
        assert metrics["stagnation_count"] == 0.0
    
    def test_reset_clears_state(self) -> None:
        """Reset clears all internal state."""
        collector = SpeciationMetricCollector(track_dynamics=True)
        
        individuals = [MockIndividual(0, MockFitness(5.0))]
        population = MockPopulation(individuals)
        
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0]},
        )
        collector.collect(context)
        
        # Verify state is populated
        assert len(collector._previous_species_ids) > 0
        
        # Reset
        collector.reset()
        
        # Verify state is cleared
        assert len(collector._previous_species_ids) == 0
        assert len(collector._previous_best_fitness) == 0
        assert len(collector._stagnation_counters) == 0


class TestSpeciationEdgeCases:
    """Edge case tests for SpeciationMetricCollector."""
    
    def test_empty_species(self) -> None:
        """Handles empty species_info dict."""
        collector = SpeciationMetricCollector()
        
        population = MockPopulation([])
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={},
        )
        
        metrics = collector.collect(context)
        
        assert metrics["species_count"] == 0.0
    
    def test_single_species(self) -> None:
        """Works with single species."""
        collector = SpeciationMetricCollector(track_dynamics=False)
        
        individuals = [MockIndividual(i, MockFitness(float(i))) for i in range(5)]
        population = MockPopulation(individuals)
        
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1, 2, 3, 4]},
        )
        
        metrics = collector.collect(context)
        
        assert metrics["species_count"] == 1.0
        assert metrics["average_species_size"] == 5.0
        assert metrics["largest_species_fitness"] == 4.0
    
    def test_individuals_without_fitness(self) -> None:
        """Handles individuals with None fitness."""
        collector = SpeciationMetricCollector(track_dynamics=False)
        
        individuals = [
            MockIndividual(0, MockFitness(5.0)),
            MockIndividual(1, None),  # No fitness
            MockIndividual(2, MockFitness(3.0)),
        ]
        population = MockPopulation(individuals)
        
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            species_info={0: [0, 1, 2]},
        )
        
        metrics = collector.collect(context)
        
        # Should use fitness from individuals 0 and 2 only
        assert metrics["largest_species_fitness"] == 5.0
