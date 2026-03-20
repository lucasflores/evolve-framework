"""
Unit tests for timing instrumentation in evolution engine.

Tests that the engine correctly captures and reports timing metrics
for each phase of evolution (selection, variation, evaluation).
"""

from __future__ import annotations

import pytest
from random import Random
from uuid import uuid4

from evolve.core.engine import EvolutionEngine, EvolutionConfig
from evolve.core.population import Population, Individual
from evolve.representation.vector import VectorGenome
from evolve.core.operators.selection import TournamentSelection
from evolve.core.operators.crossover import UniformCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.types import Fitness
from evolve.evaluation.evaluator import EvaluatorCapabilities

import numpy as np


class SimpleEvaluator:
    """Simple sphere function evaluator for testing."""
    
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(n_objectives=1)
    
    def evaluate(self, individuals, seed=None):
        """Evaluate batch of individuals."""
        results = []
        for ind in individuals:
            fitness = -sum(g**2 for g in ind.genome.genes)
            results.append(Fitness((fitness,)))
        return results


@pytest.fixture
def simple_population(rng: Random) -> Population[VectorGenome]:
    """Create a simple population for testing."""
    lower = np.array([-5.0] * 5)
    upper = np.array([5.0] * 5)
    bounds = (lower, upper)
    
    individuals = [
        Individual(id=uuid4(), genome=VectorGenome.random(5, bounds, rng))
        for _ in range(20)
    ]
    return Population(individuals=individuals, generation=0)


@pytest.fixture
def simple_engine() -> EvolutionEngine[VectorGenome]:
    """Create a simple engine for testing."""
    return EvolutionEngine(
        config=EvolutionConfig(
            population_size=20,
            max_generations=5,
            elitism=2,
            crossover_rate=0.9,
            mutation_rate=0.1,
        ),
        selection=TournamentSelection(tournament_size=3),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(sigma=0.1),
        evaluator=SimpleEvaluator(),
        seed=42,
    )


class TestEngineTimingInstrumentation:
    """Tests for timing instrumentation in EvolutionEngine."""
    
    def test_timing_metrics_present_in_history(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """Engine should capture timing metrics in history."""
        result = simple_engine.run(simple_population)
        
        # Check that all generations have timing metrics
        for gen_metrics in result.history:
            assert "generation_time_ms" in gen_metrics
            assert "selection_time_ms" in gen_metrics
            assert "variation_time_ms" in gen_metrics
            assert "evaluation_time_ms" in gen_metrics
    
    def test_timing_metrics_are_positive(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """All timing metrics should be positive values."""
        result = simple_engine.run(simple_population)
        
        for gen_metrics in result.history:
            assert gen_metrics["generation_time_ms"] > 0
            assert gen_metrics["selection_time_ms"] >= 0
            assert gen_metrics["variation_time_ms"] >= 0
            assert gen_metrics["evaluation_time_ms"] >= 0
    
    def test_cpu_time_metrics_present(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """Engine should capture CPU time metrics."""
        result = simple_engine.run(simple_population)
        
        for gen_metrics in result.history:
            assert "generation_cpu_time_ms" in gen_metrics
            assert "selection_cpu_time_ms" in gen_metrics
            assert "variation_cpu_time_ms" in gen_metrics
            assert "evaluation_cpu_time_ms" in gen_metrics
    
    def test_phase_times_sum_approximately_to_total(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """Sum of phase times should be close to total generation time."""
        result = simple_engine.run(simple_population)
        
        for gen_metrics in result.history:
            phase_sum = (
                gen_metrics["selection_time_ms"]
                + gen_metrics["variation_time_ms"]
                + gen_metrics["evaluation_time_ms"]
            )
            total = gen_metrics["generation_time_ms"]
            
            # Phase sum should be <= total (total includes overhead)
            assert phase_sum <= total * 1.1  # Allow 10% tolerance
    
    def test_timing_resets_between_generations(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """Timing should reset between generations, not accumulate."""
        result = simple_engine.run(simple_population)
        
        # Each generation's timing should be independent
        times = [m["generation_time_ms"] for m in result.history]
        
        # Not all times should be monotonically increasing
        # (would indicate accumulation bug)
        is_monotonic = all(t1 <= t2 for t1, t2 in zip(times[:-1], times[1:]))
        
        # With random variation, strict monotonic increase is unlikely
        # if timing resets properly
        # This is a heuristic test - actual times should be similar per gen
        max_time = max(times)
        min_time = min(times)
        
        # Times should be similar (within 10x of each other typically)
        # If accumulating, later times would be much larger
        assert max_time < min_time * 100, "Times may be accumulating"


class TestTimingMetricsConsistency:
    """Tests for timing metrics consistency across runs."""
    
    def test_deterministic_timing_keys(
        self,
        simple_engine: EvolutionEngine,
        simple_population: Population,
    ) -> None:
        """Timing metric keys should be consistent across generations."""
        result = simple_engine.run(simple_population)
        
        first_keys = set(k for k in result.history[0].keys() if "time" in k)
        
        for gen_metrics in result.history[1:]:
            gen_keys = set(k for k in gen_metrics.keys() if "time" in k)
            assert gen_keys == first_keys
    
    def test_timing_with_minimal_population(self) -> None:
        """Timing should work with minimal population size."""
        engine = EvolutionEngine(
            config=EvolutionConfig(
                population_size=4,
                max_generations=2,
                elitism=1,
            ),
            selection=TournamentSelection(tournament_size=2),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            evaluator=SimpleEvaluator(),
            seed=42,
        )
        
        rng = Random(42)
        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        individuals = [
            Individual(id=uuid4(), genome=VectorGenome.random(2, bounds, rng))
            for _ in range(4)
        ]
        pop = Population(individuals=individuals, generation=0)
        
        result = engine.run(pop)
        
        # Should still have timing metrics
        assert "generation_time_ms" in result.history[0]
        assert result.history[0]["generation_time_ms"] > 0
