"""
Integration test: Simple GA optimizing sphere function.

Verifies:
1. GA converges on sphere function
2. Best fitness decreases over generations
3. Final result is reasonably close to optimum
"""

import numpy as np
import pytest

from evolve.core.engine import EvolutionEngine, EvolutionConfig, create_initial_population
from evolve.core.operators.selection import TournamentSelection
from evolve.core.operators.crossover import BlendCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.stopping import FitnessThresholdStopping, CompositeStoppingCriterion, GenerationLimitStopping
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.evaluation.reference.functions import sphere, rastrigin
from evolve.representation.vector import VectorGenome
from evolve.utils.random import create_rng


@pytest.mark.integration
class TestSimpleGA:
    """Integration tests for simple genetic algorithm."""

    def test_sphere_optimization_converges(self):
        """GA should converge on sphere function."""
        # Setup
        n_dims = 10
        bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))
        
        config = EvolutionConfig(
            population_size=50,
            max_generations=100,
            elitism=2,
            crossover_rate=0.9,
            mutation_rate=1.0,
        )
        
        evaluator = FunctionEvaluator(sphere)
        
        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.1, sigma=0.1, adaptive=True),
            seed=42,
        )
        
        # Create initial population
        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )
        
        # Run
        result = engine.run(initial_pop)
        
        # Verify convergence
        assert result.best.fitness is not None
        best_fitness = float(result.best.fitness.values[0])
        
        # Should achieve reasonable fitness on sphere
        assert best_fitness < 1.0, f"Expected fitness < 1.0, got {best_fitness}"
        
        # Fitness should improve over time
        assert len(result.history) > 0
        first_best = result.history[0]["best_fitness"]
        last_best = result.history[-1]["best_fitness"]
        assert last_best < first_best, "Fitness should improve over generations"

    def test_sphere_with_threshold_stopping(self):
        """GA should stop early when threshold reached."""
        n_dims = 5
        bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))
        threshold = 0.1
        
        config = EvolutionConfig(
            population_size=50,
            max_generations=500,  # High limit to test threshold stopping
            elitism=2,
        )
        
        stopping = CompositeStoppingCriterion()
        stopping.add(FitnessThresholdStopping(threshold=threshold))
        stopping.add(GenerationLimitStopping(max_generations=500))
        
        engine = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1, adaptive=True),
            seed=42,
            stopping=stopping,
        )
        
        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )
        
        result = engine.run(initial_pop)
        
        # Should stop before max generations
        assert result.generations < 500, "Should stop early due to threshold"
        
        # Final fitness should be at or below threshold
        best_fitness = float(result.best.fitness.values[0])
        assert best_fitness <= threshold * 1.1, f"Expected ~{threshold}, got {best_fitness}"

    def test_rastrigin_multimodal(self):
        """GA should make progress on multimodal Rastrigin function."""
        n_dims = 5
        bounds = (np.full(n_dims, -5.12), np.full(n_dims, 5.12))
        
        config = EvolutionConfig(
            population_size=100,  # Larger pop for multimodal
            max_generations=100,
            elitism=3,
        )
        
        engine = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(rastrigin),
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.3),  # Less exploration
            mutation=GaussianMutation(mutation_rate=0.15, sigma=0.2, adaptive=True),
            seed=42,
        )
        
        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )
        
        result = engine.run(initial_pop)
        
        # Should make significant progress
        initial_best = result.history[0]["best_fitness"]
        final_best = result.history[-1]["best_fitness"]
        
        # Fitness should improve significantly
        improvement = (initial_best - final_best) / initial_best
        assert improvement > 0.5, f"Expected >50% improvement, got {improvement*100:.1f}%"

    def test_elitism_preserves_best(self):
        """Elitism should preserve best individuals across generations."""
        n_dims = 5
        bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))
        
        config = EvolutionConfig(
            population_size=20,
            max_generations=20,
            elitism=2,
        )
        
        engine = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(tournament_size=2),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(mutation_rate=0.3, sigma=0.5),  # High mutation
            seed=42,
        )
        
        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )
        
        result = engine.run(initial_pop)
        
        # Best fitness should never get worse (monotonically improving)
        best_values = [h["best_fitness"] for h in result.history]
        for i in range(1, len(best_values)):
            assert best_values[i] <= best_values[i-1] + 1e-10, \
                f"Best fitness degraded from {best_values[i-1]} to {best_values[i]} at gen {i}"


@pytest.mark.integration  
class TestGAComponents:
    """Test individual GA components work correctly."""

    def test_population_statistics(self):
        """Population should compute correct statistics."""
        from evolve.core.types import Fitness, Individual
        from evolve.core.population import Population
        from evolve.representation.vector import VectorGenome
        
        # Create population with known fitness values
        genomes = [VectorGenome(genes=np.array([float(i)])) for i in range(5)]
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        individuals = [
            Individual(genome=g).with_fitness(Fitness.scalar(f))
            for g, f in zip(genomes, fitness_values)
        ]
        
        pop = Population(individuals)
        stats = pop.statistics
        
        assert stats.size == 5
        assert stats.best_fitness is not None
        assert float(stats.best_fitness.values[0]) == 1.0  # Min fitness
        assert stats.mean_fitness is not None
        assert abs(float(stats.mean_fitness.values[0]) - 3.0) < 0.01

    def test_tournament_selection_pressure(self):
        """Larger tournaments should select better individuals more often."""
        from evolve.core.types import Fitness, Individual
        from evolve.core.population import Population
        from evolve.core.operators.selection import TournamentSelection
        from evolve.representation.vector import VectorGenome
        
        # Create population with varying fitness
        n = 100
        individuals = [
            Individual(genome=VectorGenome(genes=np.array([float(i)])))
            .with_fitness(Fitness.scalar(float(i)))  # fitness = index
            for i in range(n)
        ]
        pop = Population(individuals)
        rng = create_rng(42)
        
        # Small tournament
        small = TournamentSelection(tournament_size=2)
        small_selected = small.select(pop, 100, rng)
        small_mean = np.mean([float(s.fitness.values[0]) for s in small_selected])
        
        # Large tournament
        rng = create_rng(42)
        large = TournamentSelection(tournament_size=10)
        large_selected = large.select(pop, 100, rng)
        large_mean = np.mean([float(s.fitness.values[0]) for s in large_selected])
        
        # Larger tournament should select lower (better) fitness more often
        assert large_mean < small_mean, \
            f"Large tournament mean {large_mean} should be < small {small_mean}"
