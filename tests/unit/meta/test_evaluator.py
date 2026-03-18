"""Unit tests for evolve/meta/evaluator.py (T079)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evolve.config.unified import UnifiedConfig
from evolve.config.meta import MetaEvolutionConfig, ParameterSpec
from evolve.meta.evaluator import MetaEvaluator


@pytest.fixture
def base_config() -> UnifiedConfig:
    """Create base configuration for evaluator tests."""
    return UnifiedConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection="tournament",
        crossover="sbx",
        mutation="gaussian",
        genome_type="vector",
        genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        minimize=True,
    )


@pytest.fixture
def meta_config() -> MetaEvolutionConfig:
    """Create meta-evolution configuration."""
    return MetaEvolutionConfig(
        evolvable_params=(
            ParameterSpec(
                path="mutation_rate",
                param_type="continuous",
                bounds=(0.01, 0.5),
            ),
        ),
        outer_population_size=10,
        outer_generations=5,
        trials_per_config=1,
        aggregation="mean",
    )


@pytest.fixture
def fitness_fn():
    """Simple fitness function for testing."""
    return lambda genome: sum(genome.genes) if hasattr(genome, 'genes') else 0.0


@dataclass
class MockResult:
    """Mock evolution result for testing."""
    best_fitness: float
    best_individual: Any = None


class TestMetaEvaluatorBasic:
    """Basic tests for MetaEvaluator."""
    
    def test_evaluator_creation(
        self,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test MetaEvaluator can be created."""
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
            seed=42,
        )
        
        assert evaluator.base_config is base_config
        assert evaluator.meta_config is meta_config
        assert evaluator.seed == 42
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_evaluate_runs_inner_loop(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test that evaluate runs the inner evolution loop."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine.run.return_value = MockResult(best_fitness=0.5)
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        fitness = evaluator.evaluate(base_config)
        
        # Should have called engine.run
        mock_engine.run.assert_called_once()
        assert fitness == 0.5


class TestMetaEvaluatorCaching:
    """Tests for MetaEvaluator caching behavior."""
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_results_are_cached(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test that results are cached for same config."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = MockResult(best_fitness=0.5)
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        # Evaluate twice
        fitness1 = evaluator.evaluate(base_config)
        fitness2 = evaluator.evaluate(base_config)
        
        # Engine should only be run once
        assert mock_engine.run.call_count == 1
        assert fitness1 == fitness2
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_get_cached_solution(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test retrieving cached best solution."""
        best_ind = MagicMock()
        mock_engine = MagicMock()
        mock_engine.run.return_value = MockResult(
            best_fitness=0.5,
            best_individual=best_ind,
        )
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        # Before evaluation
        assert evaluator.get_cached_solution(base_config) is None
        
        # After evaluation
        evaluator.evaluate(base_config)
        cached = evaluator.get_cached_solution(base_config)
        assert cached is best_ind


class TestMetaEvaluatorTrials:
    """Tests for multi-trial evaluation."""
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_multiple_trials(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        fitness_fn,
    ) -> None:
        """Test that multiple trials are run when configured."""
        meta_config_multi = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=3,  # 3 trials
            aggregation="mean",
        )
        
        # Return different fitness each trial
        fitnesses = [0.3, 0.5, 0.4]
        call_count = [0]
        
        def mock_run(*args):
            idx = call_count[0] % len(fitnesses)
            call_count[0] += 1
            return MockResult(best_fitness=fitnesses[idx])
        
        mock_engine = MagicMock()
        mock_engine.run.side_effect = mock_run
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config_multi,
            fitness_fn=fitness_fn,
        )
        
        fitness = evaluator.evaluate(base_config)
        
        # Should have run 3 trials
        assert mock_engine.run.call_count == 3
        
        # Mean of 0.3, 0.5, 0.4 = 0.4
        assert abs(fitness - 0.4) < 0.001


class TestAggregation:
    """Tests for fitness aggregation methods."""
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_mean_aggregation(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        fitness_fn,
    ) -> None:
        """Test mean aggregation."""
        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.5)),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=3,
            aggregation="mean",
        )
        
        fitnesses = [1.0, 2.0, 3.0]
        call_count = [0]
        
        def mock_run(*args):
            idx = call_count[0]
            call_count[0] += 1
            return MockResult(best_fitness=fitnesses[idx])
        
        mock_engine = MagicMock()
        mock_engine.run.side_effect = mock_run
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        fitness = evaluator.evaluate(base_config)
        
        # Mean of 1, 2, 3 = 2
        assert abs(fitness - 2.0) < 0.001
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_median_aggregation(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        fitness_fn,
    ) -> None:
        """Test median aggregation."""
        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.5)),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=3,
            aggregation="median",
        )
        
        fitnesses = [1.0, 3.0, 2.0]
        call_count = [0]
        
        def mock_run(*args):
            idx = call_count[0]
            call_count[0] += 1
            return MockResult(best_fitness=fitnesses[idx])
        
        mock_engine = MagicMock()
        mock_engine.run.side_effect = mock_run
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        fitness = evaluator.evaluate(base_config)
        
        # Median of 1, 3, 2 = 2
        assert abs(fitness - 2.0) < 0.001
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_best_aggregation_minimize(
        self,
        mock_create_pop,
        mock_create_engine,
        fitness_fn,
    ) -> None:
        """Test best aggregation with minimization."""
        base_config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            minimize=True,
        )
        
        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.5)),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=3,
            aggregation="best",
        )
        
        fitnesses = [1.0, 0.5, 2.0]  # Best for minimize = 0.5
        call_count = [0]
        
        def mock_run(*args):
            idx = call_count[0]
            call_count[0] += 1
            return MockResult(best_fitness=fitnesses[idx])
        
        mock_engine = MagicMock()
        mock_engine.run.side_effect = mock_run
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        fitness = evaluator.evaluate(base_config)
        
        # Best (min) = 0.5
        assert abs(fitness - 0.5) < 0.001


class TestDeterministicSeeding:
    """Tests for deterministic inner seed computation."""
    
    def test_compute_inner_seed_deterministic(
        self,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test that inner seed is deterministic."""
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
            seed=42,
        )
        
        # Compute seed multiple times
        seed1 = evaluator._compute_inner_seed(base_config, trial=0)
        seed2 = evaluator._compute_inner_seed(base_config, trial=0)
        
        assert seed1 == seed2
    
    def test_compute_inner_seed_varies_with_trial(
        self,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test that different trials get different seeds."""
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
            seed=42,
        )
        
        seed0 = evaluator._compute_inner_seed(base_config, trial=0)
        seed1 = evaluator._compute_inner_seed(base_config, trial=1)
        seed2 = evaluator._compute_inner_seed(base_config, trial=2)
        
        # All seeds should be different
        assert seed0 != seed1
        assert seed1 != seed2
        assert seed0 != seed2
    
    def test_compute_inner_seed_varies_with_config(
        self,
        base_config: UnifiedConfig,
        meta_config: MetaEvolutionConfig,
        fitness_fn,
    ) -> None:
        """Test that different configs get different seeds."""
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
            seed=42,
        )
        
        config1 = base_config
        config2 = base_config.with_params(mutation_rate=0.2)
        
        seed1 = evaluator._compute_inner_seed(config1, trial=0)
        seed2 = evaluator._compute_inner_seed(config2, trial=0)
        
        assert seed1 != seed2


class TestTrialsCounter:
    """Tests for trials counter."""
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_trials_run_counter(
        self,
        mock_create_pop,
        mock_create_engine,
        base_config: UnifiedConfig,
        fitness_fn,
    ) -> None:
        """Test that trials_run counter works."""
        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.5)),
            ),
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=3,
        )
        
        mock_engine = MagicMock()
        mock_engine.run.return_value = MockResult(best_fitness=0.5)
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=fitness_fn,
        )
        
        assert evaluator.trials_run == 0
        
        # Evaluate one config with 3 trials
        evaluator.evaluate(base_config)
        assert evaluator.trials_run == 3
        
        # Evaluate different config
        other_config = base_config.with_params(mutation_rate=0.2)
        evaluator.evaluate(other_config)
        assert evaluator.trials_run == 6
