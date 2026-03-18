"""Integration tests for JSON config → engine workflow (T080)."""

from __future__ import annotations

import json
import tempfile

import pytest

from evolve.config.unified import UnifiedConfig
from evolve.config.stopping import StoppingConfig
from evolve.config.callbacks import CallbackConfig
from evolve.factory.engine import create_engine, create_initial_population
from evolve.registry.operators import reset_operator_registry
from evolve.registry.genomes import reset_genome_registry


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before and after each test."""
    reset_operator_registry()
    reset_genome_registry()
    yield
    reset_operator_registry()
    reset_genome_registry()


def sphere_fitness(phenotype) -> float:
    """Sphere benchmark function (minimization). Expects numpy array or iterable."""
    return sum(g**2 for g in phenotype)


class TestConfigToEngineWorkflow:
    """End-to-end tests for JSON config to engine workflow."""
    
    def test_minimal_config_creates_engine(self) -> None:
        """Test minimal JSON config creates working engine."""
        config_dict = {
            "population_size": 20,
            "max_generations": 5,
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 5, "bounds": [-5.0, 5.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        engine = create_engine(config, sphere_fitness)
        population = create_initial_population(config, seed=42)
        
        # Run 1 generation - should work
        result = engine.run(population)
        
        assert result is not None
        assert result.best is not None
        assert result.generations > 0
    
    def test_json_file_to_engine(self) -> None:
        """Test loading config from JSON file."""
        config_dict = {
            "population_size": 30,
            "max_generations": 3,
            "seed": 123,
            "selection": "tournament",
            "selection_params": {"tournament_size": 3},
            "crossover": "sbx",
            "crossover_params": {"eta": 20.0},
            "mutation": "gaussian",
            "mutation_params": {"sigma": 0.1},
            "genome_type": "vector",
            "genome_params": {"dimensions": 10, "bounds": [-1.0, 1.0]},
            "minimize": True,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            f.flush()
            
            # Load from file
            with open(f.name) as rf:
                loaded_dict = json.load(rf)
            
            config = UnifiedConfig.from_dict(loaded_dict)
        
        engine = create_engine(config, sphere_fitness)
        population = create_initial_population(config, seed=123)
        
        result = engine.run(population)
        
        assert result is not None
        assert result.best.fitness is not None
    
    def test_config_with_stopping_criteria(self) -> None:
        """Test config with custom stopping criteria."""
        config_dict = {
            "population_size": 20,
            "max_generations": 100,
            "stopping": {
                "fitness_threshold": 0.01,
                "stagnation_generations": 5,
            },
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
            "minimize": True,
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        engine = create_engine(config, sphere_fitness)
        population = create_initial_population(config, seed=42)
        
        result = engine.run(population)
        
        # Should stop before max_generations
        assert result.generations < 100


class TestConfigRoundTrip:
    """Test config serialization round-trip."""
    
    def test_to_json_from_json(self) -> None:
        """Test config survives JSON round-trip."""
        original = UnifiedConfig(
            population_size=100,
            max_generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
        )
        
        # Round-trip through JSON
        json_str = original.to_json()
        restored = UnifiedConfig.from_json(json_str)
        
        # Should be equal
        assert restored.population_size == original.population_size
        assert restored.max_generations == original.max_generations
        assert restored.mutation_rate == original.mutation_rate
        assert restored.selection == original.selection
        assert restored.crossover == original.crossover
        assert restored.mutation == original.mutation
        assert restored.genome_type == original.genome_type
    
    def test_config_hash_stable_after_roundtrip(self) -> None:
        """Test config hash is stable after JSON round-trip."""
        config = UnifiedConfig(
            population_size=50,
            max_generations=25,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            seed=42,
        )
        
        hash1 = config.compute_hash()
        
        # Round-trip
        restored = UnifiedConfig.from_json(config.to_json())
        hash2 = restored.compute_hash()
        
        assert hash1 == hash2


class TestDifferentGenomeTypes:
    """Test different genome types via config."""
    
    def test_vector_genome(self) -> None:
        """Test vector genome configuration."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=2,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )
        
        population = create_initial_population(config, seed=42)
        
        assert len(population) == 10
        for ind in population.individuals:
            assert hasattr(ind.genome, 'genes')
            assert len(ind.genome.genes) == 5
    
    def test_sequence_genome(self) -> None:
        """Test sequence genome configuration."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=2,
            selection="tournament",
            crossover="single_point",  # Compatible with sequence
            mutation="uniform",  # Compatible with sequence
            genome_type="sequence",
            genome_params={"length": 8, "alphabet": list(range(10))},
        )
        
        population = create_initial_population(config, seed=42)
        
        assert len(population) == 10
        for ind in population.individuals:
            assert hasattr(ind.genome, 'genes')
            assert len(ind.genome.genes) == 8


class TestDeterminism:
    """Test deterministic behavior with seeds."""
    
    def test_same_seed_same_population(self) -> None:
        """Test same seed produces same initial population."""
        config = UnifiedConfig(
            population_size=20,
            max_generations=5,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )
        
        pop1 = create_initial_population(config, seed=12345)
        pop2 = create_initial_population(config, seed=12345)
        
        for ind1, ind2 in zip(pop1.individuals, pop2.individuals):
            assert list(ind1.genome.genes) == list(ind2.genome.genes)
    
    def test_different_seeds_different_populations(self) -> None:
        """Test different seeds produce different populations."""
        config = UnifiedConfig(
            population_size=20,
            max_generations=5,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )
        
        pop1 = create_initial_population(config, seed=111)
        pop2 = create_initial_population(config, seed=222)
        
        # At least some individuals should be different
        differences = 0
        for ind1, ind2 in zip(pop1.individuals, pop2.individuals):
            if list(ind1.genome.genes) != list(ind2.genome.genes):
                differences += 1
        
        assert differences > 0
