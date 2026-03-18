"""
Unit tests for UnifiedConfig serialization.

Tests JSON round-trip, hash computation, and validation.
"""

import json
import pytest

from evolve.config.unified import UnifiedConfig
from evolve.config.stopping import StoppingConfig
from evolve.config.callbacks import CallbackConfig
from evolve.config.erp import ERPSettings
from evolve.config.multiobjective import (
    ObjectiveSpec,
    ConstraintSpec,
    MultiObjectiveConfig,
)
from evolve.config.meta import ParameterSpec, MetaEvolutionConfig


class TestUnifiedConfigSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_to_dict_minimal_config(self) -> None:
        """Test minimal config converts to dict."""
        config = UnifiedConfig(population_size=50)
        data = config.to_dict()
        
        assert data["population_size"] == 50
        assert "schema_version" in data
    
    def test_from_dict_minimal_config(self) -> None:
        """Test minimal config loads from dict."""
        data = {
            "population_size": 100,
            "max_generations": 50,
        }
        config = UnifiedConfig.from_dict(data)
        
        assert config.population_size == 100
        assert config.max_generations == 50
    
    def test_round_trip_preserves_values(self) -> None:
        """Test to_dict -> from_dict preserves all values."""
        original = UnifiedConfig(
            population_size=100,
            max_generations=200,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism=2,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            selection_params={"tournament_size": 5},
            crossover_params={"eta": 20.0},
            mutation_params={"sigma": 0.1},
            genome_type="vector",
            genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)},
            minimize=True,
            seed=42,
        )
        
        data = original.to_dict()
        restored = UnifiedConfig.from_dict(data)
        
        assert restored.population_size == original.population_size
        assert restored.max_generations == original.max_generations
        assert restored.crossover_rate == original.crossover_rate
        assert restored.mutation_rate == original.mutation_rate
        assert restored.elitism == original.elitism
        assert restored.selection == original.selection
        assert restored.crossover == original.crossover
        assert restored.mutation == original.mutation
        assert restored.selection_params == original.selection_params
        assert restored.crossover_params == original.crossover_params
        assert restored.mutation_params == original.mutation_params
        assert restored.genome_type == original.genome_type
        assert restored.minimize == original.minimize
        assert restored.seed == original.seed
    
    def test_json_round_trip(self) -> None:
        """Test to_json -> from_json preserves config."""
        original = UnifiedConfig(
            population_size=75,
            max_generations=100,
            selection="roulette",
        )
        
        json_str = original.to_json()
        restored = UnifiedConfig.from_json(json_str)
        
        assert restored.population_size == 75
        assert restored.max_generations == 100
        assert restored.selection == "roulette"
    
    def test_json_is_valid_json(self) -> None:
        """Test to_json produces valid JSON."""
        config = UnifiedConfig(population_size=50)
        json_str = config.to_json()
        
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestUnifiedConfigWithStoppingConfig:
    """Test UnifiedConfig with stopping criteria."""
    
    def test_stopping_config_serialization(self) -> None:
        """Test config with stopping criteria round-trips."""
        stopping = StoppingConfig(
            max_generations=100,
            fitness_threshold=0.99,
            stagnation_generations=10,
        )
        original = UnifiedConfig(
            population_size=50,
            stopping=stopping,
        )
        
        data = original.to_dict()
        restored = UnifiedConfig.from_dict(data)
        
        assert restored.stopping is not None
        assert restored.stopping.max_generations == 100
        assert restored.stopping.fitness_threshold == 0.99
        assert restored.stopping.stagnation_generations == 10


class TestUnifiedConfigWithCallbacks:
    """Test UnifiedConfig with callback settings."""
    
    def test_callbacks_config_serialization(self) -> None:
        """Test config with callbacks round-trips."""
        callbacks = CallbackConfig(
            enable_logging=True,
            log_level="DEBUG",
            enable_checkpointing=True,
            checkpoint_dir="./checkpoints",
            checkpoint_frequency=10,
        )
        original = UnifiedConfig(
            population_size=50,
            callbacks=callbacks,
        )
        
        data = original.to_dict()
        restored = UnifiedConfig.from_dict(data)
        
        assert restored.callbacks is not None
        assert restored.callbacks.enable_logging is True
        assert restored.callbacks.log_level == "DEBUG"
        assert restored.callbacks.checkpoint_frequency == 10


class TestUnifiedConfigHash:
    """Test deterministic hash computation."""
    
    def test_same_config_same_hash(self) -> None:
        """Same configurations produce same hash."""
        config1 = UnifiedConfig(
            population_size=100,
            crossover_rate=0.8,
            seed=42,
        )
        config2 = UnifiedConfig(
            population_size=100,
            crossover_rate=0.8,
            seed=42,
        )
        
        assert config1.compute_hash() == config2.compute_hash()
    
    def test_different_config_different_hash(self) -> None:
        """Different configurations produce different hashes."""
        config1 = UnifiedConfig(population_size=100, seed=42)
        config2 = UnifiedConfig(population_size=100, seed=43)
        
        assert config1.compute_hash() != config2.compute_hash()
    
    def test_hash_is_string(self) -> None:
        """Hash is returned as string."""
        config = UnifiedConfig(population_size=50)
        hash_val = config.compute_hash()
        
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0


class TestUnifiedConfigWithParams:
    """Test with_params method for creating modified copies."""
    
    def test_with_params_creates_new_config(self) -> None:
        """with_params creates a new config instance."""
        original = UnifiedConfig(population_size=100)
        modified = original.with_params(population_size=200)
        
        assert modified is not original
        assert modified.population_size == 200
        assert original.population_size == 100
    
    def test_with_params_preserves_unmodified_values(self) -> None:
        """with_params preserves values not specified."""
        original = UnifiedConfig(
            population_size=100,
            max_generations=50,
            crossover_rate=0.9,
        )
        modified = original.with_params(population_size=200)
        
        assert modified.max_generations == 50
        assert modified.crossover_rate == 0.9
    
    def test_with_params_multiple_values(self) -> None:
        """with_params can modify multiple values."""
        original = UnifiedConfig(population_size=100)
        modified = original.with_params(
            population_size=200,
            max_generations=500,
            mutation_rate=0.05,
        )
        
        assert modified.population_size == 200
        assert modified.max_generations == 500
        assert modified.mutation_rate == 0.05


class TestUnifiedConfigValidation:
    """Test configuration validation."""
    
    def test_valid_minimal_config(self) -> None:
        """Minimal valid config passes validation."""
        # Should not raise
        config = UnifiedConfig(population_size=10)
        assert config.population_size == 10
    
    def test_invalid_population_size_raises(self) -> None:
        """Invalid population size raises ValueError."""
        with pytest.raises(ValueError, match="population_size"):
            UnifiedConfig(population_size=0)
    
    def test_invalid_crossover_rate_raises(self) -> None:
        """Invalid crossover rate raises ValueError."""
        with pytest.raises(ValueError, match="crossover_rate"):
            UnifiedConfig(population_size=50, crossover_rate=1.5)
    
    def test_invalid_mutation_rate_raises(self) -> None:
        """Invalid mutation rate raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate"):
            UnifiedConfig(population_size=50, mutation_rate=-0.1)
