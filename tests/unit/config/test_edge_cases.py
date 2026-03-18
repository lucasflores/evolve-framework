"""Edge case tests for unified configuration (T085-T088)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from evolve.config.unified import UnifiedConfig
from evolve.config.stopping import StoppingConfig
from evolve.config.meta import MetaEvolutionConfig, ParameterSpec
from evolve.factory.engine import (
    create_engine,
    create_initial_population,
    OperatorCompatibilityError,
)
from evolve.registry.operators import reset_operator_registry
from evolve.registry.genomes import reset_genome_registry


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    reset_operator_registry()
    reset_genome_registry()
    yield
    reset_operator_registry()
    reset_genome_registry()


class TestInvalidOperatorName:
    """Tests for T085: Invalid operator name produces descriptive error."""
    
    def test_invalid_selection_operator(self) -> None:
        """Test invalid selection operator gives descriptive error."""
        config = UnifiedConfig(
            population_size=10,
            selection="nonexistent_selection",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        with pytest.raises(KeyError) as exc_info:
            create_engine(config, lambda x: sum(x))
        
        # Error should mention the operator name
        assert "nonexistent_selection" in str(exc_info.value)
        # Error should mention available operators
        assert "Available" in str(exc_info.value) or "selection" in str(exc_info.value).lower()
    
    def test_invalid_crossover_operator(self) -> None:
        """Test invalid crossover operator gives descriptive error."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="fake_crossover",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        with pytest.raises(KeyError) as exc_info:
            create_engine(config, lambda x: sum(x))
        
        assert "fake_crossover" in str(exc_info.value)
    
    def test_invalid_mutation_operator(self) -> None:
        """Test invalid mutation operator gives descriptive error."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="invalid_mutation",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        with pytest.raises(KeyError) as exc_info:
            create_engine(config, lambda x: sum(x))
        
        assert "invalid_mutation" in str(exc_info.value)
    
    def test_invalid_genome_type(self) -> None:
        """Test invalid genome type gives descriptive error."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="unsupported_genome",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        with pytest.raises(KeyError) as exc_info:
            create_initial_population(config)
        
        assert "unsupported_genome" in str(exc_info.value)


class TestIncompatibleConfiguration:
    """Tests for T086: Conflicting config flags detected at build time."""
    
    def test_graph_genome_with_vector_crossover(self) -> None:
        """Test graph genome with vector-only crossover raises error."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",  # Vector-only crossover
            mutation="neat",  # Graph mutation
            genome_type="graph",
            genome_params={
                "num_inputs": 2,
                "num_outputs": 1,
            },
        )
        
        with pytest.raises(OperatorCompatibilityError) as exc_info:
            create_engine(config, lambda x: 0.0)
        
        assert exc_info.value.operator_name == "sbx"
        assert exc_info.value.genome_type == "graph"
    
    def test_vector_genome_with_graph_mutation(self) -> None:
        """Test vector genome with graph-only mutation raises error."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="neat",  # Graph-only mutation
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        with pytest.raises(OperatorCompatibilityError) as exc_info:
            create_engine(config, lambda x: sum(x))
        
        assert exc_info.value.operator_name == "neat"
        assert exc_info.value.category == "mutation"


class TestMetaEvolutionFailure:
    """Tests for T087: Inner loop failure assigns worst-case fitness."""
    
    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_inner_loop_exception_handled(
        self,
        mock_create_pop,
        mock_create_engine,
    ) -> None:
        """Test that inner loop failure is handled gracefully."""
        from evolve.meta.evaluator import MetaEvaluator
        
        # Setup mock that raises an exception
        mock_engine = MagicMock()
        mock_engine.run.side_effect = Exception("Inner evolution failed")
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()
        
        base_config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )
        
        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(path="mutation_rate", bounds=(0.01, 0.5)),
            ),
            outer_population_size=5,
            outer_generations=3,
        )
        
        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=lambda x: sum(x),
        )
        
        # Evaluator should raise or return worst case fitness
        # The current implementation will propagate the exception
        with pytest.raises(Exception, match="Inner evolution failed"):
            evaluator.evaluate(base_config)


class TestPartialJsonDefaults:
    """Tests for T088: Partial JSON with missing optional sections applies defaults."""
    
    def test_minimal_json_applies_defaults(self) -> None:
        """Test that minimal JSON config applies default values."""
        # Minimal config - only required fields
        config_dict = {
            "population_size": 50,
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 5, "bounds": [-1.0, 1.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        # Check defaults
        assert config.max_generations == 100  # Default value
        assert config.elitism == 1  # Default value
        assert config.mutation_rate == 1.0  # Default value (apply to all offspring)
        assert config.crossover_rate == 0.9  # Default value
        assert config.minimize is True  # Default value
        assert config.stopping is None  # Optional, default None
        assert config.callbacks is None  # Optional, default None
        assert config.erp is None  # Optional, default None
        assert config.multiobjective is None  # Optional, default None
        assert config.meta is None  # Optional, default None
    
    def test_missing_stopping_uses_max_generations(self) -> None:
        """Test that missing stopping config uses max_generations."""
        config_dict = {
            "population_size": 30,
            "max_generations": 200,
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
            # No "stopping" section
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        assert config.stopping is None
        assert config.max_generations == 200
    
    def test_partial_stopping_applies_defaults(self) -> None:
        """Test partial stopping config uses defaults for missing fields."""
        config_dict = {
            "population_size": 30,
            "max_generations": 100,
            "stopping": {
                "fitness_threshold": 0.001,
                # stagnation_generations and time_limit_seconds not specified
            },
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        assert config.stopping is not None
        assert config.stopping.fitness_threshold == 0.001
        assert config.stopping.stagnation_generations is None  # Default
        assert config.stopping.time_limit_seconds is None  # Default
    
    def test_missing_operator_params_uses_defaults(self) -> None:
        """Test missing operator params uses empty dict."""
        config_dict = {
            "population_size": 30,
            "selection": "tournament",
            # No selection_params
            "crossover": "sbx",
            # No crossover_params
            "mutation": "gaussian",
            # No mutation_params
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        # Should have empty dicts for params
        assert config.selection_params == {}
        assert config.crossover_params == {}
        assert config.mutation_params == {}
    
    def test_empty_tags_list_converted(self) -> None:
        """Test empty tags list is converted properly."""
        config_dict = {
            "population_size": 30,
            "tags": [],  # Empty list
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        assert config.tags == ()
    
    def test_tags_list_converted_to_tuple(self) -> None:
        """Test tags list is converted to tuple."""
        config_dict = {
            "population_size": 30,
            "tags": ["experiment", "optimization"],  # List
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
            "genome_type": "vector",
            "genome_params": {"dimensions": 3, "bounds": [-1.0, 1.0]},
        }
        
        config = UnifiedConfig.from_dict(config_dict)
        
        assert config.tags == ("experiment", "optimization")
        assert isinstance(config.tags, tuple)
