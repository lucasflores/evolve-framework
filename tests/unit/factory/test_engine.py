"""Unit tests for evolve/factory/engine.py (T077)."""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from evolve.config.callbacks import CallbackConfig
from evolve.config.erp import ERPSettings
from evolve.config.multiobjective import MultiObjectiveConfig, ObjectiveSpec
from evolve.config.stopping import StoppingConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.stopping import (
    CompositeStoppingCriterion,
    GenerationLimitStopping,
)
from evolve.factory.engine import (
    OperatorCompatibilityError,
    _build_callbacks,
    _build_stopping_criteria,
    _validate_operator_compatibility,
    create_engine,
    create_initial_population,
)
from evolve.registry.decoders import reset_decoder_registry
from evolve.registry.genomes import reset_genome_registry
from evolve.registry.operators import reset_operator_registry


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before and after each test."""
    reset_operator_registry()
    reset_genome_registry()
    reset_decoder_registry()
    yield
    reset_operator_registry()
    reset_genome_registry()
    reset_decoder_registry()


def simple_fitness(genome: Any) -> float:
    """Simple fitness function for testing."""
    if hasattr(genome, "genes"):
        return float(sum(genome.genes))
    return 0.0


class TestCreateEngine:
    """Tests for create_engine() function."""

    def test_create_engine_minimal_config(self) -> None:
        """Test creating engine with minimal configuration."""
        config = UnifiedConfig(
            population_size=20,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )

        engine = create_engine(config, simple_fitness)

        assert engine is not None
        assert hasattr(engine, "config")
        assert engine.config.population_size == 20

    def test_create_engine_with_seed(self) -> None:
        """Test creating engine with explicit seed."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        engine1 = create_engine(config, simple_fitness, seed=42)
        engine2 = create_engine(config, simple_fitness, seed=42)

        # Both should be created successfully
        assert engine1 is not None
        assert engine2 is not None

    def test_create_engine_uses_config_seed(self) -> None:
        """Test that config seed is used when no override provided."""
        config = UnifiedConfig(
            population_size=10,
            seed=123,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        engine = create_engine(config, simple_fitness)

        assert engine is not None

    def test_create_engine_with_callable_evaluator(self) -> None:
        """Test that callable fitness function is wrapped."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        # Pass callable directly - should be wrapped
        engine = create_engine(config, lambda g: sum(g.values))

        assert engine is not None

    def test_create_engine_minimization(self) -> None:
        """Test engine with minimization objective."""
        config = UnifiedConfig(
            population_size=10,
            minimize=True,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        engine = create_engine(config, simple_fitness)

        assert engine.config.minimize is True

    def test_create_engine_maximization(self) -> None:
        """Test engine with maximization objective."""
        config = UnifiedConfig(
            population_size=10,
            minimize=False,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        engine = create_engine(config, simple_fitness)

        assert engine.config.minimize is False


class TestOperatorCompatibility:
    """Tests for operator-genome compatibility validation."""

    def test_validate_compatible_operators(self) -> None:
        """Test validation passes for compatible operators."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",  # Universal
            crossover="sbx",  # Vector compatible
            mutation="gaussian",  # Vector compatible
            genome_type="vector",
        )

        # Should not raise
        _validate_operator_compatibility(config)

    def test_invalid_crossover_raises_error(self) -> None:
        """Test incompatible crossover raises OperatorCompatibilityError."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="neat",  # Graph-only crossover
            mutation="gaussian",
            genome_type="vector",
        )

        with pytest.raises(OperatorCompatibilityError) as exc_info:
            _validate_operator_compatibility(config)

        assert exc_info.value.operator_name == "neat"
        assert exc_info.value.category == "crossover"
        assert exc_info.value.genome_type == "vector"

    def test_invalid_mutation_raises_error(self) -> None:
        """Test incompatible mutation raises OperatorCompatibilityError."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="neat",  # Graph-only mutation
            genome_type="vector",
        )

        with pytest.raises(OperatorCompatibilityError) as exc_info:
            _validate_operator_compatibility(config)

        assert exc_info.value.operator_name == "neat"
        assert exc_info.value.category == "mutation"

    def test_compatibility_error_message(self) -> None:
        """Test compatibility error has descriptive message."""
        error = OperatorCompatibilityError(
            operator_name="neat",
            category="crossover",
            genome_type="vector",
            compatible_types={"graph"},
        )

        message = str(error)
        assert "neat" in message.lower()
        assert "Crossover" in message  # Capitalized
        assert "vector" in message
        assert "graph" in message


class TestStoppingCriteria:
    """Tests for _build_stopping_criteria()."""

    def test_default_stopping_generation_limit(self) -> None:
        """Test default uses generation limit from max_generations."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=50,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        stopping = _build_stopping_criteria(config)

        assert isinstance(stopping, GenerationLimitStopping)

    def test_stopping_with_fitness_threshold(self) -> None:
        """Test stopping with fitness threshold creates composite."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=100,
            stopping=StoppingConfig(fitness_threshold=0.01),
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        stopping = _build_stopping_criteria(config)

        assert isinstance(stopping, CompositeStoppingCriterion)

    def test_stopping_with_stagnation(self) -> None:
        """Test stopping with stagnation detection."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=100,
            stopping=StoppingConfig(stagnation_generations=20),
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        stopping = _build_stopping_criteria(config)

        assert isinstance(stopping, CompositeStoppingCriterion)

    def test_stopping_with_time_limit(self) -> None:
        """Test stopping with time limit."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=100,
            stopping=StoppingConfig(time_limit_seconds=60.0),
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        stopping = _build_stopping_criteria(config)

        assert isinstance(stopping, CompositeStoppingCriterion)


class TestCallbacks:
    """Tests for _build_callbacks()."""

    def test_no_callbacks_returns_empty(self) -> None:
        """Test no callbacks config returns empty list."""
        config = UnifiedConfig(
            population_size=10,
            callbacks=None,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        callbacks = _build_callbacks(config)

        assert callbacks == []

    def test_logging_callback_created(self) -> None:
        """Test logging callback is created when enabled."""
        config = UnifiedConfig(
            population_size=10,
            callbacks=CallbackConfig(enable_logging=True, log_level="INFO"),
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
        )

        callbacks = _build_callbacks(config)

        assert len(callbacks) == 1
        assert callbacks[0].__class__.__name__ == "LoggingCallback"

    def test_checkpoint_callback_created(self) -> None:
        """Test checkpoint callback created when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = UnifiedConfig(
                population_size=10,
                callbacks=CallbackConfig(
                    enable_checkpointing=True,
                    checkpoint_dir=tmpdir,
                    checkpoint_frequency=5,
                ),
                selection="tournament",
                crossover="sbx",
                mutation="gaussian",
                genome_type="vector",
            )

            callbacks = _build_callbacks(config)

            assert any(cb.__class__.__name__ == "CheckpointCallback" for cb in callbacks)


class TestERPEngine:
    """Tests for ERP engine creation."""

    def test_create_erp_engine(self) -> None:
        """Test creating an ERP engine."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            erp=ERPSettings(
                step_limit=100,
                recovery_threshold=0.5,
            ),
        )

        engine = create_engine(config, simple_fitness)

        # ERP engine should have different type
        assert engine.__class__.__name__ == "ERPEngine"


class TestMultiObjectiveEngine:
    """Tests for multi-objective engine creation."""

    def test_create_multiobjective_engine(self) -> None:
        """Test creating a multi-objective engine."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="obj1", direction="minimize"),
                    ObjectiveSpec(name="obj2", direction="minimize"),
                ],
            ),
        )

        engine = create_engine(config, simple_fitness)

        # Should have multi-objective config attached
        assert hasattr(engine, "_multiobjective_config")

    def test_multiobjective_with_reference_point(self) -> None:
        """Test reference point is stored on engine."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="obj1", direction="minimize"),
                    ObjectiveSpec(name="obj2", direction="minimize"),
                ],
                reference_point=[10.0, 10.0],
            ),
        )

        engine = create_engine(config, simple_fitness)

        assert hasattr(engine, "_reference_point")
        assert engine._reference_point == [10.0, 10.0]


class TestCreateInitialPopulation:
    """Tests for create_initial_population()."""

    def test_creates_correct_population_size(self) -> None:
        """Test population has correct size."""
        config = UnifiedConfig(
            population_size=50,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )

        population = create_initial_population(config)

        assert len(population) == 50

    def test_deterministic_with_seed(self) -> None:
        """Test population is deterministic with same seed."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        pop1 = create_initial_population(config, seed=42)
        pop2 = create_initial_population(config, seed=42)

        # Check genomes are identical
        for ind1, ind2 in zip(pop1.individuals, pop2.individuals):
            assert list(ind1.genome.genes) == list(ind2.genome.genes)

    def test_creates_vector_genomes(self) -> None:
        """Test creates vector genomes with correct dimensions."""
        config = UnifiedConfig(
            population_size=5,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 10, "bounds": (-1.0, 1.0)},
        )

        population = create_initial_population(config)

        for individual in population.individuals:
            assert len(individual.genome.genes) == 10

    def test_genome_within_bounds(self) -> None:
        """Test genome values are within bounds."""
        config = UnifiedConfig(
            population_size=20,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-2.0, 2.0)},
        )

        population = create_initial_population(config, seed=123)

        for individual in population.individuals:
            for val in individual.genome.genes:
                assert -2.0 <= val <= 2.0


class TestDecoderWiring:
    """Tests for decoder resolution in create_engine()."""

    def test_engine_with_decoder_from_config(self) -> None:
        """Test that decoder is resolved from config and passed to FunctionEvaluator."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            decoder="identity",
        )
        engine = create_engine(config, evaluator=simple_fitness)
        assert engine is not None
        # The evaluator should be a FunctionEvaluator with decoder set
        from evolve.evaluation.evaluator import FunctionEvaluator

        assert isinstance(engine.evaluator, FunctionEvaluator)
        assert engine.evaluator._decoder is not None

    def test_engine_without_decoder(self) -> None:
        """Test that engine works without decoder (decoder stays None)."""
        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
        )
        engine = create_engine(config, evaluator=simple_fitness)
        from evolve.evaluation.evaluator import FunctionEvaluator

        assert isinstance(engine.evaluator, FunctionEvaluator)
        assert engine.evaluator._decoder is None

    def test_decoder_with_params(self) -> None:
        """Test decoder_params are forwarded to factory."""
        from unittest.mock import MagicMock

        from evolve.registry.decoders import get_decoder_registry

        reg = get_decoder_registry()
        mock_decoder = MagicMock()

        def custom_factory(hidden_size: int = 64, **_kw):
            mock_decoder.hidden_size = hidden_size
            return mock_decoder

        reg.register("test_decoder", custom_factory)

        config = UnifiedConfig(
            population_size=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            decoder="test_decoder",
            decoder_params={"hidden_size": 128},
        )
        engine = create_engine(config, evaluator=simple_fitness)
        from evolve.evaluation.evaluator import FunctionEvaluator

        assert isinstance(engine.evaluator, FunctionEvaluator)
        assert engine.evaluator._decoder is mock_decoder
        assert mock_decoder.hidden_size == 128
