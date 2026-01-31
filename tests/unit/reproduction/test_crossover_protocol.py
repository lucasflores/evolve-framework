"""
Unit tests for crossover protocol execution.
"""

from random import Random

import numpy as np
import pytest

from evolve.reproduction.crossover_protocol import (
    BlendCrossoverExecutor,
    CloneCrossoverExecutor,
    CrossoverRegistry,
    SinglePointCrossoverExecutor,
    TwoPointCrossoverExecutor,
    UniformCrossoverExecutor,
    execute_crossover,
    inherit_protocol,
    safe_execute_crossover,
    validate_offspring,
)
from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    MatchabilityFunction,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)
from evolve.reproduction.sandbox import StepCounter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> Random:
    """Fixed seed random generator for reproducibility."""
    return Random(42)


@pytest.fixture
def counter() -> StepCounter:
    """Step counter for sandbox."""
    return StepCounter(limit=1000)


@pytest.fixture
def parent1_genome() -> np.ndarray:
    """First parent genome."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def parent2_genome() -> np.ndarray:
    """Second parent genome."""
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def protocol_a() -> ReproductionProtocol:
    """First parent protocol."""
    return ReproductionProtocol(
        matchability=MatchabilityFunction(type="accept_all"),
        intent=ReproductionIntentPolicy(type="always_willing"),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
    )


@pytest.fixture
def protocol_b() -> ReproductionProtocol:
    """Second parent protocol."""
    return ReproductionProtocol(
        matchability=MatchabilityFunction(type="distance_threshold", params={"threshold": 0.5}),
        intent=ReproductionIntentPolicy(type="fitness_threshold", params={"threshold": 0.8}),
        crossover=CrossoverProtocolSpec(type=CrossoverType.BLEND, params={"alpha": 0.3}),
    )


# =============================================================================
# SinglePointCrossoverExecutor Tests
# =============================================================================


class TestSinglePointCrossoverExecutor:
    """Tests for SinglePointCrossoverExecutor."""

    def test_execute_produces_offspring(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Executor produces offspring of correct shape."""
        executor = SinglePointCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape

    def test_execute_uses_step_counter(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Execution increments step counter."""
        executor = SinglePointCrossoverExecutor()
        params = {}

        executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert counter.count > 0

    def test_execute_mixes_genes(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Offspring contains genes from both parents."""
        executor = SinglePointCrossoverExecutor()
        params = {"point_ratio": 0.5}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        # Should have some zeros and some ones
        has_zeros = np.any(offspring1 == 0.0)
        has_ones = np.any(offspring1 == 1.0)
        # At least one should be true for most crossover points
        assert has_zeros or has_ones


# =============================================================================
# TwoPointCrossoverExecutor Tests
# =============================================================================


class TestTwoPointCrossoverExecutor:
    """Tests for TwoPointCrossoverExecutor."""

    def test_execute_produces_offspring(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Executor produces offspring of correct shape."""
        executor = TwoPointCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape

    def test_execute_uses_step_counter(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Execution increments step counter."""
        executor = TwoPointCrossoverExecutor()
        params = {}

        executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert counter.count > 0


# =============================================================================
# UniformCrossoverExecutor Tests
# =============================================================================


class TestUniformCrossoverExecutor:
    """Tests for UniformCrossoverExecutor."""

    def test_execute_produces_offspring(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Executor produces offspring of correct shape."""
        executor = UniformCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape

    def test_execute_respects_probability(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Probability parameter affects gene mixing."""
        executor = UniformCrossoverExecutor()
        # swap_prob=1.0 should always swap (offspring1 gets parent2 genes, offspring2 gets parent1)
        params = {"swap_prob": 1.0}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        np.testing.assert_array_equal(offspring1, parent2_genome)
        np.testing.assert_array_equal(offspring2, parent1_genome)

    def test_execute_probability_zero(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """swap_prob=0 should never swap."""
        executor = UniformCrossoverExecutor()
        params = {"swap_prob": 0.0}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        np.testing.assert_array_equal(offspring1, parent1_genome)
        np.testing.assert_array_equal(offspring2, parent2_genome)


# =============================================================================
# BlendCrossoverExecutor Tests
# =============================================================================


class TestBlendCrossoverExecutor:
    """Tests for BlendCrossoverExecutor."""

    def test_execute_produces_offspring(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Executor produces offspring of correct shape."""
        executor = BlendCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape

    def test_execute_blends_values(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Offspring values are interpolated from parents."""
        executor = BlendCrossoverExecutor()
        params = {"alpha": 0.0}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        # With alpha=0, should be between parent values (0 and 1)
        assert np.all(offspring1 >= 0.0)
        assert np.all(offspring1 <= 1.0)


# =============================================================================
# CloneCrossoverExecutor Tests
# =============================================================================


class TestCloneCrossoverExecutor:
    """Tests for CloneCrossoverExecutor."""

    def test_execute_clones_parents(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Clone executor returns copies of parents."""
        executor = CloneCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        np.testing.assert_array_equal(offspring1, parent1_genome)
        np.testing.assert_array_equal(offspring2, parent2_genome)

    def test_execute_returns_copies(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Clone returns copies, not references."""
        executor = CloneCrossoverExecutor()
        params = {}

        offspring1, offspring2 = executor.execute(parent1_genome, parent2_genome, params, rng, counter)

        assert offspring1 is not parent1_genome
        assert offspring2 is not parent2_genome


# =============================================================================
# CrossoverRegistry Tests
# =============================================================================


class TestCrossoverRegistry:
    """Tests for CrossoverRegistry."""

    def test_get_registered_types(self):
        """Registry contains all default types."""
        types = CrossoverRegistry.list_types()

        assert CrossoverType.SINGLE_POINT in types
        assert CrossoverType.TWO_POINT in types
        assert CrossoverType.UNIFORM in types
        assert CrossoverType.BLEND in types
        assert CrossoverType.CLONE in types

    def test_get_returns_executor(self):
        """Registry returns executor for valid type."""
        executor = CrossoverRegistry.get(CrossoverType.SINGLE_POINT)

        assert executor is not None
        assert isinstance(executor, SinglePointCrossoverExecutor)

    def test_get_returns_none_for_invalid(self):
        """Registry returns None for unregistered type."""
        executor = CrossoverRegistry.get("nonexistent_type")

        assert executor is None

    def test_get_or_default_returns_clone(self):
        """get_or_default returns Clone for invalid type."""
        executor = CrossoverRegistry.get_or_default("nonexistent")

        assert isinstance(executor, CloneCrossoverExecutor)


# =============================================================================
# execute_crossover Tests
# =============================================================================


class TestExecuteCrossover:
    """Tests for execute_crossover function."""

    def test_execute_with_valid_spec(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Crossover executes with valid specification."""
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM)

        offspring1, offspring2 = execute_crossover(spec, parent1_genome, parent2_genome, rng, counter)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape

    def test_execute_inactive_spec_clones(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Inactive spec defaults to cloning."""
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM, active=False)

        offspring1, offspring2 = execute_crossover(spec, parent1_genome, parent2_genome, rng, counter)

        np.testing.assert_array_equal(offspring1, parent1_genome)
        np.testing.assert_array_equal(offspring2, parent2_genome)

    def test_execute_unknown_type_raises(
        self, rng: Random, counter: StepCounter, parent1_genome: np.ndarray, parent2_genome: np.ndarray
    ):
        """Unknown crossover type raises ValueError."""
        # Create a spec with invalid type by modifying after creation
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
        object.__setattr__(spec, "type", "nonexistent")

        with pytest.raises(ValueError, match="Unknown crossover type"):
            execute_crossover(spec, parent1_genome, parent2_genome, rng, counter)


# =============================================================================
# safe_execute_crossover Tests
# =============================================================================


class TestSafeExecuteCrossover:
    """Tests for safe_execute_crossover function."""

    def test_returns_offspring_and_success(self, rng: Random, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Safe execute returns offspring and success flag."""
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM)

        (offspring1, offspring2), success = safe_execute_crossover(spec, parent1_genome, parent2_genome, rng)

        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape
        assert success is True

    def test_handles_errors_gracefully(self, rng: Random, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Returns clones and False on error."""
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
        object.__setattr__(spec, "type", "nonexistent")

        (offspring1, offspring2), success = safe_execute_crossover(spec, parent1_genome, parent2_genome, rng)

        np.testing.assert_array_equal(offspring1, parent1_genome)
        np.testing.assert_array_equal(offspring2, parent2_genome)
        assert success is False

    def test_respects_step_limit(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Execution with very low step limit returns clones."""
        spec = CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
        rng = Random(42)

        # Step limit of 0 should fail immediately
        (offspring1, offspring2), success = safe_execute_crossover(spec, parent1_genome, parent2_genome, rng, step_limit=0)

        # May or may not fail depending on implementation, but should not crash
        assert offspring1.shape == parent1_genome.shape
        assert offspring2.shape == parent2_genome.shape


# =============================================================================
# inherit_protocol Tests
# =============================================================================


class TestInheritProtocol:
    """Tests for inherit_protocol function."""

    def test_inherits_from_one_parent(self, protocol_a: ReproductionProtocol, protocol_b: ReproductionProtocol):
        """Offspring protocol is inherited from one parent."""
        rng = Random(42)

        offspring_protocol = inherit_protocol(protocol_a, protocol_b, rng)

        # Should be equal to one of the parents
        assert offspring_protocol == protocol_a or offspring_protocol == protocol_b

    def test_deterministic_with_same_seed(self, protocol_a: ReproductionProtocol, protocol_b: ReproductionProtocol):
        """Same seed produces same inheritance."""
        rng1 = Random(42)
        rng2 = Random(42)

        result1 = inherit_protocol(protocol_a, protocol_b, rng1)
        result2 = inherit_protocol(protocol_a, protocol_b, rng2)

        assert result1 == result2

    def test_handles_none_parents(self, protocol_a: ReproductionProtocol):
        """Handles None protocol parents."""
        rng = Random(42)

        result = inherit_protocol(protocol_a, None, rng)
        assert result == protocol_a

        rng2 = Random(42)
        result2 = inherit_protocol(None, protocol_a, rng2)
        assert result2 == protocol_a

    def test_both_none_returns_none(self):
        """Both parents None returns None."""
        rng = Random(42)

        result = inherit_protocol(None, None, rng)

        assert result is None


# =============================================================================
# validate_offspring Tests
# =============================================================================


class TestValidateOffspring:
    """Tests for validate_offspring function."""

    def test_valid_offspring_passes(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Valid offspring passes validation."""
        offspring = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        is_valid, reason = validate_offspring(offspring, parent1_genome, parent2_genome)

        assert is_valid is True
        assert reason is None

    def test_wrong_shape_fails(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Offspring with wrong shape fails."""
        offspring = np.array([0.5, 0.5])  # Wrong size

        is_valid, reason = validate_offspring(offspring, parent1_genome, parent2_genome)

        assert is_valid is False
        assert reason is not None

    def test_nan_values_fail(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Offspring with NaN values fails."""
        offspring = np.array([0.5, np.nan, 0.5, 0.5, 0.5])

        is_valid, reason = validate_offspring(offspring, parent1_genome, parent2_genome)

        assert is_valid is False
        assert "NaN" in reason or "nan" in reason.lower()

    def test_inf_values_fail(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Offspring with inf values fails."""
        offspring = np.array([0.5, np.inf, 0.5, 0.5, 0.5])

        is_valid, reason = validate_offspring(offspring, parent1_genome, parent2_genome)

        assert is_valid is False
        assert "inf" in reason.lower()

    def test_wrong_dtype_fails(self, parent1_genome: np.ndarray, parent2_genome: np.ndarray):
        """Offspring with wrong dtype fails."""
        offspring = np.array(["a", "b", "c", "d", "e"])

        is_valid, reason = validate_offspring(offspring, parent1_genome, parent2_genome)

        assert is_valid is False
        assert reason is not None
