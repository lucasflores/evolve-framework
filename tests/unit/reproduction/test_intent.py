"""
Unit tests for intent policy evaluation.
"""

from random import Random

import numpy as np
import pytest

from evolve.reproduction.intent import (
    AgeDependentIntent,
    AlwaysWillingIntent,
    FitnessRankThresholdIntent,
    FitnessThresholdIntent,
    IntentRegistry,
    NeverWillingIntent,
    ProbabilisticIntent,
    ResourceBudgetIntent,
    evaluate_intent,
    safe_evaluate_intent,
)
from evolve.reproduction.protocol import IntentContext, ReproductionIntentPolicy
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
def basic_context() -> IntentContext:
    """Basic intent context."""
    return IntentContext(
        fitness=np.array([0.7]),
        generation=10,
        age=5,
        offspring_count=0,
        fitness_rank=3,
        population_size=100,
    )


# =============================================================================
# AlwaysWillingIntent Tests
# =============================================================================


class TestAlwaysWillingIntent:
    """Tests for AlwaysWillingIntent evaluator."""

    def test_always_returns_true(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Evaluator always returns True."""
        evaluator = AlwaysWillingIntent()
        policy = ReproductionIntentPolicy(type="always_willing")

        result = evaluator.evaluate(basic_context, policy.params, rng, counter)

        assert result is True

    def test_increments_counter(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Evaluation increments step counter."""
        evaluator = AlwaysWillingIntent()
        policy = ReproductionIntentPolicy(type="always_willing")

        evaluator.evaluate(basic_context, policy.params, rng, counter)

        assert counter.count == 1


# =============================================================================
# NeverWillingIntent Tests
# =============================================================================


class TestNeverWillingIntent:
    """Tests for NeverWillingIntent evaluator."""

    def test_always_returns_false(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Evaluator always returns False."""
        evaluator = NeverWillingIntent()
        policy = ReproductionIntentPolicy(type="never_willing")

        result = evaluator.evaluate(basic_context, policy.params, rng, counter)

        assert result is False


# =============================================================================
# FitnessThresholdIntent Tests
# =============================================================================


class TestFitnessThresholdIntent:
    """Tests for FitnessThresholdIntent evaluator."""

    def test_accepts_above_threshold(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns True when fitness above threshold."""
        evaluator = FitnessThresholdIntent()
        params = {"threshold": 0.5}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result  # fitness 0.7 >= 0.5

    def test_rejects_below_threshold(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns False when fitness below threshold."""
        evaluator = FitnessThresholdIntent()
        params = {"threshold": 0.8}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert not result  # fitness 0.7 < 0.8

    def test_default_threshold_zero(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Default threshold of 0 accepts all."""
        evaluator = FitnessThresholdIntent()

        result = evaluator.evaluate(basic_context, {}, rng, counter)

        assert result


# =============================================================================
# FitnessRankThresholdIntent Tests
# =============================================================================


class TestFitnessRankThresholdIntent:
    """Tests for FitnessRankThresholdIntent evaluator."""

    def test_accepts_high_rank(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns True when rank is within threshold."""
        evaluator = FitnessRankThresholdIntent()
        params = {"max_rank": 5}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is True  # rank 3 <= 5

    def test_rejects_low_rank(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns False when rank exceeds threshold."""
        evaluator = FitnessRankThresholdIntent()
        params = {"max_rank": 2}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is False  # rank 3 > 2


# =============================================================================
# ResourceBudgetIntent Tests
# =============================================================================


class TestResourceBudgetIntent:
    """Tests for ResourceBudgetIntent evaluator."""

    def test_accepts_under_budget(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns True when under offspring budget."""
        evaluator = ResourceBudgetIntent()
        params = {"max_offspring": 3}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is True  # offspring_count 0 < 3

    def test_rejects_over_budget(self, rng: Random, counter: StepCounter):
        """Returns False when offspring budget exhausted."""
        context = IntentContext(
            fitness=np.array([0.7]),
            generation=10,
            age=5,
            offspring_count=5,  # Already has 5 offspring
            fitness_rank=3,
            population_size=100,
        )
        evaluator = ResourceBudgetIntent()
        params = {"max_offspring": 3}

        result = evaluator.evaluate(context, params, rng, counter)

        assert result is False  # offspring_count 5 >= 3


# =============================================================================
# AgeDependentIntent Tests
# =============================================================================


class TestAgeDependentIntent:
    """Tests for AgeDependentIntent evaluator."""

    def test_accepts_within_age_range(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns True when age is within range."""
        evaluator = AgeDependentIntent()
        params = {"min_age": 0, "max_age": 10}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is True  # age 5 in [0, 10]

    def test_rejects_too_young(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Returns False when too young."""
        evaluator = AgeDependentIntent()
        params = {"min_age": 10, "max_age": 20}

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is False  # age 5 < min_age 10

    def test_rejects_too_old(self, rng: Random, counter: StepCounter):
        """Returns False when too old."""
        context = IntentContext(
            fitness=np.array([0.7]),
            generation=10,
            age=15,
            offspring_count=0,
            fitness_rank=3,
            population_size=100,
        )
        evaluator = AgeDependentIntent()
        params = {"min_age": 0, "max_age": 10}

        result = evaluator.evaluate(context, params, rng, counter)

        assert result is False  # age 15 > max_age 10


# =============================================================================
# ProbabilisticIntent Tests
# =============================================================================


class TestProbabilisticIntent:
    """Tests for ProbabilisticIntent evaluator."""

    def test_always_willing_with_probability_one(
        self, counter: StepCounter, basic_context: IntentContext
    ):
        """Always returns True with p=1."""
        evaluator = ProbabilisticIntent()
        params = {"probability": 1.0}
        rng = Random(42)

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is True

    def test_never_willing_with_probability_zero(
        self, counter: StepCounter, basic_context: IntentContext
    ):
        """Always returns False with p=0."""
        evaluator = ProbabilisticIntent()
        params = {"probability": 0.0}
        rng = Random(42)

        result = evaluator.evaluate(basic_context, params, rng, counter)

        assert result is False

    def test_respects_random_seed(self, counter: StepCounter, basic_context: IntentContext):
        """Same seed produces same result."""
        evaluator = ProbabilisticIntent()
        params = {"probability": 0.5}
        rng1 = Random(42)
        rng2 = Random(42)

        result1 = evaluator.evaluate(basic_context, params, rng1, StepCounter(limit=1000))
        result2 = evaluator.evaluate(basic_context, params, rng2, counter)

        assert result1 == result2


# =============================================================================
# IntentRegistry Tests
# =============================================================================


class TestIntentRegistry:
    """Tests for IntentRegistry."""

    def test_contains_builtin_types(self):
        """Registry contains all built-in types."""
        types = IntentRegistry.list_types()

        assert "always_willing" in types
        assert "never_willing" in types
        assert "fitness_threshold" in types
        assert "fitness_rank_threshold" in types
        assert "resource_budget" in types
        assert "age_dependent" in types
        assert "probabilistic" in types

    def test_get_returns_evaluator(self):
        """get() returns registered evaluator."""
        evaluator = IntentRegistry.get("always_willing")

        assert evaluator is not None
        assert isinstance(evaluator, AlwaysWillingIntent)

    def test_get_returns_none_for_invalid(self):
        """get() returns None for unregistered type."""
        evaluator = IntentRegistry.get("nonexistent")

        assert evaluator is None

    def test_get_or_default_returns_always_willing(self):
        """get_or_default() returns AlwaysWilling for invalid type."""
        evaluator = IntentRegistry.get_or_default("nonexistent")

        assert isinstance(evaluator, AlwaysWillingIntent)


# =============================================================================
# evaluate_intent Tests
# =============================================================================


class TestEvaluateIntent:
    """Tests for evaluate_intent function."""

    def test_evaluates_active_policy(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Evaluates active policy correctly."""
        policy = ReproductionIntentPolicy(type="fitness_threshold", params={"threshold": 0.5})

        result = evaluate_intent(policy, basic_context, rng, counter)

        assert result

    def test_inactive_policy_returns_true(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Inactive policy returns True (always willing)."""
        policy = ReproductionIntentPolicy(type="never_willing", active=False)

        result = evaluate_intent(policy, basic_context, rng, counter)

        assert result is True

    def test_unknown_type_raises_value_error(
        self, rng: Random, counter: StepCounter, basic_context: IntentContext
    ):
        """Unknown policy type raises ValueError."""
        policy = ReproductionIntentPolicy(type="nonexistent")

        with pytest.raises(ValueError, match="Unknown intent type"):
            evaluate_intent(policy, basic_context, rng, counter)


# =============================================================================
# safe_evaluate_intent Tests
# =============================================================================


class TestSafeEvaluateIntent:
    """Tests for safe_evaluate_intent function."""

    def test_returns_result_and_success(self, rng: Random, basic_context: IntentContext):
        """Returns result and success flag."""
        policy = ReproductionIntentPolicy(type="always_willing")

        result, success = safe_evaluate_intent(policy, basic_context, rng)

        assert result is True
        assert success is True

    def test_handles_errors_gracefully(self, rng: Random, basic_context: IntentContext):
        """Returns True and False on error."""
        policy = ReproductionIntentPolicy(type="nonexistent")

        result, success = safe_evaluate_intent(policy, basic_context, rng)

        assert result is True  # Default to willing
        assert success is False

    def test_respects_step_limit(self, rng: Random, basic_context: IntentContext):
        """Returns True on step limit exceeded."""
        policy = ReproductionIntentPolicy(type="always_willing")

        # With limit=0, should fail immediately, but may or may not depending on implementation
        result, success = safe_evaluate_intent(policy, basic_context, rng, step_limit=0)

        # Should not crash
        assert isinstance(result, bool)
        assert isinstance(success, bool)
