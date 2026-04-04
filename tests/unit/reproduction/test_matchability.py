"""
Unit tests for matchability evaluators.

Tests cover:
- All built-in matchability evaluators
- MatchabilityRegistry functionality
- evaluate_matchability with sandboxing
- safe_evaluate_matchability with error handling
- Asymmetric matchability behavior
"""

from random import Random

import numpy as np
import pytest

from evolve.reproduction.matchability import (
    AcceptAllMatchability,
    DistanceThresholdMatchability,
    DiversitySeekingMatchability,
    FitnessRatioMatchability,
    MatchabilityRegistry,
    ProbabilisticMatchability,
    RejectAllMatchability,
    SimilarityThresholdMatchability,
    evaluate_matchability,
    safe_evaluate_matchability,
)
from evolve.reproduction.protocol import MatchabilityFunction, MateContext
from evolve.reproduction.sandbox import StepCounter


@pytest.fixture
def rng() -> Random:
    """Seeded RNG for deterministic tests."""
    return Random(42)


@pytest.fixture
def counter() -> StepCounter:
    """Fresh step counter."""
    return StepCounter(limit=1000)


@pytest.fixture
def basic_context() -> MateContext:
    """Basic mate context for testing."""
    return MateContext(
        partner_distance=0.5,
        partner_fitness_rank=3,
        partner_fitness_ratio=1.2,
        partner_niche_id=None,
        population_diversity=0.7,
        crowding_distance=0.4,
        self_fitness=np.array([0.8]),
        partner_fitness=np.array([0.96]),
    )


class TestAcceptAllMatchability:
    """Tests for AcceptAllMatchability evaluator."""

    def test_always_accepts(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        evaluator = AcceptAllMatchability()
        result = evaluator.evaluate(basic_context, {}, rng, counter)
        assert result is True

    def test_increments_counter(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        evaluator = AcceptAllMatchability()
        evaluator.evaluate(basic_context, {}, rng, counter)
        assert counter.count == 1


class TestRejectAllMatchability:
    """Tests for RejectAllMatchability evaluator."""

    def test_always_rejects(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        evaluator = RejectAllMatchability()
        result = evaluator.evaluate(basic_context, {}, rng, counter)
        assert result is False


class TestDistanceThresholdMatchability:
    """Tests for DistanceThresholdMatchability evaluator."""

    def test_accepts_above_threshold(self, rng: Random, counter: StepCounter) -> None:
        evaluator = DistanceThresholdMatchability()
        context = MateContext(
            partner_distance=0.6,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"min_distance": 0.5}, rng, counter)
        assert result is True

    def test_rejects_below_threshold(self, rng: Random, counter: StepCounter) -> None:
        evaluator = DistanceThresholdMatchability()
        context = MateContext(
            partner_distance=0.3,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"min_distance": 0.5}, rng, counter)
        assert result is False

    def test_accepts_at_threshold(self, rng: Random, counter: StepCounter) -> None:
        evaluator = DistanceThresholdMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"min_distance": 0.5}, rng, counter)
        assert result is True


class TestSimilarityThresholdMatchability:
    """Tests for SimilarityThresholdMatchability evaluator."""

    def test_accepts_below_threshold(self, rng: Random, counter: StepCounter) -> None:
        evaluator = SimilarityThresholdMatchability()
        context = MateContext(
            partner_distance=0.3,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"max_distance": 0.5}, rng, counter)
        assert result is True

    def test_rejects_above_threshold(self, rng: Random, counter: StepCounter) -> None:
        evaluator = SimilarityThresholdMatchability()
        context = MateContext(
            partner_distance=0.7,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"max_distance": 0.5}, rng, counter)
        assert result is False


class TestFitnessRatioMatchability:
    """Tests for FitnessRatioMatchability evaluator."""

    def test_accepts_within_range(self, rng: Random, counter: StepCounter) -> None:
        evaluator = FitnessRatioMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.1,  # Within [0.8, 1.2]
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.55]),
        )
        params = {"min_ratio": 0.8, "max_ratio": 1.2}
        result = evaluator.evaluate(context, params, rng, counter)
        assert result is True

    def test_rejects_below_range(self, rng: Random, counter: StepCounter) -> None:
        evaluator = FitnessRatioMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=0.5,  # Below 0.8
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([0.5]),
        )
        params = {"min_ratio": 0.8, "max_ratio": 1.2}
        result = evaluator.evaluate(context, params, rng, counter)
        assert result is False

    def test_rejects_above_range(self, rng: Random, counter: StepCounter) -> None:
        evaluator = FitnessRatioMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.5,  # Above 1.2
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.75]),
        )
        params = {"min_ratio": 0.8, "max_ratio": 1.2}
        result = evaluator.evaluate(context, params, rng, counter)
        assert result is False


class TestProbabilisticMatchability:
    """Tests for ProbabilisticMatchability evaluator."""

    def test_returns_probability(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        evaluator = ProbabilisticMatchability()
        result = evaluator.evaluate(basic_context, {"base_prob": 0.5}, rng, counter)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_base_prob_only(
        self, _basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        evaluator = ProbabilisticMatchability()
        context = MateContext(
            partner_distance=0.0,  # No distance effect
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(
            context, {"base_prob": 0.7, "distance_weight": 0.0}, rng, counter
        )
        assert result == pytest.approx(0.7)

    def test_distance_increases_probability(self, rng: Random, counter: StepCounter) -> None:
        evaluator = ProbabilisticMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(
            context, {"base_prob": 0.5, "distance_weight": 0.4}, rng, counter
        )
        # 0.5 + 0.4 * 0.5 = 0.7
        assert result == pytest.approx(0.7)

    def test_probability_clamped(self, rng: Random, counter: StepCounter) -> None:
        evaluator = ProbabilisticMatchability()
        context = MateContext(
            partner_distance=1.0,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        # Would be 0.8 + 1.0 * 0.5 = 1.3, should clamp to 1.0
        result = evaluator.evaluate(
            context, {"base_prob": 0.8, "distance_weight": 0.5}, rng, counter
        )
        assert result == 1.0


class TestDiversitySeekingMatchability:
    """Tests for DiversitySeekingMatchability evaluator."""

    def test_accepts_high_crowding(self, rng: Random, counter: StepCounter) -> None:
        evaluator = DiversitySeekingMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=0.8,  # High crowding
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"crowding_threshold": 0.5}, rng, counter)
        assert result == 1.0

    def test_fallback_for_no_crowding(self, rng: Random, counter: StepCounter) -> None:
        evaluator = DiversitySeekingMatchability()
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,  # No crowding info
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        result = evaluator.evaluate(context, {"fallback_prob": 0.4}, rng, counter)
        assert result == 0.4


class TestMatchabilityRegistry:
    """Tests for MatchabilityRegistry."""

    def test_builtin_types_registered(self) -> None:
        types = MatchabilityRegistry.list_types()
        assert "accept_all" in types
        assert "reject_all" in types
        assert "distance_threshold" in types
        assert "similarity_threshold" in types
        assert "fitness_ratio" in types
        assert "different_niche" in types
        assert "probabilistic" in types
        assert "diversity_seeking" in types

    def test_get_returns_evaluator(self) -> None:
        evaluator = MatchabilityRegistry.get("accept_all")
        assert evaluator is not None
        assert isinstance(evaluator, AcceptAllMatchability)

    def test_get_returns_none_for_unknown(self) -> None:
        evaluator = MatchabilityRegistry.get("unknown_type")
        assert evaluator is None

    def test_get_or_default_returns_reject_for_unknown(self) -> None:
        evaluator = MatchabilityRegistry.get_or_default("unknown_type")
        assert isinstance(evaluator, RejectAllMatchability)

    def test_register_custom_evaluator(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        class CustomMatchability:
            def evaluate(
                self, _context: MateContext, params: dict, _rng: Random, counter: StepCounter
            ) -> bool:
                counter.step()
                return params.get("result", False)

        MatchabilityRegistry.register("custom_test", CustomMatchability())
        evaluator = MatchabilityRegistry.get("custom_test")
        assert evaluator is not None
        result = evaluator.evaluate(basic_context, {"result": True}, rng, counter)
        assert result is True


class TestEvaluateMatchability:
    """Tests for evaluate_matchability function."""

    def test_evaluates_active_function(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        func = MatchabilityFunction(type="accept_all", params={}, active=True)
        result = evaluate_matchability(func, basic_context, rng, counter)
        assert result is True

    def test_inactive_function_rejects(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        func = MatchabilityFunction(type="accept_all", params={}, active=False)
        result = evaluate_matchability(func, basic_context, rng, counter)
        assert result is False

    def test_unknown_type_raises(
        self, basic_context: MateContext, rng: Random, counter: StepCounter
    ) -> None:
        func = MatchabilityFunction(type="nonexistent_type", params={}, active=True)
        with pytest.raises(ValueError, match="Unknown matchability type"):
            evaluate_matchability(func, basic_context, rng, counter)

    def test_probabilistic_resolves_to_bool(self, rng: Random, counter: StepCounter) -> None:
        func = MatchabilityFunction(type="probabilistic", params={"base_prob": 1.0}, active=True)
        context = MateContext(
            partner_distance=0.0,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        # With base_prob=1.0, should always accept
        for _ in range(10):
            counter.reset()
            result = evaluate_matchability(func, context, rng, counter)
            assert result is True


class TestSafeEvaluateMatchability:
    """Tests for safe_evaluate_matchability function."""

    def test_normal_evaluation(self, basic_context: MateContext, rng: Random) -> None:
        func = MatchabilityFunction(type="accept_all", params={}, active=True)
        result, success = safe_evaluate_matchability(func, basic_context, rng)
        assert result is True
        assert success is True

    def test_step_limit_returns_false(self, rng: Random) -> None:
        # Create a context and function that would work normally
        func = MatchabilityFunction(type="accept_all", params={}, active=True)
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        # With step_limit=0, any evaluation should fail
        result, success = safe_evaluate_matchability(func, context, rng, step_limit=0)
        assert result is False
        assert success is False

    def test_unknown_type_returns_false(self, basic_context: MateContext, rng: Random) -> None:
        func = MatchabilityFunction(type="nonexistent", params={}, active=True)
        result, success = safe_evaluate_matchability(func, basic_context, rng)
        assert result is False
        assert success is False


class TestAsymmetricMatchability:
    """Tests demonstrating asymmetric matchability behavior."""

    def test_asymmetric_acceptance(self, rng: Random) -> None:
        """
        Verify that A accepting B doesn't imply B accepts A.

        Setup: A has distance threshold 0.3, B has threshold 0.7
        Distance between them: 0.5
        Result: A accepts B (0.5 >= 0.3), B rejects A (0.5 < 0.7)
        """
        # Individual A's matchability
        func_a = MatchabilityFunction(
            type="distance_threshold",
            params={"min_distance": 0.3},
            active=True,
        )
        # Individual B's matchability
        func_b = MatchabilityFunction(
            type="distance_threshold",
            params={"min_distance": 0.7},
            active=True,
        )

        # Context for A evaluating B (same distance either way)
        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )

        # A evaluates B
        a_accepts_b, _ = safe_evaluate_matchability(func_a, context, rng)
        # B evaluates A
        b_accepts_a, _ = safe_evaluate_matchability(func_b, context, rng)

        assert a_accepts_b is True, "A should accept B (0.5 >= 0.3)"
        assert b_accepts_a is False, "B should reject A (0.5 < 0.7)"
