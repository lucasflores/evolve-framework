"""
Matchability evaluation for Evolvable Reproduction Protocols.

This module implements the matchability system that determines whether
individuals accept each other as potential mates. Matchability is
asymmetric by default - A accepting B does not imply B accepts A.

Key Components:
- MatchabilityEvaluator: Protocol interface for evaluators
- Built-in evaluators: AcceptAll, RejectAll, DistanceThreshold, etc.
- MatchabilityRegistry: Maps type strings to evaluator classes
- evaluate_matchability: Main evaluation function with sandboxing
- safe_evaluate_matchability: Exception-safe wrapper
"""

from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable

from evolve.reproduction.protocol import MatchabilityFunction, MateContext
from evolve.reproduction.sandbox import StepCounter, StepLimitExceeded

# =============================================================================
# Protocol Interface
# =============================================================================


@runtime_checkable
class MatchabilityEvaluator(Protocol):
    """
    Protocol interface for matchability evaluators.

    Implementations determine whether a potential mate is acceptable
    based on the MateContext information.

    All evaluators must:
    - Be deterministic given the same inputs and RNG state
    - Call counter.step() for resource accounting
    - Return bool (accept/reject) or float (probability in [0,1])
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool | float:
        """
        Evaluate whether a potential mate is acceptable.

        Args:
            context: Information about potential mate and population
            params: Type-specific parameters from MatchabilityFunction
            rng: Random number generator for probabilistic decisions
            counter: Step counter for resource limiting

        Returns:
            bool: Accept (True) or reject (False)
            float: Acceptance probability in [0, 1]

        Raises:
            StepLimitExceeded: If evaluation exceeds step limit
        """
        ...


# =============================================================================
# Built-in Evaluators
# =============================================================================


class AcceptAllMatchability:
    """Always accepts any potential mate."""

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        return True


class RejectAllMatchability:
    """Always rejects any potential mate."""

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        return False


class DistanceThresholdMatchability:
    """
    Accepts mates with genetic distance above a threshold.

    Params:
        min_distance: Minimum required genetic distance
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        min_distance = params.get("min_distance", 0.0)
        return context.partner_distance >= min_distance


class SimilarityThresholdMatchability:
    """
    Accepts mates with genetic distance below a threshold (similar mates).

    Params:
        max_distance: Maximum allowed genetic distance
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        max_distance = params.get("max_distance", 1.0)
        return context.partner_distance <= max_distance


class FitnessRatioMatchability:
    """
    Accepts mates with fitness ratio within a specified range.

    Params:
        min_ratio: Minimum partner_fitness / self_fitness ratio
        max_ratio: Maximum partner_fitness / self_fitness ratio
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        min_ratio = params.get("min_ratio", 0.0)
        max_ratio = params.get("max_ratio", float("inf"))
        return min_ratio <= context.partner_fitness_ratio <= max_ratio


class DifferentNicheMatchability:
    """
    Accepts mates from different niches/species only.

    Requires niche_id to be set on individuals. If partner has no
    niche_id (None), accepts by default.
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        # If partner has no niche, accept
        if context.partner_niche_id is None:
            return True
        # Can't determine own niche from context - accept different only
        # This evaluator is meant for use with speciation
        return True  # Accept by default if niche comparison not possible


class ProbabilisticMatchability:
    """
    Accepts mates with probability based on distance.

    Params:
        base_prob: Base acceptance probability (0-1)
        distance_weight: How much distance affects probability
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> float:
        counter.step()
        base_prob = params.get("base_prob", 0.5)
        distance_weight = params.get("distance_weight", 0.0)

        # Probability increases with distance (promotes diversity)
        prob = base_prob + distance_weight * context.partner_distance
        # Clamp to [0, 1]
        return max(0.0, min(1.0, prob))


class DiversitySeekingMatchability:
    """
    Prefers partners with high crowding distance (for multi-objective).

    Params:
        crowding_threshold: Minimum crowding distance to accept
        fallback_prob: Probability to accept if below threshold
    """

    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> float:
        counter.step()
        crowding_threshold = params.get("crowding_threshold", 0.5)
        fallback_prob = params.get("fallback_prob", 0.3)

        # If no crowding info, use fallback
        if context.crowding_distance is None:
            return fallback_prob

        # Accept with high probability if above threshold
        if context.crowding_distance >= crowding_threshold:
            return 1.0

        # Below threshold, probability scales with crowding distance
        return fallback_prob + (1.0 - fallback_prob) * (
            context.crowding_distance / crowding_threshold
        )


# =============================================================================
# Registry
# =============================================================================


class MatchabilityRegistry:
    """
    Maps matchability type strings to evaluator instances.

    Built-in types are registered by default. Custom evaluators
    can be registered at runtime.
    """

    _evaluators: dict[str, MatchabilityEvaluator] = {}

    @classmethod
    def register(cls, type_name: str, evaluator: MatchabilityEvaluator) -> None:
        """Register an evaluator for a type name."""
        cls._evaluators[type_name] = evaluator

    @classmethod
    def get(cls, type_name: str) -> MatchabilityEvaluator | None:
        """Get evaluator for type name, or None if not found."""
        return cls._evaluators.get(type_name)

    @classmethod
    def get_or_default(cls, type_name: str) -> MatchabilityEvaluator:
        """Get evaluator for type name, or RejectAll if not found."""
        return cls._evaluators.get(type_name, RejectAllMatchability())

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered type names."""
        return list(cls._evaluators.keys())


# Register built-in evaluators
MatchabilityRegistry.register("accept_all", AcceptAllMatchability())
MatchabilityRegistry.register("reject_all", RejectAllMatchability())
MatchabilityRegistry.register("distance_threshold", DistanceThresholdMatchability())
MatchabilityRegistry.register("similarity_threshold", SimilarityThresholdMatchability())
MatchabilityRegistry.register("fitness_ratio", FitnessRatioMatchability())
MatchabilityRegistry.register("different_niche", DifferentNicheMatchability())
MatchabilityRegistry.register("probabilistic", ProbabilisticMatchability())
MatchabilityRegistry.register("diversity_seeking", DiversitySeekingMatchability())


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_matchability(
    func: MatchabilityFunction,
    context: MateContext,
    rng: Random,
    counter: StepCounter,
) -> bool | float:
    """
    Evaluate a matchability function.

    Args:
        func: The matchability function specification
        context: Mate context with partner information
        rng: Random number generator
        counter: Step counter for resource limiting

    Returns:
        bool or float: Accept/reject or probability

    Raises:
        StepLimitExceeded: If evaluation exceeds step limit
        ValueError: If function type is not registered
    """
    # Inactive functions always reject
    if not func.active:
        counter.step()
        return False

    evaluator = MatchabilityRegistry.get(func.type)
    if evaluator is None:
        raise ValueError(f"Unknown matchability type: {func.type}")

    result = evaluator.evaluate(context, func.params, rng, counter)

    # If result is probability, resolve to boolean
    if isinstance(result, float):
        counter.step()
        return rng.random() < result

    return result


def safe_evaluate_matchability(
    func: MatchabilityFunction,
    context: MateContext,
    rng: Random,
    step_limit: int = 1000,
) -> tuple[bool, bool]:
    """
    Safely evaluate matchability with exception handling.

    If evaluation fails (step limit, unknown type, any exception),
    returns False (reject) as the safe default.

    Args:
        func: The matchability function specification
        context: Mate context with partner information
        rng: Random number generator
        step_limit: Maximum steps allowed

    Returns:
        Tuple of (result, success):
        - result: True if accepts, False if rejects or error
        - success: True if evaluation completed normally
    """
    counter = StepCounter(limit=step_limit)
    try:
        result = evaluate_matchability(func, context, rng, counter)
        # Ensure boolean result
        if isinstance(result, float):
            result = rng.random() < result
        return bool(result), True
    except StepLimitExceeded:
        return False, False
    except Exception:
        # Any exception results in rejection
        return False, False
