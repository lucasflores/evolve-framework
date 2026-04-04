"""
Intent policy evaluation for Evolvable Reproduction Protocols.

This module implements the intent system that determines when individuals
are willing to attempt reproduction, separate from mate compatibility.
Intent is evaluated before matchability.

Key Components:
- IntentEvaluator: Protocol interface for evaluators
- Built-in evaluators: AlwaysWilling, NeverWilling, FitnessThreshold, etc.
- IntentRegistry: Maps type strings to evaluator classes
- evaluate_intent: Main evaluation function with sandboxing
- safe_evaluate_intent: Exception-safe wrapper
"""

from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable

from evolve.reproduction.protocol import IntentContext, ReproductionIntentPolicy
from evolve.reproduction.sandbox import StepCounter, StepLimitExceeded

# =============================================================================
# Protocol Interface
# =============================================================================


@runtime_checkable
class IntentEvaluator(Protocol):
    """
    Protocol interface for intent evaluators.

    Implementations determine whether an individual is willing to
    attempt reproduction based on the IntentContext information.

    All evaluators must:
    - Be deterministic given the same inputs and RNG state
    - Call counter.step() for resource accounting
    - Return bool (willing/unwilling)
    """

    def evaluate(
        self,
        context: IntentContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        """
        Evaluate whether individual is willing to reproduce.

        Args:
            context: Information about self and population
            params: Type-specific parameters from ReproductionIntentPolicy
            rng: Random number generator
            counter: Step counter for resource limiting

        Returns:
            True if willing to attempt reproduction

        Raises:
            StepLimitExceeded: If evaluation exceeds step limit
        """
        ...


# =============================================================================
# Built-in Evaluators
# =============================================================================


class AlwaysWillingIntent:
    """Always willing to reproduce."""

    def evaluate(
        self,
        _context: IntentContext,
        _params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        return True


class NeverWillingIntent:
    """Never willing to reproduce."""

    def evaluate(
        self,
        _context: IntentContext,
        _params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        return False


class FitnessThresholdIntent:
    """
    Willing to reproduce only if fitness meets threshold.

    Params:
        threshold: Minimum fitness value required
    """

    def evaluate(
        self,
        context: IntentContext,
        params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        threshold = params.get("threshold", 0.0)
        # Use first fitness value for comparison (single-objective)
        fitness_val = context.fitness[0] if len(context.fitness) > 0 else 0.0
        return fitness_val >= threshold


class FitnessRankThresholdIntent:
    """
    Willing to reproduce only if fitness rank is high enough.

    Params:
        max_rank: Maximum rank to be willing (0 = best)
    """

    def evaluate(
        self,
        context: IntentContext,
        params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        max_rank = int(params.get("max_rank", 10))
        return context.fitness_rank <= max_rank


class ResourceBudgetIntent:
    """
    Willing to reproduce until offspring budget exhausted.

    Params:
        max_offspring: Maximum offspring per generation
    """

    def evaluate(
        self,
        context: IntentContext,
        params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        max_offspring = int(params.get("max_offspring", 3))
        return context.offspring_count < max_offspring


class AgeDependentIntent:
    """
    Willing to reproduce only within certain age range.

    Params:
        min_age: Minimum age to be willing
        max_age: Maximum age to be willing
    """

    def evaluate(
        self,
        context: IntentContext,
        params: dict[str, float],
        _rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        min_age = int(params.get("min_age", 0))
        max_age = int(params.get("max_age", 1000))
        return min_age <= context.age <= max_age


class ProbabilisticIntent:
    """
    Willing to reproduce with fixed probability.

    Params:
        probability: Probability of being willing (0-1)
    """

    def evaluate(
        self,
        _context: IntentContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        counter.step()
        probability = params.get("probability", 0.5)
        return rng.random() < probability


# =============================================================================
# Registry
# =============================================================================


class IntentRegistry:
    """
    Maps intent type strings to evaluator instances.

    Built-in types are registered by default. Custom evaluators
    can be registered at runtime.
    """

    _evaluators: dict[str, IntentEvaluator] = {}

    @classmethod
    def register(cls, type_name: str, evaluator: IntentEvaluator) -> None:
        """Register an evaluator for a type name."""
        cls._evaluators[type_name] = evaluator

    @classmethod
    def get(cls, type_name: str) -> IntentEvaluator | None:
        """Get evaluator for type name, or None if not found."""
        return cls._evaluators.get(type_name)

    @classmethod
    def get_or_default(cls, type_name: str) -> IntentEvaluator:
        """Get evaluator for type name, or AlwaysWilling if not found."""
        return cls._evaluators.get(type_name, AlwaysWillingIntent())

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered type names."""
        return list(cls._evaluators.keys())


# Register built-in evaluators
IntentRegistry.register("always_willing", AlwaysWillingIntent())
IntentRegistry.register("never_willing", NeverWillingIntent())
IntentRegistry.register("fitness_threshold", FitnessThresholdIntent())
IntentRegistry.register("fitness_rank_threshold", FitnessRankThresholdIntent())
IntentRegistry.register("resource_budget", ResourceBudgetIntent())
IntentRegistry.register("age_dependent", AgeDependentIntent())
IntentRegistry.register("probabilistic", ProbabilisticIntent())


# =============================================================================
# Evaluation Functions
# =============================================================================


def evaluate_intent(
    policy: ReproductionIntentPolicy,
    context: IntentContext,
    rng: Random,
    counter: StepCounter,
) -> bool:
    """
    Evaluate an intent policy.

    Args:
        policy: The intent policy specification
        context: Intent context with individual information
        rng: Random number generator
        counter: Step counter for resource limiting

    Returns:
        True if willing to reproduce

    Raises:
        StepLimitExceeded: If evaluation exceeds step limit
        ValueError: If policy type is not registered
    """
    # Inactive policies = always willing
    if not policy.active:
        counter.step()
        return True

    evaluator = IntentRegistry.get(policy.type)
    if evaluator is None:
        raise ValueError(f"Unknown intent type: {policy.type}")

    return evaluator.evaluate(context, policy.params, rng, counter)


def safe_evaluate_intent(
    policy: ReproductionIntentPolicy,
    context: IntentContext,
    rng: Random,
    step_limit: int = 1000,
) -> tuple[bool, bool]:
    """
    Safely evaluate intent with exception handling.

    If evaluation fails (step limit, unknown type, any exception),
    returns True (willing) as the safe default to avoid blocking
    reproduction entirely.

    Args:
        policy: The intent policy specification
        context: Intent context with individual information
        rng: Random number generator
        step_limit: Maximum steps allowed

    Returns:
        Tuple of (result, success):
        - result: True if willing, False if unwilling or error
        - success: True if evaluation completed normally
    """
    counter = StepCounter(limit=step_limit)
    try:
        result = evaluate_intent(policy, context, rng, counter)
        return result, True
    except StepLimitExceeded:
        return True, False  # Default to willing on timeout
    except Exception:
        return True, False  # Default to willing on error
