"""
Crossover protocol execution for Evolvable Reproduction Protocols.

This module implements the crossover system that determines how offspring
genomes are constructed from parent genomes. Different crossover strategies
can be encoded and evolved.

Key Components:
- CrossoverExecutor: Protocol interface for executors
- Built-in executors: SinglePoint, TwoPoint, Uniform, Blend, Clone
- CrossoverRegistry: Maps type enums to executor classes
- execute_crossover: Main execution function with sandboxing
- inherit_protocol: Protocol inheritance logic (50/50 single-parent)
- validate_offspring: Offspring validation before population entry
"""

from __future__ import annotations

from random import Random
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from evolve.reproduction.protocol import CrossoverProtocolSpec, CrossoverType, ReproductionProtocol
from evolve.reproduction.sandbox import StepCounter, StepLimitExceeded

G = TypeVar("G")


# =============================================================================
# Protocol Interface
# =============================================================================


@runtime_checkable
class CrossoverExecutor(Protocol[G]):
    """
    Protocol interface for crossover executors.

    Implementations create offspring genomes from two parent genomes.
    All executors must:
    - Return new genome instances (not modify parents)
    - Be deterministic given same inputs and RNG state
    - Call counter.step() for resource accounting
    """

    def execute(
        self,
        parent1: G,
        parent2: G,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[G, G]:
        """
        Create two offspring genomes from two parents.

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            params: Type-specific parameters from CrossoverProtocolSpec
            rng: Random number generator
            counter: Step counter for resource limiting

        Returns:
            Tuple of two offspring genomes

        Raises:
            StepLimitExceeded: If execution exceeds step limit
        """
        ...


# =============================================================================
# Built-in Executors (for array-like genomes)
# =============================================================================


class SinglePointCrossoverExecutor:
    """
    Single-point crossover for array-like genomes.

    Splits genomes at a single point and swaps tails.

    Params:
        point_ratio: Relative position (0-1) for crossover point
    """

    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        counter.step()

        # Handle different genome lengths
        min_len = min(len(parent1), len(parent2))
        if min_len == 0:
            return parent1.copy(), parent2.copy()

        point_ratio = params.get("point_ratio", 0.5)
        point = int(point_ratio * min_len)
        point = max(0, min(point, min_len))  # Clamp

        counter.step()

        # Create offspring
        offspring1 = np.concatenate([parent1[:point], parent2[point:min_len]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:min_len]])

        return offspring1, offspring2


class TwoPointCrossoverExecutor:
    """
    Two-point crossover for array-like genomes.

    Swaps the segment between two points.

    Params:
        point1_ratio: First crossover point (0-1)
        point2_ratio: Second crossover point (0-1)
    """

    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        counter.step()

        min_len = min(len(parent1), len(parent2))
        if min_len == 0:
            return parent1.copy(), parent2.copy()

        point1_ratio = params.get("point1_ratio", 0.33)
        point2_ratio = params.get("point2_ratio", 0.67)

        point1 = int(point1_ratio * min_len)
        point2 = int(point2_ratio * min_len)

        # Ensure point1 < point2
        if point1 > point2:
            point1, point2 = point2, point1

        point1 = max(0, min(point1, min_len))
        point2 = max(0, min(point2, min_len))

        counter.step()

        offspring1 = np.concatenate(
            [parent1[:point1], parent2[point1:point2], parent1[point2:min_len]]
        )
        offspring2 = np.concatenate(
            [parent2[:point1], parent1[point1:point2], parent2[point2:min_len]]
        )

        return offspring1, offspring2


class UniformCrossoverExecutor:
    """
    Uniform crossover for array-like genomes.

    Each gene is independently swapped with probability swap_prob.

    Params:
        swap_prob: Probability of swapping each gene (0-1)
    """

    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        counter.step()

        min_len = min(len(parent1), len(parent2))
        if min_len == 0:
            return parent1.copy(), parent2.copy()

        swap_prob = params.get("swap_prob", 0.5)

        offspring1 = parent1[:min_len].copy()
        offspring2 = parent2[:min_len].copy()

        for i in range(min_len):
            counter.step()
            if rng.random() < swap_prob:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

        return offspring1, offspring2


class BlendCrossoverExecutor:
    """
    BLX-alpha blend crossover for real-valued genomes.

    Creates offspring by interpolating between parent values
    with some extension beyond the parent range.

    Params:
        alpha: Extension factor (0 = pure interpolation, 0.5 = typical)
    """

    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        counter.step()

        min_len = min(len(parent1), len(parent2))
        if min_len == 0:
            return parent1.copy(), parent2.copy()

        alpha = params.get("alpha", 0.5)

        offspring1 = np.zeros(min_len)
        offspring2 = np.zeros(min_len)

        for i in range(min_len):
            counter.step()
            p1, p2 = parent1[i], parent2[i]
            min_val, max_val = min(p1, p2), max(p1, p2)
            range_val = max_val - min_val

            # Extend range by alpha on each side
            low = min_val - alpha * range_val
            high = max_val + alpha * range_val

            offspring1[i] = low + rng.random() * (high - low)
            offspring2[i] = low + rng.random() * (high - low)

        return offspring1, offspring2


class CloneCrossoverExecutor:
    """
    Clone (no-op) crossover.

    Simply returns copies of the parents. Used as fallback when
    crossover is inactive or fails.
    """

    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        counter.step()
        return parent1.copy(), parent2.copy()


# =============================================================================
# Registry
# =============================================================================


class CrossoverRegistry:
    """
    Maps CrossoverType enum values to executor instances.

    Built-in types are registered by default. Custom executors
    can be registered at runtime.
    """

    _executors: dict[CrossoverType, CrossoverExecutor[Any]] = {}

    @classmethod
    def register(cls, ctype: CrossoverType, executor: CrossoverExecutor[Any]) -> None:
        """Register an executor for a crossover type."""
        cls._executors[ctype] = executor

    @classmethod
    def get(cls, ctype: CrossoverType) -> CrossoverExecutor[Any] | None:
        """Get executor for type, or None if not found."""
        return cls._executors.get(ctype)

    @classmethod
    def get_or_default(cls, ctype: CrossoverType) -> CrossoverExecutor[Any]:
        """Get executor for type, or Clone (no-op) if not found."""
        return cls._executors.get(ctype, CloneCrossoverExecutor())

    @classmethod
    def list_types(cls) -> list[CrossoverType]:
        """List all registered crossover types."""
        return list(cls._executors.keys())


# Register built-in executors
CrossoverRegistry.register(CrossoverType.SINGLE_POINT, SinglePointCrossoverExecutor())
CrossoverRegistry.register(CrossoverType.TWO_POINT, TwoPointCrossoverExecutor())
CrossoverRegistry.register(CrossoverType.UNIFORM, UniformCrossoverExecutor())
CrossoverRegistry.register(CrossoverType.BLEND, BlendCrossoverExecutor())
CrossoverRegistry.register(CrossoverType.CLONE, CloneCrossoverExecutor())
# MODULE_EXCHANGE not implemented yet (for graph genomes)


# =============================================================================
# Execution Functions
# =============================================================================


def execute_crossover(
    spec: CrossoverProtocolSpec,
    parent1: G,
    parent2: G,
    rng: Random,
    counter: StepCounter,
) -> tuple[G, G]:
    """
    Execute a crossover protocol.

    Args:
        spec: The crossover protocol specification
        parent1: First parent genome
        parent2: Second parent genome
        rng: Random number generator
        counter: Step counter for resource limiting

    Returns:
        Tuple of two offspring genomes

    Raises:
        StepLimitExceeded: If execution exceeds step limit
        ValueError: If crossover type is not registered
    """
    # Inactive protocol = clone
    if not spec.active:
        counter.step()
        if hasattr(parent1, "copy"):
            return parent1.copy(), parent2.copy()  # type: ignore
        return parent1, parent2

    executor = CrossoverRegistry.get(spec.type)
    if executor is None:
        raise ValueError(f"Unknown crossover type: {spec.type}")

    return executor.execute(parent1, parent2, spec.params, rng, counter)


def safe_execute_crossover(
    spec: CrossoverProtocolSpec,
    parent1: G,
    parent2: G,
    rng: Random,
    step_limit: int = 1000,
) -> tuple[tuple[G, G], bool]:
    """
    Safely execute crossover with exception handling.

    If execution fails, returns clones of parents as fallback.

    Args:
        spec: The crossover protocol specification
        parent1: First parent genome
        parent2: Second parent genome
        rng: Random number generator
        step_limit: Maximum steps allowed

    Returns:
        Tuple of ((offspring1, offspring2), success)
    """
    counter = StepCounter(limit=step_limit)
    try:
        result = execute_crossover(spec, parent1, parent2, rng, counter)
        return result, True
    except (StepLimitExceeded, ValueError, Exception):
        # Fallback to cloning
        if hasattr(parent1, "copy"):
            return (parent1.copy(), parent2.copy()), False  # type: ignore
        return (parent1, parent2), False


# =============================================================================
# Protocol Inheritance
# =============================================================================


def inherit_protocol(
    parent1_protocol: ReproductionProtocol | None,
    parent2_protocol: ReproductionProtocol | None,
    rng: Random,
) -> ReproductionProtocol | None:
    """
    Determine offspring protocol via 50/50 single-parent inheritance.

    Args:
        parent1_protocol: First parent's protocol (may be None)
        parent2_protocol: Second parent's protocol (may be None)
        rng: Random number generator

    Returns:
        Protocol inherited from one parent, or None if both parents have None
    """
    # If both None, offspring has None
    if parent1_protocol is None and parent2_protocol is None:
        return None

    # If one is None, inherit from the other
    if parent1_protocol is None:
        return parent2_protocol
    if parent2_protocol is None:
        return parent1_protocol

    # Both have protocols - random selection
    if rng.random() < 0.5:
        return parent1_protocol
    return parent2_protocol


# =============================================================================
# Offspring Validation
# =============================================================================


def validate_offspring(
    offspring: G,
    parent1: G,
    parent2: G,
) -> tuple[bool, str | None]:
    """
    Validate offspring genome before population entry.

    Checks:
    - Offspring is not None
    - Offspring has same type as parents
    - For numpy arrays: same shape, numeric dtype, no NaN or inf values

    Args:
        offspring: The offspring genome to validate
        parent1: First parent (for type comparison)
        parent2: Second parent (for type comparison)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if offspring is None:
        return False, "Offspring is None"

    if type(offspring) != type(parent1):
        return False, f"Type mismatch: {type(offspring)} vs {type(parent1)}"

    # For numpy arrays, check for invalid values
    if isinstance(offspring, np.ndarray):
        # Check shape matches
        if hasattr(parent1, "shape") and offspring.shape != parent1.shape:
            return False, f"Shape mismatch: {offspring.shape} vs {parent1.shape}"

        # Check for numeric dtype
        if not np.issubdtype(offspring.dtype, np.number):
            return False, f"Non-numeric dtype: {offspring.dtype}"

        if not np.all(np.isfinite(offspring)):
            return False, "Offspring contains NaN or inf values"

    return True, None
