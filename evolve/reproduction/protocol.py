"""
Core protocol dataclasses for Evolvable Reproduction Protocols.

This module defines the core data structures for the Reproduction Protocol Genome (RPG):
- MatchabilityFunction: Determines mate acceptability
- ReproductionIntentPolicy: Governs when reproduction is attempted  
- CrossoverProtocolSpec: Specifies offspring genome construction
- ReproductionProtocol: The complete evolvable protocol
- MateContext/IntentContext: Evaluation contexts
- ReproductionEvent: Observability record
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np


# =============================================================================
# Enums
# =============================================================================


class CrossoverType(Enum):
    """Supported crossover strategies."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    MODULE_EXCHANGE = "module_exchange"
    CLONE = "clone"  # No-op fallback


# =============================================================================
# Context Dataclasses
# =============================================================================


@dataclass(frozen=True)
class MateContext:
    """
    Context for matchability evaluation.

    Provides read-only information about potential mate and population state.
    All fields are immutable to prevent side effects during evaluation.

    Attributes:
        partner_distance: Genetic distance to potential mate
        partner_fitness_rank: Partner's rank in population (0 = best)
        partner_fitness_ratio: Partner fitness / self fitness
        partner_niche_id: Partner's species/niche label (None if N/A)
        population_diversity: Current population diversity metric (0-1)
        crowding_distance: Multi-objective crowding distance (None if single-obj)
        self_fitness: Own fitness value(s)
        partner_fitness: Partner's fitness value(s)
    """

    partner_distance: float
    partner_fitness_rank: int
    partner_fitness_ratio: float
    partner_niche_id: int | None
    population_diversity: float
    crowding_distance: float | None
    self_fitness: np.ndarray
    partner_fitness: np.ndarray

    def __post_init__(self) -> None:
        """Make numpy arrays immutable."""
        if isinstance(self.self_fitness, np.ndarray):
            self.self_fitness.flags.writeable = False
        if isinstance(self.partner_fitness, np.ndarray):
            self.partner_fitness.flags.writeable = False


@dataclass(frozen=True)
class IntentContext:
    """
    Context for intent policy evaluation.

    Provides read-only information about the individual and population state.

    Attributes:
        fitness: Own fitness value(s)
        fitness_rank: Own rank in population (0 = best)
        age: Number of generations survived
        offspring_count: Offspring produced this generation
        generation: Current generation number
        population_size: Current population size
    """

    fitness: np.ndarray
    fitness_rank: int
    age: int
    offspring_count: int
    generation: int
    population_size: int

    def __post_init__(self) -> None:
        """Make numpy arrays immutable."""
        if isinstance(self.fitness, np.ndarray):
            self.fitness.flags.writeable = False


# =============================================================================
# Protocol Component Dataclasses
# =============================================================================


@dataclass(frozen=True)
class MatchabilityFunction:
    """
    Evolvable function determining mate acceptability.

    Encodes the logic for evaluating whether a potential mate is acceptable.
    The actual evaluation is performed by a MatchabilityEvaluator that
    interprets the type and params.

    Attributes:
        type: Function type identifier (e.g., "accept_all", "distance_threshold")
        params: Type-specific parameters (e.g., {"min_distance": 0.5})
        active: Whether this function is active (inactive = always reject)

    Supported Types:
        - accept_all: Always accept
        - reject_all: Always reject
        - distance_threshold: Accept if distance > min_distance
        - similarity_threshold: Accept if distance < max_distance
        - fitness_ratio: Accept if ratio in [min_ratio, max_ratio]
        - different_niche: Accept if different niche_id
        - probabilistic: Probability based on distance
        - diversity_seeking: Prefer partners with high crowding distance

    Example:
        >>> func = MatchabilityFunction(
        ...     type="distance_threshold",
        ...     params={"min_distance": 0.3},
        ...     active=True
        ... )
    """

    type: str
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "type": self.type,
            "params": dict(self.params),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchabilityFunction:
        """Deserialize from dictionary."""
        return cls(
            type=data["type"],
            params=data.get("params", {}),
            active=data.get("active", True),
        )


@dataclass(frozen=True)
class ReproductionIntentPolicy:
    """
    Evolvable policy determining reproduction willingness.

    Encodes when an individual attempts to reproduce, separate from
    mate compatibility. The actual evaluation is performed by an
    IntentEvaluator that interprets the type and params.

    Attributes:
        type: Policy type identifier
        params: Type-specific parameters
        active: Whether policy is active (inactive = always willing)

    Supported Types:
        - always_willing: Always attempt reproduction
        - never_willing: Never attempt reproduction
        - fitness_threshold: Willing if fitness >= threshold
        - fitness_rank_threshold: Willing if rank <= max_rank
        - resource_budget: Willing until budget exhausted
        - age_dependent: Willing if age in [min_age, max_age]
        - probabilistic: Willing with fixed probability

    Example:
        >>> policy = ReproductionIntentPolicy(
        ...     type="fitness_threshold",
        ...     params={"threshold": 0.5},
        ...     active=True
        ... )
    """

    type: str
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "type": self.type,
            "params": dict(self.params),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionIntentPolicy:
        """Deserialize from dictionary."""
        return cls(
            type=data["type"],
            params=data.get("params", {}),
            active=data.get("active", True),
        )


@dataclass(frozen=True)
class CrossoverProtocolSpec:
    """
    Evolvable specification for offspring genome construction.

    Encodes how genetic material from two parents is combined to
    produce offspring. The actual crossover is performed by a
    CrossoverExecutor that interprets the type and params.

    Attributes:
        type: Crossover strategy to use
        params: Type-specific parameters
        active: Whether protocol is active (inactive = clone)

    Type-Specific Parameters:
        - SINGLE_POINT: point_ratio (0-1, relative position)
        - TWO_POINT: point1_ratio, point2_ratio
        - UNIFORM: swap_prob
        - BLEND: alpha (BLX-alpha parameter)
        - MODULE_EXCHANGE: module_prob (for graph genomes)
        - CLONE: (none)

    Example:
        >>> spec = CrossoverProtocolSpec(
        ...     type=CrossoverType.UNIFORM,
        ...     params={"swap_prob": 0.5},
        ...     active=True
        ... )
    """

    type: CrossoverType
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "type": self.type.value,
            "params": dict(self.params),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossoverProtocolSpec:
        """Deserialize from dictionary."""
        return cls(
            type=CrossoverType(data["type"]),
            params=data.get("params", {}),
            active=data.get("active", True),
        )


# =============================================================================
# Main Protocol Dataclass
# =============================================================================


@dataclass(frozen=True)
class ReproductionProtocol:
    """
    Complete evolvable reproduction protocol.

    This is the Reproduction Protocol Genome (RPG) that can be attached
    to individuals. It encodes all aspects of reproductive behavior:
    - Who is an acceptable mate (matchability)
    - When to attempt reproduction (intent)
    - How to combine genetic material (crossover)

    The protocol is immutable (frozen dataclass) to ensure it cannot be
    modified during evolution. New protocols are created via mutation
    or inheritance.

    Attributes:
        matchability: Function determining mate acceptability
        intent: Policy determining reproduction willingness
        crossover: Specification for offspring genome construction
        junk_data: Inactive parameters for neutral drift (not used in evaluation)

    Example:
        >>> protocol = ReproductionProtocol.default()
        >>> protocol.matchability.type
        'accept_all'
        >>> protocol.intent.type
        'always_willing'
    """

    matchability: MatchabilityFunction
    intent: ReproductionIntentPolicy
    crossover: CrossoverProtocolSpec
    junk_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary for checkpointing.

        Returns:
            JSON-serializable dictionary representation
        """
        return {
            "matchability": self.matchability.to_dict(),
            "intent": self.intent.to_dict(),
            "crossover": self.crossover.to_dict(),
            "junk_data": dict(self.junk_data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionProtocol:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed ReproductionProtocol
        """
        return cls(
            matchability=MatchabilityFunction.from_dict(data["matchability"]),
            intent=ReproductionIntentPolicy.from_dict(data["intent"]),
            crossover=CrossoverProtocolSpec.from_dict(data["crossover"]),
            junk_data=data.get("junk_data", {}),
        )

    @classmethod
    def default(cls) -> ReproductionProtocol:
        """
        Create default accept-all protocol.

        This is the baseline protocol used when no explicit protocol
        is assigned. It accepts all mates, is always willing to reproduce,
        and uses single-point crossover.

        Returns:
            Default ReproductionProtocol instance
        """
        return cls(
            matchability=MatchabilityFunction(
                type="accept_all",
                params={},
                active=True,
            ),
            intent=ReproductionIntentPolicy(
                type="always_willing",
                params={},
                active=True,
            ),
            crossover=CrossoverProtocolSpec(
                type=CrossoverType.SINGLE_POINT,
                params={"point_ratio": 0.5},
                active=True,
            ),
            junk_data={},
        )


# =============================================================================
# Event Dataclass
# =============================================================================


@dataclass(frozen=True)
class ReproductionEvent:
    """
    Record of a reproduction attempt.

    Emitted for logging, metrics collection, and debugging. Captures
    all information about a mating attempt including success/failure
    and the reasons.

    Attributes:
        generation: When the event occurred
        parent1_id: First parent UUID
        parent2_id: Second parent UUID
        success: Whether reproduction produced offspring
        failure_reason: Why reproduction failed (if applicable)
        offspring_ids: UUIDs of produced offspring (if successful)
        matchability_result: (parent1_accepts, parent2_accepts)
        intent_result: (parent1_willing, parent2_willing)
    """

    generation: int
    parent1_id: UUID
    parent2_id: UUID
    success: bool
    failure_reason: str | None
    offspring_ids: tuple[UUID, ...] | None
    matchability_result: tuple[bool, bool]
    intent_result: tuple[bool, bool]
