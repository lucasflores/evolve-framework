# Protocol Interfaces: Evolvable Reproduction Protocols (ERP)

"""
Protocol interface contracts for ERP.

These Protocol classes define the interfaces that must be implemented
for custom matchability functions, intent policies, and crossover protocols.

All implementations must:
- Accept explicit RNG for determinism
- Be serializable (for checkpointing)
- Not access global mutable state
- Complete within step limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.core.types import Individual

G = TypeVar("G")  # Genome type


# =============================================================================
# Context Dataclasses
# =============================================================================


@dataclass(frozen=True)
class MateContext:
    """
    Context for matchability evaluation.

    Provides read-only information about potential mate and population state.
    """

    partner_distance: float
    partner_fitness_rank: int
    partner_fitness_ratio: float
    partner_niche_id: int | None
    population_diversity: float
    crowding_distance: float | None
    self_fitness: np.ndarray
    partner_fitness: np.ndarray


@dataclass(frozen=True)
class IntentContext:
    """
    Context for intent policy evaluation.
    """

    fitness: np.ndarray
    fitness_rank: int
    age: int
    offspring_count: int
    generation: int
    population_size: int


# =============================================================================
# Step Limiting
# =============================================================================


class StepLimitExceeded(Exception):
    """Raised when protocol execution exceeds step limit."""

    def __init__(self, count: int, limit: int):
        self.count = count
        self.limit = limit
        super().__init__(f"Step limit exceeded: {count} > {limit}")


@dataclass
class StepCounter:
    """
    Counts execution steps and enforces limits.

    Thread-safe within a single protocol evaluation.
    """

    limit: int = 1000
    count: int = 0

    def step(self, n: int = 1) -> None:
        """Increment counter; raise StepLimitExceeded if limit exceeded."""
        self.count += n
        if self.count > self.limit:
            raise StepLimitExceeded(self.count, self.limit)

    def reset(self) -> None:
        """Reset counter for reuse."""
        self.count = 0


# =============================================================================
# Protocol Interfaces
# =============================================================================


@runtime_checkable
class MatchabilityEvaluator(Protocol):
    """
    Protocol for evaluating mate acceptability.

    Implementations must be deterministic given the same inputs and RNG state.
    """

    def evaluate(
        self,
        context: MateContext,
        rng: Random,
        counter: StepCounter,
    ) -> bool | float:
        """
        Evaluate whether a potential mate is acceptable.

        Args:
            context: Information about potential mate and population
            rng: Random number generator (for probabilistic decisions)
            counter: Step counter (call counter.step() for resource accounting)

        Returns:
            bool: Accept (True) or reject (False)
            float: Acceptance probability in [0, 1]

        Raises:
            StepLimitExceeded: If evaluation exceeds step limit
        """
        ...


@runtime_checkable
class IntentEvaluator(Protocol):
    """
    Protocol for evaluating reproduction willingness.

    Implementations must be deterministic given the same inputs and RNG state.
    """

    def evaluate(
        self,
        context: IntentContext,
        rng: Random,
        counter: StepCounter,
    ) -> bool:
        """
        Evaluate whether individual is willing to reproduce.

        Args:
            context: Information about self and population
            rng: Random number generator
            counter: Step counter

        Returns:
            True if willing to attempt reproduction

        Raises:
            StepLimitExceeded: If evaluation exceeds step limit
        """
        ...


@runtime_checkable
class CrossoverExecutor(Protocol[G]):
    """
    Protocol for executing crossover between parent genomes.

    Implementations must:
    - Return new genome instances (not modify parents)
    - Be deterministic given same inputs and RNG state
    """

    def execute(
        self,
        parent1: G,
        parent2: G,
        rng: Random,
        counter: StepCounter,
    ) -> tuple[G, G]:
        """
        Create two offspring genomes from two parents.

        Args:
            parent1: First parent genome
            parent2: Second parent genome
            rng: Random number generator
            counter: Step counter

        Returns:
            Tuple of two offspring genomes

        Raises:
            StepLimitExceeded: If execution exceeds step limit
        """
        ...


@runtime_checkable
class ProtocolMutator(Protocol):
    """
    Protocol for mutating reproduction protocols.

    Responsible for evolving matchability, intent, and crossover components.
    """

    def mutate(
        self,
        protocol: ReproductionProtocol,
        rng: Random,
    ) -> ReproductionProtocol:
        """
        Create mutated copy of protocol.

        Args:
            protocol: Protocol to mutate
            rng: Random number generator

        Returns:
            New protocol with mutations applied
        """
        ...


# =============================================================================
# Core Data Structures
# =============================================================================


class CrossoverType(Enum):
    """Supported crossover strategies."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    MODULE_EXCHANGE = "module_exchange"
    CLONE = "clone"


@dataclass(frozen=True)
class MatchabilityFunction:
    """
    Evolvable matchability function specification.

    Attributes:
        type: Function type identifier
        params: Type-specific parameters
        active: Whether function is active (inactive = reject)
    """

    type: str
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
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
    Evolvable reproduction intent policy specification.

    Attributes:
        type: Policy type identifier
        params: Type-specific parameters
        active: Whether policy is active (inactive = willing)
    """

    type: str
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
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
    Evolvable crossover protocol specification.

    Attributes:
        type: Crossover strategy to use
        params: Type-specific parameters
        active: Whether protocol is active (inactive = clone)
    """

    type: CrossoverType
    params: dict[str, float] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
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


@dataclass(frozen=True)
class ReproductionProtocol:
    """
    Complete evolvable reproduction protocol.

    This is the Reproduction Protocol Genome (RPG) attached to individuals.

    Attributes:
        matchability: Function determining mate acceptability
        intent: Policy determining reproduction willingness
        crossover: Specification for offspring genome construction
        junk_data: Inactive parameters for neutral drift
    """

    matchability: MatchabilityFunction
    intent: ReproductionIntentPolicy
    crossover: CrossoverProtocolSpec
    junk_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "matchability": self.matchability.to_dict(),
            "intent": self.intent.to_dict(),
            "crossover": self.crossover.to_dict(),
            "junk_data": dict(self.junk_data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionProtocol:
        """Deserialize from dictionary."""
        return cls(
            matchability=MatchabilityFunction.from_dict(data["matchability"]),
            intent=ReproductionIntentPolicy.from_dict(data["intent"]),
            crossover=CrossoverProtocolSpec.from_dict(data["crossover"]),
            junk_data=data.get("junk_data", {}),
        )

    @classmethod
    def default(cls) -> ReproductionProtocol:
        """Create default accept-all protocol."""
        return cls(
            matchability=MatchabilityFunction(type="accept_all", params={}, active=True),
            intent=ReproductionIntentPolicy(type="always_willing", params={}, active=True),
            crossover=CrossoverProtocolSpec(
                type=CrossoverType.SINGLE_POINT,
                params={"point_ratio": 0.5},
                active=True,
            ),
            junk_data={},
        )


# =============================================================================
# Events
# =============================================================================


@dataclass(frozen=True)
class ReproductionEvent:
    """
    Record of a reproduction attempt.

    Emitted for logging, metrics, and debugging.
    """

    generation: int
    parent1_id: UUID
    parent2_id: UUID
    success: bool
    failure_reason: str | None
    offspring_ids: tuple[UUID, ...] | None
    matchability_result: tuple[bool, bool]  # (parent1_accepts, parent2_accepts)
    intent_result: tuple[bool, bool]  # (parent1_willing, parent2_willing)


# =============================================================================
# Recovery Interface
# =============================================================================


@runtime_checkable
class RecoveryStrategy(Protocol[G]):
    """
    Protocol for handling reproduction failure (zero successful matings).
    """

    def recover(
        self,
        population: Population[G],
        generation: int,
        rng: Random,
    ) -> list[Individual[G]]:
        """
        Generate replacement individuals when reproduction fails.

        Args:
            population: Current population (all rejected each other)
            generation: Current generation number
            rng: Random number generator

        Returns:
            List of immigrant individuals to inject
        """
        ...
