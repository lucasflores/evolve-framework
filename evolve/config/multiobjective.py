"""
Multi-Objective Configuration.

Provides configuration for multi-objective optimization including
objective specifications, constraints, and NSGA-II settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Specification for a single optimization objective.

    Attributes:
        name: Objective identifier (used in fitness dictionaries).
        direction: Whether to minimize or maximize this objective.
        weight: Weight for weighted-sum scalarization (optional).

    Example:
        >>> obj = ObjectiveSpec(name="accuracy", direction="maximize")
    """

    name: str
    """Objective identifier used in fitness dictionaries."""

    direction: Literal["minimize", "maximize"] = "minimize"
    """Whether to minimize or maximize this objective."""

    weight: float = 1.0
    """Weight for weighted-sum scalarization."""

    def __post_init__(self) -> None:
        """Validate objective specification."""
        if not self.name:
            raise ValueError("Objective name cannot be empty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(f"direction must be 'minimize' or 'maximize', got {self.direction}")
        if self.weight <= 0:
            raise ValueError("weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveSpec:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            direction=data.get("direction", "minimize"),
            weight=data.get("weight", 1.0),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    """
    Specification for a constraint function.

    Constraints return violation magnitude (0 = feasible, >0 = infeasible).
    The actual constraint function is provided to the evaluator, not encoded
    in configuration (cannot serialize Python callables).

    Attributes:
        name: Constraint identifier.
        penalty_weight: Weight for penalty function approach (if used).

    Example:
        >>> constraint = ConstraintSpec(name="budget_limit", penalty_weight=1.0)
    """

    name: str
    """Constraint identifier."""

    penalty_weight: float = 1.0
    """Weight for penalty function approach."""

    def __post_init__(self) -> None:
        """Validate constraint specification."""
        if not self.name:
            raise ValueError("Constraint name cannot be empty")
        if self.penalty_weight <= 0:
            raise ValueError("penalty_weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "penalty_weight": self.penalty_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintSpec:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            penalty_weight=data.get("penalty_weight", 1.0),
        )


@dataclass(frozen=True)
class MultiObjectiveConfig:
    """
    Multi-objective optimization settings.

    When present in UnifiedConfig, the factory configures NSGA-II
    selection and multi-objective fitness handling.

    Attributes:
        objectives: Tuple of objective specifications.
        reference_point: Reference point for hypervolume calculation.
        constraints: Constraint specifications for constraint dominance.
        constraint_handling: How to handle constraints in Pareto ranking.

    Example:
        >>> config = MultiObjectiveConfig(
        ...     objectives=(
        ...         ObjectiveSpec(name="accuracy", direction="maximize"),
        ...         ObjectiveSpec(name="complexity", direction="minimize"),
        ...     ),
        ...     reference_point=(1.0, 100.0),
        ... )
    """

    objectives: tuple[ObjectiveSpec, ...] = ()
    """Tuple of objective specifications."""

    reference_point: tuple[float, ...] | None = None
    """Reference point for hypervolume calculation."""

    constraints: tuple[ConstraintSpec, ...] = ()
    """Constraint specifications for constraint dominance (FR-034)."""

    constraint_handling: Literal["dominance", "penalty"] = "dominance"
    """How to handle constraints in Pareto ranking (FR-035, FR-036, FR-037)."""

    def __post_init__(self) -> None:
        """Validate multi-objective configuration."""
        if len(self.objectives) < 2:
            raise ValueError("Multi-objective requires at least 2 objectives")
        if self.reference_point is not None:
            if len(self.reference_point) != len(self.objectives):
                raise ValueError("reference_point length must match objectives")
        if self.constraint_handling not in ("dominance", "penalty"):
            raise ValueError(
                f"constraint_handling must be 'dominance' or 'penalty', "
                f"got {self.constraint_handling}"
            )

    @property
    def num_objectives(self) -> int:
        """Get number of objectives."""
        return len(self.objectives)

    @property
    def has_constraints(self) -> bool:
        """Check if constraints are specified."""
        return len(self.constraints) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "objectives": [obj.to_dict() for obj in self.objectives],
            "reference_point": list(self.reference_point) if self.reference_point else None,
            "constraints": [c.to_dict() for c in self.constraints],
            "constraint_handling": self.constraint_handling,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiObjectiveConfig:
        """Create from dictionary."""
        objectives = tuple(ObjectiveSpec.from_dict(obj) for obj in data.get("objectives", []))
        constraints = tuple(ConstraintSpec.from_dict(c) for c in data.get("constraints", []))
        ref_point = data.get("reference_point")
        return cls(
            objectives=objectives,
            reference_point=tuple(ref_point) if ref_point else None,
            constraints=constraints,
            constraint_handling=data.get("constraint_handling", "dominance"),
        )
