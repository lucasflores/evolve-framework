"""
Multi-Objective Fitness representation.

Extends single-objective fitness to handle multiple objectives
with optional constraint handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiObjectiveFitness:
    """
    Fitness for multi-objective optimization.

    All objectives follow MAXIMIZATION convention.
    To minimize an objective, negate its value before creating fitness.

    Attributes:
        objectives: Array of objective values, shape (n_objectives,)
        constraint_violations: Optional constraint violation values, shape (n_constraints,)
            Positive values indicate constraint violation magnitude.
            Zero or negative means constraint is satisfied.
        metadata: Optional additional metadata

    Example:
        >>> # Two objectives: maximize f1, minimize f2 (negate)
        >>> fitness = MultiObjectiveFitness(
        ...     objectives=np.array([f1_value, -f2_value])
        ... )
        >>>
        >>> # With constraints: g1(x) <= 0, g2(x) <= 0
        >>> constrained = MultiObjectiveFitness(
        ...     objectives=np.array([f1, f2]),
        ...     constraint_violations=np.array([g1_value, g2_value])
        ... )
    """

    objectives: np.ndarray
    constraint_violations: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Make arrays immutable after creation."""
        # Ensure objectives is a numpy array
        if not isinstance(self.objectives, np.ndarray):
            object.__setattr__(self, "objectives", np.asarray(self.objectives))

        # Make objectives read-only
        self.objectives.flags.writeable = False

        # Handle constraint violations
        if self.constraint_violations is not None:
            if not isinstance(self.constraint_violations, np.ndarray):
                object.__setattr__(
                    self, "constraint_violations", np.asarray(self.constraint_violations)
                )
            self.constraint_violations.flags.writeable = False

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def n_constraints(self) -> int:
        """Number of constraints (0 if unconstrained)."""
        if self.constraint_violations is None:
            return 0
        return len(self.constraint_violations)

    @property
    def is_feasible(self) -> bool:
        """
        Check if all constraints are satisfied.

        Returns True if:
        - No constraints defined, OR
        - All constraint violations <= 0
        """
        if self.constraint_violations is None:
            return True
        return bool(np.all(self.constraint_violations <= 0))

    @property
    def total_constraint_violation(self) -> float:
        """
        Sum of positive constraint violations.

        Returns 0.0 if feasible or no constraints.
        """
        if self.constraint_violations is None:
            return 0.0
        return float(np.sum(np.maximum(self.constraint_violations, 0)))

    @property
    def values(self) -> tuple[float, ...]:
        """Objective values as tuple (for compatibility with single-objective)."""
        return tuple(self.objectives.tolist())

    def __eq__(self, other: object) -> bool:
        """Equality based on objective values only."""
        if not isinstance(other, MultiObjectiveFitness):
            return False
        return np.array_equal(self.objectives, other.objectives)

    def __hash__(self) -> int:
        """Hash based on objective values."""
        return hash(self.objectives.tobytes())

    def __repr__(self) -> str:
        """String representation."""
        obj_str = np.array2string(self.objectives, precision=4, separator=", ")
        if self.constraint_violations is not None:
            cv_str = np.array2string(self.constraint_violations, precision=4, separator=", ")
            return f"MultiObjectiveFitness(objectives={obj_str}, cv={cv_str})"
        return f"MultiObjectiveFitness(objectives={obj_str})"

    def with_metadata(self, **kwargs: Any) -> MultiObjectiveFitness:
        """Create copy with updated metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return MultiObjectiveFitness(
            objectives=self.objectives.copy(),
            constraint_violations=(
                self.constraint_violations.copy()
                if self.constraint_violations is not None
                else None
            ),
            metadata=new_metadata,
        )
