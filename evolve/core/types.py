"""
Core types - Fitness and Individual dataclasses.

These are the fundamental building blocks of the evolutionary framework.
All types use NumPy arrays and Python stdlib only - NO ML FRAMEWORK IMPORTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

import numpy as np

# Type variable for genome types (bounded by Genome protocol)
G = TypeVar("G")


@dataclass(frozen=True)
class IndividualMetadata:
    """
    Optional tracking information for an individual.

    Attributes:
        age: Number of generations the individual has survived
        parent_ids: UUIDs of parent individuals (for lineage tracking)
        species_id: Species assignment (for NEAT-style speciation)
        origin: How this individual was created
    """

    age: int = 0
    parent_ids: tuple[UUID, ...] | None = None
    species_id: int | None = None
    origin: str = "init"  # "init" | "crossover" | "mutation" | "migration"


@dataclass(frozen=True)
class Fitness:
    """
    Vector-valued fitness with optional constraints.

    Fitness values are stored as NumPy arrays to support multi-objective
    optimization. Single-objective problems use a 1D array.

    Attributes:
        values: Array of objective values, shape (n_objectives,)
        constraints: Optional constraint violations, shape (n_constraints,).
                    Values ≤ 0 are considered feasible.
        metadata: Optional additional information (timing, diagnostics)

    Example:
        >>> # Single objective
        >>> f = Fitness.scalar(0.5)
        >>> f.values
        array([0.5])

        >>> # Multi-objective
        >>> f = Fitness(values=np.array([0.5, 0.3]))
        >>> len(f)  # number of objectives
        2
    """

    values: np.ndarray
    constraints: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fitness values."""
        # Ensure values is a numpy array
        if not isinstance(self.values, np.ndarray):
            object.__setattr__(self, "values", np.asarray(self.values))

        # Ensure 1D array
        if self.values.ndim != 1:
            object.__setattr__(self, "values", self.values.flatten())

        # Make immutable
        self.values.flags.writeable = False

        if self.constraints is not None:
            if not isinstance(self.constraints, np.ndarray):
                object.__setattr__(self, "constraints", np.asarray(self.constraints))
            self.constraints.flags.writeable = False

    @property
    def is_feasible(self) -> bool:
        """True if all constraints satisfied (or no constraints)."""
        if self.constraints is None:
            return True
        return bool(np.all(self.constraints <= 0))

    @property
    def is_valid(self) -> bool:
        """True if no NaN or inf values in fitness."""
        return bool(np.all(np.isfinite(self.values)))

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.values)

    def __len__(self) -> int:
        """Number of objectives."""
        return len(self.values)

    def __getitem__(self, idx: int) -> float:
        """Access individual objective value."""
        return float(self.values[idx])

    def __lt__(self, other: Fitness) -> bool:
        """
        Less-than comparison for single-objective fitness.

        For multi-objective, use dominates() instead.
        """
        if len(self.values) != 1 or len(other.values) != 1:
            raise ValueError("Use dominates() for multi-objective comparison, not < operator")
        return float(self.values[0]) < float(other.values[0])

    def __le__(self, other: Fitness) -> bool:
        """Less-than-or-equal for single-objective fitness."""
        if len(self.values) != 1 or len(other.values) != 1:
            raise ValueError("Use dominates() for multi-objective comparison")
        return float(self.values[0]) <= float(other.values[0])

    def __gt__(self, other: Fitness) -> bool:
        """Greater-than for single-objective fitness."""
        if len(self.values) != 1 or len(other.values) != 1:
            raise ValueError("Use dominates() for multi-objective comparison")
        return float(self.values[0]) > float(other.values[0])

    def __ge__(self, other: Fitness) -> bool:
        """Greater-than-or-equal for single-objective fitness."""
        if len(self.values) != 1 or len(other.values) != 1:
            raise ValueError("Use dominates() for multi-objective comparison")
        return float(self.values[0]) >= float(other.values[0])

    def dominates(self, other: Fitness, minimize: bool = True) -> bool:
        """
        Pareto dominance check.

        For single-objective: simple comparison.
        For multi-objective: dominates if better in at least one objective
        and not worse in any.

        Args:
            other: Fitness to compare against
            minimize: If True, lower values are better

        Returns:
            True if self dominates other
        """
        if len(self.values) != len(other.values):
            raise ValueError(
                f"Cannot compare fitness with different dimensions: "
                f"{len(self.values)} vs {len(other.values)}"
            )

        # Handle feasibility first (feasible always dominates infeasible)
        self_feasible = self.is_feasible
        other_feasible = other.is_feasible

        if self_feasible and not other_feasible:
            return True
        if not self_feasible and other_feasible:
            return False

        # Both infeasible: compare by constraint violation
        if not self_feasible and not other_feasible:
            if self.constraints is not None and other.constraints is not None:
                self_violation = float(np.sum(np.maximum(0, self.constraints)))
                other_violation = float(np.sum(np.maximum(0, other.constraints)))
                return self_violation < other_violation
            return False

        # Both feasible: Pareto dominance
        if minimize:
            better = self.values <= other.values
            strictly_better = self.values < other.values
        else:
            better = self.values >= other.values
            strictly_better = self.values > other.values

        return bool(np.all(better) and np.any(strictly_better))

    @classmethod
    def scalar(cls, value: float, metadata: dict[str, Any] | None = None) -> Fitness:
        """
        Create single-objective fitness.

        Args:
            value: The fitness value
            metadata: Optional metadata dictionary

        Returns:
            Fitness with single objective
        """
        return cls(values=np.array([value]), metadata=metadata or {})

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"values": self.values.tolist()}
        if self.constraints is not None:
            result["constraints"] = self.constraints.tolist()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fitness:
        """Reconstruct from dict."""
        constraints = None
        if "constraints" in data and data["constraints"] is not None:
            constraints = np.array(data["constraints"])
        return cls(
            values=np.array(data["values"]),
            constraints=constraints,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Individual(Generic[G]):
    """
    A candidate solution in the population.

    Individuals are the fundamental unit of evolution. They contain:
    - A genome (genetic representation)
    - Optional fitness (computed by evaluator)
    - Metadata for tracking lineage and age
    - Optional reproduction protocol (ERP feature)

    Attributes:
        id: Unique identifier
        genome: Genetic representation (type parameter G)
        fitness: Evaluated fitness (None until evaluated)
        metadata: Tracking information (age, parents, species)
        created_at: Generation when this individual was created
        protocol: Optional reproduction protocol (matchability, intent, crossover)

    Example:
        >>> from evolve.representation.vector import VectorGenome
        >>> genome = VectorGenome(genes=np.array([1.0, 2.0]))
        >>> ind = Individual(genome=genome)
        >>> ind.fitness  # None until evaluated
        >>> ind = ind.with_fitness(Fitness.scalar(0.5))
        >>> ind.fitness.values
        array([0.5])
    """

    # Import here to avoid circular dependency
    from evolve.reproduction.protocol import ReproductionProtocol as _RP

    genome: G
    id: UUID = field(default_factory=uuid4)
    fitness: Fitness | None = None
    metadata: IndividualMetadata = field(default_factory=IndividualMetadata)
    created_at: int = 0
    protocol: _RP | None = None

    def with_fitness(self, fitness: Fitness) -> Individual[G]:
        """
        Return a copy with fitness set.

        Args:
            fitness: The evaluated fitness value

        Returns:
            New Individual with fitness assigned
        """
        return Individual(
            id=self.id,
            genome=self.genome,
            fitness=fitness,
            metadata=self.metadata,
            created_at=self.created_at,
            protocol=self.protocol,
        )

    def with_metadata(self, **updates: Any) -> Individual[G]:
        """
        Return a copy with updated metadata.

        Args:
            **updates: Metadata fields to update

        Returns:
            New Individual with updated metadata
        """
        current = {
            "age": self.metadata.age,
            "parent_ids": self.metadata.parent_ids,
            "species_id": self.metadata.species_id,
            "origin": self.metadata.origin,
        }
        current.update(updates)
        new_metadata = IndividualMetadata(**current)
        return Individual(
            id=self.id,
            genome=self.genome,
            fitness=self.fitness,
            metadata=new_metadata,
            created_at=self.created_at,
            protocol=self.protocol,
        )

    def with_protocol(self, protocol: _RP | None) -> Individual[G]:
        """
        Return a copy with protocol set.

        Args:
            protocol: The reproduction protocol (or None)

        Returns:
            New Individual with protocol assigned
        """
        return Individual(
            id=self.id,
            genome=self.genome,
            fitness=self.fitness,
            metadata=self.metadata,
            created_at=self.created_at,
            protocol=protocol,
        )

    def increment_age(self) -> Individual[G]:
        """Return a copy with age incremented by 1."""
        return self.with_metadata(age=self.metadata.age + 1)

    @property
    def is_evaluated(self) -> bool:
        """True if fitness has been computed."""
        return self.fitness is not None

    def __hash__(self) -> int:
        """Hash by ID for set/dict membership."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality by ID."""
        if not isinstance(other, Individual):
            return False
        return self.id == other.id
