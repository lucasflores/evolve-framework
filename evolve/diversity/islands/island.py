"""
Island dataclass for island-model parallelism.

An island represents an isolated subpopulation that evolves
independently with periodic migration to/from neighboring islands.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from evolve.core.types import Individual

G = TypeVar("G")


@dataclass
class Island(Generic[G]):
    """
    Isolated subpopulation for island-model parallelism.
    
    Islands evolve independently with periodic migration.
    Each island can have its own configuration (selection,
    mutation rate) for heterogeneous island models.
    
    Attributes:
        id: Unique island identifier
        population: List of individuals on this island
        topology: IDs of connected (neighbor) islands
        migration_rate: Fraction of population to migrate (0-1)
        isolation_time: Generations since last migration
        selection_operator: Optional island-specific selection
        mutation_rate: Optional island-specific mutation rate
        metadata: Additional island-specific data
    
    Example:
        >>> island = Island(
        ...     id=0,
        ...     population=individuals,
        ...     topology=[1, 3],  # Connected to islands 1 and 3
        ...     migration_rate=0.1
        ... )
    """
    
    id: int
    population: list[Individual[G]]
    topology: list[int] = field(default_factory=list)
    migration_rate: float = 0.1
    isolation_time: int = 0
    
    # Per-island configuration (overrides global if set)
    selection_operator: Any | None = None
    mutation_rate: float | None = None
    crossover_rate: float | None = None
    
    # Tracking
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Number of individuals on this island."""
        return len(self.population)
    
    @property
    def best_individual(self) -> Individual[G] | None:
        """Best individual by fitness (assumes maximization)."""
        if not self.population:
            return None
        
        evaluated = [
            ind for ind in self.population
            if ind.fitness is not None
        ]
        
        if not evaluated:
            return self.population[0]
        
        return max(
            evaluated,
            key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf")
        )
    
    @property
    def average_fitness(self) -> float:
        """Average fitness of evaluated individuals."""
        evaluated = [
            ind.fitness.values[0]
            for ind in self.population
            if ind.fitness is not None
        ]
        
        if not evaluated:
            return 0.0
        
        return sum(evaluated) / len(evaluated)
    
    @property
    def fitness_variance(self) -> float:
        """Variance of fitness values (diversity metric)."""
        evaluated = [
            ind.fitness.values[0]
            for ind in self.population
            if ind.fitness is not None
        ]
        
        if len(evaluated) < 2:
            return 0.0
        
        mean = sum(evaluated) / len(evaluated)
        return sum((f - mean) ** 2 for f in evaluated) / len(evaluated)
    
    def increment_isolation(self) -> None:
        """Increment isolation time counter."""
        self.isolation_time += 1
    
    def reset_isolation(self) -> None:
        """Reset isolation time after migration."""
        self.isolation_time = 0
    
    def __repr__(self) -> str:
        return (
            f"Island(id={self.id}, size={self.size}, "
            f"neighbors={self.topology}, isolation={self.isolation_time})"
        )
