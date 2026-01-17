# Core Interfaces Contract

**Module**: `evolve.core`  
**Purpose**: Define the backend-agnostic evolution engine interfaces

---

## Individual Protocol

```python
from typing import Protocol, TypeVar, Generic, Any
from dataclasses import dataclass
from uuid import UUID
import numpy as np

G = TypeVar('G', bound='Genome')

@dataclass(frozen=True)
class IndividualMetadata:
    """Optional tracking information for an individual."""
    age: int = 0
    parent_ids: tuple[UUID, ...] | None = None
    species_id: int | None = None
    origin: str = "init"  # "init" | "crossover" | "mutation" | "migration"

@dataclass
class Individual(Generic[G]):
    """A candidate solution in the population."""
    id: UUID
    genome: G
    fitness: 'Fitness | None' = None
    metadata: IndividualMetadata = IndividualMetadata()
    created_at: int = 0
    
    def with_fitness(self, fitness: 'Fitness') -> 'Individual[G]':
        """Return a copy with fitness set."""
        ...
    
    def with_metadata(self, **updates) -> 'Individual[G]':
        """Return a copy with updated metadata."""
        ...
```

---

## Fitness Value Object

```python
@dataclass(frozen=True)
class Fitness:
    """Vector-valued fitness with optional constraints."""
    values: np.ndarray  # Shape: (n_objectives,)
    constraints: np.ndarray | None = None  # Shape: (n_constraints,), ≤0 = feasible
    
    @property
    def is_feasible(self) -> bool:
        """True if all constraints satisfied (or no constraints)."""
        if self.constraints is None:
            return True
        return bool(np.all(self.constraints <= 0))
    
    @property
    def is_valid(self) -> bool:
        """True if no NaN values in fitness."""
        return bool(not np.any(np.isnan(self.values)))
    
    def dominates(self, other: 'Fitness', minimize: bool = True) -> bool:
        """Pareto dominance check (single-objective: simple comparison)."""
        ...
    
    def __getitem__(self, idx: int) -> float:
        """Access individual objective value."""
        return float(self.values[idx])
    
    @classmethod
    def scalar(cls, value: float) -> 'Fitness':
        """Create single-objective fitness."""
        return cls(values=np.array([value]))
```

---

## Population Container

```python
from typing import Sequence, Iterator

@dataclass
class PopulationStatistics:
    """Computed population metrics."""
    size: int
    best_fitness: Fitness
    mean_fitness: Fitness
    diversity: float
    species_count: int = 0
    front_sizes: list[int] | None = None  # For multi-objective

class Population(Generic[G]):
    """Ordered collection of individuals with statistics."""
    
    def __init__(
        self,
        individuals: Sequence[Individual[G]],
        generation: int = 0
    ) -> None:
        """
        Create population from individuals.
        
        Raises:
            ValueError: If individuals is empty
        """
        ...
    
    @property
    def individuals(self) -> Sequence[Individual[G]]:
        """Immutable view of individuals."""
        ...
    
    @property
    def generation(self) -> int:
        """Current generation number."""
        ...
    
    @property
    def statistics(self) -> PopulationStatistics:
        """Computed statistics (cached, recomputed on mutation)."""
        ...
    
    def __len__(self) -> int:
        """Number of individuals."""
        ...
    
    def __iter__(self) -> Iterator[Individual[G]]:
        """Iterate over individuals."""
        ...
    
    def __getitem__(self, idx: int) -> Individual[G]:
        """Access individual by index."""
        ...
    
    def best(self, n: int = 1) -> Sequence[Individual[G]]:
        """Return n best individuals by fitness."""
        ...
    
    def with_individuals(
        self,
        individuals: Sequence[Individual[G]],
        generation: int | None = None
    ) -> 'Population[G]':
        """Return new population with updated individuals."""
        ...
```

---

## Evolution Engine

```python
from typing import Callable
from random import Random

class EvolutionEngine(Generic[G]):
    """
    Main evolution loop orchestrator.
    
    The engine coordinates:
    1. Population initialization
    2. Fitness evaluation
    3. Selection
    4. Variation (crossover + mutation)
    5. Replacement
    6. Termination checking
    
    All randomness flows through explicit RNG instances.
    """
    
    def __init__(
        self,
        config: 'ExperimentConfig',
        evaluator: 'Evaluator[G]',
        seed: int
    ) -> None:
        """
        Initialize engine with configuration.
        
        Args:
            config: Complete experiment configuration
            evaluator: Fitness evaluator (may be accelerated)
            seed: Master random seed for reproducibility
        """
        ...
    
    def run(
        self,
        initial_population: Population[G] | None = None,
        callbacks: Sequence['Callback'] | None = None
    ) -> 'ExperimentResult[G]':
        """
        Execute full evolution run.
        
        Args:
            initial_population: Optional starting population (otherwise random init)
            callbacks: Optional event handlers
            
        Returns:
            ExperimentResult with final population and metrics
        """
        ...
    
    def step(self) -> Population[G]:
        """
        Execute single generation.
        
        Returns:
            New population after one generation
        """
        ...
    
    def checkpoint(self) -> 'Checkpoint':
        """
        Create checkpoint of current state.
        
        Returns:
            Checkpoint that can resume this exact state
        """
        ...
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: 'Checkpoint',
        evaluator: 'Evaluator[G]'
    ) -> 'EvolutionEngine[G]':
        """
        Resume from checkpoint.
        
        Args:
            checkpoint: Previously saved state
            evaluator: Evaluator (must match original)
            
        Returns:
            Engine ready to continue from checkpoint
        """
        ...
```

---

## Callback Protocol

```python
class Callback(Protocol):
    """Event handler for evolution lifecycle."""
    
    def on_generation_start(
        self,
        generation: int,
        population: Population
    ) -> None:
        """Called at start of each generation."""
        ...
    
    def on_generation_end(
        self,
        generation: int,
        population: Population,
        metrics: dict[str, Any]
    ) -> None:
        """Called at end of each generation with metrics."""
        ...
    
    def on_evaluation_complete(
        self,
        generation: int,
        evaluated: Sequence[Individual]
    ) -> None:
        """Called after batch evaluation."""
        ...
    
    def on_experiment_end(
        self,
        result: 'ExperimentResult'
    ) -> None:
        """Called when experiment completes."""
        ...
```

---

## Stopping Criterion Protocol

```python
class StoppingCriterion(Protocol):
    """Early stopping condition."""
    
    def should_stop(
        self,
        generation: int,
        population: Population,
        history: Sequence[PopulationStatistics]
    ) -> bool:
        """
        Check if evolution should stop.
        
        Args:
            generation: Current generation number
            population: Current population
            history: Statistics from all previous generations
            
        Returns:
            True if evolution should terminate
        """
        ...
    
    @property
    def reason(self) -> str:
        """Human-readable reason if stopped."""
        ...
```

**Built-in Implementations**:
- `GenerationLimit(max_generations: int)`
- `FitnessThreshold(threshold: float, objective: int = 0)`
- `StagnationDetector(patience: int, min_improvement: float)`
- `CompositeStop(*criteria, mode: 'any' | 'all')`
