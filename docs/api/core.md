# Core Module API

The `evolve.core` module provides the foundational types and orchestration for evolutionary algorithms.

## Types

### `Fitness`

```python
from evolve.core import Fitness

# Single objective
fitness = Fitness.scalar(0.5)

# Multi-objective
fitness = Fitness(values=np.array([0.5, 0.3]))

# With constraints
fitness = Fitness(
    values=np.array([0.5]),
    constraints=np.array([-0.1, 0.2])  # ≤0 is feasible
)
```

**Properties:**
- `values: np.ndarray` - Objective values
- `constraints: np.ndarray | None` - Constraint violations
- `is_feasible: bool` - True if all constraints satisfied
- `is_valid: bool` - True if no NaN/inf values

**Methods:**
- `dominates(other)` - Pareto dominance comparison
- `scalar(value)` - Create single-objective fitness

---

### `Individual`

```python
from evolve.core import Individual

individual = Individual(
    id=uuid4(),
    genome=my_genome,
    fitness=Fitness.scalar(0.5),
    metadata=IndividualMetadata(origin="crossover"),
    created_at=generation
)
```

**Attributes:**
- `id: UUID` - Unique identifier
- `genome: G` - The genotype (generic type)
- `fitness: Fitness | None` - Evaluated fitness
- `metadata: IndividualMetadata` - Tracking information
- `created_at: int` - Generation created

**Methods:**
- `with_fitness(fitness)` - Return copy with fitness assigned

---

### `IndividualMetadata`

```python
from evolve.core import IndividualMetadata

metadata = IndividualMetadata(
    age=0,
    parent_ids=(parent1.id, parent2.id),
    species_id=3,
    origin="crossover"  # "init" | "crossover" | "mutation" | "migration"
)
```

---

### `Population`

```python
from evolve.core import Population

population = Population(
    individuals=[...],
    generation=0
)

# Access
best = population.best(n=1, minimize=True)
stats = population.statistics
```

**Properties:**
- `individuals: list[Individual]` - Members
- `generation: int` - Current generation
- `statistics: PopulationStatistics` - Computed stats

**Methods:**
- `best(n, minimize)` - Get top N individuals
- `filter(predicate)` - Filter by condition

---

## Engine

### `EvolutionEngine`

The main orchestrator for evolutionary runs.

```python
from evolve.core import EvolutionEngine, EvolutionConfig

config = EvolutionConfig(
    population_size=100,
    max_generations=100,
    elitism=1,
    crossover_rate=0.9,
    mutation_rate=0.1,
    minimize=True
)

engine = EvolutionEngine(
    config=config,
    evaluator=my_evaluator,
    selection=TournamentSelection(),
    crossover=UniformCrossover(),
    mutation=GaussianMutation(),
    seed=42
)

result = engine.run(initial_population)
```

**Methods:**
- `run(population, callbacks)` - Execute full evolution
- `get_rng_state()` - Get RNG state for checkpointing
- `set_rng_state(state)` - Restore RNG state

---

### `EvolutionConfig`

```python
from evolve.core import EvolutionConfig

config = EvolutionConfig(
    population_size=100,      # Number of individuals
    max_generations=100,      # Stopping criterion
    elitism=1,                # Preserved best individuals
    crossover_rate=0.9,       # Probability of crossover
    mutation_rate=0.1,        # Probability of mutation
    minimize=True             # Optimization direction
)
```

---

### `EvolutionResult`

```python
result = engine.run(population)

print(result.best)           # Best individual
print(result.population)     # Final population
print(result.history)        # Metrics per generation
print(result.generations)    # Total generations
print(result.stop_reason)    # Why it stopped
```

---

## Callbacks

### `Callback` Protocol

```python
class MyCallback:
    def on_run_start(self, config):
        """Called before evolution starts."""
        pass
    
    def on_generation_start(self, generation, population):
        """Called at start of each generation."""
        pass
    
    def on_generation_end(self, generation, population, metrics):
        """Called after each generation."""
        pass
    
    def on_run_end(self, population, stop_reason):
        """Called when evolution completes."""
        pass
```

---

## Stopping Criteria

### `StoppingCriterion` Protocol

```python
from evolve.core.stopping import (
    GenerationLimitStopping,
    FitnessTargetStopping,
    StagnationStopping,
    CompositeStoppingCriterion
)

# Generation limit
stopping = GenerationLimitStopping(max_generations=100)

# Target fitness
stopping = FitnessTargetStopping(target=0.99, minimize=False)

# Stagnation detection
stopping = StagnationStopping(patience=20, min_delta=1e-6)

# Combine multiple criteria
stopping = CompositeStoppingCriterion([
    GenerationLimitStopping(100),
    FitnessTargetStopping(0.99)
])
```
