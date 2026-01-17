# Operators Module API

The `evolve.core.operators` module provides genetic operators for selection, crossover, and mutation.

## Selection Operators

### Tournament Selection

```python
from evolve.core.operators import TournamentSelection

selection = TournamentSelection(
    tournament_size=3,    # Individuals per tournament
    minimize=True         # Optimization direction
)

# Select n individuals
selected = selection.select(population, n=50, rng=rng)
```

### Roulette Wheel Selection

```python
from evolve.core.operators import RouletteSelection

selection = RouletteSelection(minimize=True)
selected = selection.select(population, n=50, rng=rng)
```

### Rank Selection

```python
from evolve.core.operators import RankSelection

selection = RankSelection(
    selection_pressure=2.0,  # Higher = more selective
    minimize=True
)
selected = selection.select(population, n=50, rng=rng)
```

### Truncation Selection

```python
from evolve.core.operators import TruncationSelection

selection = TruncationSelection(
    top_fraction=0.5,  # Keep top 50%
    minimize=True
)
selected = selection.select(population, n=50, rng=rng)
```

---

## Crossover Operators

### Single-Point Crossover

```python
from evolve.core.operators import SinglePointCrossover

crossover = SinglePointCrossover()
child1, child2 = crossover.crossover(parent1.genome, parent2.genome, rng)
```

### Two-Point Crossover

```python
from evolve.core.operators import TwoPointCrossover

crossover = TwoPointCrossover()
child1, child2 = crossover.crossover(parent1.genome, parent2.genome, rng)
```

### Uniform Crossover

```python
from evolve.core.operators import UniformCrossover

crossover = UniformCrossover(
    swap_probability=0.5  # Per-gene swap probability
)
child1, child2 = crossover.crossover(parent1.genome, parent2.genome, rng)
```

### SBX (Simulated Binary Crossover)

For real-valued optimization, preserves parent distribution.

```python
from evolve.core.operators import SBXCrossover

crossover = SBXCrossover(
    eta=20.0  # Distribution index (higher = more like parents)
)
child1, child2 = crossover.crossover(parent1.genome, parent2.genome, rng)
```

### Tree Crossover

For genetic programming.

```python
from evolve.core.operators import SubtreeCrossover

crossover = SubtreeCrossover(max_depth=10)
child1, child2 = crossover.crossover(tree1, tree2, rng)
```

### Graph Crossover (NEAT)

```python
from evolve.core.operators import NEATCrossover

crossover = NEATCrossover()
child = crossover.crossover(
    parent1.genome, parent2.genome, 
    fitness1=parent1.fitness, fitness2=parent2.fitness,
    rng=rng
)
```

---

## Mutation Operators

### Gaussian Mutation

```python
from evolve.core.operators import GaussianMutation

mutation = GaussianMutation(
    sigma=0.1,           # Standard deviation
    mutation_rate=0.1    # Per-gene probability
)
mutated = mutation.mutate(genome, rng)
```

### Polynomial Mutation

```python
from evolve.core.operators import PolynomialMutation

mutation = PolynomialMutation(
    eta=20.0,            # Distribution index
    mutation_rate=0.1
)
mutated = mutation.mutate(genome, rng)
```

### Adaptive Mutation

```python
from evolve.core.operators import AdaptiveMutation

mutation = AdaptiveMutation(
    initial_sigma=0.1,
    adaptation_rate=0.1  # 1/5 success rule
)
mutated = mutation.mutate(genome, rng)

# Adapt based on success rate
mutation.adapt(success_rate=0.3)
```

### Tree Mutation

```python
from evolve.core.operators import SubtreeMutation

mutation = SubtreeMutation(
    functions=func_set,
    terminals=term_set,
    max_depth=5
)
mutated = mutation.mutate(tree, rng)
```

### Graph Mutation (NEAT)

```python
from evolve.core.operators import NEATMutation

mutation = NEATMutation(
    weight_mutation_rate=0.8,
    add_node_rate=0.03,
    add_connection_rate=0.05,
    weight_perturbation_power=2.5
)
mutated = mutation.mutate(genome, innovation_counter, rng)
```

### Permutation Mutation

```python
from evolve.core.operators import SwapMutation, InversionMutation

# Swap two positions
swap = SwapMutation(n_swaps=1)
mutated = swap.mutate(permutation, rng)

# 2-opt (invert segment)
inversion = InversionMutation()
mutated = inversion.mutate(permutation, rng)
```

---

## Operator Protocols

### Selection Protocol

```python
class SelectionOperator(Protocol[G]):
    def select(
        self, 
        population: Population[G], 
        n: int, 
        rng: Random
    ) -> list[Individual[G]]:
        """Select n individuals from population."""
        ...
```

### Crossover Protocol

```python
class CrossoverOperator(Protocol[G]):
    def crossover(
        self, 
        parent1: G, 
        parent2: G, 
        rng: Random
    ) -> tuple[G, G]:
        """Create two offspring from parents."""
        ...
```

### Mutation Protocol

```python
class MutationOperator(Protocol[G]):
    def mutate(
        self, 
        genome: G, 
        rng: Random
    ) -> G:
        """Mutate genome, returning new genome."""
        ...
```
