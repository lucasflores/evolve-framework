# Operator Interfaces Contract

**Module**: `evolve.core.operators`  
**Purpose**: Define selection, crossover, and mutation operator protocols

---

## Selection Operators

```python
from typing import Protocol, TypeVar, Sequence
from random import Random

G = TypeVar('G', bound='Genome')

class SelectionOperator(Protocol[G]):
    """
    Selects individuals from population for reproduction.
    
    Selection operators MUST:
    - Accept explicit RNG for determinism
    - Support elitism via separate preserve_elites() call
    - Handle multi-objective populations (Pareto ranking)
    """
    
    def select(
        self,
        population: 'Population[G]',
        n: int,
        rng: Random
    ) -> Sequence['Individual[G]']:
        """
        Select n individuals for reproduction.
        
        Args:
            population: Source population
            n: Number to select (may include duplicates)
            rng: Random number generator
            
        Returns:
            Selected individuals (references, not copies)
        """
        ...


class ElitistSelection(Protocol[G]):
    """Selection with explicit elitism support."""
    
    def select_with_elites(
        self,
        population: 'Population[G]',
        n_select: int,
        n_elites: int,
        rng: Random
    ) -> tuple[Sequence['Individual[G]'], Sequence['Individual[G]']]:
        """
        Select individuals and preserve elites.
        
        Args:
            population: Source population
            n_select: Number to select for variation
            n_elites: Number of elites to preserve unchanged
            rng: Random number generator
            
        Returns:
            (selected_for_variation, elites_to_preserve)
        """
        ...
```

### Built-in Selection Operators

```python
@dataclass
class TournamentSelection:
    """
    Tournament selection with configurable size.
    
    Selects k random individuals, returns best.
    Larger k = higher selection pressure.
    """
    tournament_size: int = 3
    
    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]:
        ...


@dataclass  
class RouletteSelection:
    """
    Fitness-proportionate selection.
    
    Probability of selection proportional to fitness.
    Only valid for positive fitness values.
    """
    
    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]:
        ...


@dataclass
class RankSelection:
    """
    Rank-based selection.
    
    Selection probability based on rank, not raw fitness.
    More robust to fitness scaling issues.
    """
    selection_pressure: float = 1.5  # 1.0 = uniform, 2.0 = strong pressure
    
    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]:
        ...


@dataclass
class NSGA2Selection:
    """
    NSGA-II style selection for multi-objective.
    
    Uses non-dominated rank as primary criterion,
    crowding distance as secondary (for diversity).
    """
    
    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]:
        ...
```

---

## Crossover Operators

```python
class CrossoverOperator(Protocol[G]):
    """
    Combines genetic material from two parents.
    
    Crossover operators MUST:
    - Accept explicit RNG for determinism
    - Return two offspring (may be identical to parents if no crossover)
    - Not modify parent genomes (return new instances)
    """
    
    def crossover(
        self,
        parent1: G,
        parent2: G,
        rng: Random
    ) -> tuple[G, G]:
        """
        Create offspring from two parents.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            rng: Random number generator
            
        Returns:
            Two offspring genomes
        """
        ...


class ConditionalCrossover(Protocol[G]):
    """Crossover that may or may not occur based on probability."""
    
    def maybe_crossover(
        self,
        parent1: G,
        parent2: G,
        rate: float,
        rng: Random
    ) -> tuple[G, G]:
        """
        Crossover with probability `rate`.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            rate: Probability of crossover (0-1)
            rng: Random number generator
            
        Returns:
            Offspring (may be copies of parents if no crossover)
        """
        ...
```

### Built-in Crossover Operators

```python
@dataclass
class OnePointCrossover:
    """
    Single crossover point for sequences/vectors.
    
    Exchanges genetic material after random point.
    """
    
    def crossover(
        self,
        parent1: VectorGenome,
        parent2: VectorGenome,
        rng: Random
    ) -> tuple[VectorGenome, VectorGenome]:
        ...


@dataclass
class TwoPointCrossover:
    """
    Two crossover points for sequences/vectors.
    
    Exchanges segment between two random points.
    """
    
    def crossover(
        self,
        parent1: VectorGenome,
        parent2: VectorGenome,
        rng: Random
    ) -> tuple[VectorGenome, VectorGenome]:
        ...


@dataclass
class UniformCrossover:
    """
    Gene-by-gene crossover with probability.
    
    Each gene independently chosen from either parent.
    """
    swap_probability: float = 0.5
    
    def crossover(
        self,
        parent1: VectorGenome,
        parent2: VectorGenome,
        rng: Random
    ) -> tuple[VectorGenome, VectorGenome]:
        ...


@dataclass
class SimulatedBinaryCrossover:
    """
    SBX crossover for real-valued vectors.
    
    Simulates single-point crossover behavior
    for continuous domains.
    """
    eta: float = 20.0  # Distribution index (higher = children closer to parents)
    
    def crossover(
        self,
        parent1: VectorGenome,
        parent2: VectorGenome,
        rng: Random
    ) -> tuple[VectorGenome, VectorGenome]:
        ...


@dataclass
class NEATCrossover:
    """
    NEAT-style crossover for graph genomes.
    
    Aligns by innovation number, inherits from fitter parent.
    """
    
    def crossover(
        self,
        parent1: GraphGenome,
        parent2: GraphGenome,
        fitness1: Fitness,
        fitness2: Fitness,
        rng: Random
    ) -> tuple[GraphGenome, GraphGenome]:
        ...
```

---

## Mutation Operators

```python
class MutationOperator(Protocol[G]):
    """
    Introduces genetic variation.
    
    Mutation operators MUST:
    - Accept explicit RNG for determinism
    - Return new genome instance (not modify in place)
    - Respect genome constraints (bounds, structure)
    """
    
    def mutate(
        self,
        genome: G,
        rng: Random
    ) -> G:
        """
        Create mutated copy of genome.
        
        Args:
            genome: Genome to mutate
            rng: Random number generator
            
        Returns:
            New mutated genome
        """
        ...


class ConditionalMutation(Protocol[G]):
    """Mutation that may or may not occur based on probability."""
    
    def maybe_mutate(
        self,
        genome: G,
        rate: float,
        rng: Random
    ) -> G:
        """
        Mutate with probability `rate`.
        
        Args:
            genome: Genome to mutate
            rate: Probability of mutation (0-1)
            rng: Random number generator
            
        Returns:
            Mutated genome or copy of original
        """
        ...
```

### Built-in Mutation Operators

```python
@dataclass
class GaussianMutation:
    """
    Gaussian noise mutation for vectors.
    
    Adds N(0, sigma) to each gene independently.
    """
    sigma: float = 0.1
    per_gene_rate: float = 1.0  # Probability of mutating each gene
    
    def mutate(
        self,
        genome: VectorGenome,
        rng: Random
    ) -> VectorGenome:
        ...


@dataclass
class UniformMutation:
    """
    Uniform random mutation for vectors.
    
    Resets gene to random value within bounds.
    """
    per_gene_rate: float = 0.1
    
    def mutate(
        self,
        genome: VectorGenome,
        rng: Random
    ) -> VectorGenome:
        ...


@dataclass
class PolynomialMutation:
    """
    Polynomial mutation for real-valued vectors.
    
    Common in NSGA-II implementations.
    """
    eta: float = 20.0  # Distribution index
    per_gene_rate: float = 1.0
    
    def mutate(
        self,
        genome: VectorGenome,
        rng: Random
    ) -> VectorGenome:
        ...


@dataclass
class BitFlipMutation:
    """
    Bit flip mutation for binary vectors.
    """
    per_gene_rate: float = 0.01
    
    def mutate(
        self,
        genome: VectorGenome,
        rng: Random
    ) -> VectorGenome:
        ...


@dataclass
class NEATMutation:
    """
    NEAT-style mutation for graph genomes.
    
    Includes:
    - Weight perturbation
    - Add node
    - Add connection
    - Enable/disable connection
    """
    weight_mutation_rate: float = 0.8
    weight_perturbation_sigma: float = 0.1
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.05
    
    def mutate(
        self,
        genome: GraphGenome,
        rng: Random
    ) -> GraphGenome:
        ...
```

---

## Composite Operators

```python
@dataclass
class OperatorPipeline(Generic[G]):
    """
    Chain multiple operators together.
    
    Useful for domain-specific operator combinations.
    """
    operators: Sequence[MutationOperator[G]]
    
    def mutate(
        self,
        genome: G,
        rng: Random
    ) -> G:
        """Apply operators in sequence."""
        result = genome
        for op in self.operators:
            result = op.mutate(result, rng)
        return result


@dataclass
class AdaptiveOperator(Generic[G]):
    """
    Operator that adapts parameters during evolution.
    
    Extension point for self-adaptive mutation rates.
    """
    base_operator: MutationOperator[G]
    adaptation_strategy: 'AdaptationStrategy'
    
    def mutate(
        self,
        genome: G,
        rng: Random,
        generation: int,
        fitness_history: Sequence[float]
    ) -> G:
        """Mutate with adapted parameters."""
        ...
```
