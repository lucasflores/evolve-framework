# Diversity Preservation Interfaces Contract

**Module**: `evolve.diversity`  
**Purpose**: Define speciation, island models, and novelty search abstractions

---

## Distance Functions

```python
from typing import Protocol, TypeVar, Callable
import numpy as np

G = TypeVar('G')


class DistanceFunction(Protocol[G]):
    """
    Calculates genetic or behavioral distance between individuals.
    
    Used for speciation, novelty, and diversity metrics.
    """
    
    def __call__(self, a: G, b: G) -> float:
        """
        Calculate distance between two genomes.
        
        Args:
            a: First genome
            b: Second genome
            
        Returns:
            Non-negative distance (0 = identical)
        """
        ...


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance for vector genomes."""
    return float(np.linalg.norm(a - b))


def hamming_distance(a: tuple, b: tuple) -> float:
    """Hamming distance for sequence genomes."""
    if len(a) != len(b):
        return float('inf')
    return sum(1 for x, y in zip(a, b) if x != y)


def neat_distance(
    a: 'GraphGenome',
    b: 'GraphGenome',
    c_disjoint: float = 1.0,
    c_excess: float = 1.0,
    c_weight: float = 0.4
) -> float:
    """
    NEAT compatibility distance for graph genomes.
    
    Distance = (c1 * E / N) + (c2 * D / N) + (c3 * W)
    
    Where:
    - E = excess genes (beyond max innovation of smaller genome)
    - D = disjoint genes (non-matching within both ranges)
    - W = average weight difference of matching genes
    - N = normalizing factor (genes in larger genome)
    
    Args:
        a: First graph genome
        b: Second graph genome
        c_disjoint: Disjoint gene coefficient
        c_excess: Excess gene coefficient
        c_weight: Weight difference coefficient
        
    Returns:
        Compatibility distance
    """
    a_innov = {c.innovation for c in a.connections}
    b_innov = {c.innovation for c in b.connections}
    
    matching = a_innov & b_innov
    
    max_a = max(a_innov) if a_innov else 0
    max_b = max(b_innov) if b_innov else 0
    
    if max_a < max_b:
        excess = len([i for i in b_innov if i > max_a])
        smaller_max = max_a
    else:
        excess = len([i for i in a_innov if i > max_b])
        smaller_max = max_b
    
    disjoint = len([
        i for i in (a_innov ^ b_innov)
        if i <= smaller_max
    ])
    
    # Weight difference for matching genes
    if matching:
        a_weights = {c.innovation: c.weight for c in a.connections}
        b_weights = {c.innovation: c.weight for c in b.connections}
        weight_diff = sum(
            abs(a_weights[i] - b_weights[i])
            for i in matching
        ) / len(matching)
    else:
        weight_diff = 0.0
    
    n = max(len(a.connections), len(b.connections), 1)
    
    return (c_excess * excess / n) + (c_disjoint * disjoint / n) + (c_weight * weight_diff)
```

---

## Species Protocol

```python
from dataclasses import dataclass, field
from typing import Sequence
from evolve.core import Individual


@dataclass
class Species(Generic[G]):
    """
    A group of genetically similar individuals.
    
    Used in speciated evolution (NEAT, island models).
    """
    id: int
    representative: Individual[G]
    members: list[Individual[G]] = field(default_factory=list)
    age: int = 0  # Generations since creation
    best_fitness_ever: float = float('-inf')
    stagnation_counter: int = 0
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    @property
    def average_fitness(self) -> float:
        if not self.members:
            return 0.0
        return sum(
            m.fitness.value for m in self.members
            if m.fitness is not None
        ) / self.size
    
    def update_stagnation(self) -> None:
        """Update stagnation counter after evaluation."""
        current_best = max(
            (m.fitness.value for m in self.members if m.fitness),
            default=float('-inf')
        )
        if current_best > self.best_fitness_ever:
            self.best_fitness_ever = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def is_stagnant(self, threshold: int = 15) -> bool:
        """Check if species has been stagnant too long."""
        return self.stagnation_counter >= threshold
```

---

## Speciation Protocol

```python
class Speciator(Protocol[G]):
    """
    Groups individuals into species based on genetic similarity.
    """
    
    def speciate(
        self,
        population: Sequence[Individual[G]],
        existing_species: list[Species[G]]
    ) -> list[Species[G]]:
        """
        Assign individuals to species.
        
        Args:
            population: All individuals to speciate
            existing_species: Species from previous generation
            
        Returns:
            Updated list of species (may add/remove)
        """
        ...


@dataclass
class ThresholdSpeciator(Generic[G]):
    """
    Species assignment using distance threshold.
    
    An individual belongs to the first species whose
    representative is within the compatibility threshold.
    """
    distance_fn: DistanceFunction[G]
    threshold: float
    
    def speciate(
        self,
        population: Sequence[Individual[G]],
        existing_species: list[Species[G]]
    ) -> list[Species[G]]:
        # Clear existing members
        for species in existing_species:
            species.members = []
        
        species_list = list(existing_species)
        next_species_id = max((s.id for s in species_list), default=0) + 1
        
        for individual in population:
            placed = False
            
            for species in species_list:
                dist = self.distance_fn(
                    individual.genome,
                    species.representative.genome
                )
                if dist < self.threshold:
                    species.members.append(individual)
                    placed = True
                    break
            
            if not placed:
                # Create new species
                new_species = Species(
                    id=next_species_id,
                    representative=individual,
                    members=[individual]
                )
                species_list.append(new_species)
                next_species_id += 1
        
        # Remove empty species
        species_list = [s for s in species_list if s.members]
        
        # Update representatives
        for species in species_list:
            if species.members:
                species.representative = species.members[0]
                species.age += 1
        
        return species_list
```

---

## Fitness Sharing

```python
def explicit_fitness_sharing(
    individuals: Sequence[Individual[G]],
    distance_fn: DistanceFunction[G],
    sigma_share: float,
    alpha: float = 1.0
) -> list[float]:
    """
    Calculate shared fitness for each individual.
    
    Reduces fitness of individuals in crowded regions
    to promote diversity.
    
    shared_fitness[i] = raw_fitness[i] / niche_count[i]
    
    Args:
        individuals: Population
        distance_fn: Distance function
        sigma_share: Niche radius
        alpha: Shape parameter for sharing function
        
    Returns:
        Shared fitness values
    """
    n = len(individuals)
    niche_counts = [0.0] * n
    
    for i in range(n):
        for j in range(n):
            dist = distance_fn(
                individuals[i].genome,
                individuals[j].genome
            )
            if dist < sigma_share:
                # Triangular sharing function
                sharing = 1.0 - (dist / sigma_share) ** alpha
                niche_counts[i] += sharing
    
    shared_fitness = []
    for i, ind in enumerate(individuals):
        raw = ind.fitness.value if ind.fitness else 0.0
        shared = raw / max(niche_counts[i], 1.0)
        shared_fitness.append(shared)
    
    return shared_fitness
```

---

## Island Model Protocol

```python
@dataclass
class Island(Generic[G]):
    """
    Isolated subpopulation for island-model parallelism.
    
    Islands evolve independently with periodic migration.
    """
    id: int
    population: list[Individual[G]]
    topology: list[int]  # IDs of connected islands
    migration_rate: float = 0.1  # Fraction to migrate
    isolation_time: int = 0  # Generations since last migration
    
    # Per-island configuration
    selection_operator: 'SelectionOperator' | None = None
    mutation_rate: float | None = None


class MigrationPolicy(Protocol[G]):
    """
    Controls how individuals migrate between islands.
    """
    
    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: 'Random'
    ) -> list[Individual[G]]:
        """Select individuals to leave island."""
        ...
    
    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        target_island: Island[G],
        rng: 'Random'
    ) -> list[Individual[G]]:
        """Select which emigrants enter target island."""
        ...


class BestMigration:
    """Migrate best individuals (most common policy)."""
    
    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: 'Random'
    ) -> list[Individual[G]]:
        sorted_pop = sorted(
            island.population,
            key=lambda i: i.fitness.value if i.fitness else float('-inf'),
            reverse=True
        )
        return sorted_pop[:n_emigrants]
    
    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        target_island: Island[G],
        rng: 'Random'
    ) -> list[Individual[G]]:
        # Replace worst individuals
        return emigrants


class RandomMigration:
    """Migrate random individuals."""
    
    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: 'Random'
    ) -> list[Individual[G]]:
        return rng.sample(island.population, n_emigrants)
    
    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        target_island: Island[G],
        rng: 'Random'
    ) -> list[Individual[G]]:
        return emigrants
```

---

## Island Topology

```python
def ring_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Ring topology: each island connected to two neighbors.
    
    Returns:
        Mapping from island ID to connected island IDs
    """
    return {
        i: [(i - 1) % n_islands, (i + 1) % n_islands]
        for i in range(n_islands)
    }


def fully_connected_topology(n_islands: int) -> dict[int, list[int]]:
    """All islands connected to all others."""
    return {
        i: [j for j in range(n_islands) if j != i]
        for i in range(n_islands)
    }


def hypercube_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Hypercube topology for power-of-2 islands.
    
    Each island connected to others differing by one bit.
    """
    import math
    
    if n_islands & (n_islands - 1) != 0:
        raise ValueError("n_islands must be power of 2")
    
    n_bits = int(math.log2(n_islands))
    topology = {}
    
    for i in range(n_islands):
        neighbors = []
        for bit in range(n_bits):
            neighbor = i ^ (1 << bit)
            neighbors.append(neighbor)
        topology[i] = neighbors
    
    return topology
```

---

## Synchronous Migration Controller

```python
@dataclass
class MigrationController(Generic[G]):
    """
    Coordinates migration between islands.
    
    Synchronous model: all islands migrate at same time
    to maintain reproducibility.
    """
    policy: MigrationPolicy[G]
    migration_interval: int = 10  # Generations between migrations
    
    def should_migrate(self, generation: int) -> bool:
        """Check if migration should occur this generation."""
        return generation > 0 and generation % self.migration_interval == 0
    
    def migrate(
        self,
        islands: list[Island[G]],
        rng: 'Random'
    ) -> None:
        """
        Perform migration between all islands.
        
        DETERMINISM: Processes islands in fixed order,
        uses derived seeds for each island pair.
        """
        # Collect emigrants from all islands first (deterministic order)
        emigrants: dict[int, list[Individual[G]]] = {}
        
        for island in sorted(islands, key=lambda i: i.id):
            n_emigrants = int(len(island.population) * island.migration_rate)
            if n_emigrants > 0:
                # Derive seed for this island
                island_seed = rng.integers(0, 2**31)
                island_rng = np.random.default_rng(island_seed)
                emigrants[island.id] = self.policy.select_emigrants(
                    island, n_emigrants, island_rng
                )
        
        # Distribute immigrants (fixed topology order)
        for island in sorted(islands, key=lambda i: i.id):
            incoming: list[Individual[G]] = []
            
            for neighbor_id in sorted(island.topology):
                if neighbor_id in emigrants:
                    incoming.extend(emigrants[neighbor_id])
            
            if incoming:
                # Replace worst individuals
                island.population.sort(
                    key=lambda i: i.fitness.value if i.fitness else float('-inf')
                )
                n_replace = min(len(incoming), len(island.population))
                island.population[:n_replace] = incoming[:n_replace]
            
            island.isolation_time = 0
```

---

## Novelty Search Protocol

```python
@dataclass
class NoveltyArchive(Generic[G]):
    """
    Archive of novel behaviors for novelty search.
    
    Behaviors are added if sufficiently novel compared
    to current archive and population.
    """
    behaviors: list[np.ndarray] = field(default_factory=list)
    max_size: int = 1000
    k_neighbors: int = 15
    novelty_threshold: float = 0.1
    
    def novelty(
        self,
        behavior: np.ndarray,
        population_behaviors: Sequence[np.ndarray]
    ) -> float:
        """
        Calculate novelty of a behavior.
        
        Novelty = average distance to k nearest neighbors
        in archive + population.
        """
        all_behaviors = list(self.behaviors) + list(population_behaviors)
        
        if not all_behaviors:
            return float('inf')
        
        distances = [
            float(np.linalg.norm(behavior - b))
            for b in all_behaviors
        ]
        distances.sort()
        
        k = min(self.k_neighbors, len(distances))
        return sum(distances[:k]) / k
    
    def maybe_add(
        self,
        behavior: np.ndarray,
        novelty_score: float
    ) -> bool:
        """
        Add behavior to archive if sufficiently novel.
        
        Returns:
            True if behavior was added
        """
        if novelty_score >= self.novelty_threshold:
            self.behaviors.append(behavior)
            
            # Prune if over capacity
            if len(self.behaviors) > self.max_size:
                # Remove oldest
                self.behaviors = self.behaviors[-self.max_size:]
            
            return True
        return False


class BehaviorCharacterization(Protocol[G]):
    """
    Extracts behavior descriptor from individual.
    
    Used for novelty search and behavioral diversity.
    """
    
    def characterize(self, individual: Individual[G]) -> np.ndarray:
        """
        Extract behavior vector from individual.
        
        Args:
            individual: Evaluated individual
            
        Returns:
            Behavior descriptor (fixed-length vector)
        """
        ...
```

---

## Quality-Diversity (MAP-Elites) Support

```python
@dataclass
class QDArchive(Generic[G]):
    """
    Quality-Diversity archive for MAP-Elites.
    
    Maintains grid of cells, each containing the best
    individual found for that behavior niche.
    """
    dimensions: tuple[int, ...]  # Grid size per dimension
    bounds: tuple[np.ndarray, np.ndarray]  # Behavior space bounds
    archive: dict[tuple[int, ...], Individual[G]] = field(default_factory=dict)
    
    def get_cell(self, behavior: np.ndarray) -> tuple[int, ...]:
        """Map behavior to grid cell."""
        normalized = (behavior - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        indices = tuple(
            min(int(n * d), d - 1)
            for n, d in zip(normalized, self.dimensions)
        )
        return indices
    
    def try_add(self, individual: Individual[G], behavior: np.ndarray) -> bool:
        """
        Add individual if it improves its cell.
        
        Returns:
            True if individual was added
        """
        cell = self.get_cell(behavior)
        
        current = self.archive.get(cell)
        if current is None or (
            individual.fitness is not None and
            current.fitness is not None and
            individual.fitness.value > current.fitness.value
        ):
            self.archive[cell] = individual
            return True
        return False
    
    @property
    def coverage(self) -> float:
        """Fraction of cells occupied."""
        total_cells = 1
        for d in self.dimensions:
            total_cells *= d
        return len(self.archive) / total_cells
    
    def sample(self, n: int, rng: 'Random') -> list[Individual[G]]:
        """Sample n individuals uniformly from archive."""
        if not self.archive:
            return []
        keys = list(self.archive.keys())
        selected_keys = rng.sample(keys, min(n, len(keys)))
        return [self.archive[k] for k in selected_keys]
```
