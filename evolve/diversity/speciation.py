"""
Speciation for diversity preservation.

Provides mechanisms for grouping similar individuals into species:
- Distance functions for measuring genetic similarity
- Species dataclass for tracking species information
- Speciator protocol and implementations

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, Protocol, Sequence, TypeVar

import numpy as np

from evolve.core.types import Individual

if TYPE_CHECKING:
    from evolve.representation.graph import GraphGenome

G = TypeVar("G")


# ============================================================================
# Distance Functions
# ============================================================================


class DistanceFunction(Protocol[G]):
    """
    Protocol for distance functions between genomes.
    
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
    """
    Euclidean distance for vector genomes.
    
    Args:
        a: First vector (or genome with .genes attribute)
        b: Second vector (or genome with .genes attribute)
        
    Returns:
        Euclidean distance between vectors
    """
    # Handle VectorGenome objects
    if hasattr(a, "genes"):
        a = a.genes
    if hasattr(b, "genes"):
        b = b.genes
    
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def hamming_distance(a: tuple | list, b: tuple | list) -> float:
    """
    Hamming distance for sequence genomes.
    
    Counts the number of positions where elements differ.
    
    Args:
        a: First sequence
        b: Second sequence
        
    Returns:
        Number of differing positions (inf if lengths differ)
    """
    if len(a) != len(b):
        return float("inf")
    return float(sum(1 for x, y in zip(a, b) if x != y))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manhattan (L1) distance for vector genomes.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Sum of absolute differences
    """
    if hasattr(a, "genes"):
        a = a.genes
    if hasattr(b, "genes"):
        b = b.genes
    
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance for vector genomes.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        1 - cosine_similarity (range [0, 2])
    """
    if hasattr(a, "genes"):
        a = a.genes
    if hasattr(b, "genes"):
        b = b.genes
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def neat_distance(
    a: "GraphGenome",
    b: "GraphGenome",
    c_disjoint: float = 1.0,
    c_excess: float = 1.0,
    c_weight: float = 0.4,
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
        c_disjoint: Disjoint gene coefficient (default: 1.0)
        c_excess: Excess gene coefficient (default: 1.0)
        c_weight: Weight difference coefficient (default: 0.4)
        
    Returns:
        Compatibility distance
    """
    # Get innovation numbers from connections
    a_innov = {c.innovation for c in a.connections}
    b_innov = {c.innovation for c in b.connections}
    
    # Handle empty genomes
    if not a_innov and not b_innov:
        return 0.0
    if not a_innov or not b_innov:
        return float("inf")
    
    # Find matching genes
    matching = a_innov & b_innov
    
    # Find max innovation in each genome
    max_a = max(a_innov)
    max_b = max(b_innov)
    
    # Calculate excess genes (beyond max of smaller genome)
    if max_a < max_b:
        excess = len([i for i in b_innov if i > max_a])
        smaller_max = max_a
    else:
        excess = len([i for i in a_innov if i > max_b])
        smaller_max = max_b
    
    # Calculate disjoint genes (non-matching within both ranges)
    disjoint = len([i for i in (a_innov ^ b_innov) if i <= smaller_max])
    
    # Calculate weight difference for matching genes
    if matching:
        a_weights = {c.innovation: c.weight for c in a.connections}
        b_weights = {c.innovation: c.weight for c in b.connections}
        weight_diff = sum(
            abs(a_weights[i] - b_weights[i]) for i in matching
        ) / len(matching)
    else:
        weight_diff = 0.0
    
    # Normalizing factor (genes in larger genome)
    n = max(len(a.connections), len(b.connections), 1)
    
    return (c_excess * excess / n) + (c_disjoint * disjoint / n) + (c_weight * weight_diff)


# ============================================================================
# Species
# ============================================================================


@dataclass
class Species(Generic[G]):
    """
    A group of genetically similar individuals.
    
    Used in speciated evolution (NEAT, island models) to protect
    innovation by allowing new structures time to optimize.
    
    Attributes:
        id: Unique species identifier
        representative: Reference individual for distance comparison
        members: Current members of the species
        age: Generations since species creation
        best_fitness_ever: Best fitness achieved by any member
        stagnation_counter: Generations without improvement
    """
    
    id: int
    representative: Individual[G]
    members: list[Individual[G]] = field(default_factory=list)
    age: int = 0
    best_fitness_ever: float = float("-inf")
    stagnation_counter: int = 0
    
    @property
    def size(self) -> int:
        """Number of members in this species."""
        return len(self.members)
    
    @property
    def average_fitness(self) -> float:
        """Average fitness of all evaluated members."""
        if not self.members:
            return 0.0
        
        evaluated = [
            m.fitness.values[0]
            for m in self.members
            if m.fitness is not None
        ]
        
        if not evaluated:
            return 0.0
        
        return sum(evaluated) / len(evaluated)
    
    @property
    def best_fitness(self) -> float | None:
        """Best current fitness in the species."""
        evaluated = [
            m.fitness.values[0]
            for m in self.members
            if m.fitness is not None
        ]
        
        if not evaluated:
            return None
        
        return max(evaluated)
    
    @property
    def fitness_variance(self) -> float:
        """Variance of fitness values in the species."""
        evaluated = [
            m.fitness.values[0]
            for m in self.members
            if m.fitness is not None
        ]
        
        if len(evaluated) < 2:
            return 0.0
        
        mean = sum(evaluated) / len(evaluated)
        return sum((f - mean) ** 2 for f in evaluated) / len(evaluated)
    
    def update_stagnation(self, minimize: bool = False) -> None:
        """
        Update stagnation counter after evaluation.
        
        Args:
            minimize: If True, lower fitness is better
        """
        evaluated = [
            m.fitness.values[0]
            for m in self.members
            if m.fitness is not None
        ]
        
        if not evaluated:
            self.stagnation_counter += 1
            return
        
        if minimize:
            current_best = min(evaluated)
            improved = current_best < self.best_fitness_ever
        else:
            current_best = max(evaluated)
            improved = current_best > self.best_fitness_ever
        
        if improved:
            self.best_fitness_ever = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def is_stagnant(self, threshold: int = 15) -> bool:
        """
        Check if species has been stagnant too long.
        
        Args:
            threshold: Number of generations to consider stagnant
            
        Returns:
            True if stagnation counter >= threshold
        """
        return self.stagnation_counter >= threshold
    
    def get_best_member(self, minimize: bool = False) -> Individual[G] | None:
        """Get the best member by fitness."""
        evaluated = [m for m in self.members if m.fitness is not None]
        
        if not evaluated:
            return self.members[0] if self.members else None
        
        if minimize:
            return min(evaluated, key=lambda m: m.fitness.values[0])
        else:
            return max(evaluated, key=lambda m: m.fitness.values[0])
    
    def clear_members(self) -> None:
        """Clear member list for re-speciation."""
        self.members = []
    
    def __repr__(self) -> str:
        return (
            f"Species(id={self.id}, size={self.size}, age={self.age}, "
            f"stagnation={self.stagnation_counter})"
        )


# ============================================================================
# Speciator Protocol and Implementations
# ============================================================================


class Speciator(Protocol[G]):
    """
    Protocol for grouping individuals into species.
    
    Speciators assign individuals to species based on
    genetic similarity, creating and removing species
    as needed.
    """
    
    def speciate(
        self,
        population: Sequence[Individual[G]],
        existing_species: list[Species[G]],
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
    New species are created for individuals that don't
    fit any existing species.
    
    Attributes:
        distance_fn: Function to compute genome distance
        threshold: Maximum distance for same species
        dynamic_threshold: If True, adjust threshold to target species count
        target_species: Target number of species (for dynamic threshold)
        threshold_delta: Amount to adjust threshold each generation
    """
    
    distance_fn: Callable[[G, G], float]
    threshold: float
    dynamic_threshold: bool = False
    target_species: int = 10
    threshold_delta: float = 0.1
    
    def speciate(
        self,
        population: Sequence[Individual[G]],
        existing_species: list[Species[G]],
    ) -> list[Species[G]]:
        """
        Assign individuals to species based on distance threshold.
        
        Args:
            population: All individuals to speciate
            existing_species: Species from previous generation
            
        Returns:
            Updated list of species
        """
        # Clear existing members
        for species in existing_species:
            species.clear_members()
        
        species_list = list(existing_species)
        next_species_id = max((s.id for s in species_list), default=0) + 1
        
        # Assign each individual to a species
        for individual in population:
            placed = False
            
            for species in species_list:
                dist = self.distance_fn(
                    individual.genome,
                    species.representative.genome,
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
                    members=[individual],
                )
                species_list.append(new_species)
                next_species_id += 1
        
        # Remove empty species
        species_list = [s for s in species_list if s.members]
        
        # Update representatives (random member or best)
        for species in species_list:
            if species.members:
                species.representative = species.members[0]
                species.age += 1
        
        # Adjust threshold if dynamic
        if self.dynamic_threshold:
            self._adjust_threshold(len(species_list))
        
        return species_list
    
    def _adjust_threshold(self, current_count: int) -> None:
        """Adjust threshold to target species count."""
        if current_count < self.target_species:
            # Too few species, lower threshold
            self.threshold -= self.threshold_delta
            self.threshold = max(self.threshold, 0.1)
        elif current_count > self.target_species:
            # Too many species, raise threshold
            self.threshold += self.threshold_delta


@dataclass
class KMeansSpeciator(Generic[G]):
    """
    Species assignment using k-means clustering.
    
    Groups individuals into k species based on genome
    similarity using iterative centroid refinement.
    
    Attributes:
        distance_fn: Function to compute genome distance
        n_species: Number of species to create
        max_iterations: Maximum k-means iterations
    """
    
    distance_fn: Callable[[G, G], float]
    n_species: int = 5
    max_iterations: int = 10
    
    def speciate(
        self,
        population: Sequence[Individual[G]],
        existing_species: list[Species[G]],
    ) -> list[Species[G]]:
        """
        Assign individuals to k species using clustering.
        
        Args:
            population: All individuals to speciate
            existing_species: Species from previous generation (used for initial centroids)
            
        Returns:
            List of k species
        """
        if len(population) < self.n_species:
            # Not enough individuals, put each in own species
            return [
                Species(id=i, representative=ind, members=[ind])
                for i, ind in enumerate(population)
            ]
        
        # Initialize centroids
        if existing_species and len(existing_species) == self.n_species:
            centroids = [s.representative for s in existing_species]
        else:
            # Use first k individuals as initial centroids
            centroids = list(population[: self.n_species])
        
        # Iterative refinement
        for _ in range(self.max_iterations):
            # Assign to nearest centroid
            assignments: list[list[Individual[G]]] = [[] for _ in range(self.n_species)]
            
            for individual in population:
                min_dist = float("inf")
                nearest = 0
                
                for i, centroid in enumerate(centroids):
                    dist = self.distance_fn(individual.genome, centroid.genome)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
                
                assignments[nearest].append(individual)
            
            # Update centroids (use member closest to mean)
            new_centroids = []
            for i, members in enumerate(assignments):
                if members:
                    # Find member with minimum total distance to others
                    best_member = min(
                        members,
                        key=lambda m: sum(
                            self.distance_fn(m.genome, other.genome)
                            for other in members
                        ),
                    )
                    new_centroids.append(best_member)
                else:
                    new_centroids.append(centroids[i])
            
            centroids = new_centroids
        
        # Create species from final assignments
        species_list = []
        for i, members in enumerate(assignments):
            if members:
                species = Species(
                    id=i,
                    representative=centroids[i],
                    members=members,
                )
                species_list.append(species)
        
        return species_list
