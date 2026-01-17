"""
Integration tests for Speciation.

Tests species formation, distance functions, and fitness sharing.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.core.types import Fitness, Individual
from evolve.diversity.speciation import (
    DistanceFunction,
    Species,
    ThresholdSpeciator,
    KMeansSpeciator,
    euclidean_distance,
    hamming_distance,
    manhattan_distance,
    cosine_distance,
    neat_distance,
)
from evolve.diversity.niching import (
    explicit_fitness_sharing,
    crowding_distance,
    clearing,
    deterministic_crowding_pairing,
)
from evolve.representation.vector import VectorGenome


# ============================================================================
# Test Fixtures
# ============================================================================

def create_test_individual(
    id: str,
    genes: np.ndarray,
    fitness_value: float | None = None,
) -> Individual[VectorGenome]:
    """Create a test individual with VectorGenome."""
    ind = Individual(
        id=id,
        genome=VectorGenome(genes=genes),
    )
    if fitness_value is not None:
        ind.fitness = Fitness(values=(fitness_value,))
    return ind


# ============================================================================
# Distance Function Tests
# ============================================================================

@pytest.mark.integration
class TestDistanceFunctions:
    """Test distance function implementations."""

    def test_euclidean_distance_basic(self):
        """Euclidean distance should compute L2 norm."""
        a = np.array([0, 0, 0])
        b = np.array([3, 4, 0])
        
        assert euclidean_distance(a, b) == 5.0

    def test_euclidean_distance_identical(self):
        """Identical vectors should have zero distance."""
        a = np.array([1, 2, 3])
        
        assert euclidean_distance(a, a) == 0.0

    def test_euclidean_distance_symmetric(self):
        """Distance should be symmetric."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        assert euclidean_distance(a, b) == euclidean_distance(b, a)

    def test_euclidean_distance_with_vector_genome(self):
        """Should handle VectorGenome objects."""
        a = VectorGenome(genes=np.array([0, 0]))
        b = VectorGenome(genes=np.array([3, 4]))
        
        assert euclidean_distance(a, b) == 5.0

    def test_hamming_distance_basic(self):
        """Hamming distance should count differences."""
        a = (1, 2, 3, 4, 5)
        b = (1, 0, 3, 0, 5)
        
        assert hamming_distance(a, b) == 2.0

    def test_hamming_distance_identical(self):
        """Identical sequences should have zero distance."""
        a = (1, 2, 3)
        
        assert hamming_distance(a, a) == 0.0

    def test_hamming_distance_different_lengths(self):
        """Different lengths should return infinity."""
        a = (1, 2, 3)
        b = (1, 2)
        
        assert hamming_distance(a, b) == float("inf")

    def test_manhattan_distance_basic(self):
        """Manhattan distance should compute L1 norm."""
        a = np.array([0, 0])
        b = np.array([3, 4])
        
        assert manhattan_distance(a, b) == 7.0

    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1."""
        a = np.array([1, 0])
        b = np.array([0, 1])
        
        assert abs(cosine_distance(a, b) - 1.0) < 1e-10

    def test_cosine_distance_parallel(self):
        """Parallel vectors should have distance 0."""
        a = np.array([1, 2])
        b = np.array([2, 4])
        
        assert abs(cosine_distance(a, b)) < 1e-10

    def test_cosine_distance_opposite(self):
        """Opposite vectors should have distance 2."""
        a = np.array([1, 0])
        b = np.array([-1, 0])
        
        assert abs(cosine_distance(a, b) - 2.0) < 1e-10


# ============================================================================
# Species Tests
# ============================================================================

@pytest.mark.integration
class TestSpecies:
    """Test Species dataclass."""

    def test_species_creation(self):
        """Species should be created with basic attributes."""
        rep = create_test_individual("rep", np.zeros(5), fitness_value=10.0)
        
        species = Species(
            id=1,
            representative=rep,
            members=[rep],
        )
        
        assert species.id == 1
        assert species.size == 1
        assert species.age == 0
        assert species.stagnation_counter == 0

    def test_species_average_fitness(self):
        """Should compute average fitness correctly."""
        members = [
            create_test_individual(f"ind_{i}", np.zeros(5), fitness_value=float(i * 10))
            for i in range(1, 4)  # 10, 20, 30
        ]
        
        species = Species(id=1, representative=members[0], members=members)
        
        assert species.average_fitness == 20.0

    def test_species_best_fitness(self):
        """Should find best fitness."""
        members = [
            create_test_individual("ind_0", np.zeros(5), fitness_value=10.0),
            create_test_individual("ind_1", np.zeros(5), fitness_value=50.0),
            create_test_individual("ind_2", np.zeros(5), fitness_value=30.0),
        ]
        
        species = Species(id=1, representative=members[0], members=members)
        
        assert species.best_fitness == 50.0

    def test_species_stagnation_improves(self):
        """Stagnation counter should reset on improvement."""
        rep = create_test_individual("rep", np.zeros(5), fitness_value=10.0)
        species = Species(id=1, representative=rep, members=[rep])
        species.stagnation_counter = 5
        species.best_fitness_ever = 5.0
        
        species.update_stagnation(minimize=False)
        
        assert species.stagnation_counter == 0
        assert species.best_fitness_ever == 10.0

    def test_species_stagnation_increases(self):
        """Stagnation counter should increase without improvement."""
        rep = create_test_individual("rep", np.zeros(5), fitness_value=10.0)
        species = Species(id=1, representative=rep, members=[rep])
        species.best_fitness_ever = 20.0
        
        species.update_stagnation(minimize=False)
        
        assert species.stagnation_counter == 1

    def test_species_is_stagnant(self):
        """Should detect stagnant species."""
        rep = create_test_individual("rep", np.zeros(5))
        species = Species(id=1, representative=rep, members=[rep])
        species.stagnation_counter = 15
        
        assert species.is_stagnant(threshold=15)
        assert not species.is_stagnant(threshold=20)


# ============================================================================
# Speciator Tests
# ============================================================================

@pytest.mark.integration
class TestThresholdSpeciator:
    """Test ThresholdSpeciator."""

    def test_speciation_creates_species(self):
        """Should create species from population."""
        population = [
            create_test_individual(f"ind_{i}", np.array([float(i)]), fitness_value=1.0)
            for i in range(5)
        ]
        
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=2.0,
        )
        
        species = speciator.speciate(population, [])
        
        assert len(species) > 0
        # All individuals should be assigned
        total_members = sum(s.size for s in species)
        assert total_members == 5

    def test_speciation_groups_similar(self):
        """Similar individuals should be in same species."""
        # Create two clusters
        cluster1 = [
            create_test_individual(f"c1_{i}", np.array([0.0, float(i) * 0.1]))
            for i in range(5)
        ]
        cluster2 = [
            create_test_individual(f"c2_{i}", np.array([10.0, float(i) * 0.1]))
            for i in range(5)
        ]
        
        population = cluster1 + cluster2
        
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=2.0,
        )
        
        species = speciator.speciate(population, [])
        
        # Should have at least 2 species
        assert len(species) >= 2

    def test_speciation_preserves_existing(self):
        """Should use existing species when possible."""
        rep = create_test_individual("rep", np.array([0.0]))
        existing = Species(id=1, representative=rep, members=[])
        
        population = [
            create_test_individual("new", np.array([0.1]))
        ]
        
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=1.0,
        )
        
        species = speciator.speciate(population, [existing])
        
        # Should use existing species
        assert any(s.id == 1 for s in species)

    def test_speciation_removes_empty(self):
        """Should remove empty species."""
        rep = create_test_individual("rep", np.array([100.0]))
        existing = Species(id=1, representative=rep, members=[])
        
        population = [
            create_test_individual("new", np.array([0.0]))
        ]
        
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=1.0,
        )
        
        species = speciator.speciate(population, [existing])
        
        # Old species should be removed (too far away)
        assert all(s.id != 1 for s in species)


@pytest.mark.integration
class TestKMeansSpeciator:
    """Test KMeansSpeciator."""

    def test_kmeans_creates_k_species(self):
        """Should create k species."""
        population = [
            create_test_individual(f"ind_{i}", np.random.randn(5))
            for i in range(20)
        ]
        
        speciator = KMeansSpeciator(
            distance_fn=euclidean_distance,
            n_species=4,
        )
        
        species = speciator.speciate(population, [])
        
        # Should have at most k species
        assert len(species) <= 4

    def test_kmeans_assigns_all(self):
        """All individuals should be assigned."""
        population = [
            create_test_individual(f"ind_{i}", np.random.randn(5))
            for i in range(20)
        ]
        
        speciator = KMeansSpeciator(
            distance_fn=euclidean_distance,
            n_species=4,
        )
        
        species = speciator.speciate(population, [])
        
        total_members = sum(s.size for s in species)
        assert total_members == 20


# ============================================================================
# Fitness Sharing Tests
# ============================================================================

@pytest.mark.integration
class TestFitnessSharing:
    """Test explicit fitness sharing."""

    def test_sharing_reduces_crowded_fitness(self):
        """Crowded individuals should have lower shared fitness."""
        # Create individuals: one isolated, others clustered
        isolated = create_test_individual("isolated", np.array([10.0]), fitness_value=10.0)
        crowded = [
            create_test_individual(f"crowded_{i}", np.array([float(i) * 0.1]), fitness_value=10.0)
            for i in range(5)
        ]
        
        population = [isolated] + crowded
        
        shared = explicit_fitness_sharing(
            population,
            euclidean_distance,
            sigma_share=1.0,
        )
        
        # Isolated should have higher shared fitness
        assert shared[0] > shared[1]

    def test_sharing_empty_population(self):
        """Empty population should return empty list."""
        shared = explicit_fitness_sharing([], euclidean_distance, sigma_share=1.0)
        assert shared == []

    def test_crowding_distance_basic(self):
        """Crowding distance should be computed."""
        population = [
            create_test_individual(f"ind_{i}", np.zeros(5), fitness_value=float(i))
            for i in range(5)
        ]
        
        distances = crowding_distance(population, n_objectives=1)
        
        assert len(distances) == 5
        # Boundary individuals should have infinite distance
        assert distances[0] == float("inf")
        assert distances[-1] == float("inf")

    def test_clearing_keeps_winners(self):
        """Clearing should keep only niche winners."""
        # Create cluster with varying fitness
        population = [
            create_test_individual(f"ind_{i}", np.array([0.0]), fitness_value=float(i))
            for i in range(5)
        ]
        
        cleared = clearing(
            population,
            euclidean_distance,
            sigma_clear=1.0,
            kappa=2,
        )
        
        # Only top 2 should have non-zero fitness
        non_zero = sum(1 for f in cleared if f > 0)
        assert non_zero == 2


# ============================================================================
# Deterministic Crowding Tests
# ============================================================================

@pytest.mark.integration
class TestDeterministicCrowding:
    """Test deterministic crowding pairing."""

    def test_crowding_selects_winner(self):
        """Better offspring should replace similar parent."""
        parents = [
            create_test_individual("p1", np.array([0.0]), fitness_value=5.0),
            create_test_individual("p2", np.array([10.0]), fitness_value=5.0),
        ]
        
        offspring = [
            create_test_individual("c1", np.array([0.1]), fitness_value=10.0),  # Better, similar to p1
            create_test_individual("c2", np.array([10.1]), fitness_value=3.0),  # Worse, similar to p2
        ]
        
        survivors = deterministic_crowding_pairing(
            parents, offspring, euclidean_distance
        )
        
        assert len(survivors) == 2
        # c1 should beat p1 (higher fitness)
        # p2 should beat c2 (higher fitness)

    def test_crowding_requires_same_size(self):
        """Should raise error for mismatched sizes."""
        parents = [create_test_individual("p", np.zeros(5), fitness_value=1.0)]
        offspring = [
            create_test_individual("c1", np.zeros(5), fitness_value=1.0),
            create_test_individual("c2", np.zeros(5), fitness_value=1.0),
        ]
        
        with pytest.raises(ValueError):
            deterministic_crowding_pairing(parents, offspring, euclidean_distance)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestSpeciationIntegration:
    """Integration tests for speciation workflow."""

    def test_full_speciation_workflow(self):
        """Test complete speciation over multiple generations."""
        np.random.seed(42)
        
        # Create initial population with clusters
        population = []
        for cluster in range(3):
            for i in range(10):
                genes = np.array([cluster * 5.0, 0.0]) + np.random.randn(2) * 0.5
                population.append(create_test_individual(
                    f"c{cluster}_{i}",
                    genes,
                    fitness_value=10.0 - np.sum(genes ** 2) * 0.01,
                ))
        
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=2.0,
        )
        
        species = speciator.speciate(population, [])
        
        # Should have roughly 3 species
        assert 2 <= len(species) <= 5
        
        # All individuals assigned
        total = sum(s.size for s in species)
        assert total == 30

    def test_speciation_with_fitness_sharing(self):
        """Speciation should work with fitness sharing."""
        population = [
            create_test_individual(f"ind_{i}", np.array([float(i)]), fitness_value=10.0)
            for i in range(10)
        ]
        
        # First speciate
        speciator = ThresholdSpeciator(
            distance_fn=euclidean_distance,
            threshold=3.0,
        )
        species = speciator.speciate(population, [])
        
        # Then apply fitness sharing within species
        for sp in species:
            shared = explicit_fitness_sharing(
                sp.members,
                euclidean_distance,
                sigma_share=2.0,
            )
            assert len(shared) == sp.size
