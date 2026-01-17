"""
Integration tests for NSGA-II multi-objective optimization.

Tests verify:
- Non-dominated sorting correctness
- Pareto front approximation quality
- Hypervolume convergence
- Deterministic reproduction
"""

from __future__ import annotations

import numpy as np
import pytest
from random import Random

from evolve.core.types import Individual
from evolve.core.operators import UniformCrossover, GaussianMutation
from evolve.evaluation.reference.functions import zdt1, zdt2
from evolve.multiobjective import (
    MultiObjectiveFitness,
    dominates,
    pareto_front,
    fast_non_dominated_sort,
    crowding_distance,
    NSGA2Selector,
    CrowdedTournamentSelection,
    hypervolume_2d,
)
from evolve.representation.vector import VectorGenome


@pytest.fixture
def rng() -> Random:
    """Seeded RNG for deterministic tests."""
    return Random(42)


@pytest.fixture
def zdt1_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Standard bounds for ZDT1 (30 dimensions)."""
    n_dims = 30
    return (np.zeros(n_dims), np.ones(n_dims))


class TestParetoOperations:
    """Test basic Pareto dominance and front operations."""
    
    def test_dominance_basic(self):
        """Test basic Pareto dominance."""
        # f1 dominates f2: better on both objectives
        f1 = MultiObjectiveFitness(np.array([3.0, 2.0]))
        f2 = MultiObjectiveFitness(np.array([2.0, 1.0]))
        
        assert dominates(f1, f2)
        assert not dominates(f2, f1)
    
    def test_dominance_non_dominated(self):
        """Test that neither dominates when on same front."""
        f1 = MultiObjectiveFitness(np.array([3.0, 1.0]))
        f2 = MultiObjectiveFitness(np.array([1.0, 3.0]))
        
        assert not dominates(f1, f2)
        assert not dominates(f2, f1)
    
    def test_dominance_equal(self):
        """Test equal fitness."""
        f1 = MultiObjectiveFitness(np.array([2.0, 2.0]))
        f2 = MultiObjectiveFitness(np.array([2.0, 2.0]))
        
        # Strict dominance requires at least one strictly better
        assert not dominates(f1, f2, strict=True)
        # Weak dominance allows equality
        assert dominates(f1, f2, strict=False)
    
    def test_constrained_dominance(self):
        """Test feasibility-first constraint handling."""
        # Feasible dominates infeasible regardless of objectives
        feasible = MultiObjectiveFitness(
            np.array([1.0, 1.0]),
            constraint_violations=np.array([-1.0])  # Satisfied
        )
        infeasible = MultiObjectiveFitness(
            np.array([10.0, 10.0]),  # Better objectives but infeasible
            constraint_violations=np.array([1.0])  # Violated
        )
        
        assert dominates(feasible, infeasible)
        assert not dominates(infeasible, feasible)
    
    def test_pareto_front_simple(self):
        """Test Pareto front extraction."""
        fitnesses = [
            MultiObjectiveFitness(np.array([3.0, 1.0])),  # 0: on front
            MultiObjectiveFitness(np.array([2.0, 2.0])),  # 1: on front
            MultiObjectiveFitness(np.array([1.0, 3.0])),  # 2: on front
            MultiObjectiveFitness(np.array([1.5, 1.5])),  # 3: dominated by 1
        ]
        
        front = pareto_front(fitnesses)
        assert set(front) == {0, 1, 2}


class TestNonDominatedSorting:
    """Test NSGA-II non-dominated sorting."""
    
    def test_single_front(self):
        """Test when all solutions are non-dominated."""
        fitnesses = [
            MultiObjectiveFitness(np.array([3.0, 1.0])),
            MultiObjectiveFitness(np.array([2.0, 2.0])),
            MultiObjectiveFitness(np.array([1.0, 3.0])),
        ]
        
        fronts = fast_non_dominated_sort(fitnesses)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}
    
    def test_multiple_fronts(self):
        """Test sorting into multiple fronts."""
        fitnesses = [
            MultiObjectiveFitness(np.array([4.0, 1.0])),  # 0: front 0
            MultiObjectiveFitness(np.array([3.0, 2.0])),  # 1: front 0
            MultiObjectiveFitness(np.array([2.0, 3.0])),  # 2: front 0
            MultiObjectiveFitness(np.array([2.5, 1.5])),  # 3: front 1 (dominated by 1)
            MultiObjectiveFitness(np.array([1.5, 2.5])),  # 4: front 1 (dominated by 2)
            MultiObjectiveFitness(np.array([1.0, 1.0])),  # 5: front 2 (dominated by 3,4)
        ]
        
        fronts = fast_non_dominated_sort(fitnesses)
        
        assert len(fronts) >= 2
        assert set(fronts[0]) == {0, 1, 2}
        assert 3 in fronts[1] or 4 in fronts[1]


class TestCrowdingDistance:
    """Test crowding distance calculation."""
    
    def test_boundary_infinite(self):
        """Test that boundary solutions get infinite distance."""
        fitnesses = [
            MultiObjectiveFitness(np.array([1.0, 4.0])),  # 0: boundary (min f1)
            MultiObjectiveFitness(np.array([2.0, 3.0])),  # 1: interior
            MultiObjectiveFitness(np.array([3.0, 2.0])),  # 2: interior
            MultiObjectiveFitness(np.array([4.0, 1.0])),  # 3: boundary (max f1)
        ]
        
        distances = crowding_distance(fitnesses, [0, 1, 2, 3])
        
        assert distances[0] == float('inf')
        assert distances[3] == float('inf')
        assert 0 < distances[1] < float('inf')
        assert 0 < distances[2] < float('inf')
    
    def test_two_points_infinite(self):
        """Test that with only 2 points, both get infinite distance."""
        fitnesses = [
            MultiObjectiveFitness(np.array([1.0, 2.0])),
            MultiObjectiveFitness(np.array([2.0, 1.0])),
        ]
        
        distances = crowding_distance(fitnesses, [0, 1])
        
        assert distances[0] == float('inf')
        assert distances[1] == float('inf')


class TestNSGA2Selection:
    """Test NSGA-II selection operator."""
    
    def test_selection_prefers_lower_rank(self, rng):
        """Test that selection prefers lower Pareto rank."""
        # Create population with clear ranking
        population = []
        
        # Front 0 individuals
        for i in range(5):
            genome = VectorGenome(
                genes=np.array([0.1 * i, 0.1 * (4 - i)]),
                bounds=(np.zeros(2), np.ones(2))
            )
            fitness = MultiObjectiveFitness(np.array([3.0 - 0.5 * i, 1.0 + 0.5 * i]))
            population.append(Individual(genome=genome, fitness=fitness))
        
        # Front 1 individuals (dominated)
        for i in range(5):
            genome = VectorGenome(
                genes=np.array([0.5 + 0.1 * i, 0.5 - 0.1 * i]),
                bounds=(np.zeros(2), np.ones(2))
            )
            fitness = MultiObjectiveFitness(np.array([1.0 + 0.1 * i, 0.5 + 0.1 * i]))
            population.append(Individual(genome=genome, fitness=fitness))
        
        selector = NSGA2Selector()
        selected = selector.select(population, n_select=5, rng=rng)
        
        # Should select all from front 0
        selected_indices = [population.index(ind) for ind in selected]
        assert all(idx < 5 for idx in selected_indices)
    
    def test_crowded_tournament_selection(self, rng):
        """Test crowded tournament selection."""
        population = []
        for i in range(10):
            genome = VectorGenome(
                genes=np.array([0.1 * i]),
                bounds=(np.zeros(1), np.ones(1))
            )
            fitness = MultiObjectiveFitness(np.array([float(i), 10.0 - i]))
            population.append(Individual(genome=genome, fitness=fitness))
        
        # Compute ranks and crowding
        selector = NSGA2Selector()
        ranks, crowding = selector.get_ranking_info(population)
        
        # Tournament selection
        tournament = CrowdedTournamentSelection(tournament_size=2)
        selected = tournament.select(population, 5, ranks, crowding, rng)
        
        assert len(selected) == 5


class TestHypervolume:
    """Test hypervolume calculation."""
    
    def test_single_point(self):
        """Test hypervolume of single point."""
        points = np.array([[3.0, 2.0]])
        reference = np.array([0.0, 0.0])
        
        hv = hypervolume_2d(points, reference)
        
        # Area = 3 * 2 = 6
        assert hv == pytest.approx(6.0)
    
    def test_two_points(self):
        """Test hypervolume of two points."""
        points = np.array([[3.0, 1.0], [1.0, 3.0]])
        reference = np.array([0.0, 0.0])
        
        hv = hypervolume_2d(points, reference)
        
        # Area = 3*1 + 1*2 = 3 + 2 = 5... let me recalculate
        # Point (3,1): dominates 3x1 = 3 area from ref
        # Point (1,3): adds 1x2 = 2 area (from y=1 to y=3, x=0 to x=1)
        assert hv == pytest.approx(5.0)
    
    def test_empty_front(self):
        """Test hypervolume of empty front."""
        points = np.array([]).reshape(0, 2)
        reference = np.array([0.0, 0.0])
        
        hv = hypervolume_2d(points, reference)
        assert hv == 0.0


@pytest.mark.integration
class TestNSGA2Integration:
    """Integration test for NSGA-II on ZDT1."""
    
    def test_zdt1_pareto_front_approximation(self, rng, zdt1_bounds):
        """Test that NSGA-II produces reasonable Pareto front on ZDT1."""
        lower, upper = zdt1_bounds
        n_dims = len(lower)
        pop_size = 50
        n_generations = 50
        
        # Initialize population
        population: list[Individual[VectorGenome]] = []
        for _ in range(pop_size):
            genes = np.array([rng.uniform(lower[i], upper[i]) for i in range(n_dims)])
            genome = VectorGenome(genes=genes, bounds=(lower, upper))
            obj_values = zdt1(genome.genes)
            # Negate for maximization (ZDT minimizes)
            fitness = MultiObjectiveFitness(objectives=-obj_values)
            population.append(Individual(genome=genome, fitness=fitness))
        
        # Operators
        crossover = UniformCrossover(swap_prob=0.5)
        mutation = GaussianMutation(mutation_rate=0.1, sigma=0.1)
        selector = NSGA2Selector[VectorGenome]()
        tournament = CrowdedTournamentSelection[VectorGenome](tournament_size=2)
        
        # Evolution loop
        for gen in range(n_generations):
            # Get ranking info for tournament selection
            ranks, crowding = selector.get_ranking_info(population)
            
            # Generate offspring
            offspring: list[Individual[VectorGenome]] = []
            while len(offspring) < pop_size:
                # Select parents
                parents = tournament.select(population, 2, ranks, crowding, rng)
                p1, p2 = parents[0], parents[1]
                
                # Crossover (returns tuple of two children)
                child1_genome, child2_genome = crossover.crossover(p1.genome, p2.genome, rng)
                
                # Mutation and evaluate both children
                for child_genome in [child1_genome, child2_genome]:
                    if len(offspring) >= pop_size:
                        break
                    child_genome = mutation.mutate(child_genome, rng)
                    obj_values = zdt1(child_genome.genes)
                    fitness = MultiObjectiveFitness(objectives=-obj_values)
                    offspring.append(Individual(genome=child_genome, fitness=fitness))
            
            # Environmental selection (NSGA-II)
            combined = list(population) + offspring
            population = selector.select(combined, pop_size, rng)
        
        # Verify Pareto front quality
        fitnesses = [ind.fitness for ind in population]
        front_indices = pareto_front(fitnesses)
        
        # Should have reasonable front size
        assert len(front_indices) >= 10, f"Expected at least 10 on front, got {len(front_indices)}"
        
        # Extract front points (negate back to minimization for checking)
        front_points = np.array([
            -population[i].fitness.objectives 
            for i in front_indices
        ])
        
        # Front should be roughly along f2 = 1 - sqrt(f1)
        # Check that points are in reasonable range
        assert np.all(front_points[:, 0] >= 0)
        assert np.all(front_points[:, 0] <= 1)
        assert np.all(front_points[:, 1] >= 0)
        
        # Calculate hypervolume (with maximization objectives)
        max_front = -front_points  # Back to maximization
        reference = np.array([-1.5, -1.5])  # Worse than any feasible point
        hv = hypervolume_2d(max_front, reference)
        
        # Should have reasonable hypervolume
        assert hv > 0.5, f"Hypervolume {hv} too low"
    
    def test_deterministic_reproduction(self, zdt1_bounds):
        """Test that same seed produces identical results."""
        results = []
        
        for _ in range(2):
            rng = Random(12345)
            lower, upper = zdt1_bounds
            n_dims = len(lower)
            
            # Create identical starting population
            population: list[Individual[VectorGenome]] = []
            for _ in range(20):
                genes = np.array([rng.uniform(lower[i], upper[i]) for i in range(n_dims)])
                genome = VectorGenome(genes=genes, bounds=(lower, upper))
                obj_values = zdt1(genome.genes)
                fitness = MultiObjectiveFitness(objectives=-obj_values)
                population.append(Individual(genome=genome, fitness=fitness))
            
            # Run a few generations
            selector = NSGA2Selector[VectorGenome]()
            tournament = CrowdedTournamentSelection[VectorGenome]()
            crossover = UniformCrossover()
            mutation = GaussianMutation(mutation_rate=0.1, sigma=0.05)
            
            for _ in range(5):
                ranks, crowding = selector.get_ranking_info(population)
                offspring = []
                for _ in range(10):
                    parents = tournament.select(population, 2, ranks, crowding, rng)
                    child1, child2 = crossover.crossover(parents[0].genome, parents[1].genome, rng)
                    child1 = mutation.mutate(child1, rng)
                    fitness = MultiObjectiveFitness(objectives=-zdt1(child1.genes))
                    offspring.append(Individual(genome=child1, fitness=fitness))
                combined = list(population) + offspring
                population = selector.select(combined, 20, rng)
            
            # Record final state
            results.append([ind.fitness.objectives.tolist() for ind in population])
        
        # Both runs should produce identical results
        assert results[0] == results[1]
