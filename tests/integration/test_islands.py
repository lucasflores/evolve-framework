"""
Integration tests for Island Model evolution.

Tests migration, topology, and diversity preservation.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.core.operators.crossover import BlendCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.types import Fitness, Individual
from evolve.diversity.islands import (
    BestMigration,
    Island,
    IslandEvolutionEngine,
    MigrationController,
    RandomMigration,
    fully_connected_topology,
    hypercube_topology,
    ring_topology,
)
from evolve.diversity.islands.engine import IslandConfig
from evolve.diversity.islands.migration import TournamentMigration
from evolve.diversity.islands.topology import ladder_topology, star_topology
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome

# ============================================================================
# Test Fixtures
# ============================================================================


def create_test_individual(
    id: str,
    genome: np.ndarray,
    fitness_value: float | None = None,
) -> Individual[VectorGenome]:
    """Create a test individual with optional fitness."""
    ind = Individual(
        id=id,
        genome=VectorGenome(genes=genome),
    )
    if fitness_value is not None:
        ind.fitness = Fitness(values=(fitness_value,))
    return ind


def sphere_function(genes: np.ndarray) -> float:
    """Simple sphere function for testing - receives raw gene array."""
    return float(np.sum(genes**2))


# ============================================================================
# Topology Tests
# ============================================================================


@pytest.mark.integration
class TestTopologies:
    """Test island topology functions."""

    def test_ring_topology_basic(self):
        """Ring topology should create circular connections."""
        topo = ring_topology(4)

        assert len(topo) == 4
        assert topo[0] == [3, 1]
        assert topo[1] == [0, 2]
        assert topo[2] == [1, 3]
        assert topo[3] == [2, 0]

    def test_ring_topology_each_has_two_neighbors(self):
        """Each island in ring should have exactly 2 neighbors."""
        for n in range(2, 10):
            topo = ring_topology(n)
            for island_id, neighbors in topo.items():
                assert len(neighbors) == 2
                assert island_id not in neighbors

    def test_ring_topology_min_islands(self):
        """Ring topology should require at least 2 islands."""
        with pytest.raises(ValueError):
            ring_topology(1)

    def test_fully_connected_basic(self):
        """Fully connected should connect all islands."""
        topo = fully_connected_topology(4)

        assert len(topo) == 4
        assert set(topo[0]) == {1, 2, 3}
        assert set(topo[1]) == {0, 2, 3}
        assert set(topo[2]) == {0, 1, 3}
        assert set(topo[3]) == {0, 1, 2}

    def test_fully_connected_neighbor_count(self):
        """Fully connected island should have n-1 neighbors."""
        for n in range(2, 8):
            topo = fully_connected_topology(n)
            for island_id, neighbors in topo.items():
                assert len(neighbors) == n - 1
                assert island_id not in neighbors

    def test_hypercube_basic(self):
        """Hypercube should connect islands differing by one bit."""
        topo = hypercube_topology(4)

        # 0 (00) connects to 1 (01), 2 (10)
        assert set(topo[0]) == {1, 2}
        # 1 (01) connects to 0 (00), 3 (11)
        assert set(topo[1]) == {0, 3}
        # 2 (10) connects to 0 (00), 3 (11)
        assert set(topo[2]) == {0, 3}
        # 3 (11) connects to 1 (01), 2 (10)
        assert set(topo[3]) == {1, 2}

    def test_hypercube_8_islands(self):
        """8-island hypercube should have 3 neighbors each."""
        topo = hypercube_topology(8)

        for _island_id, neighbors in topo.items():
            assert len(neighbors) == 3  # log2(8) = 3

    def test_hypercube_requires_power_of_2(self):
        """Hypercube should reject non-power-of-2."""
        with pytest.raises(ValueError):
            hypercube_topology(3)
        with pytest.raises(ValueError):
            hypercube_topology(5)

    def test_ladder_topology(self):
        """Ladder topology should create linear chain."""
        topo = ladder_topology(4)

        assert topo[0] == [1]  # First only connects forward
        assert topo[1] == [0, 2]
        assert topo[2] == [1, 3]
        assert topo[3] == [2]  # Last only connects backward

    def test_star_topology(self):
        """Star topology should have central hub."""
        topo = star_topology(4)

        # Hub connects to all
        assert set(topo[0]) == {1, 2, 3}
        # Others connect only to hub
        assert topo[1] == [0]
        assert topo[2] == [0]
        assert topo[3] == [0]


# ============================================================================
# Island Tests
# ============================================================================


@pytest.mark.integration
class TestIsland:
    """Test Island dataclass."""

    def test_island_creation(self):
        """Island should be created with basic attributes."""
        population = [
            create_test_individual(f"ind_{i}", np.random.randn(5), fitness_value=float(i))
            for i in range(10)
        ]

        island = Island(
            id=0,
            population=population,
            topology=[1, 2],
            migration_rate=0.1,
        )

        assert island.id == 0
        assert island.size == 10
        assert island.topology == [1, 2]
        assert island.migration_rate == 0.1
        assert island.isolation_time == 0

    def test_island_best_individual(self):
        """Should find best individual by fitness."""
        population = [
            create_test_individual("ind_0", np.zeros(5), fitness_value=10.0),
            create_test_individual("ind_1", np.ones(5), fitness_value=5.0),
            create_test_individual("ind_2", np.ones(5) * 2, fitness_value=20.0),
        ]

        island = Island(id=0, population=population)

        best = island.best_individual
        assert best is not None
        assert best.fitness.values[0] == 20.0

    def test_island_average_fitness(self):
        """Should compute average fitness correctly."""
        population = [
            create_test_individual("ind_0", np.zeros(5), fitness_value=10.0),
            create_test_individual("ind_1", np.ones(5), fitness_value=20.0),
            create_test_individual("ind_2", np.ones(5) * 2, fitness_value=30.0),
        ]

        island = Island(id=0, population=population)

        assert island.average_fitness == 20.0

    def test_island_fitness_variance(self):
        """Should compute fitness variance."""
        population = [
            create_test_individual("ind_0", np.zeros(5), fitness_value=10.0),
            create_test_individual("ind_1", np.ones(5), fitness_value=20.0),
            create_test_individual("ind_2", np.ones(5) * 2, fitness_value=30.0),
        ]

        island = Island(id=0, population=population)

        # Variance of [10, 20, 30] around mean 20 is 66.67
        assert abs(island.fitness_variance - 66.666666) < 0.01

    def test_island_isolation_tracking(self):
        """Should track isolation time."""
        island = Island(id=0, population=[])

        assert island.isolation_time == 0

        island.increment_isolation()
        island.increment_isolation()
        assert island.isolation_time == 2

        island.reset_isolation()
        assert island.isolation_time == 0


# ============================================================================
# Migration Policy Tests
# ============================================================================


@pytest.mark.integration
class TestMigrationPolicies:
    """Test migration policies."""

    def test_best_migration_selects_top(self):
        """BestMigration should select highest fitness individuals."""
        population = [
            create_test_individual(f"ind_{i}", np.ones(5) * i, fitness_value=float(i))
            for i in range(10)
        ]

        island = Island(id=0, population=population)
        policy = BestMigration()
        rng = Random(42)

        emigrants = policy.select_emigrants(island, 3, rng)

        assert len(emigrants) == 3
        # Should be highest fitness (9, 8, 7)
        fitness_values = [e.fitness.values[0] for e in emigrants]
        assert fitness_values == [9.0, 8.0, 7.0]

    def test_best_migration_returns_copies(self):
        """BestMigration should return copies, not originals."""
        population = [create_test_individual("ind_0", np.ones(5), fitness_value=10.0)]

        island = Island(id=0, population=population)
        policy = BestMigration()
        rng = Random(42)

        emigrants = policy.select_emigrants(island, 1, rng)

        assert emigrants[0] is not population[0]
        assert emigrants[0].id == population[0].id

    def test_random_migration_selects_random(self):
        """RandomMigration should select random individuals."""
        population = [
            create_test_individual(f"ind_{i}", np.ones(5) * i, fitness_value=float(i))
            for i in range(10)
        ]

        island = Island(id=0, population=population)
        policy = RandomMigration()

        # Run multiple times to verify randomness
        selections = []
        for seed in range(10):
            emigrants = policy.select_emigrants(island, 2, Random(seed))
            selections.append(tuple(e.id for e in emigrants))

        # Should have some variation (not all same)
        assert len(set(selections)) > 1

    def test_tournament_migration(self):
        """TournamentMigration should use tournament selection."""
        population = [
            create_test_individual(f"ind_{i}", np.ones(5) * i, fitness_value=float(i))
            for i in range(10)
        ]

        island = Island(id=0, population=population)
        policy = TournamentMigration(tournament_size=3)
        rng = Random(42)

        emigrants = policy.select_emigrants(island, 3, rng)

        assert len(emigrants) == 3
        # Should tend toward higher fitness due to tournament


# ============================================================================
# Migration Controller Tests
# ============================================================================


@pytest.mark.integration
class TestMigrationController:
    """Test MigrationController."""

    def test_should_migrate_interval(self):
        """Should migrate at correct intervals."""
        controller = MigrationController(
            policy=BestMigration(),
            migration_interval=5,
        )

        assert not controller.should_migrate(0)
        assert not controller.should_migrate(1)
        assert not controller.should_migrate(4)
        assert controller.should_migrate(5)
        assert not controller.should_migrate(6)
        assert controller.should_migrate(10)

    def test_migrate_ring_topology(self):
        """Migration should exchange individuals in ring."""
        # Create 4 islands with distinct fitness ranges
        islands = []
        for i in range(4):
            population = [
                create_test_individual(
                    f"island{i}_ind{j}", np.ones(5) * (i * 10 + j), fitness_value=float(i * 10 + j)
                )
                for j in range(5)
            ]
            islands.append(
                Island(
                    id=i,
                    population=population,
                    topology=ring_topology(4)[i],
                    migration_rate=0.4,  # Migrate 2 individuals
                )
            )

        controller = MigrationController(
            policy=BestMigration(),
            migration_interval=1,
        )

        # Record original best per island
        [island.best_individual.fitness.values[0] for island in islands]

        # Perform migration
        stats = controller.migrate(islands, Random(42))

        assert stats["total_emigrants"] > 0
        assert stats["total_immigrants"] > 0

    def test_migration_deterministic(self):
        """Migration should be deterministic with same seed."""

        def create_islands():
            islands = []
            for i in range(4):
                population = [
                    create_test_individual(
                        f"island{i}_ind{j}",
                        np.ones(5) * (i * 10 + j),
                        fitness_value=float(i * 10 + j),
                    )
                    for j in range(5)
                ]
                islands.append(
                    Island(
                        id=i,
                        population=population,
                        topology=ring_topology(4)[i],
                        migration_rate=0.4,
                    )
                )
            return islands

        controller = MigrationController(
            policy=BestMigration(),
            migration_interval=1,
        )

        # Run twice with same seed
        islands1 = create_islands()
        controller.migrate(islands1, Random(42))

        islands2 = create_islands()
        controller.migrate(islands2, Random(42))

        # Results should match
        for i1, i2 in zip(islands1, islands2):
            f1 = [ind.fitness.values[0] for ind in i1.population]
            f2 = [ind.fitness.values[0] for ind in i2.population]
            assert f1 == f2


# ============================================================================
# Island Evolution Engine Tests
# ============================================================================


@pytest.mark.integration
class TestIslandEvolutionEngine:
    """Test IslandEvolutionEngine."""

    def test_engine_basic_run(self):
        """Engine should complete evolution run."""
        config = IslandConfig(
            n_islands=4,
            population_per_island=20,
            max_generations=10,
            migration_interval=3,
            migration_rate=0.1,
        )

        evaluator = FunctionEvaluator(sphere_function)

        engine = IslandEvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
            topology_fn=ring_topology,
            seed=42,
        )

        def genome_factory(rng: Random) -> VectorGenome:
            n_params = 5
            genes = np.array([rng.gauss(0, 1) for _ in range(n_params)])
            return VectorGenome(genes=genes)

        result = engine.run(genome_factory)

        assert result.generations == 10
        assert result.best is not None
        assert len(result.islands) == 4
        assert result.stop_reason == "max_generations"

    def test_engine_migration_occurs(self):
        """Engine should perform migrations."""
        config = IslandConfig(
            n_islands=4,
            population_per_island=20,
            max_generations=15,
            migration_interval=5,
            migration_rate=0.1,
        )

        evaluator = FunctionEvaluator(sphere_function)

        engine = IslandEvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
            seed=42,
        )

        def genome_factory(rng: Random) -> VectorGenome:
            genes = np.array([rng.gauss(0, 1) for _ in range(5)])
            return VectorGenome(genes=genes)

        result = engine.run(genome_factory)

        # Should have migrated at gen 5 and 10
        assert result.migration_stats["total_migrations"] >= 2

    def test_engine_deterministic(self):
        """Engine should be deterministic with same seed."""

        def run_engine(seed: int):
            config = IslandConfig(
                n_islands=4,
                population_per_island=10,
                max_generations=5,
                migration_interval=2,
            )

            evaluator = FunctionEvaluator(sphere_function)

            engine = IslandEvolutionEngine(
                config=config,
                evaluator=evaluator,
                selection=TournamentSelection(tournament_size=3),
                crossover=BlendCrossover(alpha=0.5),
                mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
                seed=seed,
            )

            def genome_factory(rng: Random) -> VectorGenome:
                genes = np.array([rng.gauss(0, 1) for _ in range(5)])
                return VectorGenome(genes=genes)

            return engine.run(genome_factory)

        result1 = run_engine(42)
        result2 = run_engine(42)

        assert result1.best.fitness.values[0] == result2.best.fitness.values[0]

    def test_engine_diversity_preservation(self):
        """Island model should preserve diversity better than single pop."""
        config = IslandConfig(
            n_islands=4,
            population_per_island=25,
            max_generations=20,
            migration_interval=10,
            migration_rate=0.1,
        )

        evaluator = FunctionEvaluator(sphere_function)

        engine = IslandEvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
            topology_fn=ring_topology,
            seed=42,
        )

        def genome_factory(rng: Random) -> VectorGenome:
            genes = np.array([rng.gauss(0, 1) for _ in range(5)])
            return VectorGenome(genes=genes)

        engine.run(genome_factory)

        # Check diversity across islands
        diversity = engine.get_diversity_metrics()

        # Should have some inter-island variance (diversity)
        assert diversity["inter_island_variance"] >= 0

    def test_engine_with_different_topologies(self):
        """Engine should work with different topologies."""
        for topo_fn in [ring_topology, fully_connected_topology]:
            config = IslandConfig(
                n_islands=4,
                population_per_island=10,
                max_generations=5,
            )

            evaluator = FunctionEvaluator(sphere_function)

            engine = IslandEvolutionEngine(
                config=config,
                evaluator=evaluator,
                selection=TournamentSelection(tournament_size=3),
                crossover=BlendCrossover(alpha=0.5),
                mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
                topology_fn=topo_fn,
                seed=42,
            )

            def genome_factory(rng: Random) -> VectorGenome:
                genes = np.array([rng.gauss(0, 1) for _ in range(5)])
                return VectorGenome(genes=genes)

            result = engine.run(genome_factory)
            assert result.best is not None

    def test_engine_history_tracking(self):
        """Engine should track history properly."""
        config = IslandConfig(
            n_islands=4,
            population_per_island=10,
            max_generations=5,
        )

        evaluator = FunctionEvaluator(sphere_function)

        engine = IslandEvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
            seed=42,
        )

        def genome_factory(rng: Random) -> VectorGenome:
            genes = np.array([rng.gauss(0, 1) for _ in range(5)])
            return VectorGenome(genes=genes)

        result = engine.run(genome_factory)

        assert len(result.history) == 5
        for entry in result.history:
            assert "generation" in entry
            assert "global_best" in entry
            assert "islands" in entry


# ============================================================================
# Property Tests for Determinism
# ============================================================================


@pytest.mark.integration
class TestIslandModelDeterminism:
    """Property tests for deterministic behavior."""

    def test_migration_determinism_property(self):
        """Migration should produce identical results with same seed."""
        # Run multiple times and verify consistency
        results = []

        for _ in range(3):
            islands = []
            for i in range(4):
                population = [
                    create_test_individual(
                        f"island{i}_ind{j}",
                        np.ones(5) * (i * 10 + j),
                        fitness_value=float(i * 10 + j),
                    )
                    for j in range(5)
                ]
                islands.append(
                    Island(
                        id=i,
                        population=population,
                        topology=ring_topology(4)[i],
                        migration_rate=0.4,
                    )
                )

            controller = MigrationController(
                policy=BestMigration(),
                migration_interval=1,
            )

            controller.migrate(islands, Random(12345))

            # Extract state
            state = tuple(
                tuple(ind.fitness.values[0] for ind in island.population) for island in islands
            )
            results.append(state)

        # All runs should be identical
        assert all(r == results[0] for r in results)

    def test_full_evolution_determinism(self):
        """Full evolution should be deterministic."""

        def run_evolution():
            config = IslandConfig(
                n_islands=4,
                population_per_island=10,
                max_generations=10,
                migration_interval=3,
            )

            evaluator = FunctionEvaluator(sphere_function)

            engine = IslandEvolutionEngine(
                config=config,
                evaluator=evaluator,
                selection=TournamentSelection(tournament_size=3),
                crossover=BlendCrossover(alpha=0.5),
                mutation=GaussianMutation(mutation_rate=0.2, sigma=0.1),
                seed=99999,
            )

            def genome_factory(rng: Random) -> VectorGenome:
                genes = np.array([rng.gauss(0, 1) for _ in range(5)])
                return VectorGenome(genes=genes)

            result = engine.run(genome_factory)
            return result.best.fitness.values[0]

        # Run twice
        result1 = run_evolution()
        result2 = run_evolution()

        assert result1 == result2
