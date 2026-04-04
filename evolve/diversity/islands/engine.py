"""
Island Evolution Engine.

Coordinates multi-population evolution with migration.
Each island evolves independently with periodic migration
to neighboring islands based on the configured topology.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any, Generic, TypeVar
from uuid import uuid4

import numpy as np

from evolve.core.callbacks import Callback
from evolve.core.population import Population
from evolve.core.types import Individual
from evolve.diversity.islands.island import Island
from evolve.diversity.islands.migration import BestMigration, MigrationController
from evolve.diversity.islands.topology import ring_topology
from evolve.evaluation.evaluator import Evaluator
from evolve.utils.random import create_rng

G = TypeVar("G")


@dataclass
class IslandConfig:
    """
    Configuration for island model evolution.

    Attributes:
        n_islands: Number of islands
        population_per_island: Individuals per island
        max_generations: Maximum generations to run
        migration_interval: Generations between migrations
        migration_rate: Fraction of population to migrate
        elitism: Elites to preserve per island
        crossover_rate: Base crossover probability
        mutation_rate: Base mutation probability
        minimize: If True, lower fitness is better
    """

    n_islands: int = 4
    population_per_island: int = 50
    max_generations: int = 100
    migration_interval: int = 10
    migration_rate: float = 0.1
    elitism: int = 1
    crossover_rate: float = 0.9
    mutation_rate: float = 1.0
    minimize: bool = True


@dataclass
class IslandResult(Generic[G]):
    """
    Result of island model evolution.

    Attributes:
        best: Global best individual across all islands
        islands: Final state of all islands
        history: Metrics from each generation
        generations: Number of generations completed
        stop_reason: Why evolution terminated
        migration_stats: Cumulative migration statistics
    """

    best: Individual[G]
    islands: list[Island[G]]
    history: list[dict[str, Any]]
    generations: int
    stop_reason: str
    migration_stats: dict[str, int] = field(default_factory=dict)

    @property
    def total_population(self) -> int:
        """Total individuals across all islands."""
        return sum(island.size for island in self.islands)

    def get_all_individuals(self) -> list[Individual[G]]:
        """Get all individuals from all islands."""
        return [ind for island in self.islands for ind in island.population]


class IslandEvolutionEngine(Generic[G]):
    """
    Multi-population evolution engine with island model.

    Each island evolves independently using standard GA operators.
    Periodic migration exchanges individuals between connected
    islands based on the topology graph.

    Features:
    - Configurable topologies (ring, fully connected, hypercube)
    - Pluggable migration policies
    - Per-island configuration overrides
    - Deterministic with fixed seed

    Example:
        >>> from evolve.diversity.islands import (
        ...     IslandEvolutionEngine, IslandConfig, ring_topology
        ... )
        >>>
        >>> config = IslandConfig(n_islands=4, population_per_island=50)
        >>> engine = IslandEvolutionEngine(
        ...     config=config,
        ...     evaluator=evaluator,
        ...     selection=TournamentSelection(),
        ...     crossover=BlendCrossover(),
        ...     mutation=GaussianMutation(),
        ...     topology_fn=ring_topology,
        ...     seed=42
        ... )
        >>> result = engine.run(genome_factory)
    """

    def __init__(
        self,
        config: IslandConfig,
        evaluator: Evaluator[G],
        selection: Any,  # SelectionOperator[G]
        crossover: Any,  # CrossoverOperator[G]
        mutation: Any,  # MutationOperator[G]
        topology_fn: Callable[[int], dict[int, list[int]]] | None = None,
        migration_controller: MigrationController[G] | None = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize island evolution engine.

        Args:
            config: Island model configuration
            evaluator: Fitness evaluator
            selection: Selection operator
            crossover: Crossover operator
            mutation: Mutation operator
            topology_fn: Function to generate topology (default: ring)
            migration_controller: Migration controller (default: BestMigration)
            seed: Master random seed
        """
        self.config = config
        self.evaluator = evaluator
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.seed = seed
        self.rng = create_rng(seed)

        # Set up topology
        if topology_fn is None:
            topology_fn = ring_topology
        self.topology = topology_fn(config.n_islands)

        # Set up migration
        if migration_controller is None:
            migration_controller = MigrationController(
                policy=BestMigration(),
                migration_interval=config.migration_interval,
            )
        self.migration_controller = migration_controller

        # State
        self._generation = 0
        self._history: list[dict[str, Any]] = []
        self._callbacks: list[Callback[G]] = []
        self._islands: list[Island[G]] = []
        self._migration_stats: dict[str, int] = {
            "total_migrations": 0,
            "total_emigrants": 0,
            "total_immigrants": 0,
        }

    def run(
        self,
        genome_factory: Callable[[Random], G],
        callbacks: Sequence[Callback[G]] | None = None,
        initial_islands: list[Island[G]] | None = None,
    ) -> IslandResult[G]:
        """
        Execute island model evolution.

        Args:
            genome_factory: Function to create random genomes
            callbacks: Optional event callbacks
            initial_islands: Pre-configured islands (optional)

        Returns:
            IslandResult with best individual and all islands
        """
        self._callbacks = list(callbacks) if callbacks else []
        self._history = []
        self._generation = 0
        self._migration_stats = {
            "total_migrations": 0,
            "total_emigrants": 0,
            "total_immigrants": 0,
        }

        # Initialize islands
        if initial_islands is not None:
            self._islands = initial_islands
            # Update topology if not set
            for island in self._islands:
                if not island.topology:
                    island.topology = self.topology.get(island.id, [])
        else:
            self._islands = self._initialize_islands(genome_factory)

        # Evaluate all islands
        self._evaluate_all_islands()

        stop_reason = "max_generations"

        while self._generation < self.config.max_generations:
            # Check for migration
            if self.migration_controller.should_migrate(self._generation):
                stats = self.migration_controller.migrate(
                    self._islands, Random(self.rng.randint(0, 2**31 - 1))
                )
                self._migration_stats["total_migrations"] += 1
                self._migration_stats["total_emigrants"] += stats["total_emigrants"]
                self._migration_stats["total_immigrants"] += stats["total_immigrants"]

            # Evolve each island
            for island in self._islands:
                self._evolve_island(island)
                island.increment_isolation()

            # Re-evaluate all islands
            self._evaluate_all_islands()

            # Record history
            self._record_generation_stats()

            self._generation += 1

            # Notify callbacks
            for cb in self._callbacks:
                if hasattr(cb, "on_generation_end"):
                    cb.on_generation_end(
                        self._generation,
                        self._get_combined_population(),
                        self._history[-1] if self._history else {},
                    )

        # Find global best
        best = self._find_global_best()

        return IslandResult(
            best=best,
            islands=self._islands,
            history=self._history,
            generations=self._generation,
            stop_reason=stop_reason,
            migration_stats=self._migration_stats,
        )

    def _initialize_islands(
        self,
        genome_factory: Callable[[Random], G],
    ) -> list[Island[G]]:
        """Create initial islands with random populations."""
        islands = []

        for island_id in range(self.config.n_islands):
            # Derive seed for this island
            island_seed = self.rng.randint(0, 2**31 - 1)
            island_rng = Random(island_seed)

            # Create population
            population = []
            for _ in range(self.config.population_per_island):
                genome = genome_factory(island_rng)
                individual = Individual(
                    id=str(uuid4()),
                    genome=genome,
                )
                population.append(individual)

            # Create island
            island = Island(
                id=island_id,
                population=population,
                topology=self.topology.get(island_id, []),
                migration_rate=self.config.migration_rate,
            )
            islands.append(island)

        return islands

    def _evaluate_all_islands(self) -> None:
        """Evaluate all individuals on all islands."""
        for island in self._islands:
            # Collect unevaluated individuals
            unevaluated = [ind for ind in island.population if ind.fitness is None]

            if unevaluated:
                # Evaluate batch
                eval_seed = self.rng.randint(0, 2**31 - 1)
                fitnesses = self.evaluator.evaluate(unevaluated, seed=eval_seed)

                # Assign fitness
                for ind, fitness in zip(unevaluated, fitnesses):
                    ind.fitness = fitness

    def _evolve_island(self, island: Island[G]) -> None:
        """
        Perform one generation of evolution on an island.

        Args:
            island: Island to evolve
        """
        # Get operators (island-specific or default)
        selection = island.selection_operator or self.selection
        crossover_rate = island.crossover_rate or self.config.crossover_rate
        mutation_rate = island.mutation_rate or self.config.mutation_rate

        # Derive seed for this island's generation
        gen_seed = self.rng.randint(0, 2**31 - 1)
        gen_rng = Random(gen_seed)

        # Sort by fitness for elitism
        sorted_pop = sorted(
            island.population,
            key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf"),
            reverse=not self.config.minimize,
        )

        # Preserve elites
        new_population = []
        for i in range(min(self.config.elitism, len(sorted_pop))):
            elite = sorted_pop[i]
            new_population.append(
                Individual(
                    id=str(uuid4()),
                    genome=elite.genome,  # Keep same genome
                    fitness=elite.fitness,  # Keep fitness
                )
            )

        # Generate offspring
        target_size = self.config.population_per_island
        n_offspring = target_size - len(new_population)

        # Convert island population to Population object for selection
        island_pop = Population(individuals=island.population)

        # Select parent pairs
        n_parents = n_offspring * 2
        parents = list(selection.select(island_pop, n_parents, gen_rng))

        # Create offspring via crossover and mutation
        for i in range(0, len(parents) - 1, 2):
            if len(new_population) >= target_size:
                break

            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if gen_rng.random() < crossover_rate:
                child1_genome, child2_genome = self.crossover.crossover(
                    parent1.genome, parent2.genome, gen_rng
                )
            else:
                child1_genome = parent1.genome
                child2_genome = parent2.genome

            # Mutation
            if gen_rng.random() < mutation_rate:
                child1_genome = self.mutation.mutate(child1_genome, gen_rng)
            if gen_rng.random() < mutation_rate:
                child2_genome = self.mutation.mutate(child2_genome, gen_rng)

            # Create offspring
            new_population.append(
                Individual(
                    id=str(uuid4()),
                    genome=child1_genome,
                )
            )

            if len(new_population) < target_size:
                new_population.append(
                    Individual(
                        id=str(uuid4()),
                        genome=child2_genome,
                    )
                )

        island.population = new_population[:target_size]

    def _find_global_best(self) -> Individual[G]:
        """Find best individual across all islands."""
        all_individuals = self._get_combined_population()

        if self.config.minimize:
            return min(
                all_individuals,
                key=lambda ind: ind.fitness.values[0] if ind.fitness else float("inf"),
            )
        else:
            return max(
                all_individuals,
                key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf"),
            )

    def _get_combined_population(self) -> Population[G]:
        """Get all individuals from all islands as a Population."""
        all_individuals = [ind for island in self._islands for ind in island.population]
        return Population(individuals=all_individuals)

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        stats: dict[str, Any] = {
            "generation": self._generation,
            "islands": {},
        }

        all_fitness = []

        for island in self._islands:
            island_fitness = [
                ind.fitness.values[0] for ind in island.population if ind.fitness is not None
            ]

            if island_fitness:
                stats["islands"][island.id] = {
                    "size": island.size,
                    "best_fitness": min(island_fitness)
                    if self.config.minimize
                    else max(island_fitness),
                    "avg_fitness": sum(island_fitness) / len(island_fitness),
                    "fitness_variance": np.var(island_fitness),
                    "isolation_time": island.isolation_time,
                }
                all_fitness.extend(island_fitness)

        if all_fitness:
            stats["global_best"] = min(all_fitness) if self.config.minimize else max(all_fitness)
            stats["global_avg"] = sum(all_fitness) / len(all_fitness)
            stats["global_variance"] = float(np.var(all_fitness))

        self._history.append(stats)

    def get_diversity_metrics(self) -> dict[str, float]:
        """
        Calculate diversity metrics across islands.

        Returns:
            Dictionary with diversity metrics:
            - inter_island_variance: Variance of island mean fitness
            - intra_island_variance: Average within-island variance
            - diversity_ratio: inter/intra variance ratio
        """
        island_means = []
        island_variances = []

        for island in self._islands:
            fitness_values = [
                ind.fitness.values[0] for ind in island.population if ind.fitness is not None
            ]

            if fitness_values:
                island_means.append(np.mean(fitness_values))
                island_variances.append(np.var(fitness_values))

        if len(island_means) < 2:
            return {
                "inter_island_variance": 0.0,
                "intra_island_variance": 0.0,
                "diversity_ratio": 1.0,
            }

        inter_variance = float(np.var(island_means))
        intra_variance = float(np.mean(island_variances))

        return {
            "inter_island_variance": inter_variance,
            "intra_island_variance": intra_variance,
            "diversity_ratio": inter_variance / max(intra_variance, 1e-10),
        }
