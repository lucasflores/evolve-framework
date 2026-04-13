"""
EvolutionEngine - Main evolution loop orchestrator.

The engine coordinates:
1. Population initialization
2. Fitness evaluation
3. Selection
4. Variation (crossover + mutation)
5. Replacement
6. Termination checking

All randomness flows through explicit RNG instances.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any, Generic, TypeVar
from uuid import uuid4

import numpy as np

from evolve.core.callbacks import Callback
from evolve.core.population import Population
from evolve.core.stopping import (
    GenerationLimitStopping,
)
from evolve.core.types import Individual, IndividualMetadata
from evolve.evaluation.evaluator import Evaluator
from evolve.utils.random import create_rng
from evolve.utils.timing import GenerationTimer

G = TypeVar("G")


@dataclass
class EvolutionConfig:
    """
    Configuration for evolution run.

    Attributes:
        population_size: Number of individuals
        max_generations: Maximum generations
        elitism: Number of elites to preserve
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation per individual
        minimize: If True, lower fitness is better
        metric_categories: Set of metric category names to compute
    """

    population_size: int = 100
    max_generations: int = 100
    elitism: int = 1
    crossover_rate: float = 0.9
    mutation_rate: float = 1.0  # Per individual, not per gene
    minimize: bool = True
    metric_categories: frozenset[str] = field(default_factory=lambda: frozenset({"core"}))


@dataclass
class EvolutionResult(Generic[G]):
    """
    Result of an evolution run.

    Attributes:
        best: Best individual found
        population: Final population
        history: Metrics from each generation
        generations: Number of generations completed
        stop_reason: Why evolution terminated
    """

    best: Individual[G]
    population: Population[G]
    history: list[dict[str, Any]]
    generations: int
    stop_reason: str


class EvolutionEngine(Generic[G]):
    """
    Main evolution loop orchestrator.

    The engine coordinates the evolutionary process:
    1. Population initialization
    2. Fitness evaluation
    3. Selection
    4. Variation (crossover + mutation)
    5. Replacement
    6. Termination checking

    All randomness flows through explicit RNG instances for reproducibility.

    Example:
        >>> from evolve.core.engine import EvolutionEngine, EvolutionConfig
        >>> from evolve.core.operators import TournamentSelection, UniformCrossover, GaussianMutation
        >>> from evolve.evaluation import FunctionEvaluator
        >>> from evolve.evaluation.reference.functions import sphere
        >>>
        >>> config = EvolutionConfig(population_size=50, max_generations=100)
        >>> evaluator = FunctionEvaluator(sphere)
        >>> engine = EvolutionEngine(
        ...     config=config,
        ...     evaluator=evaluator,
        ...     selection=TournamentSelection(),
        ...     crossover=UniformCrossover(),
        ...     mutation=GaussianMutation(),
        ...     seed=42
        ... )
        >>> result = engine.run(initial_population=pop)
        >>> print(f"Best fitness: {result.best.fitness}")
    """

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: Evaluator[G],
        selection: Any,  # SelectionOperator[G]
        crossover: Any,  # CrossoverOperator[G]
        mutation: Any,  # MutationOperator[G]
        seed: int = 42,
        stopping: Any | None = None,
        callbacks: Sequence[Callback[G]] | None = None,
    ) -> None:
        """
        Initialize engine with configuration.

        Args:
            config: Evolution parameters
            evaluator: Fitness evaluator
            selection: Selection operator
            crossover: Crossover operator
            mutation: Mutation operator
            seed: Master random seed
            stopping: Optional stopping criterion (default: generation limit)
            callbacks: Optional callbacks to persist across all run() calls
        """
        self.config = config
        self.evaluator = evaluator
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.seed = seed
        self.rng = create_rng(seed)

        # Set up stopping criterion
        if stopping is None:
            self.stopping = GenerationLimitStopping(config.max_generations)
        else:
            self.stopping = stopping

        # State
        self._generation = 0
        self._history: list[dict[str, Any]] = []
        self._creation_callbacks: list[Callback[G]] = list(callbacks) if callbacks else []
        self._callbacks: list[Callback[G]] = []
        self._timer = GenerationTimer()
        self._prev_centroid: np.ndarray | None = None
        self._prev_best_genome: Any = None

    def run(
        self,
        initial_population: Population[G],
        callbacks: Sequence[Callback[G]] | None = None,
    ) -> EvolutionResult[G]:
        """
        Execute full evolution run.

        Args:
            initial_population: Starting population
            callbacks: Optional event callbacks

        Returns:
            EvolutionResult with best individual and history
        """
        # Merge creation-time callbacks with run-time callbacks, sort by priority
        run_callbacks = list(callbacks) if callbacks else []
        merged = self._creation_callbacks + run_callbacks
        self._callbacks = sorted(merged, key=lambda cb: getattr(cb, "priority", 0))
        self._history = []
        self._generation = 0

        # Reset stopping criteria if they have state
        if hasattr(self.stopping, "reset"):
            self.stopping.reset()

        # Notify run start
        for cb in self._callbacks:
            if hasattr(cb, "on_run_start"):
                cb.on_run_start(self.config)

        # Evaluate initial population
        population = self._evaluate_population(initial_population)

        stop_reason = "Unknown"

        while True:
            # Check stopping condition
            if self.stopping.should_stop(self._generation, population, self._history):
                stop_reason = self.stopping.reason
                break

            # Notify generation start
            for cb in self._callbacks:
                if hasattr(cb, "on_generation_start"):
                    cb.on_generation_start(self._generation, population)

            # Evolution step
            population = self._step(population)

            # Compute metrics
            metrics = self._compute_metrics(population)
            self._history.append(metrics)

            # Notify generation end
            for cb in self._callbacks:
                if hasattr(cb, "on_generation_end"):
                    cb.on_generation_end(self._generation, population, metrics)

            self._generation += 1

        # Find best individual
        best = self._get_best(population)

        # Notify run end
        for cb in self._callbacks:
            if hasattr(cb, "on_run_end"):
                cb.on_run_end(population, stop_reason)

        return EvolutionResult(
            best=best,
            population=population,
            history=self._history,
            generations=self._generation,
            stop_reason=stop_reason,
        )

    def _step(self, population: Population[G]) -> Population[G]:
        """
        Perform one evolution step with timing instrumentation.

        1. Select parents
        2. Apply crossover
        3. Apply mutation
        4. Preserve elites
        5. Create new generation
        6. Evaluate

        Timing is captured via self._timer for:
        - selection_time_ms
        - variation_time_ms (crossover + mutation)
        - evaluation_time_ms
        """
        # Reset timer for this generation
        self._timer.reset()
        self._timer.start_generation()

        pop_size = self.config.population_size
        n_elites = self.config.elitism
        n_offspring = pop_size - n_elites

        # Get elites (best individuals preserved unchanged)
        elites = list(population.best(n_elites, minimize=self.config.minimize))

        # Time selection phase
        self._timer.start("selection")
        n_parents = n_offspring * 2
        parents = list(self.selection.select(population, n_parents, self.rng))
        self._timer.stop("selection")

        # Time variation phase (crossover + mutation)
        self._timer.start("variation")
        offspring: list[Individual[G]] = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1_genome, child2_genome = self.crossover.crossover(
                    parent1.genome, parent2.genome, self.rng
                )
            else:
                child1_genome = parent1.genome.copy()
                child2_genome = parent2.genome.copy()

            # Mutation
            if self.rng.random() < self.config.mutation_rate:
                child1_genome = self.mutation.mutate(child1_genome, self.rng)
            if self.rng.random() < self.config.mutation_rate:
                child2_genome = self.mutation.mutate(child2_genome, self.rng)

            # Create individuals
            child1 = Individual(
                id=uuid4(),
                genome=child1_genome,
                metadata=IndividualMetadata(
                    parent_ids=(parent1.id, parent2.id),
                    origin="crossover",
                ),
                created_at=self._generation + 1,
            )
            child2 = Individual(
                id=uuid4(),
                genome=child2_genome,
                metadata=IndividualMetadata(
                    parent_ids=(parent1.id, parent2.id),
                    origin="crossover",
                ),
                created_at=self._generation + 1,
            )

            offspring.extend([child1, child2])

        # Trim to exact size
        offspring = offspring[:n_offspring]
        self._timer.stop("variation")

        # Combine elites and offspring
        new_individuals = elites + offspring

        # Create new population
        new_population = Population(
            individuals=new_individuals,
            generation=self._generation + 1,
            minimize=self.config.minimize,
        )

        # Time evaluation phase
        self._timer.start("evaluation")
        evaluated_population = self._evaluate_population(new_population)
        self._timer.stop("evaluation")

        # End generation timing
        self._timer.end_generation()

        return evaluated_population

    def _evaluate_population(self, population: Population[G]) -> Population[G]:
        """Evaluate unevaluated individuals."""
        # Find individuals needing evaluation
        to_evaluate = [ind for ind in population.individuals if ind.fitness is None]

        if not to_evaluate:
            return population

        # Evaluate
        fitness_values = self.evaluator.evaluate(to_evaluate, seed=self.rng.randint(0, 2**31))

        # Update individuals with fitness
        fitness_map = {to_evaluate[i].id: fitness_values[i] for i in range(len(to_evaluate))}

        updated = [
            ind.with_fitness(fitness_map[ind.id]) if ind.id in fitness_map else ind
            for ind in population.individuals
        ]

        return Population(
            individuals=updated, generation=population.generation, minimize=self.config.minimize
        )

    def _compute_metrics(self, population: Population[G]) -> dict[str, Any]:
        """Compute generation metrics including timing."""
        stats = population.statistics
        categories = self.config.metric_categories

        metrics: dict[str, Any] = {
            "generation": self._generation,
            "population_size": len(population),
            "evaluated_count": stats.evaluated_count,
        }

        if stats.best_fitness is not None:
            metrics["best_fitness"] = float(stats.best_fitness.values[0])

        if stats.worst_fitness is not None:
            metrics["worst_fitness"] = float(stats.worst_fitness.values[0])

        if stats.mean_fitness is not None:
            metrics["mean_fitness"] = float(stats.mean_fitness.values[0])

        if stats.std_fitness is not None:
            metrics["std_fitness"] = stats.std_fitness

        # Extended population metrics (fitness distribution)
        if "extended_population" in categories:
            evaluated = [
                ind
                for ind in population.individuals
                if ind.fitness is not None and ind.fitness.n_objectives == 1
            ]
            if evaluated:
                values = np.array(
                    [float(ind.fitness.values[0]) for ind in evaluated if ind.fitness is not None]
                )
                metrics["median_fitness"] = float(np.median(values))
                metrics["q1_fitness"] = float(np.percentile(values, 25))
                metrics["q3_fitness"] = float(np.percentile(values, 75))
                metrics["min_fitness"] = float(np.min(values))
                metrics["max_fitness"] = float(np.max(values))
                metrics["fitness_range"] = float(np.max(values) - np.min(values))
                metrics["unique_fitness_count"] = int(len(np.unique(values)))

        # Diversity metrics
        if "diversity" in categories:
            self._compute_diversity_metrics(population, metrics)

        # Add timing metrics (selection, variation, evaluation, total)
        timing_metrics = self._timer.get_metrics(breakdown=True)
        metrics.update(timing_metrics)

        return metrics

    def _compute_diversity_metrics(
        self, population: Population[G], metrics: dict[str, Any]
    ) -> None:
        """Compute genome diversity and search movement metrics."""
        from evolve.representation.vector import VectorGenome

        evaluated = [ind for ind in population.individuals if ind.fitness is not None]
        if not evaluated:
            return

        genomes = [ind.genome for ind in evaluated]

        # Gene-level statistics (VectorGenome only)
        if isinstance(genomes[0], VectorGenome):
            gene_matrix = np.array([g.genes for g in genomes])  # type: ignore[attr-defined]
            gene_stds = np.std(gene_matrix, axis=0)
            metrics["mean_gene_std"] = float(np.mean(gene_stds))

            centroid = np.mean(gene_matrix, axis=0)
            dists_from_centroid = np.linalg.norm(gene_matrix - centroid, axis=1)
            metrics["mean_distance_from_centroid"] = float(np.mean(dists_from_centroid))

            # Centroid drift
            if self._prev_centroid is not None:
                metrics["centroid_drift"] = float(np.linalg.norm(centroid - self._prev_centroid))
            self._prev_centroid = centroid

        # Sampled pairwise distance via Genome.distance() protocol
        if hasattr(genomes[0], "distance"):
            n = len(genomes)
            if n >= 2:
                # Sample up to 50 pairs for efficiency
                max_pairs = min(50, n * (n - 1) // 2)
                rng = self.rng
                total_dist = 0.0
                count = 0
                for _ in range(max_pairs):
                    i = rng.randint(0, n - 1)
                    j = rng.randint(0, n - 2)
                    if j >= i:
                        j += 1
                    total_dist += genomes[i].distance(genomes[j])  # type: ignore[attr-defined]
                    count += 1
                if count > 0:
                    metrics["mean_pairwise_distance"] = total_dist / count

        # Best genome similarity / best_changed
        best_list = population.best(1, minimize=self.config.minimize)
        if best_list:
            current_best_genome = best_list[0].genome
            if self._prev_best_genome is not None and hasattr(current_best_genome, "distance"):
                with contextlib.suppress(TypeError, ValueError):
                    metrics["best_genome_similarity"] = current_best_genome.distance(
                        self._prev_best_genome
                    )
            metrics["best_changed"] = (
                self._prev_best_genome is None or current_best_genome != self._prev_best_genome
            )
            self._prev_best_genome = current_best_genome

    def _get_best(self, population: Population[G]) -> Individual[G]:
        """Get best individual from population."""
        best_list = population.best(1, minimize=self.config.minimize)
        return best_list[0]

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def history(self) -> list[dict[str, Any]]:
        """History of generation metrics."""
        return self._history.copy()

    def get_rng_state(self) -> tuple[Any, ...]:
        """Get RNG state for checkpointing."""
        return self.rng.getstate()

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        """Restore RNG state from checkpoint."""
        self.rng.setstate(state)


def create_initial_population(
    genome_factory: Callable[[Random], G],
    population_size: int,
    rng: Random,
) -> Population[G]:
    """
    Create initial population using a genome factory.

    Args:
        genome_factory: Function that creates random genomes
        population_size: Number of individuals
        rng: Random number generator

    Returns:
        Initial population (unevaluated)
    """
    individuals = [
        Individual(
            id=uuid4(),
            genome=genome_factory(rng),
            metadata=IndividualMetadata(origin="init"),
            created_at=0,
        )
        for _ in range(population_size)
    ]
    return Population(individuals=individuals, generation=0)
