"""
Meta-Evolution Evaluator.

Provides infrastructure for evaluating configurations by running
inner evolutionary loops and aggregating fitness.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from statistics import mean, median
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from evolve.config.meta import MetaEvolutionConfig
    from evolve.config.unified import UnifiedConfig
    from evolve.meta.result import MetaEvolutionResult


@dataclass
class MetaEvaluator:
    """
    Evaluate configurations by running inner evolution (T063).

    For each configuration, runs the inner evolutionary loop and
    returns fitness based on the inner loop's best solution.

    Attributes:
        base_config: Base configuration template.
        meta_config: Meta-evolution settings.
        fitness_fn: Inner loop fitness function.
        seed: Base random seed.

    Example:
        >>> evaluator = MetaEvaluator(
        ...     base_config=base_config,
        ...     meta_config=meta_config,
        ...     fitness_fn=my_fitness,
        ... )
        >>> fitness = evaluator.evaluate(candidate_config)
    """

    base_config: UnifiedConfig
    """Base configuration template."""

    meta_config: MetaEvolutionConfig
    """Meta-evolution settings."""

    fitness_fn: Callable[[Any], float]
    """Inner loop fitness function."""

    seed: int = 42
    """Base random seed."""

    _cache: dict[str, tuple[float, Any]] = field(default_factory=dict)
    """Cache mapping config hash -> (fitness, best_solution)."""

    _trials_run: int = 0
    """Total number of trials run."""

    def evaluate(self, config: UnifiedConfig) -> float:
        """
        Evaluate a configuration by running inner evolution (T064).

        Runs multiple trials if configured, aggregating results.

        Args:
            config: Configuration to evaluate.

        Returns:
            Aggregated fitness across trials.
        """
        from evolve.factory.engine import create_engine, create_initial_population

        # Check cache first
        config_hash = config.compute_hash()
        if config_hash in self._cache:
            return self._cache[config_hash][0]

        fitnesses: list[float] = []
        best_solution: Any = None
        best_fitness: float = float("inf") if config.minimize else float("-inf")

        for trial in range(self.meta_config.trials_per_config):
            # Compute deterministic trial seed (T065)
            trial_seed = self._compute_inner_seed(config, trial)

            # Apply inner_generations override if specified
            if self.meta_config.inner_generations is not None:
                config = config.with_params(max_generations=self.meta_config.inner_generations)

            # Create and run inner evolution
            engine = create_engine(config, self.fitness_fn, seed=trial_seed)
            population = create_initial_population(config, seed=trial_seed)

            result = engine.run(population)
            self._trials_run += 1

            # Extract fitness
            trial_fitness = self._extract_fitness(result)
            fitnesses.append(trial_fitness)

            # Track best solution
            if self._is_better(trial_fitness, best_fitness, config.minimize):
                best_fitness = trial_fitness
                best_solution = self._extract_best_individual(result)

        # Aggregate fitness (T066)
        aggregated = self._aggregate_fitness(fitnesses)

        # Cache result
        self._cache[config_hash] = (aggregated, best_solution)

        return aggregated

    def _compute_inner_seed(self, config: UnifiedConfig, trial: int) -> int:
        """
        Compute deterministic seed for inner evolution (T065).

        Combines base seed, config hash, and trial number for
        reproducible yet varied trials.

        Args:
            config: Configuration being evaluated.
            trial: Trial index (0-based).

        Returns:
            Deterministic seed for this trial.
        """
        config_hash = config.compute_hash()
        combined = f"{self.seed}:{config_hash}:{trial}"
        hash_bytes = hashlib.md5(combined.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") % (2**31)

    def _aggregate_fitness(self, fitnesses: list[float]) -> float:
        """
        Aggregate fitness values across trials (T066).

        Supports mean, median, and best aggregation.

        Args:
            fitnesses: List of fitness values from trials.

        Returns:
            Aggregated fitness.
        """
        if not fitnesses:
            return float("nan")

        aggregation = self.meta_config.aggregation

        if aggregation == "mean":
            return mean(fitnesses)
        elif aggregation == "median":
            return median(fitnesses)
        elif aggregation == "best":
            minimize = self.base_config.minimize
            return min(fitnesses) if minimize else max(fitnesses)
        else:
            return mean(fitnesses)

    def get_cached_solution(self, config: UnifiedConfig) -> Any | None:
        """
        Get cached best solution for configuration (T067).

        Args:
            config: Configuration to look up.

        Returns:
            Best solution if cached, None otherwise.
        """
        config_hash = config.compute_hash()
        if config_hash in self._cache:
            return self._cache[config_hash][1]
        return None

    @property
    def trials_run(self) -> int:
        """Get total number of trials run."""
        return self._trials_run

    def _extract_fitness(self, result: Any) -> float:
        """Extract fitness from evolution result."""
        # Try different result formats

        # EvolutionResult has result.best.fitness
        if hasattr(result, "best") and hasattr(result.best, "fitness"):
            fitness = result.best.fitness
            if fitness is not None:
                if hasattr(fitness, "values"):
                    return float(fitness.values[0])
                return float(fitness)

        if hasattr(result, "best_fitness"):
            fitness = result.best_fitness
            if hasattr(fitness, "values"):
                return float(fitness.values[0])
            return float(fitness)

        if hasattr(result, "statistics"):
            stats = result.statistics
            if hasattr(stats, "best_fitness") and stats.best_fitness is not None:
                if hasattr(stats.best_fitness, "values"):
                    return float(stats.best_fitness.values[0])
                return float(stats.best_fitness)

        # Fallback: assume result is population
        if hasattr(result, "__iter__"):
            fitnesses = []
            for ind in result:
                if hasattr(ind, "fitness") and ind.fitness is not None:
                    if hasattr(ind.fitness, "values"):
                        fitnesses.append(ind.fitness.values[0])
                    else:
                        fitnesses.append(float(ind.fitness))
            if fitnesses:
                return cast(float, min(fitnesses) if self.base_config.minimize else max(fitnesses))

        raise ValueError("Cannot extract fitness from result")

    def _extract_best_individual(self, result: Any) -> Any:
        """Extract best individual from evolution result."""
        if hasattr(result, "best_individual"):
            return result.best_individual

        if hasattr(result, "__iter__"):
            minimize = self.base_config.minimize
            best = None
            best_fit = float("inf") if minimize else float("-inf")

            for ind in result:
                if hasattr(ind, "fitness") and ind.fitness is not None:
                    if hasattr(ind.fitness, "values"):
                        fit = ind.fitness.values[0]
                    else:
                        fit = float(ind.fitness)

                    if self._is_better(fit, best_fit, minimize):
                        best_fit = fit
                        best = ind

            return best

        return None

    def _is_better(self, a: float, b: float, minimize: bool) -> bool:
        """Check if fitness a is better than fitness b."""
        if minimize:
            return a < b
        return a > b


def run_meta_evolution(
    base_config: UnifiedConfig,
    fitness_fn: Callable[[Any], float],
    seed: int = 42,
) -> MetaEvolutionResult:
    """
    Run meta-evolution to optimize configuration parameters (T071).

    Uses an outer evolutionary loop to evolve configuration parameters,
    evaluating each candidate by running inner evolution.

    Args:
        base_config: Base configuration with meta settings.
        fitness_fn: Fitness function for inner evolution.
        seed: Random seed for reproducibility.

    Returns:
        MetaEvolutionResult with best configuration and solution.

    Raises:
        ValueError: If base_config has no meta_evolution settings.

    Example:
        >>> result = run_meta_evolution(config, fitness_fn)
        >>> print(result.best_config.mutation_rate)
        >>> result.export_best_config("best.json")
    """
    from evolve.meta.codec import ConfigCodec
    from evolve.meta.result import MetaEvolutionResult

    meta_config = base_config.meta
    if meta_config is None:
        raise ValueError("base_config must have meta settings")

    # Create codec for parameter encoding
    codec = ConfigCodec(base_config, meta_config.evolvable_params)

    # Create meta-evaluator
    meta_evaluator = MetaEvaluator(
        base_config=base_config,
        meta_config=meta_config,
        fitness_fn=fitness_fn,
        seed=seed,
    )

    # Run outer evolution
    rng = Random(seed)

    # Initialize population of configurations
    population: list[tuple[UnifiedConfig, float | None]] = []
    for _ in range(meta_config.outer_population_size):
        # Generate random vector in [0, 1] for each dimension
        vector = [rng.random() for _ in range(codec.dimensions)]
        config = codec.decode(vector)
        population.append((config, None))

    # Evolution history
    history: list[dict[str, Any]] = []
    best_overall: tuple[UnifiedConfig, float] | None = None

    # Outer evolution loop
    for gen in range(meta_config.outer_generations):
        # Evaluate population
        evaluated: list[tuple[UnifiedConfig, float]] = []
        for config, _ in population:
            fitness = meta_evaluator.evaluate(config)
            evaluated.append((config, fitness))

        # Sort by fitness
        minimize = base_config.minimize
        evaluated.sort(key=lambda x: x[1], reverse=not minimize)

        # Track best
        gen_best = evaluated[0]
        if best_overall is None or (
            (minimize and gen_best[1] < best_overall[1])
            or (not minimize and gen_best[1] > best_overall[1])
        ):
            best_overall = gen_best

        # Record history
        fitnesses = [f for _, f in evaluated]
        history.append(
            {
                "generation": gen,
                "best_fitness": gen_best[1],
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "min_fitness": min(fitnesses),
                "max_fitness": max(fitnesses),
            }
        )

        # Selection and variation for next generation
        elite_count = max(1, meta_config.outer_population_size // 10)
        next_pop: list[tuple[UnifiedConfig, float | None]] = []

        # Elitism
        for cfg, fit in evaluated[:elite_count]:
            next_pop.append((cfg, fit))

        # Fill rest with offspring
        while len(next_pop) < meta_config.outer_population_size:
            # Tournament selection
            parent1 = _tournament_select(evaluated, 3, rng, minimize)
            parent2 = _tournament_select(evaluated, 3, rng, minimize)

            # Crossover
            child_vector = _crossover_vectors(
                codec.encode(parent1),
                codec.encode(parent2),
                rng,
            )

            # Mutation
            child_vector = _mutate_vector(child_vector, 0.1, rng)

            # Decode to config
            child_config = codec.decode(child_vector)
            next_pop.append((child_config, None))

        population = next_pop

    # Final evaluation of population
    final_pop: list[tuple[UnifiedConfig, float]] = []
    for config, cached in population:
        if cached is not None:
            final_pop.append((config, cached))
        else:
            fitness = meta_evaluator.evaluate(config)
            final_pop.append((config, fitness))

    # Get best solution from cache
    assert best_overall is not None
    best_solution = meta_evaluator.get_cached_solution(best_overall[0])

    return MetaEvolutionResult(
        best_config=best_overall[0],
        best_fitness=best_overall[1],
        best_solution=best_solution,
        final_population=final_pop,
        history=history,
        trials_run=meta_evaluator.trials_run,
    )


def _tournament_select(
    population: list[tuple[UnifiedConfig, float]],
    tournament_size: int,
    rng: Random,
    minimize: bool,
) -> UnifiedConfig:
    """Select individual via tournament selection."""
    tournament = rng.sample(population, min(tournament_size, len(population)))
    if minimize:
        winner = min(tournament, key=lambda x: x[1])
    else:
        winner = max(tournament, key=lambda x: x[1])
    return winner[0]


def _crossover_vectors(
    parent1: list[float],
    parent2: list[float],
    rng: Random,
) -> list[float]:
    """Blend crossover between two vectors."""
    child = []
    for v1, v2 in zip(parent1, parent2):
        if rng.random() < 0.5:
            # Blend
            w = rng.random()
            value = w * v1 + (1 - w) * v2
        else:
            # Inherit from one parent
            value = v1 if rng.random() < 0.5 else v2
        child.append(max(0.0, min(1.0, value)))
    return child


def _mutate_vector(
    vector: list[float],
    mutation_rate: float,
    rng: Random,
) -> list[float]:
    """Gaussian mutation on vector."""
    mutated = []
    for v in vector:
        if rng.random() < mutation_rate:
            # Gaussian mutation
            delta = rng.gauss(0, 0.1)
            v = max(0.0, min(1.0, v + delta))
        mutated.append(v)
    return mutated
