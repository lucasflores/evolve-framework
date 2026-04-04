"""
Meta-Evolution Contracts

Defines interfaces for meta-evolution (hyperparameter optimization).
Includes ConfigCodec, MetaEvaluator, and MetaEvolutionResult.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Generic, TypeVar

import numpy as np

from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import Evaluator
from evolve.representation.vector import VectorGenome

# Import from sibling contract
from .unified_config import MetaEvolutionConfig, ParameterSpec, UnifiedConfig

G = TypeVar("G")


# =============================================================================
# Configuration Codec
# =============================================================================


class ConfigCodec:
    """
    Encodes UnifiedConfig to VectorGenome and decodes back.

    Works relative to a base configuration: only evolvable parameters
    are encoded; fixed parameters come from the base.

    Each ParameterSpec maps to one dimension in the vector genome:
    - continuous: value normalized to bounds
    - integer: float value, rounded on decode
    - categorical: index as float, rounded and looked up on decode

    Example:
        >>> codec = ConfigCodec(
        ...     base_config=base,
        ...     param_specs=(
        ...         ParameterSpec("population_size", "integer", bounds=(50, 500)),
        ...         ParameterSpec("mutation", "categorical", choices=("gaussian", "polynomial")),
        ...     )
        ... )
        >>> genome = codec.encode(config)
        >>> recovered = codec.decode(genome)
    """

    def __init__(
        self,
        base_config: UnifiedConfig,
        param_specs: tuple[ParameterSpec, ...],
    ) -> None:
        """
        Initialize codec.

        Args:
            base_config: Template configuration with fixed parameter values.
            param_specs: Parameters being evolved.
        """
        self.base_config = base_config
        self.param_specs = param_specs
        self._dimension_mapping = list(param_specs)

        # Precompute bounds
        self._lower_bounds, self._upper_bounds = self._compute_bounds()

    def _compute_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute lower and upper bounds for vector genome."""
        lower = []
        upper = []

        for spec in self.param_specs:
            if spec.param_type in ("continuous", "integer"):
                assert spec.bounds is not None
                lower.append(spec.bounds[0])
                upper.append(spec.bounds[1])
            elif spec.param_type == "categorical":
                assert spec.choices is not None
                lower.append(0.0)
                upper.append(float(len(spec.choices) - 1))

        return np.array(lower), np.array(upper)

    @property
    def dimensions(self) -> int:
        """Number of dimensions in encoded vector."""
        return len(self.param_specs)

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get bounds for vector genome.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays.
        """
        return self._lower_bounds.copy(), self._upper_bounds.copy()

    def encode(self, config: UnifiedConfig) -> VectorGenome:
        """
        Extract evolvable parameters and encode to vector.

        Args:
            config: Configuration to encode.

        Returns:
            VectorGenome representing evolvable parameters.
        """
        values = []

        for spec in self.param_specs:
            value = self._get_param(config, spec.path)

            if spec.param_type == "continuous" or spec.param_type == "integer":
                values.append(float(value))
            elif spec.param_type == "categorical":
                assert spec.choices is not None
                idx = spec.choices.index(value)
                values.append(float(idx))

        return VectorGenome(genes=np.array(values))

    def decode(self, genome: VectorGenome) -> UnifiedConfig:
        """
        Decode vector and merge with base configuration.

        Args:
            genome: Vector genome to decode.

        Returns:
            New UnifiedConfig with decoded parameter values.
        """
        updates: dict[str, Any] = {}

        for i, spec in enumerate(self.param_specs):
            raw_value = genome.genes[i]

            if spec.param_type == "continuous":
                value = float(raw_value)
                if spec.log_scale and spec.bounds:
                    # Log-scale decoding
                    import math

                    log_low = math.log(spec.bounds[0])
                    log_high = math.log(spec.bounds[1])
                    normalized = (raw_value - spec.bounds[0]) / (spec.bounds[1] - spec.bounds[0])
                    value = math.exp(log_low + normalized * (log_high - log_low))
            elif spec.param_type == "integer":
                value = int(round(raw_value))
            elif spec.param_type == "categorical":
                assert spec.choices is not None
                idx = int(round(raw_value))
                idx = max(0, min(idx, len(spec.choices) - 1))
                value = spec.choices[idx]

            self._set_param_update(updates, spec.path, value)

        # Apply updates to base config
        return self._apply_updates(self.base_config, updates)

    def _get_param(self, config: UnifiedConfig, path: str) -> Any:
        """Get parameter value by dot-notation path."""
        parts = path.split(".")
        obj: Any = config

        for part in parts:
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)

        return obj

    def _set_param_update(self, updates: dict, path: str, value: Any) -> None:
        """Set parameter in updates dict by dot-notation path."""
        parts = path.split(".")

        if len(parts) == 1:
            updates[parts[0]] = value
        else:
            # Handle nested params (e.g., "mutation_params.sigma")
            root = parts[0]
            rest = ".".join(parts[1:])

            if root not in updates:
                updates[root] = {}

            if isinstance(updates[root], dict):
                self._set_param_update(updates[root], rest, value)

    def _apply_updates(self, config: UnifiedConfig, updates: dict) -> UnifiedConfig:
        """Apply updates dictionary to create new config."""
        # Handle dict params specially
        final_updates = {}

        for key, value in updates.items():
            if key.endswith("_params") and isinstance(value, dict):
                # Merge with existing params
                existing = dict(getattr(config, key))
                existing.update(value)
                final_updates[key] = existing
            else:
                final_updates[key] = value

        return config.with_params(**final_updates)


# =============================================================================
# Meta-Evaluator
# =============================================================================


class MetaEvaluator(Evaluator[VectorGenome]):
    """
    Evaluator for meta-evolution that treats configurations as individuals.

    Evaluates a configuration by:
    1. Decoding the vector genome to a configuration
    2. Running the inner evolutionary loop
    3. Extracting and caching the result
    4. Returning inner-loop performance as fitness

    Inner Loop Seeding:
        Seeds are computed deterministically from configuration hash
        and trial number: seed = hash(config_hash + trial_num) % 2^31

    Example:
        >>> meta_evaluator = MetaEvaluator(
        ...     codec=codec,
        ...     inner_evaluator=sphere_evaluator,
        ...     meta_config=meta_config,
        ... )
        >>> fitness = meta_evaluator.evaluate(config_genome, rng)
    """

    def __init__(
        self,
        codec: ConfigCodec,
        inner_evaluator: Evaluator,
        meta_config: MetaEvolutionConfig,
    ) -> None:
        """
        Initialize meta-evaluator.

        Args:
            codec: Encodes/decodes configurations.
            inner_evaluator: Evaluator for the actual problem.
            meta_config: Meta-evolution settings.
        """
        self.codec = codec
        self.inner_evaluator = inner_evaluator
        self.meta_config = meta_config

        # Cache: config_hash -> best solution from that config
        self._solution_cache: dict[str, Individual] = {}

        # Track total evaluations
        self._total_evaluations: int = 0

    def evaluate(self, genome: VectorGenome, rng: Random) -> Fitness:
        """
        Evaluate a configuration by running inner evolution.

        Args:
            genome: Configuration encoded as vector.
            rng: Random number generator.

        Returns:
            Aggregated fitness across trials.
        """
        # Decode configuration
        config = self.codec.decode(genome)
        config_hash = config.compute_hash()

        # Override inner generations if specified
        if self.meta_config.inner_generations is not None:
            config = config.with_params(max_generations=self.meta_config.inner_generations)

        # Run multiple trials
        trial_fitnesses: list[float] = []
        best_solution: Individual | None = None
        best_fitness: float = float("inf") if config.minimize else float("-inf")

        for trial in range(self.meta_config.trials_per_config):
            # Deterministic seed from config hash and trial
            inner_seed = self._compute_inner_seed(config_hash, trial)

            try:
                # Run inner evolution
                result = self._run_inner_evolution(config, inner_seed)

                trial_fitness = self._extract_fitness(result.best.fitness)
                trial_fitnesses.append(trial_fitness)

                # Track best solution
                is_better = (config.minimize and trial_fitness < best_fitness) or (
                    not config.minimize and trial_fitness > best_fitness
                )
                if is_better:
                    best_fitness = trial_fitness
                    best_solution = result.best

                self._total_evaluations += result.generations * config.population_size

            except Exception:
                # Invalid config produces worst-case fitness
                worst = float("inf") if config.minimize else float("-inf")
                trial_fitnesses.append(worst)

        # Cache best solution
        if best_solution is not None:
            self._solution_cache[config_hash] = best_solution

        # Aggregate across trials
        return self._aggregate_fitness(trial_fitnesses)

    def _run_inner_evolution(self, config: UnifiedConfig, seed: int) -> Any:
        """Run inner evolutionary loop."""
        # Import here to avoid circular imports
        from .factory import create_engine

        engine = create_engine(config, self.inner_evaluator, seed=seed)
        return engine.run()

    def _compute_inner_seed(self, config_hash: str, trial: int) -> int:
        """Compute deterministic seed for inner run."""
        import hashlib

        combined = f"{config_hash}:{trial}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") % (2**31)

    def _extract_fitness(self, fitness: Fitness) -> float:
        """Extract scalar from fitness (handle multi-objective)."""
        if isinstance(fitness, (int, float)):
            return float(fitness)
        elif isinstance(fitness, dict):
            # Multi-objective: return hypervolume or first objective
            if "hypervolume" in fitness:
                return float(fitness["hypervolume"])
            return float(list(fitness.values())[0])
        else:
            return float(fitness)

    def _aggregate_fitness(self, fitnesses: list[float]) -> Fitness:
        """Aggregate fitness across trials."""
        if not fitnesses:
            return float("inf")

        if self.meta_config.aggregation == "mean":
            return float(np.mean(fitnesses))
        elif self.meta_config.aggregation == "median":
            return float(np.median(fitnesses))
        elif self.meta_config.aggregation == "best":
            return min(fitnesses)  # Assuming minimization
        else:
            return float(np.mean(fitnesses))

    def get_cached_solution(self, config_hash: str) -> Individual | None:
        """
        Retrieve best solution for configuration.

        Args:
            config_hash: Configuration hash (from compute_hash()).

        Returns:
            Best solution found, or None if not cached.
        """
        return self._solution_cache.get(config_hash)

    @property
    def total_evaluations(self) -> int:
        """Total inner-loop evaluations performed."""
        return self._total_evaluations

    @property
    def solution_cache(self) -> dict[str, Individual]:
        """Read-only access to solution cache."""
        return dict(self._solution_cache)


# =============================================================================
# Meta-Evolution Result
# =============================================================================


@dataclass(frozen=True)
class MetaEvolutionResult(Generic[G]):
    """
    Result of meta-evolution run.

    Contains both the best configuration found and the best solution
    that configuration produced.

    Example:
        >>> result = run_meta_evolution(base_config, param_specs, evaluator)
        >>> print(f"Best config: {result.best_config.to_dict()}")
        >>> print(f"Best solution fitness: {result.best_solution.fitness}")
        >>> result.export_best_config("best_config.json")
    """

    best_config: UnifiedConfig
    """Best-performing configuration found."""

    best_solution: Individual[G]
    """Best solution found by the best configuration."""

    config_population: Population[VectorGenome]
    """Final population of configuration genomes."""

    solution_cache: dict[str, Individual[G]]
    """All cached solutions indexed by config hash."""

    outer_history: list[dict[str, Any]]
    """Metrics from each outer generation."""

    inner_history: list[dict[str, Any]] | None
    """Metrics from best config's final inner run."""

    outer_generations: int
    """Number of outer generations completed."""

    total_evaluations: int
    """Total inner-loop evaluations performed."""

    def get_pareto_configs(self) -> list[UnifiedConfig]:
        """
        Get Pareto-optimal configurations (for multi-objective meta).

        Returns:
            List of non-dominated configurations.
        """
        # TODO: Implement Pareto extraction from population
        return [self.best_config]

    def export_best_config(self, path: str) -> None:
        """
        Save best configuration to JSON file.

        Args:
            path: Output file path.
        """
        self.best_config.to_json(path)

    def get_config_by_hash(self, config_hash: str) -> UnifiedConfig | None:
        """
        Get configuration that produced cached solution.

        Note: Configurations themselves are not cached; only solutions.
        This method decodes from the population if possible.
        """
        # Would need codec reference to decode
        return None


# =============================================================================
# Factory Function
# =============================================================================


def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Any,  # Callable also accepted
    seed: int | None = None,
    callbacks: Any | None = None,
) -> Any:
    """
    Create a ready-to-run engine from unified configuration.

    This is the main entry point for using unified configuration.

    Args:
        config: Unified configuration specifying all parameters.
        evaluator: Fitness evaluator or callable fitness function.
        seed: Override configuration seed if provided.
        callbacks: Custom callbacks (not specifiable in config).

    Returns:
        EvolutionEngine if config.erp is None
        ERPEngine if config.erp is specified

    Raises:
        ValueError: If operator not found in registry.
        ValueError: If operator incompatible with genome type.
        ValueError: If required parameters missing.

    Example:
        >>> config = UnifiedConfig.from_json("experiment.json")
        >>> engine = create_engine(config, fitness_function)
        >>> result = engine.run()
    """
    from .registries import get_genome_registry, get_operator_registry

    # Get registries
    op_registry = get_operator_registry()
    genome_registry = get_genome_registry()

    # Determine seed
    effective_seed = seed if seed is not None else config.seed

    # Validate operator compatibility
    _validate_compatibility(config, op_registry)

    # Resolve operators
    selection = op_registry.get("selection", config.selection, **config.selection_params)
    crossover = op_registry.get("crossover", config.crossover, **config.crossover_params)
    mutation = op_registry.get("mutation", config.mutation, **config.mutation_params)

    # Wrap callable as evaluator if needed
    if callable(evaluator) and not isinstance(evaluator, Evaluator):
        from evolve.evaluation.evaluator import FunctionEvaluator

        evaluator = FunctionEvaluator(evaluator)

    # Build config for engine
    if config.erp is not None:
        # ERP Engine
        from evolve.core.engine import EvolutionConfig
        from evolve.reproduction.engine import ERPConfig, ERPEngine

        erp_config = ERPConfig(
            population_size=config.population_size,
            max_generations=config.max_generations,
            elitism=config.elitism,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            minimize=config.minimize,
            step_limit=config.erp.step_limit,
            recovery_threshold=config.erp.recovery_threshold,
            protocol_mutation_rate=config.erp.protocol_mutation_rate,
            enable_intent=config.erp.enable_intent,
            enable_recovery=config.erp.enable_recovery,
        )

        return ERPEngine(
            config=erp_config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            seed=effective_seed,
        )
    else:
        # Standard Evolution Engine
        from evolve.core.engine import EvolutionConfig, EvolutionEngine

        engine_config = EvolutionConfig(
            population_size=config.population_size,
            max_generations=config.max_generations,
            elitism=config.elitism,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            minimize=config.minimize,
        )

        return EvolutionEngine(
            config=engine_config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            seed=effective_seed,
        )


def _validate_compatibility(config: UnifiedConfig, registry: Any) -> None:
    """Validate operator-genome compatibility."""
    genome_type = config.genome_type

    for op_name in (config.selection, config.crossover, config.mutation):
        if not registry.is_compatible(op_name, genome_type):
            compatible = registry.get_compatibility(op_name)
            raise ValueError(
                f"Operator '{op_name}' is not compatible with genome type '{genome_type}'. "
                f"Compatible types: {compatible}"
            )


# =============================================================================
# Meta-Evolution Runner
# =============================================================================


def run_meta_evolution(
    base_config: UnifiedConfig,
    param_specs: tuple[ParameterSpec, ...],
    evaluator: Evaluator,
    outer_population_size: int = 20,
    outer_generations: int = 10,
    trials_per_config: int = 1,
    aggregation: str = "mean",
    inner_generations: int | None = None,
    seed: int | None = None,
) -> MetaEvolutionResult:
    """
    Run meta-evolution to find optimal hyperparameters.

    Args:
        base_config: Template configuration (fixed parameters).
        param_specs: Parameters to evolve with bounds.
        evaluator: Evaluator for the actual problem.
        outer_population_size: Population size for outer loop.
        outer_generations: Generations for outer loop.
        trials_per_config: Trials per config for robustness.
        aggregation: How to aggregate trials ("mean", "median", "best").
        inner_generations: Override inner generations for speed.
        seed: Random seed for outer loop.

    Returns:
        MetaEvolutionResult with best config and solution.

    Example:
        >>> result = run_meta_evolution(
        ...     base_config=UnifiedConfig(genome_type="vector", genome_params={"dimensions": 10}),
        ...     param_specs=(
        ...         ParameterSpec("population_size", "integer", bounds=(50, 500)),
        ...         ParameterSpec("mutation_rate", "continuous", bounds=(0.01, 0.5)),
        ...     ),
        ...     evaluator=sphere_evaluator,
        ...     outer_generations=20,
        ... )
        >>> print(result.best_config)
    """
    from random import Random

    from evolve.core.engine import EvolutionConfig, EvolutionEngine
    from evolve.core.operators.crossover import SimulatedBinaryCrossover
    from evolve.core.operators.mutation import GaussianMutation
    from evolve.core.operators.selection import TournamentSelection

    # Build meta-evolution config
    meta_config = MetaEvolutionConfig(
        evolvable_params=param_specs,
        outer_population_size=outer_population_size,
        outer_generations=outer_generations,
        trials_per_config=trials_per_config,
        aggregation=aggregation,
        inner_generations=inner_generations,
    )

    # Create codec
    codec = ConfigCodec(base_config, param_specs)

    # Create meta-evaluator
    meta_evaluator = MetaEvaluator(codec, evaluator, meta_config)

    # Configure outer evolution
    outer_config = EvolutionConfig(
        population_size=outer_population_size,
        max_generations=outer_generations,
        elitism=2,
        crossover_rate=0.9,
        mutation_rate=1.0,
        minimize=base_config.minimize,
    )

    # Create outer engine with vector genome operators
    lower, upper = codec.get_bounds()
    outer_engine = EvolutionEngine(
        config=outer_config,
        evaluator=meta_evaluator,
        selection=TournamentSelection(tournament_size=3),
        crossover=SimulatedBinaryCrossover(eta=15.0),
        mutation=GaussianMutation(sigma=0.1),
        seed=seed,
    )

    # Initialize population with random configs
    rng = Random(seed)
    initial_genomes = [
        VectorGenome.random(
            dimensions=codec.dimensions,
            bounds=(lower, upper),
            rng=rng,
        )
        for _ in range(outer_population_size)
    ]

    # Run outer evolution
    from evolve.core.population import Population

    initial_pop = Population.from_genomes(initial_genomes)
    outer_result = outer_engine.run(initial_population=initial_pop)

    # Extract best configuration
    best_genome = outer_result.best.genome
    best_config = codec.decode(best_genome)

    # Get cached solution for best config
    best_hash = best_config.compute_hash()
    best_solution = meta_evaluator.get_cached_solution(best_hash)

    if best_solution is None:
        # Re-run best config to get solution
        inner_seed = hash(best_hash) % (2**31)
        inner_engine = create_engine(best_config, evaluator, seed=inner_seed)
        inner_result = inner_engine.run()
        best_solution = inner_result.best

    return MetaEvolutionResult(
        best_config=best_config,
        best_solution=best_solution,
        config_population=outer_result.population,
        solution_cache=meta_evaluator.solution_cache,
        outer_history=outer_result.history,
        inner_history=None,  # Could add by re-running
        outer_generations=outer_result.generations,
        total_evaluations=meta_evaluator.total_evaluations,
    )
