"""
Factory Contracts

Defines the main factory function for creating engines from unified configuration.
This module serves as the primary entry point for the unified configuration system.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from evolve.core.callbacks import Callback
from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.evaluation.evaluator import Evaluator

from .registries import get_genome_registry, get_operator_registry
from .unified_config import CallbackConfig, UnifiedConfig


def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable,
    seed: int | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> EvolutionEngine | Any:  # | ERPEngine
    """
    Create a ready-to-run engine from unified configuration.

    This is the main entry point for the unified configuration system.
    Resolves operators and genomes from registries, validates compatibility,
    and returns an engine ready to call .run().

    Args:
        config: Unified configuration specifying all experiment parameters.
        evaluator: Fitness evaluator (Evaluator instance or callable function).
        seed: Override the seed in config (None uses config.seed).
        callbacks: Additional custom callbacks (beyond those in config.callbacks).

    Returns:
        - EvolutionEngine if config.erp is None
        - ERPEngine if config.erp is specified

    Raises:
        ValueError: If operator not found in registry.
        ValueError: If operator incompatible with genome type.
        ValueError: If required parameters missing for operator.
        TypeError: If evaluator is not callable or Evaluator.

    Example:
        >>> # From JSON file
        >>> config = UnifiedConfig.from_json("experiment.json")
        >>> engine = create_engine(config, fitness_function)
        >>> result = engine.run()

        >>> # Programmatic configuration
        >>> config = UnifiedConfig(
        ...     population_size=100,
        ...     selection="tournament",
        ...     selection_params={"tournament_size": 5},
        ...     crossover="sbx",
        ...     mutation="gaussian",
        ...     genome_type="vector",
        ...     genome_params={"dimensions": 10, "bounds": [-5.12, 5.12]},
        ... )
        >>> engine = create_engine(config, sphere_function)
        >>> result = engine.run()

    Process:
        1. Resolve operators from registry by name
        2. Validate operator-genome compatibility
        3. Create genome factory from registry
        4. Instantiate callbacks from config (if any)
        5. Build stopping criteria from config (if any)
        6. Construct and return appropriate engine type
    """
    # Get registries
    op_registry = get_operator_registry()
    genome_registry = get_genome_registry()

    # Determine effective seed
    effective_seed = seed if seed is not None else config.seed

    # Validate operator compatibility with genome type
    _validate_operator_compatibility(config, op_registry)

    # Resolve operators from registry
    selection = op_registry.get("selection", config.selection, **config.selection_params)
    crossover = op_registry.get("crossover", config.crossover, **config.crossover_params)
    mutation = op_registry.get("mutation", config.mutation, **config.mutation_params)

    # Wrap callable as evaluator if needed
    if callable(evaluator) and not isinstance(evaluator, Evaluator):
        from evolve.evaluation.evaluator import FunctionEvaluator

        evaluator = FunctionEvaluator(evaluator, minimize=config.minimize)

    # Build callbacks list
    all_callbacks = _build_callbacks(config.callbacks, callbacks)

    # Build stopping criteria
    stopping = _build_stopping_criteria(config)

    # Create engine based on config type
    if config.erp is not None:
        return _create_erp_engine(
            config,
            evaluator,
            selection,
            crossover,
            mutation,
            effective_seed,
            all_callbacks,
            stopping,
        )
    elif config.multiobjective is not None:
        return _create_multiobjective_engine(
            config,
            evaluator,
            selection,
            crossover,
            mutation,
            effective_seed,
            all_callbacks,
            stopping,
        )
    else:
        return _create_standard_engine(
            config,
            evaluator,
            selection,
            crossover,
            mutation,
            effective_seed,
            all_callbacks,
            stopping,
        )


def _validate_operator_compatibility(
    config: UnifiedConfig,
    registry: Any,
) -> None:
    """
    Validate all operators are compatible with the genome type.

    Raises:
        ValueError: If any operator is incompatible.
    """
    genome_type = config.genome_type
    operators = [
        ("selection", config.selection),
        ("crossover", config.crossover),
        ("mutation", config.mutation),
    ]

    for category, op_name in operators:
        if not registry.is_compatible(op_name, genome_type):
            compatible = registry.get_compatibility(op_name)
            raise ValueError(
                f"Operator '{op_name}' ({category}) is not compatible with "
                f"genome type '{genome_type}'. "
                f"Compatible genome types: {compatible}"
            )


def _build_callbacks(
    config_callbacks: CallbackConfig | None,
    custom_callbacks: Sequence[Callback] | None,
) -> list[Callback]:
    """
    Build callback list from config and custom callbacks.

    Returns:
        Combined list of all callbacks.
    """
    callbacks: list[Callback] = []

    # Add configured built-in callbacks
    if config_callbacks is not None:
        if config_callbacks.enable_logging:
            from evolve.core.callbacks import LoggingCallback

            callbacks.append(
                LoggingCallback(
                    level=config_callbacks.log_level,
                    destination=config_callbacks.log_destination,
                )
            )

        if config_callbacks.enable_checkpointing:
            from evolve.core.callbacks import CheckpointCallback

            callbacks.append(
                CheckpointCallback(
                    directory=config_callbacks.checkpoint_dir,
                    frequency=config_callbacks.checkpoint_frequency,
                )
            )

    # Add custom callbacks
    if custom_callbacks:
        callbacks.extend(custom_callbacks)

    return callbacks


def _build_stopping_criteria(config: UnifiedConfig) -> Any:
    """
    Build stopping criteria from config.

    Returns:
        Stopping criterion or None if only using max_generations.
    """
    from evolve.core.stopping import (
        CompositeStoppingCriterion,
        FitnessThresholdStopping,
        GenerationLimitStopping,
        StagnationStopping,
        TimeLimitStopping,
    )

    criteria = []

    # Always add generation limit (from config or stopping)
    max_gens = config.max_generations
    if config.stopping and config.stopping.max_generations:
        max_gens = config.stopping.max_generations
    criteria.append(GenerationLimitStopping(max_gens))

    # Add other criteria from stopping config
    if config.stopping:
        if config.stopping.fitness_threshold is not None:
            criteria.append(
                FitnessThresholdStopping(
                    threshold=config.stopping.fitness_threshold,
                    minimize=config.minimize,
                )
            )

        if config.stopping.stagnation_generations is not None:
            criteria.append(
                StagnationStopping(
                    generations=config.stopping.stagnation_generations,
                )
            )

        if config.stopping.time_limit_seconds is not None:
            criteria.append(
                TimeLimitStopping(
                    seconds=config.stopping.time_limit_seconds,
                )
            )

    # Return composite if multiple, single if one
    if len(criteria) == 1:
        return criteria[0]
    return CompositeStoppingCriterion(criteria)


def _create_standard_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int | None,
    callbacks: list[Callback],
    stopping: Any,
) -> EvolutionEngine:
    """Create standard EvolutionEngine."""
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
        seed=seed,
        callbacks=callbacks,
        stopping_criterion=stopping,
    )


def _create_erp_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int | None,
    callbacks: list[Callback],
    stopping: Any,
) -> Any:  # ERPEngine
    """Create ERPEngine for evolvable reproduction protocols."""
    from evolve.reproduction.engine import ERPConfig, ERPEngine

    assert config.erp is not None

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
        seed=seed,
        callbacks=callbacks,
        stopping_criterion=stopping,
    )


def _create_multiobjective_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int | None,
    callbacks: list[Callback],
    stopping: Any,
) -> EvolutionEngine:
    """Create engine configured for multi-objective optimization."""
    # Override selection with NSGA-II selection
    from evolve.multiobjective.selection import CrowdedTournamentSelection

    assert config.multiobjective is not None

    # Use crowded tournament selection for NSGA-II
    mo_selection = CrowdedTournamentSelection(tournament_size=2)

    engine_config = EvolutionConfig(
        population_size=config.population_size,
        max_generations=config.max_generations,
        elitism=0,  # NSGA-II handles elitism differently
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        minimize=True,  # Multi-objective handles direction per objective
    )

    # TODO: Configure reference point, constraint handling

    return EvolutionEngine(
        config=engine_config,
        evaluator=evaluator,
        selection=mo_selection,
        crossover=crossover,
        mutation=mutation,
        seed=seed,
        callbacks=callbacks,
        stopping_criterion=stopping,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def load_and_run(
    config_path: str,
    evaluator: Evaluator | Callable,
    seed: int | None = None,
) -> Any:
    """
    Load configuration from file and run evolution.

    Convenience function combining load and run.

    Args:
        config_path: Path to JSON configuration file.
        evaluator: Fitness evaluator or callable.
        seed: Override seed (None uses config seed).

    Returns:
        Evolution result from engine.run().

    Example:
        >>> result = load_and_run("experiment.json", sphere_function)
        >>> print(f"Best fitness: {result.best.fitness}")
    """
    config = UnifiedConfig.from_json(config_path)
    engine = create_engine(config, evaluator, seed=seed)
    return engine.run()


def create_config(
    genome_type: str = "vector",
    population_size: int = 100,
    max_generations: int = 100,
    selection: str = "tournament",
    crossover: str = "uniform",
    mutation: str = "gaussian",
    **kwargs: Any,
) -> UnifiedConfig:
    """
    Create configuration with common defaults.

    Convenience function with sensible defaults.

    Args:
        genome_type: Genome representation type.
        population_size: Population size.
        max_generations: Maximum generations.
        selection: Selection operator name.
        crossover: Crossover operator name.
        mutation: Mutation operator name.
        **kwargs: Additional configuration parameters.

    Returns:
        UnifiedConfig instance.

    Example:
        >>> config = create_config(
        ...     genome_type="vector",
        ...     genome_params={"dimensions": 10},
        ...     population_size=50,
        ... )
    """
    return UnifiedConfig(
        genome_type=genome_type,
        population_size=population_size,
        max_generations=max_generations,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        **kwargs,
    )
