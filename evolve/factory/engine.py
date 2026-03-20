"""
Engine Factory.

Provides one-line engine creation from unified configuration,
automatically resolving operators and genome types from registries.
"""

from __future__ import annotations

import logging
import os
from random import Random
from typing import Any, Callable, TYPE_CHECKING

from evolve.config.unified import UnifiedConfig
from evolve.config.stopping import StoppingConfig
from evolve.config.callbacks import CallbackConfig
from evolve.registry.operators import get_operator_registry
from evolve.registry.genomes import get_genome_registry
from evolve.core.engine import EvolutionEngine, EvolutionConfig
from evolve.core.population import Population
from evolve.core.stopping import (
    GenerationLimitStopping,
    FitnessThresholdStopping,
    StagnationStopping,
    TimeLimitStopping,
    CompositeStoppingCriterion,
)
from evolve.core.callbacks import Callback, LoggingCallback, CheckpointCallback
from evolve.evaluation.evaluator import Evaluator

if TYPE_CHECKING:
    from evolve.reproduction.engine import ERPEngine


class OperatorCompatibilityError(Exception):
    """Raised when operator is incompatible with genome type."""
    
    def __init__(
        self,
        operator_name: str,
        category: str,
        genome_type: str,
        compatible_types: set[str],
    ) -> None:
        self.operator_name = operator_name
        self.category = category
        self.genome_type = genome_type
        self.compatible_types = compatible_types
        
        message = (
            f"{category.capitalize()} operator '{operator_name}' is not compatible "
            f"with genome type '{genome_type}'. "
            f"Compatible types: {sorted(compatible_types)}"
        )
        super().__init__(message)


def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable[[Any], float],
    seed: int | None = None,
    callbacks: list[Callback] | None = None,
) -> EvolutionEngine:
    """
    Create a ready-to-run engine from unified configuration (FR-027).
    
    This factory resolves operators from registries, validates compatibility,
    and produces either EvolutionEngine or ERPEngine based on configuration.
    
    Args:
        config: Unified experiment configuration.
        evaluator: Fitness evaluator or callable fitness function (FR-032).
        seed: Random seed override (default: use config.seed).
        callbacks: Additional custom callbacks (FR-040).
        
    Returns:
        Configured EvolutionEngine or ERPEngine ready to run.
        
    Raises:
        OperatorCompatibilityError: If operator incompatible with genome (FR-031).
        KeyError: If operator or genome type not found in registry.
        ValueError: If configuration is invalid.
    
    Example:
        >>> config = UnifiedConfig(
        ...     population_size=100,
        ...     selection="tournament",
        ...     crossover="sbx",
        ...     mutation="gaussian",
        ...     genome_type="vector",
        ... )
        >>> engine = create_engine(config, fitness_fn)
        >>> result = engine.run(initial_population)
    """
    # Resolve seed
    effective_seed = seed if seed is not None else config.seed
    if effective_seed is None:
        effective_seed = Random().randint(0, 2**31)
    
    # Wrap callable as evaluator if needed (FR-032)
    if not isinstance(evaluator, Evaluator):
        from evolve.evaluation.evaluator import FunctionEvaluator
        evaluator = FunctionEvaluator(evaluator)
    
    # Validate operator compatibility (FR-031)
    # Note: Full validation requires compatibility metadata from T047-T049
    _validate_operator_compatibility(config)
    
    # Build operators from registry (FR-028)
    op_registry = get_operator_registry()
    
    selection = op_registry.get(
        "selection",
        config.selection,
        **config.selection_params,
    )
    crossover = op_registry.get(
        "crossover",
        config.crossover,
        **config.crossover_params,
    )
    mutation = op_registry.get(
        "mutation",
        config.mutation,
        **config.mutation_params,
    )
    
    # Build stopping criteria (FR-033)
    stopping = _build_stopping_criteria(config)
    
    # Build callbacks (FR-039)
    all_callbacks = _build_callbacks(config)
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Check for multi-objective mode (FR-030)
    if config.is_multiobjective:
        return _create_multiobjective_engine(
            config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            seed=effective_seed,
            stopping=stopping,
            callbacks=all_callbacks,
        )
    
    # Check for ERP mode (FR-029)
    if config.is_erp_enabled:
        return _create_erp_engine(
            config=config,
            evaluator=evaluator,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            seed=effective_seed,
            stopping=stopping,
            callbacks=all_callbacks,
        )
    
    # Standard evolution engine
    return _create_standard_engine(
        config=config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        seed=effective_seed,
        stopping=stopping,
        callbacks=all_callbacks,
    )


def _validate_operator_compatibility(config: UnifiedConfig) -> None:
    """
    Validate operator-genome compatibility (FR-031).
    
    Args:
        config: Unified configuration.
        
    Raises:
        OperatorCompatibilityError: If any operator is incompatible.
    
    Note:
        This is a skeleton implementation. Full validation requires
        compatibility metadata from T047-T049.
    """
    op_registry = get_operator_registry()
    
    # Check selection
    if not op_registry.is_compatible(config.selection, config.genome_type):
        raise OperatorCompatibilityError(
            config.selection,
            "selection",
            config.genome_type,
            op_registry.get_compatibility(config.selection),
        )
    
    # Check crossover
    if not op_registry.is_compatible(config.crossover, config.genome_type):
        raise OperatorCompatibilityError(
            config.crossover,
            "crossover",
            config.genome_type,
            op_registry.get_compatibility(config.crossover),
        )
    
    # Check mutation
    if not op_registry.is_compatible(config.mutation, config.genome_type):
        raise OperatorCompatibilityError(
            config.mutation,
            "mutation",
            config.genome_type,
            op_registry.get_compatibility(config.mutation),
        )


def _build_stopping_criteria(config: UnifiedConfig) -> Any:
    """
    Build stopping criteria from configuration (FR-033).
    
    Args:
        config: Unified configuration.
        
    Returns:
        Stopping criterion (single or composite).
    """
    criteria: list[Any] = []
    
    # Base generation limit
    criteria.append(GenerationLimitStopping(config.max_generations))
    
    # Additional stopping criteria from StoppingConfig
    if config.stopping:
        stop_cfg = config.stopping
        
        # Override generation limit if specified
        if stop_cfg.max_generations is not None:
            criteria = [GenerationLimitStopping(stop_cfg.max_generations)]
        
        # Fitness threshold (FR-010)
        if stop_cfg.fitness_threshold is not None:
            criteria.append(
                FitnessThresholdStopping(
                    threshold=stop_cfg.fitness_threshold,
                    minimize=config.minimize,
                )
            )
        
        # Stagnation detection (FR-011)
        if stop_cfg.stagnation_generations is not None:
            criteria.append(
                StagnationStopping(stop_cfg.stagnation_generations)
            )
        
        # Time limit (FR-012)
        if stop_cfg.time_limit_seconds is not None:
            criteria.append(
                TimeLimitStopping(stop_cfg.time_limit_seconds)
            )
    
    # Single or composite (FR-013)
    if len(criteria) == 1:
        return criteria[0]
    return CompositeStoppingCriterion(criteria)


def _build_callbacks(config: UnifiedConfig) -> list[Callback]:
    """
    Build built-in callbacks from configuration (FR-039, FR-041, FR-042).
    
    Args:
        config: Unified configuration.
        
    Returns:
        List of configured callback instances.
    """
    callbacks: list[Callback] = []
    
    # Callback config may be None, but we still process tracking
    cb_cfg = config.callbacks
    
    if cb_cfg is not None:
        # Logging callback (FR-041)
        if cb_cfg.enable_logging:
            callbacks.append(
                LoggingCallback(
                    log_level=cb_cfg.log_level,
                    log_destination=cb_cfg.log_destination,
                )
            )
        
        # Checkpoint callback (FR-042)
        if cb_cfg.enable_checkpointing:
            # Ensure directory exists
            checkpoint_dir = cb_cfg.checkpoint_dir or "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            callbacks.append(
                CheckpointCallback(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_frequency=cb_cfg.checkpoint_frequency,
                )
            )
    
    # Tracking callback (FR-019, FR-028)
    # Note: Tracking is independent of CallbackConfig
    if config.is_tracking_enabled:
        from evolve.experiment.tracking.callback import TrackingCallback
        
        tracking_config = config.tracking
        callbacks.append(
            TrackingCallback(
                config=tracking_config,
                unified_config_dict=config.to_dict(),
            )
        )
    
    return callbacks


def _create_standard_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int,
    stopping: Any,
    callbacks: list[Callback],
) -> EvolutionEngine:
    """
    Create a standard EvolutionEngine.
    
    Args:
        config: Unified configuration.
        evaluator: Fitness evaluator.
        selection: Selection operator.
        crossover: Crossover operator.
        mutation: Mutation operator.
        seed: Random seed.
        stopping: Stopping criterion.
        callbacks: Callback instances.
        
    Returns:
        Configured EvolutionEngine.
    """
    # Convert to EvolutionConfig
    evo_config = EvolutionConfig(
        population_size=config.population_size,
        max_generations=config.max_generations,
        elitism=config.elitism,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        minimize=config.minimize,
    )
    
    engine = EvolutionEngine(
        config=evo_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        seed=seed,
        stopping=stopping,
    )
    
    # Attach callbacks (runtime attribute)
    setattr(engine, "_callbacks", callbacks)
    
    return engine


def _create_erp_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int,
    stopping: Any,
    callbacks: list[Callback],
) -> "ERPEngine":
    """
    Create an ERPEngine for evolvable reproduction protocols (FR-029).
    
    Args:
        config: Unified configuration (must have erp settings).
        evaluator: Fitness evaluator.
        selection: Selection operator.
        crossover: Crossover operator.
        mutation: Mutation operator.
        seed: Random seed.
        stopping: Stopping criterion.
        callbacks: Callback instances.
        
    Returns:
        Configured ERPEngine.
    """
    from evolve.reproduction.engine import ERPEngine, ERPConfig
    
    erp_settings = config.erp
    if erp_settings is None:
        raise ValueError("ERP settings required for ERP engine")
    
    # Convert to ERPConfig
    erp_config = ERPConfig(
        population_size=config.population_size,
        max_generations=config.max_generations,
        elitism=config.elitism,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        minimize=config.minimize,
        step_limit=erp_settings.step_limit,
        recovery_threshold=erp_settings.recovery_threshold,
        enable_intent=erp_settings.enable_intent,
        enable_recovery=erp_settings.enable_recovery,
    )
    
    engine = ERPEngine(
        config=erp_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        seed=seed,
        stopping=stopping,
    )
    
    # Attach callbacks (runtime attribute)
    setattr(engine, "_callbacks", callbacks)
    
    return engine


def _create_multiobjective_engine(
    config: UnifiedConfig,
    evaluator: Evaluator,
    selection: Any,
    crossover: Any,
    mutation: Any,
    seed: int,
    stopping: Any,
    callbacks: list[Callback],
) -> EvolutionEngine:
    """
    Create a multi-objective engine with NSGA-II selection (FR-030).
    
    Handles constraint dominance (FR-036, FR-037) when constraints specified:
    - Feasible solutions dominate infeasible solutions
    - Among infeasible solutions, lower violation ranks higher
    
    Args:
        config: Unified configuration (must have multiobjective settings).
        evaluator: Fitness evaluator.
        selection: Selection operator (will be overridden if not CrowdedTournament).
        crossover: Crossover operator.
        mutation: Mutation operator.
        seed: Random seed.
        stopping: Stopping criterion.
        callbacks: Callback instances.
        
    Returns:
        Configured EvolutionEngine with multi-objective support.
    """
    from evolve.multiobjective.selection import CrowdedTournamentSelection
    
    mo_settings = config.multiobjective
    if mo_settings is None:
        raise ValueError("Multi-objective settings required")
    
    # Override selection with NSGA-II crowded tournament (T054)
    if config.selection != "crowded_tournament":
        selection = CrowdedTournamentSelection(tournament_size=2)
    
    # Convert to EvolutionConfig
    evo_config = EvolutionConfig(
        population_size=config.population_size,
        max_generations=config.max_generations,
        elitism=config.elitism,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        minimize=True,  # Multi-objective always minimizes (Pareto)
    )
    
    engine = EvolutionEngine(
        config=evo_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        seed=seed,
        stopping=stopping,
    )
    
    # Store multi-objective settings for later use (runtime attribute)
    setattr(engine, "_multiobjective_config", mo_settings)
    
    # Store reference point for hypervolume tracking (T055)
    if mo_settings.reference_point is not None:
        setattr(engine, "_reference_point", mo_settings.reference_point)
    
    # Store constraint handling settings (T056a-T056c)
    if mo_settings.has_constraints:
        setattr(engine, "_constraint_specs", mo_settings.constraints)
        setattr(engine, "_constraint_handling", mo_settings.constraint_handling)
    
    # Attach callbacks
    setattr(engine, "_callbacks", callbacks)
    
    return engine


def create_initial_population(
    config: UnifiedConfig,
    seed: int | None = None,
) -> Population:
    """
    Create initial population from configuration.
    
    Args:
        config: Unified configuration.
        seed: Random seed (default: use config.seed).
        
    Returns:
        Random initial population.
    """
    from evolve.core.types import Individual, IndividualMetadata
    from uuid import uuid4
    
    # Resolve seed
    effective_seed = seed if seed is not None else config.seed
    if effective_seed is None:
        effective_seed = Random().randint(0, 2**31)
    
    rng = Random(effective_seed)
    
    # Get genome factory
    genome_registry = get_genome_registry()
    
    # Create individuals
    individuals = []
    for _ in range(config.population_size):
        genome = genome_registry.create(
            config.genome_type,
            rng=rng,
            **config.genome_params,
        )
        individual = Individual(
            genome=genome,
            fitness=None,
            metadata=IndividualMetadata(origin="init"),
        )
        individuals.append(individual)
    
    return Population(individuals)
