"""
Experiment runner for orchestrating experiments.

Handles configuration, checkpointing, metric tracking,
and reproducible experiment execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from evolve.experiment.checkpoint import Checkpoint, CheckpointManager
from evolve.experiment.config import ExperimentConfig
from evolve.experiment.metrics import LocalTracker, MetricTracker, NullTracker

if TYPE_CHECKING:
    from evolve.core.engine import EvolutionEngine, EvolutionResult
    from evolve.core.population import Population
    from evolve.core.types import Individual

G = TypeVar("G")


@dataclass
class ExperimentRunner(Generic[G]):
    """
    Orchestrates experiment execution.

    Handles:
    - Configuration validation
    - Checkpointing
    - Metric tracking
    - Resume from checkpoint
    - Stopping criteria

    Example:
        >>> config = ExperimentConfig(
        ...     name="my_experiment",
        ...     seed=42,
        ...     population_size=100,
        ...     n_generations=100,
        ... )
        >>> runner = ExperimentRunner(
        ...     config=config,
        ...     engine=engine,
        ...     initial_population=population,
        ... )
        >>> result = runner.run()
    """

    config: ExperimentConfig
    engine: EvolutionEngine[G]
    initial_population: Population[G]
    tracker: MetricTracker = field(default_factory=NullTracker)
    checkpoint_manager: CheckpointManager | None = field(default=None)
    _started: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config and setup defaults."""
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")

        # Setup output directory
        output_dir = Path(self.config.output_dir) / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default tracker if not provided
        if isinstance(self.tracker, NullTracker):
            self.tracker = LocalTracker()

        # Default checkpoint manager
        if self.checkpoint_manager is None:
            checkpoint_dir = output_dir / "checkpoints"
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=self.config.checkpoint_interval,
            )

    def run(self, resume: bool = False) -> EvolutionResult[G]:
        """
        Execute the experiment.

        Args:
            resume: Whether to resume from checkpoint

        Returns:
            Evolution result with best individual
        """

        start_generation = 0
        population = self.initial_population

        # Try to resume from checkpoint
        if resume and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                population, start_generation = self._restore_checkpoint(checkpoint)
                print(f"Resumed from generation {checkpoint.generation}")

        # Start tracking
        self.tracker.start_run(self.config)
        self._started = True

        try:
            # Run evolution with monitoring
            result = self._run_evolution(population, start_generation)

            # Final checkpoint
            if self.checkpoint_manager:
                final_checkpoint = self._create_checkpoint(
                    result.population,
                    result.generations,
                    result.best,
                )
                self.checkpoint_manager.save(final_checkpoint)

            return result

        finally:
            self.tracker.end_run()
            self._started = False

    def _run_evolution(
        self,
        population: Population[G],
        start_generation: int,
    ) -> EvolutionResult[G]:
        """
        Run evolution loop with monitoring.

        Args:
            population: Starting population
            start_generation: Generation to start from

        Returns:
            Evolution result
        """
        from evolve.core.engine import EvolutionResult

        # Modify engine config for remaining generations
        remaining_gens = self.config.n_generations - start_generation
        self.engine.config.max_generations = remaining_gens

        # Store original stopping criterion

        # Create callback for tracking
        class TrackingCallback:
            def __init__(cb_self, runner: ExperimentRunner[G]) -> None:
                cb_self.runner = runner

            def on_generation_end(
                cb_self,
                generation: int,
                pop: Population[G],
                metrics: dict[str, Any],
            ) -> None:
                # Adjust generation number to account for resume
                actual_gen = start_generation + generation

                # Log metrics
                cb_self.runner.tracker.log_generation(actual_gen, metrics)

                # Checkpoint if needed
                if (
                    cb_self.runner.checkpoint_manager
                    and cb_self.runner.checkpoint_manager.should_checkpoint(actual_gen)
                ):
                    best = pop.best(1, minimize=cb_self.runner.engine.config.minimize)[0]
                    checkpoint = cb_self.runner._create_checkpoint(pop, actual_gen, best)
                    cb_self.runner.checkpoint_manager.save(checkpoint)

                # Check early stopping
                if cb_self.runner._should_stop(actual_gen, pop, metrics):
                    # Signal early stop
                    pass

        # Run with tracking callback
        tracking_cb = TrackingCallback(self)
        result = self.engine.run(population, callbacks=[tracking_cb])

        # Adjust result generation count
        adjusted_result = EvolutionResult(
            best=result.best,
            population=result.population,
            history=result.history,
            generations=start_generation + result.generations,
            stop_reason=result.stop_reason,
        )

        return adjusted_result

    def _should_stop(
        self,
        generation: int,
        _population: Population[G],
        metrics: dict[str, Any],
    ) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            generation: Current generation
            population: Current population
            metrics: Current metrics

        Returns:
            True if should stop early
        """
        # Target fitness reached
        if self.config.target_fitness is not None:
            best_fitness = metrics.get("best_fitness")
            if best_fitness is not None and best_fitness >= self.config.target_fitness:
                return True

        # Max evaluations exceeded
        if self.config.max_evaluations is not None:
            # Estimate total evaluations
            evaluations = generation * self.config.population_size
            if evaluations >= self.config.max_evaluations:
                return True

        return False

    def _create_checkpoint(
        self,
        population: Population[G],
        generation: int,
        best: Individual[G],
    ) -> Checkpoint:
        """
        Create checkpoint from current state.

        Args:
            population: Current population
            generation: Current generation
            best: Best individual

        Returns:
            Checkpoint object
        """
        return Checkpoint(
            experiment_name=self.config.name,
            config_hash=self.config.hash(),
            generation=generation,
            population=population.individuals,
            best_individual=best,
            rng_state=self.engine.get_rng_state(),
            fitness_history=self.engine.history,
        )

    def _restore_checkpoint(
        self,
        checkpoint: Checkpoint,
    ) -> tuple[Population[G], int]:
        """
        Restore state from checkpoint.

        Args:
            checkpoint: Checkpoint to restore

        Returns:
            Tuple of (population, start_generation)
        """
        from evolve.core.population import Population

        # Verify config hash
        if checkpoint.config_hash != self.config.hash():
            print(
                f"Warning: Config hash mismatch. "
                f"Checkpoint: {checkpoint.config_hash[:8]}, "
                f"Current: {self.config.hash()[:8]}"
            )

        # Restore RNG state
        if checkpoint.rng_state:
            self.engine.set_rng_state(checkpoint.rng_state)

        # Restore history
        if checkpoint.fitness_history:
            self.engine._history = checkpoint.fitness_history

        # Recreate population
        population = Population(
            individuals=checkpoint.population,
            generation=checkpoint.generation,
        )

        return population, checkpoint.generation + 1


@dataclass
class ExperimentComparison:
    """
    Compare results across multiple experiments.

    Loads metrics from multiple experiment directories
    and provides comparison utilities.

    Example:
        >>> comparison = ExperimentComparison({
        ...     "baseline": Path("experiments/baseline"),
        ...     "optimized": Path("experiments/optimized"),
        ... })
        >>> summary = comparison.summarize()
    """

    experiments: dict[str, Path]

    def load_metrics(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load metrics from all experiments.

        Returns:
            Dictionary mapping experiment name to metrics list
        """
        import csv

        results: dict[str, list[dict[str, Any]]] = {}

        for name, path in self.experiments.items():
            metrics_file = path / "metrics.csv"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    results[name] = [
                        {k: float(v) if v else 0.0 for k, v in row.items()} for row in reader
                    ]

        return results

    def summarize(self) -> list[dict[str, Any]]:
        """
        Create summary of all experiments.

        Returns:
            List of summary dictionaries
        """
        summaries: list[dict[str, Any]] = []

        for name, path in self.experiments.items():
            config_file = path / "config.json"
            metrics_file = path / "metrics.csv"

            if not config_file.exists() or not metrics_file.exists():
                continue

            config = ExperimentConfig.from_json(config_file)
            metrics = self.load_metrics().get(name, [])

            if metrics:
                final = metrics[-1]
                summaries.append(
                    {
                        "name": name,
                        "final_best": final.get("best_fitness", 0),
                        "final_mean": final.get("mean_fitness", 0),
                        "generations": len(metrics),
                        "seed": config.seed,
                        "population_size": config.population_size,
                    }
                )

        return summaries


@dataclass
class SweepConfig:
    """
    Configuration for hyperparameter sweep.

    Defines the parameter space for experiments.

    Example:
        >>> sweep = SweepConfig(
        ...     base_config=ExperimentConfig(...),
        ...     parameter_space={
        ...         "population_size": [50, 100, 200],
        ...         "mutation_rate": [0.01, 0.1, 0.5],
        ...     },
        ... )
        >>> configs = sweep.generate_configs()
    """

    base_config: ExperimentConfig
    parameter_space: dict[str, list[Any]]
    num_seeds: int = 3

    def generate_configs(self) -> list[ExperimentConfig]:
        """
        Generate all config combinations.

        Returns:
            List of configs for each parameter combination
        """
        import itertools

        configs: list[ExperimentConfig] = []

        # Get parameter names and values
        param_names = list(self.parameter_space.keys())
        param_values = list(self.parameter_space.values())

        # Generate all combinations
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))

            # Generate configs for each seed
            for seed_idx in range(self.num_seeds):
                # Create config dict from base
                config_dict = self.base_config.to_dict()

                # Update with sweep parameters
                config_dict.update(params)

                # Set unique seed
                base_seed = config_dict.get("seed", 42)
                config_dict["seed"] = base_seed + seed_idx

                # Generate unique name
                param_str = "_".join(f"{k}={v}" for k, v in params.items())
                config_dict["name"] = f"{self.base_config.name}_{param_str}_seed{seed_idx}"

                configs.append(ExperimentConfig.from_dict(config_dict))

        return configs


__all__ = [
    "ExperimentRunner",
    "ExperimentComparison",
    "SweepConfig",
]
