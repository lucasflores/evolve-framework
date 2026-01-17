"""
Experiment tracking and reproducibility module.

Provides:
- ExperimentConfig: Complete configuration for experiments
- Checkpoint/CheckpointManager: State saving and restoration
- MetricTracker: Interface for logging metrics
- LocalTracker: File-based tracking
- ExperimentRunner: Orchestrates experiments

Example:
    >>> from evolve.experiment import ExperimentConfig, ExperimentRunner
    >>> 
    >>> config = ExperimentConfig(
    ...     name="my_experiment",
    ...     seed=42,
    ...     population_size=100,
    ...     n_generations=100,
    ... )
    >>> runner = ExperimentRunner(config=config, engine=engine, initial_population=pop)
    >>> result = runner.run()
"""

from evolve.experiment.checkpoint import Checkpoint, CheckpointManager
from evolve.experiment.config import ConfigValidationError, ExperimentConfig
from evolve.experiment.metrics import (
    CompositeTracker,
    LocalTracker,
    MetricTracker,
    NullTracker,
    compute_generation_metrics,
)
from evolve.experiment.runner import (
    ExperimentComparison,
    ExperimentRunner,
    SweepConfig,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "ConfigValidationError",
    # Checkpointing
    "Checkpoint",
    "CheckpointManager",
    # Tracking
    "MetricTracker",
    "LocalTracker",
    "CompositeTracker",
    "NullTracker",
    "compute_generation_metrics",
    # Runner
    "ExperimentRunner",
    "ExperimentComparison",
    "SweepConfig",
]
