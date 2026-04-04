"""
Weights & Biases tracking integration.

Requires wandb to be installed:
    pip install wandb
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from evolve.experiment.config import ExperimentConfig


@dataclass
class WandbTracker:
    """
    Weights & Biases experiment tracking.

    Logs metrics, parameters, and artifacts to W&B.

    Example:
        >>> tracker = WandbTracker(project="evolve")
        >>> tracker.start_run(config)
        >>> tracker.log_generation(0, {"best_fitness": 10.5})
        >>> tracker.end_run()

    Requires:
        pip install wandb
    """

    project: str = "evolve"
    entity: str | None = field(default=None)
    run: Any = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

    def start_run(self, config: ExperimentConfig) -> None:
        """Start W&B run."""
        # Create config dict
        config_dict = config.to_dict()

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=config.name,
            config=config_dict,
            tags=[f"config:{config.hash()[:8]}"],
        )
        self._started = True

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """Log metrics to W&B."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        wandb.log(metrics, step=generation)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters to W&B config."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        wandb.config.update(params)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Log artifact to W&B."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        path = Path(path)
        artifact = wandb.Artifact(
            name=name or path.stem,
            type="experiment_artifact",
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def log_dict(self, filename: str, data: dict[str, Any]) -> None:
        """Log dictionary as table."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        # Convert to W&B table if possible
        wandb.log({filename: wandb.Table(data=list(data.items()), columns=["key", "value"])})

    def end_run(self) -> None:
        """End W&B run."""
        if not self._started:
            return

        wandb.finish()
        self._started = False
        self.run = None


__all__ = ["WandbTracker", "WANDB_AVAILABLE"]
