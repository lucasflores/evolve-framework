"""
Metric tracking for experiments.

Provides protocols and implementations for logging
experiment metrics and artifacts.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from evolve.experiment.config import ExperimentConfig


class MetricTracker(Protocol):
    """
    Abstract interface for experiment tracking.
    
    Implementations may log to:
    - Local files (default)
    - MLflow
    - Weights & Biases
    - TensorBoard
    - Custom backends
    
    Example:
        >>> tracker = LocalTracker()
        >>> tracker.start_run(config)
        >>> tracker.log_generation(0, {"best_fitness": 10.5})
        >>> tracker.end_run()
    """

    def start_run(self, config: ExperimentConfig) -> None:
        """
        Start tracking a new experiment run.
        
        Args:
            config: Experiment configuration
        """
        ...

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Log metrics for a generation.
        
        Standard metrics:
        - best_fitness
        - mean_fitness
        - std_fitness
        - diversity
        
        Args:
            generation: Generation number
            metrics: Dictionary of metric values
        """
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Parameter dictionary
        """
        ...

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """
        Log file as artifact.
        
        Args:
            path: Path to file
            name: Optional artifact name (defaults to filename)
        """
        ...

    def end_run(self) -> None:
        """Finalize tracking."""
        ...


@dataclass
class LocalTracker:
    """
    Simple local file-based tracking.
    
    Creates CSV files and JSON logs in the output directory.
    
    Output structure:
        output_dir/
            config.json
            params.json
            metrics.csv
            summary.json
            artifacts/
                ...
    
    Example:
        >>> tracker = LocalTracker()
        >>> tracker.start_run(config)
        >>> for gen in range(100):
        ...     tracker.log_generation(gen, {"best_fitness": fitness})
        >>> tracker.end_run()
    """

    output_dir: Path | None = field(default=None)
    metrics_file: Path | None = field(default=None)
    _started: bool = field(default=False, repr=False)
    _metrics_writer: Any = field(default=None, repr=False)
    _metrics_file_handle: Any = field(default=None, repr=False)

    def start_run(self, config: ExperimentConfig) -> None:
        """Start tracking with config."""
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_json(self.output_dir / "config.json")

        # Initialize metrics CSV
        self.metrics_file = self.output_dir / "metrics.csv"
        self._metrics_file_handle = open(self.metrics_file, "w", newline="")
        self._metrics_writer = csv.writer(self._metrics_file_handle)
        self._metrics_writer.writerow([
            "generation", "best_fitness", "mean_fitness", "std_fitness",
            "min_fitness", "max_fitness", "diversity"
        ])

        # Create artifacts directory
        (self.output_dir / "artifacts").mkdir(exist_ok=True)

        self._started = True

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """Log metrics to CSV."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        self._metrics_writer.writerow([
            generation,
            metrics.get("best_fitness", 0),
            metrics.get("mean_fitness", 0),
            metrics.get("std_fitness", 0),
            metrics.get("min_fitness", 0),
            metrics.get("max_fitness", 0),
            metrics.get("diversity", 0),
        ])
        # Flush to ensure data is written
        self._metrics_file_handle.flush()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to JSON."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        with open(self.output_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Copy artifact to output directory."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        path = Path(path)
        dest = self.output_dir / "artifacts" / (name or path.name)
        shutil.copy(path, dest)

    def log_dict(self, filename: str, data: dict[str, Any]) -> None:
        """
        Log arbitrary dictionary to JSON file.
        
        Args:
            filename: Output filename
            data: Data to save
        """
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        with open(self.output_dir / filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def end_run(self) -> None:
        """Finalize tracking and write summary."""
        if not self._started:
            return

        # Close CSV file
        if self._metrics_file_handle:
            self._metrics_file_handle.close()
            self._metrics_file_handle = None
            self._metrics_writer = None

        # Write summary
        summary = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self._started = False


@dataclass
class CompositeTracker:
    """
    Tracker that logs to multiple backends.
    
    Example:
        >>> tracker = CompositeTracker([
        ...     LocalTracker(),
        ...     MLflowTracker(),
        ... ])
    """

    trackers: list[MetricTracker] = field(default_factory=list)

    def start_run(self, config: ExperimentConfig) -> None:
        """Start all trackers."""
        for tracker in self.trackers:
            tracker.start_run(config)

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """Log to all trackers."""
        for tracker in self.trackers:
            tracker.log_generation(generation, metrics)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to all trackers."""
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            tracker.log_artifact(path, name)

    def end_run(self) -> None:
        """End all tracker runs."""
        for tracker in self.trackers:
            tracker.end_run()


@dataclass
class NullTracker:
    """
    No-op tracker for testing or disabled logging.
    """

    def start_run(self, config: ExperimentConfig) -> None:
        pass

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        pass

    def end_run(self) -> None:
        pass


def compute_generation_metrics(
    fitness_values: list[float],
    diversity: float | None = None,
) -> dict[str, float]:
    """
    Compute standard generation metrics from fitness values.
    
    Args:
        fitness_values: List of fitness values
        diversity: Optional diversity measure
        
    Returns:
        Dictionary of computed metrics
    """
    import numpy as np
    
    values = np.array(fitness_values)
    
    metrics = {
        "best_fitness": float(np.max(values)),
        "min_fitness": float(np.min(values)),
        "mean_fitness": float(np.mean(values)),
        "std_fitness": float(np.std(values)),
        "max_fitness": float(np.max(values)),
    }
    
    if diversity is not None:
        metrics["diversity"] = diversity
    
    return metrics


__all__ = [
    "MetricTracker",
    "LocalTracker",
    "CompositeTracker",
    "NullTracker",
    "compute_generation_metrics",
]
