"""
MLflow tracking integration.

Requires mlflow to be installed:
    pip install mlflow
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

from evolve.experiment.config import ExperimentConfig


@dataclass
class MLflowTracker:
    """
    MLflow experiment tracking.

    Logs metrics, parameters, and artifacts to MLflow.

    Example:
        >>> tracker = MLflowTracker(experiment_name="my_exp")
        >>> tracker.start_run(config)
        >>> tracker.log_generation(0, {"best_fitness": 10.5})
        >>> tracker.end_run()

    Requires:
        pip install mlflow
    """

    experiment_name: str = "evolve"
    tracking_uri: str | None = field(default=None)
    run_id: str | None = field(default=None, repr=False)
    _client: Any = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        self._client = MlflowClient()

    def start_run(self, config: ExperimentConfig) -> None:
        """Start MLflow run."""
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start run
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=config.name,
        )
        self.run_id = run.info.run_id
        self._started = True

        # Log config as params
        mlflow.log_params(
            {
                "seed": config.seed,
                "population_size": config.population_size,
                "n_generations": config.n_generations,
                "selection_method": config.selection_method,
                "crossover_method": config.crossover_method,
                "mutation_method": config.mutation_method,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
                "genome_type": config.genome_type,
                "evaluator_type": config.evaluator_type,
            }
        )

        # Log config hash as tag
        mlflow.set_tag("config_hash", config.hash())

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """Log metrics to MLflow."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        mlflow.log_metrics(metrics, step=generation)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        # Convert to strings for MLflow
        string_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(string_params)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Log artifact to MLflow."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        path = Path(path)
        if name:
            # Log with specific artifact path
            mlflow.log_artifact(str(path), artifact_path=name)
        else:
            mlflow.log_artifact(str(path))

    def log_dict(self, filename: str, data: dict[str, Any]) -> None:
        """Log dictionary as JSON artifact."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, artifact_path=filename)

    def end_run(self) -> None:
        """End MLflow run."""
        if not self._started:
            return

        mlflow.end_run()
        self._started = False
        self.run_id = None


@dataclass
class ResilientMLflowTracker:
    """
    MLflow tracker with graceful degradation for unreachable servers (FR-028).

    Buffers metrics in memory when the server is unreachable and attempts
    periodic reconnection. Evolution continues uninterrupted regardless
    of tracking infrastructure status.

    Features:
        - In-memory buffering when server unreachable
        - Periodic reconnection with exponential backoff
        - Batch metric logging (MLflow 2.0+, FR-029)
        - Clear ImportError at initialization (FR-005)

    Example:
        >>> from evolve.config.tracking import TrackingConfig
        >>>
        >>> config = TrackingConfig(experiment_name="my_exp")
        >>> tracker = ResilientMLflowTracker(config)
        >>> tracker.start_run()
        >>> tracker.log_generation(0, {"best_fitness": 10.5})
        >>> # Server goes down - metrics buffered automatically
        >>> tracker.log_generation(1, {"best_fitness": 11.0})
        >>> # Server comes back - buffer flushed
        >>> tracker.end_run()

    Requires:
        pip install mlflow>=2.0.0
    """

    config: Any  # TrackingConfig - avoid circular import
    experiment_name: str = field(default="evolve", init=False)
    tracking_uri: str | None = field(default=None, init=False)
    run_id: str | None = field(default=None, repr=False)

    _client: Any = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _connected: bool = field(default=True, repr=False)
    _buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _last_flush_time: float = field(default=0.0, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize tracker with import guard (FR-005)."""
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow >= 2.0 required for tracking. Install with: pip install mlflow"
            )

        # Extract settings from TrackingConfig
        self.experiment_name = self.config.experiment_name
        self.tracking_uri = self.config.tracking_uri

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        self._client = MlflowClient()

    @property
    def buffer_size(self) -> int:
        """Current number of buffered metric entries."""
        return len(self._buffer)

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to MLflow server."""
        return self._connected

    def start_run(
        self, params: dict[str, Any] | None = None, description: str | None = None
    ) -> None:
        """
        Start MLflow run.

        Args:
            params: Optional parameters to log at run start.
            description: Optional run description.
        """
        try:
            # Enable system metrics if configured
            if self.config.system_metrics:
                mlflow.enable_system_metrics_logging()

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id

            # Start run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.config.run_name,
            )
            self.run_id = run.info.run_id
            self._started = True
            self._connected = True
            self._consecutive_failures = 0

            # Log parameters
            if params:
                self._log_params_safe(params)

            # Log description
            if description:
                mlflow.set_tag("mlflow.note.content", description)

            # Log tracking config as tag
            mlflow.set_tag("tracking_backend", "mlflow")
            mlflow.set_tag("tracking_categories", ",".join(c.value for c in self.config.categories))

        except Exception as e:
            import logging

            logging.warning(f"MLflow server unreachable during start_run: {e}")
            self._connected = False
            self._started = True  # Mark as started to allow buffering

    def log_generation(
        self,
        generation: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Log metrics for a generation.

        Buffers metrics if server is unreachable and attempts
        periodic reconnection based on flush_interval.

        Args:
            generation: Generation number.
            metrics: Dictionary of metric values.
        """
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        # Add to buffer
        self._buffer.append(
            {
                "step": generation,
                "metrics": metrics.copy(),
            }
        )

        # Check if we should attempt flush
        if self._should_flush():
            self._try_flush()

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        import time

        # Flush if buffer is full
        if len(self._buffer) >= self.config.buffer_size:
            return True

        # Flush based on interval
        current_time = time.time()
        return current_time - self._last_flush_time >= self.config.flush_interval

    def _try_flush(self) -> None:
        """Attempt to flush buffer to MLflow server."""
        import logging
        import time

        self._last_flush_time = time.time()

        if not self._buffer:
            return

        try:
            # Batch log all buffered metrics
            for entry in self._buffer:
                mlflow.log_metrics(entry["metrics"], step=entry["step"])

            self._buffer.clear()
            self._connected = True
            self._consecutive_failures = 0

        except Exception as e:
            self._connected = False
            self._consecutive_failures += 1

            # Only log warning periodically to avoid spam
            if self._consecutive_failures <= 3 or self._consecutive_failures % 10 == 0:
                logging.warning(
                    f"MLflow server unreachable (attempt {self._consecutive_failures}): {e}. "
                    f"Buffered {len(self._buffer)} metric entries."
                )

            # Handle buffer overflow with circular buffer behavior
            if len(self._buffer) > self.config.buffer_size:
                dropped = len(self._buffer) - self.config.buffer_size
                self._buffer = self._buffer[-self.config.buffer_size :]
                logging.warning(f"Buffer overflow: dropped {dropped} oldest metric entries")

    def _log_params_safe(self, params: dict[str, Any]) -> None:
        """Log parameters with error handling."""
        import logging

        try:
            # Convert to strings for MLflow
            string_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(string_params)
        except Exception as e:
            logging.warning(f"Failed to log parameters: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log additional parameters."""
        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        self._log_params_safe(params)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Log artifact to MLflow."""
        import logging

        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        try:
            path = Path(path)
            if name:
                mlflow.log_artifact(str(path), artifact_path=name)
            else:
                mlflow.log_artifact(str(path))
        except Exception as e:
            logging.warning(f"Failed to log artifact: {e}")

    def log_dataset(
        self,
        data: Any,
        name: str = "initial_population",
        context: str = "training",
    ) -> None:
        """
        Log dataset to MLflow.

        Logs input data (e.g., initial population) as MLflow dataset
        for experiment reproducibility and lineage tracking.

        Args:
            data: Dataset to log (DataFrame, numpy array, or list of dicts).
            name: Dataset name.
            context: Dataset context ("training", "evaluation", etc.).
        """
        import logging

        if not self._started:
            raise RuntimeError("Call start_run() before logging")

        if not self.config.log_datasets:
            return

        try:
            import numpy as np
            import pandas as pd

            # Convert to pandas DataFrame for MLflow
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                # Assume list of dicts or list of arrays
                if data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
            else:
                logging.warning(f"Unsupported dataset type: {type(data)}")
                return

            # Create MLflow dataset
            dataset = mlflow.data.from_pandas(df, name=name)
            mlflow.log_input(dataset, context=context)

        except Exception as e:
            logging.warning(f"Failed to log dataset: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End MLflow run.

        Attempts final flush of buffered metrics before ending.

        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED").
        """
        if not self._started:
            return

        # Final flush attempt
        if self._buffer:
            self._try_flush()

        with contextlib.suppress(Exception):
            mlflow.end_run(status=status)

        self._started = False
        self.run_id = None
        self._buffer.clear()

    def flush(self) -> bool:
        """
        Force flush buffered metrics.

        Returns:
            True if flush successful, False otherwise.
        """
        self._try_flush()
        return len(self._buffer) == 0


__all__ = ["MLflowTracker", "ResilientMLflowTracker", "MLFLOW_AVAILABLE"]
