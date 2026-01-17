"""
MLflow tracking integration.

Requires mlflow to be installed:
    pip install mlflow
"""

from __future__ import annotations

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
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow"
            )
        
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
        mlflow.log_params({
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
        })

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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
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


__all__ = ["MLflowTracker", "MLFLOW_AVAILABLE"]
