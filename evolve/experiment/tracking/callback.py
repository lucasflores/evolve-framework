"""
Tracking callback for experiment observability.

Provides a callback that integrates with MLflow tracking
through ResilientMLflowTracker.

NO ML FRAMEWORK IMPORTS ALLOWED (except via lazy import).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evolve.config.tracking import TrackingConfig
from evolve.core.callbacks import Callback

if TYPE_CHECKING:
    from evolve.core.population import Population


logger = logging.getLogger(__name__)


@dataclass
class TrackingCallback(Callback):
    """
    Callback for logging metrics to tracking backend.

    Integrates with ResilientMLflowTracker to log metrics per generation.
    Supports graceful degradation when tracking server is unavailable.

    Attributes:
        config: Tracking configuration.
        unified_config_dict: Serialized UnifiedConfig for parameter logging.
        description: Run description for MLflow UI.
        evaluation_data: Optional dataset used for fitness evaluation.
            If provided, logged as MLflow dataset for reproducibility.
            Examples: training images, time series data, regression targets.

    Example:
        >>> from evolve.config.tracking import TrackingConfig
        >>> import pandas as pd
        >>>
        >>> # Load your evaluation data
        >>> eval_data = pd.read_csv("train.csv")
        >>>
        >>> tracking = TrackingConfig.standard("my_experiment")
        >>> callback = TrackingCallback(
        ...     config=tracking,
        ...     evaluation_data=eval_data,  # The data your fitness function uses
        ... )
    """

    config: TrackingConfig
    unified_config_dict: dict[str, Any] | None = None
    description: str | None = None
    evaluation_data: Any = None  # DataFrame, np.ndarray, or list - the data fitness is computed on
    evaluation_data_name: str = "evaluation_data"

    _tracker: Any = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _generation_counter: int = field(default=0, repr=False)
    _dataset_logged: bool = field(default=False, repr=False)

    def _ensure_tracker(self) -> None:
        """Lazily initialize tracker on first use."""
        if self._tracker is not None:
            return

        if self.config.backend == "mlflow":
            from evolve.experiment.tracking.mlflow_tracker import (
                MLFLOW_AVAILABLE,
                ResilientMLflowTracker,
            )

            if not MLFLOW_AVAILABLE:
                raise ImportError(
                    "MLflow >= 2.0 required for tracking. Install with: pip install mlflow"
                )

            self._tracker = ResilientMLflowTracker(config=self.config)

        elif self.config.backend == "null":
            # No-op tracker for testing
            self._tracker = _NullTracker()

        else:
            raise ValueError(f"Unsupported tracking backend: {self.config.backend}")

    def on_run_start(
        self,
        config: Any,
    ) -> None:
        """
        Called when evolution starts.

        Initializes tracking run and logs parameters.
        """
        if not self.config.enabled:
            return

        try:
            self._ensure_tracker()

            # Flatten unified config to params dict (including nested dicts)
            params = {}
            description = self.description

            if self.unified_config_dict:
                params = self._flatten_config(self.unified_config_dict)

                # Extract description from config if not explicitly set
                if not description:
                    description = self.unified_config_dict.get("description")

            self._tracker.start_run(params=params, description=description)
            self._started = True
            self._generation_counter = 0
            self._dataset_logged = False

            # Log full config as JSON artifact
            if self.unified_config_dict:
                self._log_config_artifact()

            # Log evaluation data if provided
            if self.evaluation_data is not None and self.config.log_datasets:
                self._log_evaluation_data()

        except ImportError:
            # Don't silently swallow missing dependencies - fail loudly
            raise
        except Exception as e:
            logger.warning(f"Failed to start tracking: {e}")
            self._started = False

    def on_generation_end(
        self,
        generation: int,
        population: Population[Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Called at the end of each generation.

        Logs metrics if tracking is enabled and log_interval is satisfied.
        """
        if not self.config.enabled or not self._started:
            return

        self._generation_counter += 1

        # Check log interval
        if self._generation_counter % self.config.log_interval != 0:
            return

        # Build metrics dict
        log_metrics: dict[str, float] = {}

        # Always log core metrics if available
        if metrics:
            log_metrics.update(metrics)

        # Log to tracker
        try:
            self._tracker.log_generation(generation, log_metrics)
        except Exception as e:
            logger.warning(f"Failed to log generation {generation}: {e}")

    def _log_evaluation_data(self) -> None:
        """
        Log evaluation data as MLflow dataset.

        This is the data that fitness functions evaluate solutions against.
        For example: training images, time series, regression targets.
        """
        if self._dataset_logged or self.evaluation_data is None:
            return

        try:
            # Log to tracker
            if hasattr(self._tracker, "log_dataset"):
                self._tracker.log_dataset(
                    self.evaluation_data,
                    name=self.evaluation_data_name,
                    context="evaluation",
                )
                self._dataset_logged = True

        except Exception as e:
            logger.warning(f"Failed to log evaluation dataset: {e}")

    def _flatten_config(
        self,
        config: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """
        Flatten nested config dict for MLflow parameters.

        Nested dicts become dot-separated keys:
            {"crossover_params": {"eta": 20}} -> {"crossover_params.eta": 20}
        """
        result = {}
        for key, value in config.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                result.update(self._flatten_config(value, f"{full_key}."))
            elif isinstance(value, (list, tuple)):
                # Convert lists to string representation
                if len(value) <= 10:  # Only for short lists
                    result[full_key] = str(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                result[full_key] = value

        return result

    def _log_config_artifact(self) -> None:
        """Log full unified config as JSON artifact."""
        if not self.unified_config_dict:
            return

        try:
            import json
            import tempfile
            from pathlib import Path

            # Write config to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(self.unified_config_dict, f, indent=2, default=str)
                temp_path = f.name

            # Log as artifact
            if hasattr(self._tracker, "log_artifact"):
                self._tracker.log_artifact(temp_path, "config")

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Failed to log config artifact: {e}")

    def _log_best_solution(self, population: Population[Any]) -> None:
        """
        Log best solution as artifact.

        Saves genome data as JSON for reproducibility.
        """
        try:
            import json
            import tempfile
            from pathlib import Path

            # Find best individual
            best = None
            best_fitness = None
            for ind in population.individuals:
                if ind.fitness is not None:
                    fitness_val = (
                        ind.fitness.values[0]
                        if hasattr(ind.fitness, "values")
                        else float(ind.fitness)
                    )
                    if best_fitness is None or fitness_val < best_fitness:  # Assume minimization
                        best = ind
                        best_fitness = fitness_val

            if best is None:
                return

            # Build solution dict
            solution = {
                "fitness": best_fitness,
            }

            # Extract genome data
            genome = best.genome
            if hasattr(genome, "genes"):
                genes = genome.genes
                if hasattr(genes, "tolist"):
                    solution["genes"] = genes.tolist()
                else:
                    solution["genes"] = list(genes)

            if hasattr(genome, "to_dict"):
                solution["genome"] = genome.to_dict()

            # Write to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(solution, f, indent=2, default=str)
                temp_path = f.name

            # Log as artifact
            if hasattr(self._tracker, "log_artifact"):
                self._tracker.log_artifact(temp_path, "best_solution")

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Failed to log best solution: {e}")

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """
        Called when evolution ends.

        Logs best solution and finalizes tracking run.
        """
        if not self._started:
            return

        try:
            # Log best solution as artifact
            self._log_best_solution(population)

            self._tracker.end_run(status="FINISHED")
        except Exception as e:
            logger.warning(f"Failed to end tracking run: {e}")
        finally:
            self._started = False

    def on_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """
        Called when an error occurs during evolution.

        Marks tracking run as failed.
        """
        if not self._started:
            return

        try:
            self._tracker.end_run(status="FAILED")
        except Exception:
            pass
        finally:
            self._started = False


class _NullTracker:
    """No-op tracker for testing and disabled tracking."""

    def start_run(self, params: dict[str, Any] | None = None) -> None:
        pass

    def log_generation(self, generation: int, metrics: dict[str, float]) -> None:
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        pass


__all__ = ["TrackingCallback"]
