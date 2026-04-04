"""
Tracking backends for experiments.

Provides integrations with external tracking services
like MLflow and Weights & Biases.
"""

from evolve.experiment.metrics import (
    CompositeTracker,
    LocalTracker,
    MetricTracker,
    NullTracker,
    compute_generation_metrics,
)

# Optional imports for third-party trackers
_MLFLOW_AVAILABLE = False
_WANDB_AVAILABLE = False

try:
    from evolve.experiment.tracking.mlflow_tracker import (
        MLflowTracker,
        ResilientMLflowTracker,
    )

    _MLFLOW_AVAILABLE = True
except ImportError:
    MLflowTracker = None  # type: ignore
    ResilientMLflowTracker = None  # type: ignore

# Tracking callback
from evolve.experiment.tracking.callback import TrackingCallback  # noqa: E402

try:
    from evolve.experiment.tracking.wandb_tracker import WandbTracker

    _WANDB_AVAILABLE = True
except ImportError:
    WandbTracker = None  # type: ignore


def is_mlflow_available() -> bool:
    """Check if MLflow tracking is available."""
    return _MLFLOW_AVAILABLE


def is_wandb_available() -> bool:
    """Check if Weights & Biases tracking is available."""
    return _WANDB_AVAILABLE


__all__ = [
    "MetricTracker",
    "LocalTracker",
    "CompositeTracker",
    "NullTracker",
    "compute_generation_metrics",
    "is_mlflow_available",
    "is_wandb_available",
    "MLflowTracker",
    "ResilientMLflowTracker",
    "TrackingCallback",
    "WandbTracker",
]
