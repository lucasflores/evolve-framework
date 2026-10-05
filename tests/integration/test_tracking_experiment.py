"""A tracked run's experiment is the current one, so nested runs land in it."""

from __future__ import annotations

from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from evolve.config.tracking import TrackingConfig  # noqa: E402
from evolve.experiment.config import ExperimentConfig  # noqa: E402
from evolve.experiment.tracking.mlflow_tracker import (  # noqa: E402
    MLflowTracker,
    ResilientMLflowTracker,
)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)  # an sqlite store writes artifacts under ./mlruns
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _nested_experiment() -> str:
    """Where a nested run started without an experiment id lands."""
    with mlflow.start_run(nested=True) as child:
        return child.info.experiment_id


def test_the_resilient_trackers_nested_runs_land_in_its_experiment(store: str) -> None:
    tracker = ResilientMLflowTracker(
        config=TrackingConfig(backend="mlflow", experiment_name="search", tracking_uri=store)
    )
    tracker.start_run()
    parent = mlflow.active_run()
    assert parent is not None
    assert parent.info.experiment_id != "0"
    assert _nested_experiment() == parent.info.experiment_id


def test_the_plain_trackers_nested_runs_land_in_its_experiment(store: str) -> None:
    tracker = MLflowTracker(experiment_name="plain", tracking_uri=store)
    tracker.start_run(ExperimentConfig(name="plain-run"))
    parent = mlflow.active_run()
    assert parent is not None
    assert parent.info.experiment_id != "0"
    assert _nested_experiment() == parent.info.experiment_id
