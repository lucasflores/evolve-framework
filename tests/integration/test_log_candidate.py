"""log_candidate(): one candidate as a nested MLflow run under the active run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from evolve.experiment.tracking import log_candidate  # noqa: E402


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)  # an sqlite store writes artifacts under ./mlruns
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.mark.usefixtures("store")
def test_logs_a_nested_run_in_the_parents_experiment() -> None:
    # The parent's experiment is named by id only, never made current
    experiment = mlflow.create_experiment("search")
    with mlflow.start_run(experiment_id=experiment) as parent:
        assert log_candidate(
            "candidate-abc",
            tags={"status": "scored"},
            metrics={"objective.brier": 0.2},
            dicts={"spec.json": {"encoder": "e5"}},
        )
    client = MlflowClient()
    runs = client.search_runs(
        [experiment], filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    )
    assert len(runs) == 1
    child = runs[0]
    assert child.info.run_name == "candidate-abc"
    assert child.data.tags["status"] == "scored"
    assert child.data.metrics == {"objective.brier": 0.2}
    assert [a.path for a in client.list_artifacts(child.info.run_id)] == ["spec.json"]


@pytest.mark.usefixtures("store")
def test_without_an_active_run_it_does_nothing() -> None:
    assert log_candidate("candidate-x", metrics={"m": 1.0}) is False
    assert len(MlflowClient().search_experiments()) == 1  # only the default, untouched
    assert mlflow.search_runs(search_all_experiments=True).empty


def test_without_mlflow_it_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # import mlflow now fails
    assert log_candidate("candidate-x", metrics={"m": 1.0}) is False


@pytest.mark.usefixtures("store")
def test_a_failure_is_a_warning_and_the_run_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_metrics: object) -> None:
        raise OSError("disk full")

    with mlflow.start_run() as parent:
        monkeypatch.setattr(mlflow, "log_metrics", broken)
        with pytest.warns(RuntimeWarning, match="candidate-x.*disk full"):
            assert log_candidate("candidate-x", metrics={"m": 1.0}) is False
        assert mlflow.active_run().info.run_id == parent.info.run_id
