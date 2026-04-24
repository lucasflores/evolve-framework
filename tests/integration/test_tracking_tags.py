"""Integration tests for TrackingCallback tags and dataset logging (T041)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

mlflow = pytest.importorskip("mlflow")

from evolve.config.tracking import TrackingConfig
from evolve.config.unified import DatasetConfig, UnifiedConfig
from evolve.experiment.tracking.callback import TrackingCallback


def _make_callback(
    *,
    tags: tuple[str, ...] = (),
    training_data: DatasetConfig | None = None,
    validation_data: DatasetConfig | None = None,
) -> TrackingCallback:
    """Helper to build a TrackingCallback with a null tracker."""
    config = UnifiedConfig(
        tags=tags,
        training_data=training_data,
        validation_data=validation_data,
        tracking=TrackingConfig(
            experiment_name="test",
            backend="null",
            log_datasets=True,
        ),
    )
    return TrackingCallback(
        config=config.tracking,
        unified_config_dict=config.to_dict(),
    )


class TestTrackingCallbackTags:
    """T039: Tags from UnifiedConfig are logged as MLflow tag."""

    @patch("mlflow.set_tag")
    def test_tags_logged_on_run_start(self, mock_set_tag: MagicMock) -> None:
        cb = _make_callback(tags=("research", "evolve"))
        cb.on_run_start(None)
        mock_set_tag.assert_any_call("evolve.tags", "research,evolve")

    @patch("mlflow.set_tag")
    def test_no_tags_logged_when_empty(self, mock_set_tag: MagicMock) -> None:
        cb = _make_callback(tags=())
        cb.on_run_start(None)
        tag_calls = [c for c in mock_set_tag.call_args_list if c.args[0] == "evolve.tags"]
        assert len(tag_calls) == 0

    @patch("mlflow.set_tag")
    def test_no_tags_logged_when_default(self, mock_set_tag: MagicMock) -> None:
        cb = _make_callback()
        cb.on_run_start(None)
        tag_calls = [c for c in mock_set_tag.call_args_list if c.args[0] == "evolve.tags"]
        assert len(tag_calls) == 0


class TestTrackingCallbackDatasetConfigs:
    """T040: Dataset configs are logged as MLflow tags."""

    @patch("mlflow.set_tag")
    def test_training_data_tags_logged(self, mock_set_tag: MagicMock) -> None:
        ds = DatasetConfig(name="mnist", path="/data/mnist", context="training")
        cb = _make_callback(training_data=ds)
        cb.on_run_start(None)
        calls = {(c.args[0], c.args[1]) for c in mock_set_tag.call_args_list}
        assert ("dataset.training_data.name", "mnist") in calls
        assert ("dataset.training_data.path", "/data/mnist") in calls
        assert ("dataset.training_data.context", "training") in calls

    @patch("mlflow.set_tag")
    def test_validation_data_tags_logged(self, mock_set_tag: MagicMock) -> None:
        ds = DatasetConfig(name="val_set", path="/data/val")
        cb = _make_callback(validation_data=ds)
        cb.on_run_start(None)
        calls = {(c.args[0], c.args[1]) for c in mock_set_tag.call_args_list}
        assert ("dataset.validation_data.name", "val_set") in calls
        assert ("dataset.validation_data.path", "/data/val") in calls

    @patch("mlflow.set_tag")
    def test_no_dataset_tags_when_none(self, mock_set_tag: MagicMock) -> None:
        cb = _make_callback()
        cb.on_run_start(None)
        dataset_calls = [c for c in mock_set_tag.call_args_list if c.args[0].startswith("dataset.")]
        assert len(dataset_calls) == 0
