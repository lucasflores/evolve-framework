"""Unit tests for DatasetConfig and UnifiedConfig dataset fields (US8 / T038)."""

from __future__ import annotations

from evolve.config.unified import DatasetConfig, UnifiedConfig


class TestDatasetConfig:
    """DatasetConfig serialization and construction."""

    def test_minimal_construction(self):
        ds = DatasetConfig(name="mnist")
        assert ds.name == "mnist"
        assert ds.path is None
        assert ds.data is None
        assert ds.context == ""

    def test_full_construction(self):
        ds = DatasetConfig(name="cifar", path="/data/cifar", data=[1, 2], context="training")
        assert ds.name == "cifar"
        assert ds.path == "/data/cifar"
        assert ds.data == [1, 2]
        assert ds.context == "training"

    def test_to_dict_minimal(self):
        ds = DatasetConfig(name="mnist")
        d = ds.to_dict()
        assert d == {"name": "mnist"}

    def test_to_dict_excludes_data(self):
        ds = DatasetConfig(name="mnist", data=[1, 2, 3])
        d = ds.to_dict()
        assert "data" not in d

    def test_to_dict_includes_path_and_context(self):
        ds = DatasetConfig(name="mnist", path="/tmp/mnist", context="training")
        d = ds.to_dict()
        assert d == {"name": "mnist", "path": "/tmp/mnist", "context": "training"}

    def test_from_dict_minimal(self):
        ds = DatasetConfig.from_dict({"name": "mnist"})
        assert ds.name == "mnist"
        assert ds.path is None
        assert ds.context == ""

    def test_from_dict_full(self):
        ds = DatasetConfig.from_dict({"name": "cifar", "path": "/data", "context": "val"})
        assert ds.name == "cifar"
        assert ds.path == "/data"
        assert ds.context == "val"

    def test_round_trip(self):
        original = DatasetConfig(name="test", path="/p", context="ctx")
        restored = DatasetConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.context == original.context

    def test_frozen(self):
        ds = DatasetConfig(name="x")
        try:
            ds.name = "y"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestUnifiedConfigDatasetFields:
    """UnifiedConfig training_data / validation_data integration."""

    def test_defaults_none(self):
        cfg = UnifiedConfig()
        assert cfg.training_data is None
        assert cfg.validation_data is None

    def test_set_training_data(self):
        ds = DatasetConfig(name="train")
        cfg = UnifiedConfig(training_data=ds)
        assert cfg.training_data is ds

    def test_to_dict_with_datasets(self):
        train = DatasetConfig(name="train", path="/t")
        val = DatasetConfig(name="val", path="/v")
        cfg = UnifiedConfig(training_data=train, validation_data=val)
        d = cfg.to_dict()
        assert d["training_data"] == {"name": "train", "path": "/t"}
        assert d["validation_data"] == {"name": "val", "path": "/v"}

    def test_to_dict_none_datasets(self):
        cfg = UnifiedConfig()
        d = cfg.to_dict()
        assert d["training_data"] is None
        assert d["validation_data"] is None

    def test_from_dict_with_datasets(self):
        d = UnifiedConfig().to_dict()
        d["training_data"] = {"name": "train", "path": "/t"}
        d["validation_data"] = {"name": "val", "context": "validation"}
        cfg = UnifiedConfig.from_dict(d)
        assert cfg.training_data is not None
        assert cfg.training_data.name == "train"
        assert cfg.training_data.path == "/t"
        assert cfg.validation_data is not None
        assert cfg.validation_data.name == "val"
        assert cfg.validation_data.context == "validation"

    def test_from_dict_none_datasets(self):
        d = UnifiedConfig().to_dict()
        cfg = UnifiedConfig.from_dict(d)
        assert cfg.training_data is None
        assert cfg.validation_data is None

    def test_round_trip(self):
        train = DatasetConfig(name="train", path="/t", context="training")
        val = DatasetConfig(name="val", path="/v", context="validation")
        original = UnifiedConfig(training_data=train, validation_data=val)
        restored = UnifiedConfig.from_dict(original.to_dict())
        assert restored.training_data is not None
        assert restored.training_data.name == "train"
        assert restored.validation_data is not None
        assert restored.validation_data.name == "val"

    def test_hash_differs_with_datasets(self):
        c1 = UnifiedConfig()
        c2 = UnifiedConfig(training_data=DatasetConfig(name="mnist"))
        assert c1.compute_hash() != c2.compute_hash()

    def test_hash_same_for_same_datasets(self):
        ds = DatasetConfig(name="mnist", path="/p")
        c1 = UnifiedConfig(training_data=ds)
        c2 = UnifiedConfig(training_data=ds)
        assert c1.compute_hash() == c2.compute_hash()
