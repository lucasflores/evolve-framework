"""
Unit tests for TrackingConfig.

Tests configuration validation, serialization, factory methods,
and category manipulation.
"""

from __future__ import annotations

import json

import pytest

from evolve.config.tracking import MetricCategory, TrackingConfig


class TestMetricCategory:
    """Tests for MetricCategory enum."""

    def test_all_categories_have_string_values(self) -> None:
        """All categories should have string values."""
        for category in MetricCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0

    def test_category_values_are_unique(self) -> None:
        """Category values should be unique."""
        values = [c.value for c in MetricCategory]
        assert len(values) == len(set(values))

    def test_expected_categories_exist(self) -> None:
        """Expected categories should exist."""
        expected = {
            "core",
            "extended_population",
            "diversity",
            "timing",
            "speciation",
            "multiobjective",
            "erp",
            "metadata",
            "derived",
        }
        actual = {c.value for c in MetricCategory}
        assert expected == actual


class TestTrackingConfigValidation:
    """Tests for TrackingConfig validation."""

    def test_default_config_is_valid(self) -> None:
        """Default configuration should be valid."""
        config = TrackingConfig()
        assert config.enabled is True
        assert config.backend == "mlflow"
        assert config.log_interval == 1

    def test_invalid_log_interval_raises(self) -> None:
        """log_interval < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="log_interval"):
            TrackingConfig(log_interval=0)

        with pytest.raises(ValueError, match="log_interval"):
            TrackingConfig(log_interval=-1)

    def test_invalid_metadata_threshold_raises(self) -> None:
        """metadata_threshold outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="metadata_threshold"):
            TrackingConfig(metadata_threshold=-0.1)

        with pytest.raises(ValueError, match="metadata_threshold"):
            TrackingConfig(metadata_threshold=1.1)

    def test_valid_metadata_threshold_boundaries(self) -> None:
        """metadata_threshold at boundaries should be valid."""
        config_zero = TrackingConfig(metadata_threshold=0.0)
        assert config_zero.metadata_threshold == 0.0

        config_one = TrackingConfig(metadata_threshold=1.0)
        assert config_one.metadata_threshold == 1.0

    def test_invalid_diversity_sample_size_raises(self) -> None:
        """diversity_sample_size < 10 should raise ValueError."""
        with pytest.raises(ValueError, match="diversity_sample_size"):
            TrackingConfig(diversity_sample_size=9)

    def test_invalid_buffer_size_raises(self) -> None:
        """buffer_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="buffer_size"):
            TrackingConfig(buffer_size=0)

    def test_invalid_flush_interval_raises(self) -> None:
        """flush_interval <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="flush_interval"):
            TrackingConfig(flush_interval=0)

        with pytest.raises(ValueError, match="flush_interval"):
            TrackingConfig(flush_interval=-1.0)

    def test_invalid_velocity_window_raises(self) -> None:
        """velocity_window < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="velocity_window"):
            TrackingConfig(velocity_window=0)


class TestTrackingConfigCategories:
    """Tests for category manipulation."""

    def test_default_has_core_only(self) -> None:
        """Default config should have CORE category only."""
        config = TrackingConfig()
        assert config.categories == frozenset({MetricCategory.CORE})

    def test_has_category(self) -> None:
        """has_category should return correct values."""
        config = TrackingConfig()
        assert config.has_category(MetricCategory.CORE) is True
        assert config.has_category(MetricCategory.TIMING) is False

    def test_with_category_adds_category(self) -> None:
        """with_category should add new category."""
        config = TrackingConfig()
        new_config = config.with_category(MetricCategory.TIMING)

        assert new_config.has_category(MetricCategory.CORE)
        assert new_config.has_category(MetricCategory.TIMING)
        # Original unchanged
        assert not config.has_category(MetricCategory.TIMING)

    def test_with_category_multiple(self) -> None:
        """with_category should handle multiple categories."""
        config = TrackingConfig()
        new_config = config.with_category(
            MetricCategory.TIMING,
            MetricCategory.DIVERSITY,
        )

        assert new_config.has_category(MetricCategory.CORE)
        assert new_config.has_category(MetricCategory.TIMING)
        assert new_config.has_category(MetricCategory.DIVERSITY)

    def test_without_category_removes_category(self) -> None:
        """without_category should remove category."""
        config = TrackingConfig(categories=frozenset({MetricCategory.CORE, MetricCategory.TIMING}))
        new_config = config.without_category(MetricCategory.TIMING)

        assert new_config.has_category(MetricCategory.CORE)
        assert not new_config.has_category(MetricCategory.TIMING)

    def test_frozen_config_is_hashable(self) -> None:
        """Frozen config should be hashable."""
        config1 = TrackingConfig()
        config2 = TrackingConfig()

        # Should not raise
        hash(config1)
        hash(config2)

        # Same config should have same hash
        assert hash(config1) == hash(config2)


class TestTrackingConfigSerialization:
    """Tests for JSON serialization."""

    def test_to_dict_basic(self) -> None:
        """to_dict should return all fields."""
        config = TrackingConfig(experiment_name="test_exp")
        data = config.to_dict()

        assert data["enabled"] is True
        assert data["backend"] == "mlflow"
        assert data["experiment_name"] == "test_exp"
        assert data["log_interval"] == 1

    def test_to_dict_categories_are_sorted_strings(self) -> None:
        """Categories should be serialized as sorted list of strings."""
        config = TrackingConfig(categories=frozenset({MetricCategory.TIMING, MetricCategory.CORE}))
        data = config.to_dict()

        # Should be sorted alphabetically
        assert data["categories"] == ["core", "timing"]

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict output should be JSON serializable."""
        config = TrackingConfig.comprehensive("test")
        data = config.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_from_dict_roundtrip(self) -> None:
        """from_dict(to_dict()) should recreate config."""
        original = TrackingConfig(
            experiment_name="test_exp",
            run_name="run_1",
            tracking_uri="http://localhost:5000",
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.TIMING,
                    MetricCategory.DIVERSITY,
                }
            ),
            log_interval=5,
            buffer_size=500,
            timing_breakdown=True,
            velocity_window=10,
        )

        data = original.to_dict()
        restored = TrackingConfig.from_dict(data)

        assert restored.experiment_name == original.experiment_name
        assert restored.run_name == original.run_name
        assert restored.tracking_uri == original.tracking_uri
        assert restored.categories == original.categories
        assert restored.log_interval == original.log_interval
        assert restored.buffer_size == original.buffer_size
        assert restored.timing_breakdown == original.timing_breakdown
        assert restored.velocity_window == original.velocity_window

    def test_from_dict_with_hypervolume_reference(self) -> None:
        """from_dict should handle hypervolume_reference."""
        config = TrackingConfig(hypervolume_reference=(1.0, 2.0, 3.0))
        data = config.to_dict()

        # Should be list in JSON
        assert data["hypervolume_reference"] == [1.0, 2.0, 3.0]

        restored = TrackingConfig.from_dict(data)
        assert restored.hypervolume_reference == (1.0, 2.0, 3.0)


class TestTrackingConfigFactoryMethods:
    """Tests for factory methods."""

    def test_minimal_has_core_only(self) -> None:
        """minimal() should have CORE category only."""
        config = TrackingConfig.minimal()
        assert config.categories == frozenset({MetricCategory.CORE})
        assert config.enabled is True

    def test_standard_has_expected_categories(self) -> None:
        """standard() should have core, extended, and timing."""
        config = TrackingConfig.standard("test_exp")

        assert config.experiment_name == "test_exp"
        assert config.has_category(MetricCategory.CORE)
        assert config.has_category(MetricCategory.EXTENDED_POPULATION)
        assert config.has_category(MetricCategory.TIMING)
        assert not config.has_category(MetricCategory.ERP)

    def test_comprehensive_has_all_categories(self) -> None:
        """comprehensive() should have all categories."""
        config = TrackingConfig.comprehensive("test_exp")

        assert config.experiment_name == "test_exp"
        for category in MetricCategory:
            assert config.has_category(category), f"Missing {category}"


class TestTrackingConfigDefaults:
    """Tests for default values."""

    def test_default_values(self) -> None:
        """Default values should match specification."""
        config = TrackingConfig()

        assert config.enabled is True
        assert config.backend == "mlflow"
        assert config.experiment_name == "evolve"
        assert config.run_name is None
        assert config.tracking_uri is None
        assert config.log_interval == 1
        assert config.buffer_size == 1000
        assert config.flush_interval == 30.0
        assert config.metadata_threshold == 0.5
        assert config.metadata_prefix == "meta_"
        assert config.timing_breakdown is False
        assert config.diversity_sample_size == 1000
        assert config.hypervolume_reference is None
        assert config.velocity_window == 5
