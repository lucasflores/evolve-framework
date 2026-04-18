"""Tests for MergeConfig validation."""

from __future__ import annotations

import pytest

from evolve.config.merge import MergeConfig
from evolve.config.tracking import MetricCategory, TrackingConfig
from evolve.config.unified import UnifiedConfig


class TestMergeConfigDefaults:
    """Test default values and construction."""

    def test_default_values(self) -> None:
        cfg = MergeConfig()
        assert cfg.operator == "graph_symbiogenetic"
        assert cfg.merge_rate == 0.0
        assert cfg.symbiont_source == "cross_species"
        assert cfg.symbiont_fate == "consumed"
        assert cfg.archive_size == 50
        assert cfg.interface_count == 4
        assert cfg.interface_ratio == 0.5
        assert cfg.weight_method == "mean"
        assert cfg.weight_mean == 0.0
        assert cfg.weight_std == 1.0
        assert cfg.max_complexity is None
        assert cfg.operator_params == {}

    def test_frozen(self) -> None:
        cfg = MergeConfig()
        with pytest.raises(AttributeError):
            cfg.merge_rate = 0.5  # type: ignore[misc]


class TestMergeConfigValidation:
    """Test __post_init__ validation."""

    def test_merge_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="merge_rate"):
            MergeConfig(merge_rate=-0.1)

    def test_merge_rate_above_one(self) -> None:
        with pytest.raises(ValueError, match="merge_rate"):
            MergeConfig(merge_rate=1.1)

    def test_merge_rate_boundaries(self) -> None:
        MergeConfig(merge_rate=0.0)
        MergeConfig(merge_rate=1.0)

    def test_interface_count_zero(self) -> None:
        with pytest.raises(ValueError, match="interface_count"):
            MergeConfig(interface_count=0)

    def test_interface_count_negative(self) -> None:
        with pytest.raises(ValueError, match="interface_count"):
            MergeConfig(interface_count=-1)

    def test_interface_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="interface_ratio"):
            MergeConfig(interface_ratio=1.5)
        with pytest.raises(ValueError, match="interface_ratio"):
            MergeConfig(interface_ratio=-0.1)

    def test_archive_size_zero(self) -> None:
        with pytest.raises(ValueError, match="archive_size"):
            MergeConfig(archive_size=0)

    def test_weight_std_zero(self) -> None:
        with pytest.raises(ValueError, match="weight_std"):
            MergeConfig(weight_std=0.0)

    def test_max_complexity_zero(self) -> None:
        with pytest.raises(ValueError, match="max_complexity"):
            MergeConfig(max_complexity=0)

    def test_max_complexity_none_ok(self) -> None:
        cfg = MergeConfig(max_complexity=None)
        assert cfg.max_complexity is None

    def test_max_complexity_positive_ok(self) -> None:
        cfg = MergeConfig(max_complexity=100)
        assert cfg.max_complexity == 100


class TestMergeConfigSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip_defaults(self) -> None:
        cfg = MergeConfig()
        data = cfg.to_dict()
        restored = MergeConfig.from_dict(data)
        assert restored == cfg

    def test_round_trip_custom(self) -> None:
        cfg = MergeConfig(
            operator="graph_symbiogenetic",
            merge_rate=0.2,
            symbiont_source="archive",
            symbiont_fate="survives",
            archive_size=100,
            interface_count=8,
            interface_ratio=0.7,
            weight_method="random",
            weight_mean=0.5,
            weight_std=0.2,
            max_complexity=200,
            operator_params={"foo": "bar"},
        )
        data = cfg.to_dict()
        restored = MergeConfig.from_dict(data)
        assert restored == cfg

    def test_from_dict_missing_keys_uses_defaults(self) -> None:
        cfg = MergeConfig.from_dict({})
        assert cfg.merge_rate == 0.0
        assert cfg.operator == "graph_symbiogenetic"


class TestUnifiedConfigMergeIntegration:
    """Test MergeConfig integration with UnifiedConfig."""

    def test_merge_field_none_by_default(self) -> None:
        cfg = UnifiedConfig()
        assert cfg.merge is None
        assert not cfg.is_merge_enabled

    def test_merge_enabled(self) -> None:
        cfg = UnifiedConfig(merge=MergeConfig(merge_rate=0.1))
        assert cfg.merge is not None
        assert cfg.is_merge_enabled

    def test_merge_zero_rate_not_enabled(self) -> None:
        cfg = UnifiedConfig(merge=MergeConfig(merge_rate=0.0))
        assert cfg.merge is not None
        assert not cfg.is_merge_enabled

    def test_with_merge_convenience(self) -> None:
        base = UnifiedConfig()
        cfg = base.with_merge(merge_rate=0.15, interface_count=6)
        assert cfg.merge is not None
        assert cfg.merge.merge_rate == 0.15
        assert cfg.merge.interface_count == 6

    def test_with_merge_auto_tracking(self) -> None:
        base = UnifiedConfig(tracking=TrackingConfig())
        cfg = base.with_merge(merge_rate=0.1)
        assert cfg.tracking is not None
        assert cfg.tracking.has_category(MetricCategory.SYMBIOGENESIS)

    def test_serialization_round_trip_with_merge(self) -> None:
        cfg = UnifiedConfig(merge=MergeConfig(merge_rate=0.1, interface_count=6))
        data = cfg.to_dict()
        restored = UnifiedConfig.from_dict(data)
        assert restored.merge is not None
        assert restored.merge.merge_rate == 0.1
        assert restored.merge.interface_count == 6


class TestMetricCategorySymbiogenesis:
    """Test SYMBIOGENESIS metric category."""

    def test_enum_value(self) -> None:
        assert MetricCategory.SYMBIOGENESIS.value == "symbiogenesis"

    def test_tracking_config_with_category(self) -> None:
        tc = TrackingConfig()
        tc2 = tc.with_category(MetricCategory.SYMBIOGENESIS)
        assert tc2.has_category(MetricCategory.SYMBIOGENESIS)
