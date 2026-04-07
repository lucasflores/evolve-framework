"""Unit tests for UnifiedConfig extensions: hash and serialization."""

from __future__ import annotations

import tempfile

from evolve.config.unified import UnifiedConfig


class TestComputeHashNewFields:
    """US4: Experiment hash reflects full specification."""

    def test_hash_differs_by_evaluator_name(self):
        c1 = UnifiedConfig(evaluator="benchmark")
        c2 = UnifiedConfig(evaluator="function")
        assert c1.compute_hash() != c2.compute_hash()

    def test_hash_differs_by_evaluator_params(self):
        c1 = UnifiedConfig(
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )
        c2 = UnifiedConfig(
            evaluator="benchmark",
            evaluator_params={"function_name": "rastrigin"},
        )
        assert c1.compute_hash() != c2.compute_hash()

    def test_hash_differs_by_custom_callbacks(self):
        c1 = UnifiedConfig(
            custom_callbacks=({"name": "print"},),
        )
        c2 = UnifiedConfig(
            custom_callbacks=({"name": "history"},),
        )
        assert c1.compute_hash() != c2.compute_hash()

    def test_backward_compatible_hash_when_defaults(self):
        """Legacy configs (no new fields) produce identical hashes."""
        legacy = UnifiedConfig(name="test", population_size=50)
        new = UnifiedConfig(
            name="test",
            population_size=50,
            evaluator=None,
            evaluator_params={},
            custom_callbacks=(),
        )
        assert legacy.compute_hash() == new.compute_hash()

    def test_hash_is_deterministic(self):
        c = UnifiedConfig(
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )
        assert c.compute_hash() == c.compute_hash()

    def test_hash_length_is_16(self):
        c = UnifiedConfig(evaluator="benchmark")
        assert len(c.compute_hash()) == 16


class TestSerializationRoundtrip:
    """US6: Full serialization roundtrip."""

    def _make_config_with_all_new_fields(self) -> UnifiedConfig:
        return UnifiedConfig(
            name="roundtrip_test",
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere", "minimize": True},
            custom_callbacks=(
                {"name": "print", "params": {"print_every": 5}},
                {"name": "history"},
            ),
        )

    def test_to_dict_from_dict_roundtrip(self):
        original = self._make_config_with_all_new_fields()
        d = original.to_dict()
        restored = UnifiedConfig.from_dict(d)
        assert restored.evaluator == original.evaluator
        assert restored.evaluator_params == original.evaluator_params
        assert restored.custom_callbacks == original.custom_callbacks

    def test_to_dict_includes_new_keys(self):
        c = self._make_config_with_all_new_fields()
        d = c.to_dict()
        assert "evaluator" in d
        assert "evaluator_params" in d
        assert "custom_callbacks" in d

    def test_to_json_from_json_roundtrip(self):
        original = self._make_config_with_all_new_fields()
        json_str = original.to_json()
        restored = UnifiedConfig.from_json(json_str)
        assert restored.evaluator == original.evaluator
        assert restored.evaluator_params == original.evaluator_params
        assert restored.custom_callbacks == original.custom_callbacks

    def test_to_file_from_file_roundtrip(self):
        original = self._make_config_with_all_new_fields()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        original.to_file(path)
        restored = UnifiedConfig.from_file(path)
        assert restored.evaluator == original.evaluator
        assert restored.evaluator_params == original.evaluator_params
        assert restored.custom_callbacks == original.custom_callbacks

    def test_legacy_json_missing_new_fields_loads_defaults(self):
        """Legacy JSON that doesn't have new fields loads with defaults."""
        legacy_data = {
            "name": "legacy",
            "population_size": 100,
            "max_generations": 50,
        }
        config = UnifiedConfig.from_dict(legacy_data)
        assert config.evaluator is None
        assert config.evaluator_params == {}
        assert config.custom_callbacks == ()

    def test_hash_preserved_after_roundtrip(self):
        original = self._make_config_with_all_new_fields()
        json_str = original.to_json()
        restored = UnifiedConfig.from_json(json_str)
        assert original.compute_hash() == restored.compute_hash()

    def test_custom_callbacks_list_to_tuple_conversion(self):
        """from_dict converts custom_callbacks list to tuple."""
        data = {
            "custom_callbacks": [
                {"name": "print", "params": {"print_every": 5}},
            ],
        }
        config = UnifiedConfig.from_dict(data)
        assert isinstance(config.custom_callbacks, tuple)
        assert len(config.custom_callbacks) == 1
