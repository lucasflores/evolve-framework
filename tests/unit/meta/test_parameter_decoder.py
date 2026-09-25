"""Unit tests for decode_parameters() and the "parameters" decoder."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from evolve.config.meta import ParameterSpec
from evolve.config.unified import UnifiedConfig
from evolve.factory.engine import create_engine, create_initial_population
from evolve.meta.codec import ParameterDecoder, decode_parameters, decode_value
from evolve.registry.decoders import get_decoder_registry, reset_decoder_registry
from evolve.representation.vector import VectorGenome

POLICY = ParameterSpec(
    path="holding.policy", param_type="categorical", choices=("hold", "switch_for_gain")
)
THRESHOLD = ParameterSpec(
    path="holding.switch_threshold",
    bounds=(0.0, 1.0),
    parent="holding.policy",
    active_values=("switch_for_gain",),
)
ENCODER = ParameterSpec(path="encoder.name", param_type="categorical", choices=("bge", "e5", "x"))
READING_LENGTH = ParameterSpec(
    path="encoder.reading_length",
    param_type="categorical",
    parent="encoder.name",
    choices_by_parent={"bge": (128, 256, 512), "e5": (512,)},
)
TRAINING = ParameterSpec(path="training_pool", param_type="subset", choices=("a", "b", "c"))
SERVING = ParameterSpec(
    path="serving_pool", param_type="subset", choices=("a", "b", "c"), parent="training_pool"
)
FORECASTER = ParameterSpec(path="forecaster", param_type="categorical", choices=("f1", "f2", "f3"))
POOL_BY_FORECASTER = ParameterSpec(
    path="pool",
    param_type="subset",
    choices=("a", "b", "c", "d"),
    parent="forecaster",
    choices_by_parent={"f1": ("a", "c"), "f2": ("d", "b")},
)


class TestSubsetSpec:
    """The subset parameter type."""

    def test_one_position_per_choice(self) -> None:
        assert TRAINING.num_dimensions == 3
        assert POLICY.num_dimensions == 1

    def test_requires_choices(self) -> None:
        with pytest.raises(ValueError, match="choices required for subset"):
            ParameterSpec(path="pool", param_type="subset")

    def test_includes_choices_at_or_above_half_in_choices_order(self) -> None:
        assert decode_parameters([0.5, 0.49, 1.0], [TRAINING]) == {"training_pool": ["a", "c"]}
        assert decode_parameters([0.0, 0.0, 0.0], [TRAINING]) == {"training_pool": []}


class TestDependentSpecValidation:
    """Per-spec checks of the dependency fields."""

    def test_active_values_require_parent(self) -> None:
        with pytest.raises(ValueError, match="require a parent"):
            ParameterSpec(path="t", bounds=(0.0, 1.0), active_values=("x",))

    def test_parent_must_be_used(self) -> None:
        with pytest.raises(ValueError, match="parent of 't' is unused"):
            ParameterSpec(path="t", bounds=(0.0, 1.0), parent="p")

    def test_is_relative_only_for_subset_without_condition_or_map(self) -> None:
        assert SERVING.is_relative
        assert not TRAINING.is_relative
        assert not POOL_BY_FORECASTER.is_relative
        assert not THRESHOLD.is_relative

    def test_choices_by_parent_not_for_continuous(self) -> None:
        with pytest.raises(ValueError, match="only for categorical and subset"):
            ParameterSpec(path="t", bounds=(0.0, 1.0), parent="p", choices_by_parent={"x": (1,)})

    def test_choices_and_choices_by_parent_exclusive(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            ParameterSpec(
                path="t",
                param_type="categorical",
                choices=(1,),
                parent="p",
                choices_by_parent={"x": (1,)},
            )

    def test_subset_allowed_choices_must_be_in_choices(self) -> None:
        with pytest.raises(ValueError, match="not in choices: \\['e'\\]"):
            ParameterSpec(
                path="pool",
                param_type="subset",
                choices=("a", "b"),
                parent="forecaster",
                choices_by_parent={"f1": ("a", "e")},
            )

    @pytest.mark.parametrize(
        "spec", [THRESHOLD, READING_LENGTH, SERVING, TRAINING, POLICY, POOL_BY_FORECASTER]
    )
    def test_dict_round_trip_through_json(self, spec: ParameterSpec) -> None:
        assert ParameterSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


class TestConditionRule:
    """A spec active only for some of its parent's values."""

    def test_active_when_parent_value_listed(self) -> None:
        assert decode_parameters([0.9, 0.25], [POLICY, THRESHOLD]) == {
            "holding": {"policy": "switch_for_gain", "switch_threshold": 0.25}
        }

    def test_inactive_spec_is_omitted(self) -> None:
        assert decode_parameters([0.1, 0.25], [POLICY, THRESHOLD]) == {
            "holding": {"policy": "hold"}
        }

    def test_inactive_positions_have_no_effect(self) -> None:
        specs = [POLICY, THRESHOLD, TRAINING, SERVING]
        a = decode_parameters([0.1, 0.25, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], specs)
        b = decode_parameters([0.1, 0.75, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], specs)
        assert a == b

    def test_child_of_inactive_parent_is_inactive(self) -> None:
        middle = ParameterSpec(
            path="holding.switch_unit",
            param_type="categorical",
            choices=("day", "week"),
            parent="holding.policy",
            active_values=("switch_for_gain",),
        )
        grandchild = ParameterSpec(
            path="holding.switch_every",
            bounds=(1.0, 7.0),
            parent="holding.switch_unit",
            active_values=("day",),
        )
        specs = [POLICY, middle, grandchild]
        assert decode_parameters([0.9, 0.0, 0.5], specs)["holding"]["switch_every"] == 4.0
        assert decode_parameters([0.1, 0.0, 0.5], specs) == {"holding": {"policy": "hold"}}


class TestDependentCategoricalRule:
    """A categorical whose options come from its parent's value."""

    @pytest.mark.parametrize("position", [0.0, 0.34, 0.5, 0.99, 1.0])
    def test_same_index_math_as_plain_categorical(self, position: float) -> None:
        plain = ParameterSpec(path="r", param_type="categorical", choices=(128, 256, 512))
        decoded = decode_parameters([0.0, position], [ENCODER, READING_LENGTH])
        assert decoded["encoder"]["reading_length"] == decode_value(plain, [position])

    def test_options_follow_parent_value(self) -> None:
        decoded = decode_parameters([0.5, 0.0], [ENCODER, READING_LENGTH])
        assert decoded == {"encoder": {"name": "e5", "reading_length": 512}}

    def test_parent_value_without_entry_is_inactive(self) -> None:
        assert decode_parameters([0.9, 0.0], [ENCODER, READING_LENGTH]) == {
            "encoder": {"name": "x"}
        }


class TestRelativeSubsetRule:
    """A subset intersected with its subset parent's value."""

    def test_intersects_parent_keeping_choices_order(self) -> None:
        serving = ParameterSpec(
            path="serving_pool",
            param_type="subset",
            choices=("c", "b", "a"),
            parent="training_pool",
        )
        decoded = decode_parameters([1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [TRAINING, serving])
        assert decoded == {"training_pool": ["a", "c"], "serving_pool": ["c", "a"]}

    def test_positions_outside_parent_have_no_effect(self) -> None:
        a = decode_parameters([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [TRAINING, SERVING])
        b = decode_parameters([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [TRAINING, SERVING])
        assert a == b == {"training_pool": ["a", "c"], "serving_pool": ["a"]}


class TestSubsetByParentRule:
    """A subset limited to the choices allowed for its categorical parent's value."""

    def test_intersects_allowed_keeping_choices_order(self) -> None:
        decoded = decode_parameters([0.5, 1.0, 1.0, 1.0, 1.0], [FORECASTER, POOL_BY_FORECASTER])
        assert decoded == {"forecaster": "f2", "pool": ["b", "d"]}

    def test_parent_value_without_entry_is_inactive(self) -> None:
        decoded = decode_parameters([0.9, 1.0, 1.0, 1.0, 1.0], [FORECASTER, POOL_BY_FORECASTER])
        assert decoded == {"forecaster": "f3"}

    def test_positions_of_disallowed_choices_have_no_effect(self) -> None:
        specs = [FORECASTER, POOL_BY_FORECASTER]
        a = decode_parameters([0.0, 1.0, 0.0, 0.0, 0.0], specs)
        b = decode_parameters([0.0, 1.0, 1.0, 0.0, 1.0], specs)
        assert a == b == {"forecaster": "f1", "pool": ["a"]}

    def test_keys_must_be_values_the_parent_can_take(self) -> None:
        typo = ParameterSpec(
            path="pool",
            param_type="subset",
            choices=("a",),
            parent="forecaster",
            choices_by_parent={"f4": ("a",)},
        )
        with pytest.raises(ValueError, match="cannot take: \\['f4'\\]"):
            ParameterDecoder((FORECASTER, typo))


class TestLayoutAndOrder:
    """Positions follow list order; decoding follows dependency order."""

    def test_child_listed_before_parent(self) -> None:
        # positions: threshold first, then policy
        assert decode_parameters([0.25, 0.9], [THRESHOLD, POLICY]) == {
            "holding": {"switch_threshold": 0.25, "policy": "switch_for_gain"}
        }

    def test_dimensions_sum_positions(self) -> None:
        decoder = ParameterDecoder((POLICY, THRESHOLD, ENCODER, READING_LENGTH, TRAINING, SERVING))
        assert decoder.dimensions == 10

    def test_wrong_vector_length_refused(self) -> None:
        with pytest.raises(ValueError, match="Expected vector of length 4"):
            decode_parameters([0.5], [POLICY, TRAINING])


class TestSpecSetRefusals:
    """Inconsistent spec sets are refused when the decoder is built."""

    def test_cycle(self) -> None:
        a = ParameterSpec(
            path="a", param_type="categorical", choices=("x",), parent="b", active_values=("x",)
        )
        b = ParameterSpec(
            path="b", param_type="categorical", choices=("x",), parent="a", active_values=("x",)
        )
        with pytest.raises(ValueError, match="cycle: .*a"):
            ParameterDecoder((a, b))

    def test_unknown_parent(self) -> None:
        with pytest.raises(ValueError, match="Unknown parent 'holding.policy'"):
            ParameterDecoder((THRESHOLD,))

    def test_condition_needs_categorical_parent(self) -> None:
        policy = ParameterSpec(path="holding.policy", bounds=(0.0, 1.0))
        with pytest.raises(ValueError, match="is continuous; it must be categorical"):
            ParameterDecoder((policy, THRESHOLD))

    def test_relative_subset_needs_subset_parent(self) -> None:
        training = ParameterSpec(path="training_pool", param_type="categorical", choices=("a",))
        with pytest.raises(ValueError, match="is categorical; it must be subset"):
            ParameterDecoder((training, SERVING))

    def test_values_parent_cannot_take(self) -> None:
        typo = ParameterSpec(
            path="holding.switch_threshold",
            bounds=(0.0, 1.0),
            parent="holding.policy",
            active_values=("switch-for-gain",),
        )
        with pytest.raises(ValueError, match="cannot take: \\['switch-for-gain'\\]"):
            ParameterDecoder((POLICY, typo))

    def test_duplicate_path(self) -> None:
        with pytest.raises(ValueError, match="Duplicate parameter path"):
            ParameterDecoder((POLICY, POLICY))

    def test_path_nested_under_another(self) -> None:
        encoder = ParameterSpec(path="encoder", param_type="categorical", choices=("bge",))
        with pytest.raises(ValueError, match="'encoder.name' is nested under parameter 'encoder'"):
            ParameterDecoder((encoder, ENCODER))


class TestRegisteredDecoder:
    """The "parameters" built-in decoder."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        reset_decoder_registry()
        yield
        reset_decoder_registry()

    def test_built_from_spec_dicts(self) -> None:
        decoder = get_decoder_registry().get(
            "parameters", params=[POLICY.to_dict(), THRESHOLD.to_dict()]
        )
        assert isinstance(decoder, ParameterDecoder)
        assert decoder.decode(VectorGenome(genes=np.array([0.9, 0.5]))) == {
            "holding": {"policy": "switch_for_gain", "switch_threshold": 0.5}
        }

    def test_genome_length_mismatch_refused(self) -> None:
        decoder = ParameterDecoder((POLICY, THRESHOLD))
        with pytest.raises(ValueError, match="Expected vector of length 2, got 3"):
            decoder.decode(VectorGenome(genes=np.zeros(3)))

    def test_create_engine_evaluates_decoded_dicts(self) -> None:
        specs = [POLICY, THRESHOLD, ENCODER, READING_LENGTH, TRAINING, SERVING]
        params = [spec.to_dict() for spec in specs]
        dimensions = ParameterDecoder(tuple(specs)).dimensions
        config = UnifiedConfig.from_dict(
            json.loads(
                json.dumps(
                    UnifiedConfig(
                        population_size=6,
                        max_generations=2,
                        selection="tournament",
                        crossover="sbx",
                        mutation="gaussian",
                        genome_type="vector",
                        genome_params={"dimensions": dimensions, "bounds": (0.0, 1.0)},
                        decoder="parameters",
                        decoder_params={"params": params},
                        seed=5,
                    ).to_dict()
                )
            )
        )
        seen: list[dict[str, Any]] = []

        def fitness(spec: dict[str, Any]) -> float:
            seen.append(spec)
            return float(len(spec["serving_pool"]))

        engine = create_engine(config, evaluator=fitness)
        engine.run(create_initial_population(config))

        assert seen
        for spec in seen:
            assert set(spec) == {"holding", "encoder", "training_pool", "serving_pool"}
            assert set(spec["serving_pool"]) <= set(spec["training_pool"])
            assert ("switch_threshold" in spec["holding"]) == (
                spec["holding"]["policy"] == "switch_for_gain"
            )
