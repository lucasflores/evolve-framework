"""Unit tests for decode_parameters() and the "parameters" decoder."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from evolve.config.meta import ParameterSpec
from evolve.config.unified import UnifiedConfig
from evolve.factory.engine import create_engine, create_initial_population
from evolve.meta.codec import ParameterDecoder, decode_parameters, decode_value, resolve
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
FORECASTER_X = ParameterSpec(path="f", param_type="categorical", choices=("x",))
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

    def test_empty_active_values_refused(self) -> None:
        with pytest.raises(ValueError, match="active_values is empty"):
            ParameterSpec(path="t", bounds=(0.0, 1.0), parent="p", active_values=())

    @pytest.mark.parametrize("empty", [{}, ()])
    def test_empty_choices_by_parent_refused(self, empty: Any) -> None:
        with pytest.raises(ValueError, match="choices_by_parent is empty"):
            ParameterSpec(path="t", param_type="categorical", parent="p", choices_by_parent=empty)

    def test_subset_allowed_list_may_be_empty(self) -> None:
        spec = ParameterSpec(
            path="pool",
            param_type="subset",
            choices=("a",),
            parent="f",
            choices_by_parent={"x": ()},
        )
        assert decode_parameters([0.0, 1.0], [FORECASTER_X, spec]) == {"f": "x", "pool": []}

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


class TestChoicesByParentStorage:
    """choices_by_parent is stored as (parent value, options) pairs."""

    WINDOW = ParameterSpec(path="window", param_type="categorical", choices=(128, 512))
    STRIDE = ParameterSpec(
        path="stride",
        param_type="categorical",
        parent="window",
        choices_by_parent={128: (16, 32), 512: (64,)},
    )

    def test_dict_and_pairs_give_the_same_spec(self) -> None:
        pairs = ParameterSpec(
            path="stride",
            param_type="categorical",
            parent="window",
            choices_by_parent=[(128, [16, 32]), (512, [64])],  # type: ignore[arg-type]
        )
        assert pairs == self.STRIDE
        assert self.STRIDE.choices_by_parent == ((128, (16, 32)), (512, (64,)))

    def test_spec_is_hashable(self) -> None:
        assert hash(self.STRIDE) == hash(ParameterSpec.from_dict(self.STRIDE.to_dict()))

    def test_to_dict_emits_pairs(self) -> None:
        assert self.STRIDE.to_dict()["choices_by_parent"] == [[128, [16, 32]], [512, [64]]]

    def test_integer_parent_values_survive_json(self) -> None:
        specs = [self.WINDOW, self.STRIDE]
        loaded = [ParameterSpec.from_dict(json.loads(json.dumps(s.to_dict()))) for s in specs]
        assert loaded == specs
        assert decode_parameters([0.9, 0.0], loaded) == {"window": 512, "stride": 64}

    def test_from_dict_accepts_a_hand_written_mapping(self) -> None:
        data = {**self.STRIDE.to_dict(), "choices_by_parent": {128: [16, 32], 512: [64]}}
        assert ParameterSpec.from_dict(data) == self.STRIDE

    def test_duplicate_parent_values_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once: \\[128\\]"):
            ParameterSpec(
                path="stride",
                param_type="categorical",
                parent="window",
                choices_by_parent=((128, (16,)), (128, (32,))),
            )


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

    def test_relative_subset_choices_outside_parent_choices(self) -> None:
        serving = ParameterSpec(
            path="serving_pool",
            param_type="subset",
            choices=("a", "z", "b"),
            parent="training_pool",
        )
        with pytest.raises(ValueError, match="choices its parent 'training_pool' lacks: \\['z'\\]"):
            ParameterDecoder((TRAINING, serving))

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


class TestParameterDecoderValidatesOnce:
    """ParameterDecoder validates and orders its specs once, when built."""

    SPECS = (
        THRESHOLD,
        POLICY,
        ENCODER,
        READING_LENGTH,
        TRAINING,
        SERVING,
        FORECASTER,
        POOL_BY_FORECASTER,
    )

    def test_decode_matches_decode_parameters(self) -> None:
        decoder = ParameterDecoder(self.SPECS)
        rng = np.random.default_rng(0)
        for _ in range(50):
            genes = rng.uniform(0.0, 1.0, decoder.dimensions)
            assert decoder.decode(VectorGenome(genes=genes)) == decode_parameters(
                genes.tolist(), self.SPECS
            )

    def test_decode_does_not_revalidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        decoder = ParameterDecoder(self.SPECS)

        def fail(_specs: Any) -> None:
            raise AssertionError("specs validated again on decode")

        monkeypatch.setattr("evolve.meta.codec._dependency_order", fail)
        decoder.decode(VectorGenome(genes=np.full(decoder.dimensions, 0.5)))


class TestResolve:
    """resolve(): the one rule for whether a spec is active and what it chooses from."""

    def test_spec_without_parent_uses_its_own_choices(self) -> None:
        assert resolve(ENCODER, {}) == (True, None)

    def test_inactive_parent_makes_the_child_inactive(self) -> None:
        assert resolve(THRESHOLD, {}) == (False, None)

    def test_parent_value_outside_active_values(self) -> None:
        assert resolve(THRESHOLD, {"holding.policy": "hold"}) == (False, None)
        assert resolve(THRESHOLD, {"holding.policy": "switch_for_gain"}) == (True, None)

    def test_options_for_the_parent_value(self) -> None:
        assert resolve(READING_LENGTH, {"encoder.name": "bge"}) == (True, (128, 256, 512))
        assert resolve(READING_LENGTH, {"encoder.name": "x"}) == (False, None)

    def test_relative_subset_keeps_what_its_parent_holds(self) -> None:
        assert resolve(SERVING, {"training_pool": ["a", "c"]}) == (True, ["a", "c"])


class TestValuesAndOrder:
    """ParameterDecoder.values() and .order, which callers walking the specs use."""

    SPECS = TestParameterDecoderValidatesOnce.SPECS

    def test_values_are_the_decoded_values_by_path(self) -> None:
        decoder = ParameterDecoder(self.SPECS)
        rng = np.random.default_rng(1)
        for _ in range(50):
            genes = rng.uniform(0.0, 1.0, decoder.dimensions).tolist()
            values = decoder.values(genes)
            decoded = decode_parameters(genes, self.SPECS)
            for path, value in values.items():
                node: Any = decoded
                for part in path.split("."):
                    node = node[part]
                assert node == value
            assert len(values) == sum(_leaves(decoded))

    def test_order_puts_parents_first(self) -> None:
        order = [s.path for s in ParameterDecoder(self.SPECS).order]
        for spec in self.SPECS:
            if spec.parent is not None:
                assert order.index(spec.parent) < order.index(spec.path)


def _leaves(node: Any) -> list[int]:
    if isinstance(node, dict):
        return [n for child in node.values() for n in _leaves(child)]
    return [1]


EPOCHS = ParameterSpec(path="train.epochs", param_type="integer", bounds=(1, 5))
RATE = ParameterSpec(path="train.rate", bounds=(1e-4, 1e-1), log_scale=True)


class TestEncode:
    """ParameterDecoder.encode(): the genome that decodes to given values."""

    SPECS = (*TestParameterDecoderValidatesOnce.SPECS, EPOCHS, RATE)

    def test_decoding_what_was_encoded_gives_the_values_back(self) -> None:
        decoder = ParameterDecoder(self.SPECS)
        rng = np.random.default_rng(3)
        for _ in range(200):
            values = decoder.decode(VectorGenome(genes=rng.uniform(0, 1, decoder.dimensions)))
            again = decoder.decode(VectorGenome(genes=np.array(decoder.encode(values))))
            rate = again["train"].pop("rate")
            assert rate == pytest.approx(values["train"].pop("rate"), rel=1e-12)
            assert again == values

    def test_positions_sit_where_a_known_candidate_is_placed(self) -> None:
        decoder = ParameterDecoder((POLICY, THRESHOLD, EPOCHS, TRAINING, RATE))
        genome = decoder.encode(
            {
                "holding": {"policy": "hold"},
                "train": {"epochs": 2, "rate": 1e-3},
                "training_pool": ["a", "c"],
            }
        )
        # policy mid-bin; threshold inactive at 0.5; epochs at its own point;
        # subset choices at 0.75 in, 0.25 out; rate a third of the way in logs
        assert genome[:6] == [0.25, 0.5, 0.25, 0.75, 0.25, 0.75]
        assert genome[6] == pytest.approx(1 / 3)

    def test_options_follow_the_parents_value(self) -> None:
        decoder = ParameterDecoder((ENCODER, READING_LENGTH))
        assert decoder.encode({"encoder": {"name": "bge", "reading_length": 256}}) == [
            pytest.approx(1 / 6),
            0.5,
        ]

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            ({"holding": {}}, "'holding.policy' is active and has no value"),
            (
                {"holding": {"policy": "hold", "switch_threshold": 0.5}},
                "'holding.switch_threshold' is inactive",
            ),
            ({"holding": {"policy": "hold"}, "extra": 1}, r"no parameter: \['extra'\]"),
            ({"holding": {"policy": "sell"}}, "'sell' isn't one of"),
        ],
    )
    def test_values_no_genome_decodes_to_are_refused(self, values: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ParameterDecoder((POLICY, THRESHOLD)).encode(values)

    @pytest.mark.parametrize(
        ("value", "match"),
        [(0, "outside 1..5"), (6, "outside 1..5"), (2.5, "whole number"), (True, "whole number")],
    )
    def test_an_integer_must_be_one_it_can_take(self, value: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ParameterDecoder((EPOCHS,)).encode({"train": {"epochs": value}})

    def test_a_number_outside_its_bounds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            ParameterDecoder((RATE,)).encode({"train": {"rate": 0.5}})

    def test_an_option_closed_under_the_parents_value_is_refused(self) -> None:
        # "e5" offers only 512
        with pytest.raises(ValueError, match=r"128 isn't one of \[512\]"):
            ParameterDecoder((ENCODER, READING_LENGTH)).encode(
                {"encoder": {"name": "e5", "reading_length": 128}}
            )

    def test_subset_choices_must_be_its_own_and_allowed_by_its_parent(self) -> None:
        decoder = ParameterDecoder((TRAINING, SERVING))
        with pytest.raises(ValueError, match=r"\['z'\] aren't among its choices"):
            decoder.encode({"training_pool": ["a", "z"], "serving_pool": []})
        # serving may only keep what training holds
        with pytest.raises(ValueError, match=r"\['c'\] aren't allowed by its parent"):
            decoder.encode({"training_pool": ["a", "b"], "serving_pool": ["a", "c"]})
        assert decoder.encode({"training_pool": ["a", "b"], "serving_pool": ["b"]}) == [
            0.75,
            0.75,
            0.25,
            0.25,
            0.75,
            0.25,
        ]


class TestIdentity:
    """ParameterDecoder.identity(): genomes that decode alike are one candidate."""

    def test_genomes_decoding_alike_share_it(self) -> None:
        decoder = ParameterDecoder((POLICY, THRESHOLD, TRAINING))
        a = VectorGenome(genes=np.array([0.1, 0.9, 0.6, 0.2, 0.7]))
        b = VectorGenome(genes=np.array([0.4, 0.1, 0.9, 0.0, 0.5]))  # threshold inactive
        assert decoder.identity(a) == decoder.identity(b)
        assert hash(decoder.identity(a)) == hash(decoder.identity(b))

    def test_any_decoded_difference_separates_them(self) -> None:
        decoder = ParameterDecoder((POLICY, THRESHOLD, TRAINING))
        a = VectorGenome(genes=np.array([0.9, 0.3, 0.6, 0.2, 0.7]))
        b = VectorGenome(genes=np.array([0.9, 0.3000001, 0.6, 0.2, 0.7]))
        c = VectorGenome(genes=np.array([0.9, 0.3, 0.6, 0.6, 0.7]))
        assert len({decoder.identity(g) for g in (a, b, c)}) == 3


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
