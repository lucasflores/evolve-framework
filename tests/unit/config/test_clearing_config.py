"""The clearing section of UnifiedConfig, and how the factory builds it."""

from __future__ import annotations

import json

import pytest

from evolve.config.meta import ParameterSpec
from evolve.config.unified import ClearingConfig, UnifiedConfig
from evolve.diversity.speciation import GenomeDistance
from evolve.factory.engine import (
    OperatorCompatibilityError,
    create_engine,
    create_initial_population,
)
from evolve.meta.codec import ParameterDistance
from evolve.registry.operators import reset_operator_registry
from tests.integration.test_fixed_seed_runs import (
    GOLDEN,
    TwoObjectives,
    multiobjective_config,
    run,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_operator_registry()
    yield
    reset_operator_registry()


class TestSection:
    def test_defaults_measure_only(self) -> None:
        settings = ClearingConfig()
        assert settings.copies is None
        assert settings.closeness == 0.0
        assert settings.distance is None
        assert settings.distance_params == {}

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"copies": 0}, "copies"),
            ({"closeness": -0.5}, "closeness"),
            ({"distance": ""}, "distance"),
        ],
    )
    def test_bad_settings_are_refused(self, fields: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ClearingConfig(**fields)

    def test_round_trip_through_json(self) -> None:
        config = multiobjective_config().with_clearing(
            copies=2, closeness=0.1, distance="parameters", distance_params={"x": 1}
        )
        back = UnifiedConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert back.clearing == config.clearing

    def test_an_empty_section_is_clearing_that_measures_only(self) -> None:
        data = multiobjective_config().to_dict()
        data["clearing"] = {}
        assert UnifiedConfig.from_dict(data).clearing == ClearingConfig()

    def test_absent_settings_leave_the_config_as_it_was(self) -> None:
        config = multiobjective_config()
        assert config.clearing is None
        assert "clearing" not in config.to_dict()
        assert config.compute_hash() == GOLDEN["multiobjective_hash"]

    def test_present_settings_change_the_hash(self) -> None:
        config = multiobjective_config()
        assert config.with_clearing(copies=1).compute_hash() != config.compute_hash()


class TestFactory:
    def test_builds_the_engines_clearing(self) -> None:
        config = multiobjective_config().with_clearing(copies=1, closeness=0.2)
        engine = create_engine(config, evaluator=TwoObjectives())
        assert engine._clearing.copies == 1
        assert engine._clearing.closeness == 0.2
        assert isinstance(engine._clearing.distance, GenomeDistance)

    def test_a_distance_that_names_decoder_receives_the_declared_one(self) -> None:
        spec = ParameterSpec(path="x", bounds=(0.0, 1.0))
        config = multiobjective_config().with_params(
            genome_params={"dimensions": 1, "bounds": (0.0, 1.0)},
            decoder="parameters",
            decoder_params={"params": [spec.to_dict()]},
        )
        config = config.with_clearing(copies=1, distance="parameters")
        # TwoObjectives reads raw genes, so the factory says the evaluator
        # doesn't get the decoder; the distance still does
        with pytest.warns(UserWarning, match="is ignored"):
            engine = create_engine(config, evaluator=TwoObjectives())
        assert isinstance(engine._clearing.distance, ParameterDistance)
        assert engine._clearing.distance._parameters.specs == (spec,)

    def test_single_objective_mode_is_refused(self) -> None:
        config = UnifiedConfig(
            selection="tournament", crossover="sbx", genome_params={"dimensions": 2}
        ).with_clearing(copies=1)
        with pytest.raises(ValueError, match="multi-objective"):
            create_engine(config, evaluator=lambda genes: float(sum(genes)))

    def test_a_distance_for_another_genome_type_is_refused(self) -> None:
        config = multiobjective_config().with_clearing(copies=1, distance="neat")
        with pytest.raises(OperatorCompatibilityError):
            create_engine(config, evaluator=TwoObjectives())

    def test_the_default_distance_follows_the_genome_type(self) -> None:
        from evolve.factory.engine import _distance_name

        assert _distance_name(UnifiedConfig(genome_type="graph").with_clearing()) == "neat"
        assert _distance_name(UnifiedConfig(genome_type="vector").with_clearing()) == "genome"
        named = UnifiedConfig(genome_type="graph").with_clearing(distance="custom")
        assert _distance_name(named) == "custom"

    def test_an_unknown_distance_is_refused(self) -> None:
        config = multiobjective_config().with_clearing(copies=1, distance="nowhere")
        with pytest.raises(KeyError, match="nowhere"):
            create_engine(config, evaluator=TwoObjectives())


class TestRuns:
    def test_measuring_only_reproduces_main(self) -> None:
        config = multiobjective_config().with_clearing()
        assert run("multiobjective", config) == GOLDEN["multiobjective"]

    def test_a_run_logs_its_groups_every_generation(self) -> None:
        config = multiobjective_config().with_clearing(copies=1)
        engine = create_engine(config, evaluator=TwoObjectives())
        result = engine.run(create_initial_population(config))
        assert len(result.history) == config.max_generations
        for metrics in result.history:
            assert {"clearing_groups", "clearing_largest", "clearing_held_back"} <= set(metrics)
