"""Integration tests for multi-objective workflow (T056d) and meta-evolution (T081)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evolve.config.meta import MetaEvolutionConfig, ParameterSpec
from evolve.config.multiobjective import MultiObjectiveConfig, ObjectiveSpec
from evolve.config.unified import UnifiedConfig
from evolve.factory.engine import create_engine, create_initial_population
from evolve.registry.genomes import reset_genome_registry
from evolve.registry.operators import reset_operator_registry


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset registries before each test."""
    reset_operator_registry()
    reset_genome_registry()
    yield
    reset_operator_registry()
    reset_genome_registry()


class TestMultiObjectiveWorkflow:
    """Integration tests for multi-objective optimization (T056d)."""

    def test_multiobjective_config_creates_engine(self) -> None:
        """Test that multi-objective config creates working engine."""
        config = UnifiedConfig(
            population_size=20,
            max_generations=3,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-5.0, 5.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="f1", direction="minimize"),
                    ObjectiveSpec(name="f2", direction="minimize"),
                ],
            ),
        )

        def multi_fitness(x):
            """Simple multi-objective function."""
            return [sum(xi**2 for xi in x), sum((xi - 1) ** 2 for xi in x)]

        engine = create_engine(config, multi_fitness)
        create_initial_population(config, seed=42)

        assert engine is not None
        assert hasattr(engine, "_multiobjective_config")

    def test_multiobjective_with_reference_point(self) -> None:
        """Test multi-objective engine with reference point."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=2,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="obj1", direction="minimize"),
                    ObjectiveSpec(name="obj2", direction="minimize"),
                ],
                reference_point=[5.0, 5.0],
            ),
        )

        def fitness(x):
            return [sum(x), sum(xi**2 for xi in x)]

        engine = create_engine(config, fitness)

        assert hasattr(engine, "_reference_point")
        assert engine._reference_point == [5.0, 5.0]

    def test_multiobjective_config_serialization(self) -> None:
        """Test multi-objective config survives JSON round-trip."""
        config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="obj1", direction="minimize"),
                    ObjectiveSpec(name="obj2", direction="maximize"),
                ],
            ),
        )

        # Round-trip through JSON
        json_str = config.to_json()
        restored = UnifiedConfig.from_json(json_str)

        assert restored.is_multiobjective
        assert len(restored.multiobjective.objectives) == 2
        assert restored.multiobjective.objectives[0].name == "obj1"
        assert restored.multiobjective.objectives[1].direction == "maximize"


class TestMetaEvolutionWorkflow:
    """Integration tests for meta-evolution (T081)."""

    def test_meta_config_creation(self) -> None:
        """Test meta-evolution configuration is valid."""
        param_specs = (
            ParameterSpec(
                path="mutation_rate",
                param_type="continuous",
                bounds=(0.01, 0.5),
            ),
            ParameterSpec(
                path="population_size",
                param_type="integer",
                bounds=(20, 200),
            ),
        )

        meta_config = MetaEvolutionConfig(
            evolvable_params=param_specs,
            outer_population_size=10,
            outer_generations=5,
            trials_per_config=2,
            aggregation="mean",
        )

        assert len(meta_config.evolvable_params) == 2
        assert meta_config.outer_population_size == 10
        assert meta_config.trials_per_config == 2

    def test_meta_config_in_unified_config(self) -> None:
        """Test meta-evolution config integrates with unified config."""
        param_specs = (
            ParameterSpec(
                path="mutation_rate",
                param_type="continuous",
                bounds=(0.01, 0.5),
            ),
        )

        config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            meta=MetaEvolutionConfig(
                evolvable_params=param_specs,
                outer_population_size=5,
                outer_generations=3,
            ),
        )

        assert config.is_meta_evolution
        assert config.meta is not None
        assert len(config.meta.evolvable_params) == 1

    def test_meta_config_serialization(self) -> None:
        """Test meta-evolution config survives JSON round-trip."""
        param_specs = (
            ParameterSpec(
                path="mutation_rate",
                param_type="continuous",
                bounds=(0.01, 0.5),
            ),
            ParameterSpec(
                path="selection",
                param_type="categorical",
                choices=("tournament", "roulette", "rank"),
            ),
        )

        config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            meta=MetaEvolutionConfig(
                evolvable_params=param_specs,
                outer_population_size=8,
                outer_generations=4,
                aggregation="median",
            ),
        )

        # Round-trip through JSON
        json_str = config.to_json()
        restored = UnifiedConfig.from_json(json_str)

        assert restored.is_meta_evolution
        assert restored.meta.outer_population_size == 8
        assert restored.meta.aggregation == "median"
        assert len(restored.meta.evolvable_params) == 2

    @patch("evolve.factory.engine.create_engine")
    @patch("evolve.factory.engine.create_initial_population")
    def test_meta_evaluator_integration(
        self,
        mock_create_pop,
        mock_create_engine,
    ) -> None:
        """Test MetaEvaluator can evaluate configurations."""
        from dataclasses import dataclass

        from evolve.meta.evaluator import MetaEvaluator

        @dataclass
        class MockResult:
            best_fitness: float

        # Setup mocks
        mock_engine = MagicMock()
        mock_engine.run.return_value = MockResult(best_fitness=0.25)
        mock_create_engine.return_value = mock_engine
        mock_create_pop.return_value = MagicMock()

        base_config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            mutation_rate=0.1,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            minimize=True,
        )

        meta_config = MetaEvolutionConfig(
            evolvable_params=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
            ),
            outer_population_size=5,
            outer_generations=3,
            trials_per_config=1,
        )

        evaluator = MetaEvaluator(
            base_config=base_config,
            meta_config=meta_config,
            fitness_fn=lambda x: sum(x),
            seed=42,
        )

        # Evaluate
        fitness = evaluator.evaluate(base_config)

        assert fitness == 0.25
        mock_engine.run.assert_called_once()


class TestConstrainedMultiObjective:
    """Tests for constrained multi-objective optimization (FR-034→FR-037)."""

    def test_constraint_spec_serialization(self) -> None:
        """Test constraint specs are serialized in config."""
        from evolve.config.multiobjective import ConstraintSpec

        config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-5.0, 5.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="f1", direction="minimize"),
                    ObjectiveSpec(name="f2", direction="minimize"),
                ],
                constraints=[
                    ConstraintSpec(name="g1"),
                    ConstraintSpec(name="g2"),
                ],
            ),
        )

        # Round-trip
        restored = UnifiedConfig.from_json(config.to_json())

        assert restored.multiobjective.has_constraints
        assert len(restored.multiobjective.constraints) == 2

    def test_constraint_handling_stored_on_engine(self) -> None:
        """Test constraint handling settings stored on engine."""
        from evolve.config.multiobjective import ConstraintSpec

        config = UnifiedConfig(
            population_size=10,
            max_generations=2,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 2, "bounds": (-1.0, 1.0)},
            multiobjective=MultiObjectiveConfig(
                objectives=[
                    ObjectiveSpec(name="f1", direction="minimize"),
                    ObjectiveSpec(name="f2", direction="minimize"),  # Need 2+ objectives
                ],
                constraints=[
                    ConstraintSpec(name="g1"),
                ],
                constraint_handling="penalty",
            ),
        )

        def fitness(x):
            return [sum(x), sum(xi**2 for xi in x)]

        engine = create_engine(config, fitness)

        assert hasattr(engine, "_constraint_specs")
        assert hasattr(engine, "_constraint_handling")
        assert engine._constraint_handling == "penalty"
