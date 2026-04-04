"""
Unit tests for FitnessMetadataCollector.

Tests:
- Numeric field extraction
- Aggregation (best, mean, std)
- Metadata prefix configuration
- Threshold-based field inclusion
- Non-numeric field skipping
- Nested metadata flattening
- Reset functionality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from evolve.core.types import Fitness
from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.metadata import FitnessMetadataCollector


@dataclass
class MockIndividual:
    """Mock individual for testing."""

    id: Any = None
    fitness: Any = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()


@dataclass
class MockPopulation:
    """Mock population for testing."""

    individuals: list[MockIndividual]
    _minimize: bool = False

    def __len__(self) -> int:
        return len(self.individuals)

    def best(self, n: int = 1, minimize: bool = False) -> list[MockIndividual]:
        """Return best individuals by fitness."""
        sorted_inds = sorted(
            [i for i in self.individuals if i.fitness is not None],
            key=lambda x: float(x.fitness.values[0]),
            reverse=not minimize,
        )
        return sorted_inds[:n]


def make_individual(
    fitness_value: float,
    metadata: dict[str, Any] | None = None,
) -> MockIndividual:
    """Create individual with fitness and metadata."""
    meta = metadata or {}
    fitness = Fitness(values=np.array([fitness_value]), metadata=meta)
    return MockIndividual(fitness=fitness)


def make_context(
    individuals: list[MockIndividual] | None = None,
    generation: int = 1,
) -> CollectionContext:
    """Create test collection context."""
    if individuals is None:
        individuals = []

    population = MockPopulation(individuals=individuals)

    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore
    )


class TestNumericExtraction:
    """Tests for numeric field extraction."""

    def test_extracts_int_fields(self):
        """Test extraction of integer metadata fields."""
        individuals = [
            make_individual(1.0, {"count": 10}),
            make_individual(2.0, {"count": 20}),
            make_individual(3.0, {"count": 30}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_count_mean" in metrics
        assert np.isclose(metrics["meta_count_mean"], 20.0)

    def test_extracts_float_fields(self):
        """Test extraction of float metadata fields."""
        individuals = [
            make_individual(1.0, {"latency": 0.1}),
            make_individual(2.0, {"latency": 0.2}),
            make_individual(3.0, {"latency": 0.3}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_latency_mean" in metrics
        assert np.isclose(metrics["meta_latency_mean"], 0.2)

    def test_extracts_numpy_scalar(self):
        """Test extraction of numpy scalar values."""
        individuals = [
            make_individual(1.0, {"score": np.float64(1.5)}),
            make_individual(2.0, {"score": np.float64(2.5)}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_score_mean" in metrics
        assert np.isclose(metrics["meta_score_mean"], 2.0)

    def test_extracts_single_element_array(self):
        """Test extraction of single-element numpy arrays."""
        individuals = [
            make_individual(1.0, {"value": np.array([10.0])}),
            make_individual(2.0, {"value": np.array([20.0])}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_value_mean" in metrics
        assert np.isclose(metrics["meta_value_mean"], 15.0)


class TestAggregation:
    """Tests for aggregation functions."""

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {"x": 20}),
            make_individual(3.0, {"x": 30}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(aggregations=("mean",))
        metrics = collector.collect(context)

        assert "meta_x_mean" in metrics
        assert np.isclose(metrics["meta_x_mean"], 20.0)

    def test_std_aggregation(self):
        """Test std aggregation."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {"x": 20}),
            make_individual(3.0, {"x": 30}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(aggregations=("std",))
        metrics = collector.collect(context)

        assert "meta_x_std" in metrics
        # std of [10, 20, 30] = sqrt((100 + 0 + 100) / 3) ≈ 8.16
        assert metrics["meta_x_std"] > 0

    def test_min_max_aggregation(self):
        """Test min and max aggregation."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {"x": 50}),
            make_individual(3.0, {"x": 30}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(aggregations=("min", "max"))
        metrics = collector.collect(context)

        assert metrics["meta_x_min"] == 10.0
        assert metrics["meta_x_max"] == 50.0

    def test_best_aggregation(self):
        """Test best individual aggregation."""
        individuals = [
            make_individual(1.0, {"x": 100}),  # Worst fitness
            make_individual(3.0, {"x": 300}),  # Best fitness
            make_individual(2.0, {"x": 200}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(aggregations=("best",))
        metrics = collector.collect(context)

        assert "meta_x_best" in metrics
        # Best individual has fitness 3.0 and x=300
        assert metrics["meta_x_best"] == 300.0

    def test_multiple_aggregations(self):
        """Test multiple aggregation functions."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {"x": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(aggregations=("mean", "std", "min", "max"))
        metrics = collector.collect(context)

        assert "meta_x_mean" in metrics
        assert "meta_x_std" in metrics
        assert "meta_x_min" in metrics
        assert "meta_x_max" in metrics


class TestMetadataPrefix:
    """Tests for metadata prefix configuration."""

    def test_default_prefix(self):
        """Test default 'meta_' prefix."""
        individuals = [make_individual(1.0, {"x": 10})]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert any(k.startswith("meta_") for k in metrics)

    def test_custom_prefix(self):
        """Test custom prefix configuration."""
        individuals = [make_individual(1.0, {"x": 10})]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(prefix="domain_")
        metrics = collector.collect(context)

        assert any(k.startswith("domain_") for k in metrics)
        assert not any(k.startswith("meta_") for k in metrics)

    def test_empty_prefix(self):
        """Test empty prefix (no prefix)."""
        individuals = [make_individual(1.0, {"x": 10})]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(prefix="")
        metrics = collector.collect(context)

        assert "x_mean" in metrics


class TestThreshold:
    """Tests for coverage threshold."""

    def test_field_above_threshold_included(self):
        """Test field with sufficient coverage is included."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {"x": 20}),
            make_individual(3.0, {}),  # Missing x
        ]
        context = make_context(individuals)

        # 2/3 = 67% coverage, above 50% threshold
        collector = FitnessMetadataCollector(threshold=0.5)
        metrics = collector.collect(context)

        assert "meta_x_mean" in metrics

    def test_field_below_threshold_excluded(self):
        """Test field with insufficient coverage is excluded."""
        individuals = [
            make_individual(1.0, {"x": 10}),
            make_individual(2.0, {}),
            make_individual(3.0, {}),
            make_individual(4.0, {}),
        ]
        context = make_context(individuals)

        # 1/4 = 25% coverage, below 50% threshold
        collector = FitnessMetadataCollector(threshold=0.5)
        metrics = collector.collect(context)

        assert "meta_x_mean" not in metrics

    def test_threshold_zero_includes_all(self):
        """Test zero threshold includes all fields."""
        individuals = [
            make_individual(1.0, {"rare_field": 10}),
            make_individual(2.0, {}),
            make_individual(3.0, {}),
            make_individual(4.0, {}),
            make_individual(5.0, {}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(threshold=0.0)
        metrics = collector.collect(context)

        assert "meta_rare_field_mean" in metrics

    def test_threshold_one_requires_all(self):
        """Test threshold 1.0 requires all individuals to have field."""
        individuals = [
            make_individual(1.0, {"common": 10}),
            make_individual(2.0, {"common": 20}),
            make_individual(3.0, {}),  # Missing
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(threshold=1.0)
        metrics = collector.collect(context)

        # 2/3 < 100%, so excluded
        assert "meta_common_mean" not in metrics


class TestNonNumericSkipping:
    """Tests for non-numeric field handling."""

    def test_skips_string_fields(self):
        """Test non-numeric string fields are skipped."""
        individuals = [
            make_individual(1.0, {"name": "individual_1", "score": 10}),
            make_individual(2.0, {"name": "individual_2", "score": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_score_mean" in metrics
        assert "meta_name_mean" not in metrics

    def test_skips_list_fields(self):
        """Test multi-element list fields are skipped."""
        individuals = [
            make_individual(1.0, {"values": [1, 2, 3], "score": 10}),
            make_individual(2.0, {"values": [4, 5, 6], "score": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_score_mean" in metrics
        assert "meta_values_mean" not in metrics

    def test_skips_dict_fields_at_max_depth(self):
        """Test nested dicts beyond max_depth are skipped."""
        deep_nested = {"level1": {"level2": {"level3": {"level4": 10}}}}
        individuals = [make_individual(1.0, deep_nested)]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(max_depth=2)
        metrics = collector.collect(context)

        # Should not reach level4
        assert not any("level4" in k for k in metrics)

    def test_warning_logged_once(self, caplog):
        """Test debug warning logged only once per non-numeric field."""
        individuals = [
            make_individual(1.0, {"name": "a"}),
            make_individual(2.0, {"name": "b"}),
            make_individual(3.0, {"name": "c"}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()

        with caplog.at_level(logging.DEBUG):
            collector.collect(context)

        # Should only log once for "name"
        assert caplog.text.count("Skipping non-numeric") <= 1


class TestNestedMetadata:
    """Tests for nested metadata flattening."""

    def test_flattens_nested_dict(self):
        """Test nested dicts are flattened with dot notation."""
        individuals = [
            make_individual(1.0, {"timing": {"eval": 0.1, "select": 0.2}}),
            make_individual(2.0, {"timing": {"eval": 0.2, "select": 0.3}}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert "meta_timing.eval_mean" in metrics
        assert "meta_timing.select_mean" in metrics

    def test_deep_nesting(self):
        """Test multi-level nesting."""
        individuals = [
            make_individual(1.0, {"a": {"b": {"c": 10}}}),
            make_individual(2.0, {"a": {"b": {"c": 20}}}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(max_depth=3)
        metrics = collector.collect(context)

        assert "meta_a.b.c_mean" in metrics
        assert np.isclose(metrics["meta_a.b.c_mean"], 15.0)

    def test_flatten_disabled(self):
        """Test flattening can be disabled."""
        individuals = [
            make_individual(1.0, {"nested": {"value": 10}}),
            make_individual(2.0, {"nested": {"value": 20}}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(flatten_nested=False)
        metrics = collector.collect(context)

        # Nested dicts should be skipped entirely
        assert not any("nested" in k for k in metrics)


class TestSkipFields:
    """Tests for field skipping configuration."""

    def test_skips_configured_fields(self):
        """Test explicitly configured fields are skipped."""
        individuals = [
            make_individual(1.0, {"internal": 100, "public": 10}),
            make_individual(2.0, {"internal": 200, "public": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(skip_fields=frozenset({"internal"}))
        metrics = collector.collect(context)

        assert "meta_public_mean" in metrics
        assert "meta_internal_mean" not in metrics


class TestReset:
    """Tests for collector reset functionality."""

    def test_reset_clears_warning_state(self, caplog):
        """Test reset clears non-numeric warning state."""
        individuals = [make_individual(1.0, {"name": "test"})]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()

        with caplog.at_level(logging.DEBUG):
            collector.collect(context)

        assert "name" in collector._warned_non_numeric

        collector.reset()

        assert "name" not in collector._warned_non_numeric


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_population(self):
        """Test empty population returns no metrics."""
        context = make_context([])

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert metrics == {}

    def test_no_metadata(self):
        """Test population with no metadata returns no metrics."""
        individuals = [
            make_individual(1.0, {}),
            make_individual(2.0, {}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        assert metrics == {}

    def test_none_fitness(self):
        """Test individuals with None fitness are handled."""
        individuals = [
            MockIndividual(fitness=None),
            make_individual(2.0, {"x": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector()
        metrics = collector.collect(context)

        # Should still work for valid individuals
        # 1/2 = 50%, meets threshold
        assert "meta_x_mean" in metrics

    def test_nan_values_excluded(self):
        """Test NaN values are excluded."""
        individuals = [
            make_individual(1.0, {"x": float("nan")}),
            make_individual(2.0, {"x": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(threshold=0.0)
        metrics = collector.collect(context)

        # Only the valid value should be included
        assert "meta_x_mean" in metrics
        assert metrics["meta_x_mean"] == 20.0

    def test_inf_values_excluded(self):
        """Test infinite values are excluded."""
        individuals = [
            make_individual(1.0, {"x": float("inf")}),
            make_individual(2.0, {"x": 20}),
        ]
        context = make_context(individuals)

        collector = FitnessMetadataCollector(threshold=0.0)
        metrics = collector.collect(context)

        assert "meta_x_mean" in metrics
        assert metrics["meta_x_mean"] == 20.0
