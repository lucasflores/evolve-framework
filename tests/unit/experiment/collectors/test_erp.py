"""
Unit tests for ERPMetricCollector.

Tests ERP mating statistics tracking and metrics computation.
"""

import logging
from dataclasses import dataclass

import pytest

from evolve.experiment.collectors.base import CollectionContext, MatingStats
from evolve.experiment.collectors.erp import ERPMetricCollector


@dataclass
class MockFitness:
    """Mock Fitness class for testing."""

    _value: float

    def __getitem__(self, idx: int) -> float:
        return self._value


@dataclass
class MockIndividual:
    """Mock Individual class for testing."""

    id: int
    fitness: MockFitness | None = None


class MockPopulation:
    """Mock Population class for testing."""

    def __init__(self, individuals: list[MockIndividual]):
        self._individuals = individuals

    def __getitem__(self, idx: int) -> MockIndividual:
        return self._individuals[idx]

    def __len__(self) -> int:
        return len(self._individuals)


class TestERPMetricCollectorBasic:
    """Basic tests for ERPMetricCollector."""

    def test_returns_empty_when_no_mating_stats(self) -> None:
        """Returns empty dict when mating_stats is None."""
        collector = ERPMetricCollector()

        population = MockPopulation([])
        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=None,
        )

        metrics = collector.collect(context)

        assert metrics == {}

    def test_basic_mating_metrics(self) -> None:
        """Collects basic mating metrics correctly."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([MockIndividual(i) for i in range(10)])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=85,
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert metrics["attempted_matings"] == 100.0
        assert metrics["successful_matings"] == 85.0
        assert metrics["mating_success_rate"] == 0.85

    def test_zero_attempts_gives_zero_success_rate(self) -> None:
        """Zero attempts results in zero success rate."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=0,
            successful_matings=0,
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert metrics["mating_success_rate"] == 0.0

    def test_per_protocol_success_rates(self) -> None:
        """Collects per-protocol success rates."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=70,
            protocol_attempts={"symmetric": 50, "asymmetric": 50},
            protocol_successes={"symmetric": 45, "asymmetric": 25},
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert metrics["erp_protocol_symmetric_success_rate"] == 0.9  # 45/50
        assert metrics["erp_protocol_asymmetric_success_rate"] == 0.5  # 25/50
        assert metrics["erp_protocol_symmetric_attempts"] == 50.0
        assert metrics["erp_protocol_asymmetric_attempts"] == 50.0
        assert metrics["erp_protocol_symmetric_successes"] == 45.0
        assert metrics["erp_protocol_asymmetric_successes"] == 25.0


class TestERPMetricCollectorWarnings:
    """Tests for zero success rate warning behavior."""

    def test_warns_on_zero_success_rate(self, caplog: pytest.LogCaptureFixture) -> None:
        """Logs warning when success rate drops to zero."""
        collector = ERPMetricCollector(warn_on_zero_success=True)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=50,
            successful_matings=0,  # Zero successes!
        )

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        with caplog.at_level(logging.WARNING):
            collector.collect(context)

        assert "dropped to zero" in caplog.text
        assert "generation 5" in caplog.text

    def test_warns_only_once_per_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """Only logs zero success warning once per run."""
        collector = ERPMetricCollector(warn_on_zero_success=True)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=50,
            successful_matings=0,
        )

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        with caplog.at_level(logging.WARNING):
            collector.collect(context)
            collector.collect(context)  # Second call
            collector.collect(context)  # Third call

        # Should only have one warning
        assert caplog.text.count("dropped to zero") == 1

    def test_reset_clears_warning_state(self, caplog: pytest.LogCaptureFixture) -> None:
        """Reset clears the warning state."""
        collector = ERPMetricCollector(warn_on_zero_success=True)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=50,
            successful_matings=0,
        )

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        with caplog.at_level(logging.WARNING):
            collector.collect(context)
            collector.reset()  # Reset state
            collector.collect(context)  # Should warn again

        # Should have two warnings (before and after reset)
        assert caplog.text.count("dropped to zero") == 2

    def test_no_warning_when_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when warn_on_zero_success is False."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=50,
            successful_matings=0,
        )

        context = CollectionContext(
            generation=5,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        with caplog.at_level(logging.WARNING):
            collector.collect(context)

        assert "dropped to zero" not in caplog.text


class TestERPMetricCollectorProtocolNames:
    """Tests for protocol name sanitization."""

    def test_sanitizes_spaces_in_protocol_names(self) -> None:
        """Spaces in protocol names are replaced with underscores."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=50,
            protocol_attempts={"my protocol name": 100},
            protocol_successes={"my protocol name": 50},
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert "erp_protocol_my_protocol_name_success_rate" in metrics

    def test_sanitizes_special_characters(self) -> None:
        """Special characters in protocol names are handled."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=50,
            protocol_attempts={"fast-crossover.v2": 100},
            protocol_successes={"fast-crossover.v2": 50},
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert "erp_protocol_fast_crossover_v2_success_rate" in metrics

    def test_lowercases_protocol_names(self) -> None:
        """Protocol names are lowercased for consistency."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=50,
            protocol_attempts={"SymmetricMatcher": 100},
            protocol_successes={"SymmetricMatcher": 50},
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        metrics = collector.collect(context)

        assert "erp_protocol_symmetricmatcher_success_rate" in metrics


class TestERPMetricCollectorReset:
    """Tests for reset functionality."""

    def test_reset_clears_previous_success_rate(self) -> None:
        """Reset clears the previous success rate tracking."""
        collector = ERPMetricCollector(warn_on_zero_success=False)

        population = MockPopulation([])
        mating_stats = MatingStats(
            attempted_matings=100,
            successful_matings=85,
        )

        context = CollectionContext(
            generation=0,
            population=population,  # type: ignore
            mating_stats=mating_stats,
        )

        collector.collect(context)
        assert collector._previous_success_rate == 0.85

        collector.reset()
        assert collector._previous_success_rate is None


class TestMatingStatsDataclass:
    """Tests for MatingStats dataclass."""

    def test_success_rate_property(self) -> None:
        """success_rate property computes correctly."""
        stats = MatingStats(attempted_matings=100, successful_matings=75)
        assert stats.success_rate == 0.75

    def test_success_rate_with_zero_attempts(self) -> None:
        """success_rate returns 0 when no attempts."""
        stats = MatingStats(attempted_matings=0, successful_matings=0)
        assert stats.success_rate == 0.0

    def test_protocol_success_rate(self) -> None:
        """protocol_success_rate computes per-protocol rate."""
        stats = MatingStats(
            attempted_matings=100,
            successful_matings=80,
            protocol_attempts={"a": 50, "b": 50},
            protocol_successes={"a": 40, "b": 40},
        )

        assert stats.protocol_success_rate("a") == 0.8
        assert stats.protocol_success_rate("b") == 0.8

    def test_protocol_success_rate_missing_protocol(self) -> None:
        """protocol_success_rate returns 0 for unknown protocol."""
        stats = MatingStats(attempted_matings=100, successful_matings=80)

        assert stats.protocol_success_rate("unknown") == 0.0
