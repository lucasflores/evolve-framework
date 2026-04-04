"""
Tests for recovery strategies.

These tests verify that recovery mechanisms properly:
- Detect population stalls based on success rate
- Inject immigrants when needed
- Boost mutation temporarily
- Relax matchability constraints
- Compose multiple strategies
"""

from random import Random

import numpy as np
import pytest

from evolve.reproduction.protocol import ReproductionProtocol
from evolve.reproduction.recovery import (
    CompositeRecovery,
    ImmigrationRecovery,
    MutationBoostRecovery,
    RelaxedMatchingRecovery,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Provide seeded random number generator."""
    return Random(42)


@pytest.fixture
def genome_factory():
    """Provide a factory for creating random genomes."""

    def factory(rng: Random) -> np.ndarray:
        return rng.random() * np.ones(10)

    return factory


@pytest.fixture
def protocol_factory():
    """Provide a factory for creating random protocols."""

    def factory(rng: Random) -> ReproductionProtocol:
        return ReproductionProtocol.default()

    return factory


# =============================================================================
# ImmigrationRecovery Tests
# =============================================================================


class TestImmigrationRecovery:
    """Tests for ImmigrationRecovery strategy."""

    def test_should_not_trigger_high_success_rate(self):
        """Should not trigger when success rate is high."""
        recovery = ImmigrationRecovery(trigger_threshold=0.1)

        assert not recovery.should_trigger(
            successful_matings=9,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_should_trigger_low_success_rate(self):
        """Should trigger when success rate is below threshold."""
        recovery = ImmigrationRecovery(
            trigger_threshold=0.1,
            min_generations=0,  # Disable generation check
        )

        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_respects_min_generations(self):
        """Should not trigger before min_generations."""
        recovery = ImmigrationRecovery(
            trigger_threshold=0.1,
            min_generations=10,
        )

        # Low success but early generation
        assert not recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=5,
        )

        # Low success and past min_generations
        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=15,
        )

    def test_respects_cooldown(self, rng, genome_factory, protocol_factory):
        """Should not trigger during cooldown period."""
        recovery = ImmigrationRecovery(
            trigger_threshold=0.1,
            min_generations=0,
            cooldown_generations=5,
        )

        # First trigger at generation 10
        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

        # Do recovery to start cooldown (sets _last_trigger to generation)
        survivors = [np.ones(10) for _ in range(5)]
        recovery.recover(survivors, genome_factory, protocol_factory, rng, generation=10)

        # Should not trigger during cooldown (generation 10 + 5 = 15)
        assert not recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=12,  # Within cooldown (12 - 10 = 2 < 5)
        )

    def test_recover_adds_immigrants(self, rng, genome_factory, protocol_factory):
        """Should add immigrants to population."""
        recovery = ImmigrationRecovery(immigration_rate=0.2)

        survivors = [np.ones(10) for _ in range(10)]
        result = recovery.recover(survivors, genome_factory, protocol_factory, rng)

        # Should return tuple of (survivors, immigrants)
        assert isinstance(result, tuple)
        survivors_out, immigrants = result

        # Should have immigrants (20% of 10 = 2)
        assert len(immigrants) >= 1

    def test_zero_attempts_triggers_recovery(self):
        """With zero attempts, should trigger recovery."""
        recovery = ImmigrationRecovery(trigger_threshold=0.1, min_generations=0)

        # Zero attempts means we can't compute success rate, default to trigger
        result = recovery.should_trigger(
            successful_matings=0,
            attempted_matings=0,
            population_size=50,
            generation=10,
        )
        assert result  # Zero attempts triggers recovery


# =============================================================================
# MutationBoostRecovery Tests
# =============================================================================


class TestMutationBoostRecovery:
    """Tests for MutationBoostRecovery strategy."""

    def test_should_trigger_low_success(self):
        """Should trigger when success rate is low."""
        recovery = MutationBoostRecovery(
            boost_multiplier=3.0,
            boost_duration=5,
        )

        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_should_not_trigger_when_boosted(self):
        """Should not trigger when already boosted."""
        recovery = MutationBoostRecovery(
            boost_multiplier=3.0,
            boost_duration=5,
        )

        # First trigger
        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

        # Activate boost
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should not trigger again while boosted
        assert not recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=11,
        )

    def test_get_mutation_multiplier_active(self):
        """Should return boost when active."""
        recovery = MutationBoostRecovery(
            boost_multiplier=3.0,
            boost_duration=5,
        )

        # Activate boost
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should return boosted multiplier
        multiplier = recovery.get_mutation_multiplier()
        assert multiplier == 3.0

    def test_boost_expires_after_duration(self):
        """Boost should expire after specified duration."""
        recovery = MutationBoostRecovery(
            boost_multiplier=3.0,
            boost_duration=3,
        )

        # Activate boost
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should return boost for duration
        assert recovery.get_mutation_multiplier() == 3.0  # Call 1
        assert recovery.get_mutation_multiplier() == 3.0  # Call 2
        assert recovery.get_mutation_multiplier() == 3.0  # Call 3
        # Should expire after duration
        assert recovery.get_mutation_multiplier() == 1.0  # Call 4

    def test_recover_returns_population(self, rng, genome_factory, protocol_factory):
        """Mutation boost returns population unchanged."""
        recovery = MutationBoostRecovery()

        survivors = [np.ones(10) for _ in range(5)]
        result = recovery.recover(survivors, genome_factory, protocol_factory, rng)

        # Should return population unchanged (not a tuple)
        assert len(result) == 5


# =============================================================================
# RelaxedMatchingRecovery Tests
# =============================================================================


class TestRelaxedMatchingRecovery:
    """Tests for RelaxedMatchingRecovery strategy."""

    def test_should_trigger_low_success(self):
        """Should trigger when success rate is low."""
        recovery = RelaxedMatchingRecovery(relaxation_duration=3)

        assert recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_should_not_trigger_when_relaxed(self):
        """Should not trigger when already relaxed."""
        recovery = RelaxedMatchingRecovery(relaxation_duration=3)

        # Activate relaxation
        recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should not trigger again while relaxed
        assert not recovery.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=11,
        )

    def test_is_relaxed_when_active(self):
        """Should indicate relaxation is active."""
        recovery = RelaxedMatchingRecovery(relaxation_duration=3)

        # Activate relaxation
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should be relaxed
        assert recovery.is_relaxed()

    def test_relaxation_expires(self):
        """Relaxation should expire after duration."""
        recovery = RelaxedMatchingRecovery(relaxation_duration=3)

        # Activate relaxation
        recovery.recover([], lambda rng: None, lambda rng: None, Random(42))

        # Should be relaxed during duration
        assert recovery.is_relaxed()  # Call 1
        assert recovery.is_relaxed()  # Call 2
        assert recovery.is_relaxed()  # Call 3
        # Should expire after duration
        assert not recovery.is_relaxed()  # Call 4

    def test_recover_returns_population(self, rng, genome_factory, protocol_factory):
        """Relaxed matching returns population unchanged."""
        recovery = RelaxedMatchingRecovery()

        survivors = [np.ones(10) for _ in range(5)]
        result = recovery.recover(survivors, genome_factory, protocol_factory, rng)

        # Returns population unchanged (not a tuple)
        assert len(result) == 5


# =============================================================================
# CompositeRecovery Tests
# =============================================================================


class TestCompositeRecovery:
    """Tests for CompositeRecovery strategy."""

    def test_triggers_any_strategy(self):
        """Should trigger if any sub-strategy triggers."""
        # Immigration triggers on low success
        immigration = ImmigrationRecovery(
            trigger_threshold=0.5,  # Higher threshold
            min_generations=0,
        )
        # Mutation boost with different threshold
        mutation = MutationBoostRecovery()

        composite = CompositeRecovery(strategies=[immigration, mutation])

        # Should trigger
        assert composite.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_applies_first_triggered_strategy(self, rng, genome_factory, protocol_factory):
        """Should apply first triggered strategy."""
        immigration = ImmigrationRecovery(
            immigration_rate=0.2,
            min_generations=0,
            trigger_threshold=0.5,
        )
        mutation = MutationBoostRecovery(boost_multiplier=2.0)

        composite = CompositeRecovery(strategies=[immigration, mutation])

        survivors = [np.ones(10) for _ in range(10)]
        result = composite.recover(
            survivors,
            genome_factory,
            protocol_factory,
            rng,
            successful_matings=0,
            attempted_matings=10,
            population_size=10,
            generation=10,
        )

        # Should apply immigration (first strategy to trigger)
        # Immigration returns tuple
        assert isinstance(result, tuple)
        survivors_out, immigrants = result
        assert len(immigrants) >= 1

    def test_empty_strategies_list(self):
        """Should handle empty strategies list."""
        composite = CompositeRecovery(strategies=[])

        # Should not trigger with no strategies
        assert not composite.should_trigger(
            successful_matings=0,
            attempted_matings=10,
            population_size=50,
            generation=10,
        )

    def test_returns_unchanged_if_no_trigger(self, rng, genome_factory, protocol_factory):
        """Should return population unchanged if no strategy triggers."""
        # High threshold means won't trigger
        immigration = ImmigrationRecovery(
            trigger_threshold=0.0,  # Never triggers
            min_generations=100,
        )

        composite = CompositeRecovery(strategies=[immigration])

        survivors = [np.ones(10) for _ in range(5)]
        result = composite.recover(
            survivors,
            genome_factory,
            protocol_factory,
            rng,
            successful_matings=10,  # High success
            attempted_matings=10,
            population_size=5,
            generation=10,
        )

        # Should return unchanged population
        assert len(result) == 5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_success_rate_handled(self):
        """Should handle edge case of negative values."""
        recovery = ImmigrationRecovery(trigger_threshold=0.1, min_generations=0)

        # Negative values compute negative rate which is < threshold
        result = recovery.should_trigger(
            successful_matings=-1,  # Invalid but computes as < threshold
            attempted_matings=10,
            population_size=50,
            generation=10,
        )
        assert isinstance(result, bool)
        assert result  # Negative rate triggers recovery

    def test_large_population(self, rng, genome_factory, protocol_factory):
        """Should handle large populations."""
        recovery = ImmigrationRecovery(immigration_rate=0.01)

        # Large population
        survivors = [np.ones(10) for _ in range(1000)]
        result = recovery.recover(survivors, genome_factory, protocol_factory, rng)

        survivors_out, immigrants = result
        # Should add 1% = 10 immigrants
        assert len(immigrants) >= 5

    def test_recovery_with_empty_survivors(self, rng, genome_factory, protocol_factory):
        """Should handle empty survivor list."""
        recovery = ImmigrationRecovery(immigration_rate=0.5)

        survivors = []
        result = recovery.recover(survivors, genome_factory, protocol_factory, rng)

        # Should still work, at least 1 immigrant
        survivors_out, immigrants = result
        assert len(survivors_out) == 0
        assert len(immigrants) >= 1
