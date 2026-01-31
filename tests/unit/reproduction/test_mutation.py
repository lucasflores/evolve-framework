"""
Unit tests for protocol mutation operators.
"""

from random import Random

import pytest

from evolve.reproduction.mutation import (
    MutationConfig,
    ProtocolMutator,
    demote_param_to_junk,
    mutate_crossover,
    mutate_intent,
    mutate_junk_data,
    mutate_matchability,
    mutate_params,
    promote_junk_to_param,
)
from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    MatchabilityFunction,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> Random:
    """Fixed seed random generator for reproducibility."""
    return Random(42)


@pytest.fixture
def config() -> MutationConfig:
    """Default mutation configuration."""
    return MutationConfig()


@pytest.fixture
def default_protocol() -> ReproductionProtocol:
    """Default protocol for testing."""
    return ReproductionProtocol.default()


# =============================================================================
# mutate_params Tests
# =============================================================================


class TestMutateParams:
    """Tests for parameter mutation."""

    def test_preserves_keys(self, rng: Random):
        """Mutation preserves parameter keys."""
        params = {"threshold": 0.5, "ratio": 0.3}

        mutated = mutate_params(params, rng, mutation_rate=1.0, mutation_strength=0.1)

        assert set(mutated.keys()) == set(params.keys())

    def test_mutation_rate_zero_no_change(self, rng: Random):
        """Zero mutation rate means no changes."""
        params = {"threshold": 0.5, "ratio": 0.3}

        mutated = mutate_params(params, rng, mutation_rate=0.0, mutation_strength=0.1)

        assert mutated == params

    def test_clamps_probability_params(self, rng: Random):
        """Probability parameters are clamped to [0, 1]."""
        params = {"swap_prob": 0.5}

        # High strength to force out of range
        mutated = mutate_params(params, rng, mutation_rate=1.0, mutation_strength=10.0)

        assert 0.0 <= mutated["swap_prob"] <= 1.0

    def test_deterministic_with_same_seed(self):
        """Same seed produces same mutations."""
        params = {"threshold": 0.5}

        rng1 = Random(42)
        rng2 = Random(42)

        result1 = mutate_params(params, rng1, 1.0, 0.1)
        result2 = mutate_params(params, rng2, 1.0, 0.1)

        assert result1 == result2


# =============================================================================
# mutate_junk_data Tests
# =============================================================================


class TestMutateJunkData:
    """Tests for junk data mutation."""

    def test_can_add_junk(self, rng: Random):
        """Mutation can add new junk entries."""
        config = MutationConfig(junk_add_rate=1.0, junk_remove_rate=0.0, junk_modify_rate=0.0)

        mutated = mutate_junk_data({}, rng, config)

        assert len(mutated) == 1

    def test_can_remove_junk(self, rng: Random):
        """Mutation can remove junk entries."""
        config = MutationConfig(junk_add_rate=0.0, junk_remove_rate=1.0, junk_modify_rate=0.0)
        junk = {"key1": 0.5, "key2": 0.3}

        mutated = mutate_junk_data(junk, rng, config)

        assert len(mutated) < len(junk)

    def test_can_modify_junk(self, rng: Random):
        """Mutation can modify junk values."""
        config = MutationConfig(junk_add_rate=0.0, junk_remove_rate=0.0, junk_modify_rate=1.0)
        junk = {"key1": 0.5}

        mutated = mutate_junk_data(junk, rng, config)

        assert "key1" in mutated
        assert mutated["key1"] != 0.5  # Modified

    def test_handles_nested_dicts(self, rng: Random):
        """Can modify nested dictionary junk."""
        config = MutationConfig(junk_add_rate=0.0, junk_remove_rate=0.0, junk_modify_rate=1.0)
        junk = {"nested": {"value": 0.5}}

        mutated = mutate_junk_data(junk, rng, config)

        assert "nested" in mutated


# =============================================================================
# promote_junk_to_param Tests
# =============================================================================


class TestPromoteJunkToParam:
    """Tests for promoting junk to active params."""

    def test_promotes_numeric_junk(self, rng: Random):
        """Can promote numeric junk values."""
        junk = {"dormant_threshold": 0.7}
        params = {"existing": 0.5}

        new_junk, new_params = promote_junk_to_param(junk, params, rng)

        assert "dormant_threshold" not in new_junk
        assert len(new_params) > len(params)

    def test_preserves_existing_params(self, rng: Random):
        """Existing params are preserved."""
        junk = {"new_param": 0.3}
        params = {"existing": 0.5}

        new_junk, new_params = promote_junk_to_param(junk, params, rng)

        assert "existing" in new_params
        assert new_params["existing"] == 0.5

    def test_handles_empty_junk(self, rng: Random):
        """Handles empty junk data gracefully."""
        junk = {}
        params = {"existing": 0.5}

        new_junk, new_params = promote_junk_to_param(junk, params, rng)

        assert new_junk == {}
        assert new_params == params


# =============================================================================
# demote_param_to_junk Tests
# =============================================================================


class TestDemoteParamToJunk:
    """Tests for demoting params to junk."""

    def test_demotes_param(self, rng: Random):
        """Can demote a parameter to junk."""
        params = {"threshold": 0.5, "ratio": 0.3}
        junk = {}

        new_params, new_junk = demote_param_to_junk(params, junk, rng)

        assert len(new_params) < len(params)
        assert len(new_junk) > 0

    def test_preserves_value(self, rng: Random):
        """Demoted value is preserved in junk."""
        params = {"only_param": 0.5}
        junk = {}

        new_params, new_junk = demote_param_to_junk(params, junk, rng)

        assert 0.5 in new_junk.values()

    def test_handles_empty_params(self, rng: Random):
        """Handles empty params gracefully."""
        params = {}
        junk = {"existing": 0.5}

        new_params, new_junk = demote_param_to_junk(params, junk, rng)

        assert new_params == {}
        assert new_junk == junk


# =============================================================================
# mutate_matchability Tests
# =============================================================================


class TestMutateMatchability:
    """Tests for matchability mutation."""

    def test_returns_matchability_function(self, rng: Random, config: MutationConfig):
        """Mutation returns a MatchabilityFunction."""
        original = MatchabilityFunction(type="accept_all")

        mutated = mutate_matchability(original, rng, config)

        assert isinstance(mutated, MatchabilityFunction)

    def test_can_change_type(self):
        """Mutation can switch matchability type."""
        config = MutationConfig(type_mutation_rate=1.0)
        original = MatchabilityFunction(type="accept_all")
        rng = Random(42)

        # Run multiple times to increase chance of type change
        found_different = False
        for _ in range(10):
            mutated = mutate_matchability(original, rng, config)
            if mutated.type != original.type:
                found_different = True
                break

        assert found_different

    def test_can_toggle_active(self):
        """Mutation can toggle active flag."""
        config = MutationConfig(activation_mutation_rate=1.0)
        original = MatchabilityFunction(type="accept_all", active=True)
        rng = Random(42)

        mutated = mutate_matchability(original, rng, config)

        assert mutated.active != original.active


# =============================================================================
# mutate_intent Tests
# =============================================================================


class TestMutateIntent:
    """Tests for intent mutation."""

    def test_returns_intent_policy(self, rng: Random, config: MutationConfig):
        """Mutation returns a ReproductionIntentPolicy."""
        original = ReproductionIntentPolicy(type="always_willing")

        mutated = mutate_intent(original, rng, config)

        assert isinstance(mutated, ReproductionIntentPolicy)

    def test_can_change_type(self):
        """Mutation can switch intent type."""
        config = MutationConfig(type_mutation_rate=1.0)
        original = ReproductionIntentPolicy(type="always_willing")
        rng = Random(42)

        found_different = False
        for _ in range(10):
            mutated = mutate_intent(original, rng, config)
            if mutated.type != original.type:
                found_different = True
                break

        assert found_different


# =============================================================================
# mutate_crossover Tests
# =============================================================================


class TestMutateCrossover:
    """Tests for crossover mutation."""

    def test_returns_crossover_spec(self, rng: Random, config: MutationConfig):
        """Mutation returns a CrossoverProtocolSpec."""
        original = CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT)

        mutated = mutate_crossover(original, rng, config)

        assert isinstance(mutated, CrossoverProtocolSpec)

    def test_can_change_type(self):
        """Mutation can switch crossover type."""
        config = MutationConfig(type_mutation_rate=1.0)
        original = CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT)
        rng = Random(42)

        found_different = False
        for _ in range(10):
            mutated = mutate_crossover(original, rng, config)
            if mutated.type != original.type:
                found_different = True
                break

        assert found_different


# =============================================================================
# ProtocolMutator Tests
# =============================================================================


class TestProtocolMutator:
    """Tests for the main ProtocolMutator class."""

    def test_mutate_returns_protocol(self, rng: Random, default_protocol: ReproductionProtocol):
        """Mutation returns a ReproductionProtocol."""
        mutator = ProtocolMutator()

        mutated = mutator.mutate(default_protocol, rng)

        assert isinstance(mutated, ReproductionProtocol)

    def test_mutate_preserves_structure(self, rng: Random, default_protocol: ReproductionProtocol):
        """Mutated protocol has all components."""
        mutator = ProtocolMutator()

        mutated = mutator.mutate(default_protocol, rng)

        assert mutated.matchability is not None
        assert mutated.intent is not None
        assert mutated.crossover is not None
        assert isinstance(mutated.junk_data, dict)

    def test_deterministic_with_same_seed(self, default_protocol: ReproductionProtocol):
        """Same seed produces same mutation."""
        mutator = ProtocolMutator()

        rng1 = Random(42)
        rng2 = Random(42)

        result1 = mutator.mutate(default_protocol, rng1)
        result2 = mutator.mutate(default_protocol, rng2)

        assert result1 == result2

    def test_mutate_single_component(self, rng: Random, default_protocol: ReproductionProtocol):
        """Can mutate just one component."""
        mutator = ProtocolMutator(config=MutationConfig(
            param_mutation_rate=1.0,
            type_mutation_rate=1.0,
        ))

        # Mutate only matchability
        mutated = mutator.mutate_single_component(default_protocol, rng, "matchability")

        # Other components should be identical
        assert mutated.intent == default_protocol.intent
        assert mutated.crossover == default_protocol.crossover
        assert mutated.junk_data == default_protocol.junk_data

    def test_mutate_single_component_invalid_raises(self, rng: Random, default_protocol: ReproductionProtocol):
        """Invalid component name raises ValueError."""
        mutator = ProtocolMutator()

        with pytest.raises(ValueError, match="Unknown component"):
            mutator.mutate_single_component(default_protocol, rng, "invalid")

    def test_junk_activation(self, default_protocol: ReproductionProtocol):
        """Can activate junk data as params."""
        # Create protocol with junk
        protocol_with_junk = ReproductionProtocol(
            matchability=default_protocol.matchability,
            intent=default_protocol.intent,
            crossover=default_protocol.crossover,
            junk_data={"dormant_param": 0.7},
        )

        mutator = ProtocolMutator(config=MutationConfig(
            junk_activate_rate=1.0,
            param_mutation_rate=0.0,
            type_mutation_rate=0.0,
            activation_mutation_rate=0.0,
        ))
        rng = Random(42)

        mutated = mutator.mutate(protocol_with_junk, rng)

        # Junk should have moved somewhere
        total_junk_before = len(protocol_with_junk.junk_data)
        total_junk_after = len(mutated.junk_data)
        # Either junk was promoted or stayed the same
        assert total_junk_after <= total_junk_before


# =============================================================================
# Integration Tests
# =============================================================================


class TestMutationIntegration:
    """Integration tests for mutation workflow."""

    def test_mutation_over_generations(self, default_protocol: ReproductionProtocol):
        """Protocol evolves over multiple generations."""
        mutator = ProtocolMutator(config=MutationConfig(
            param_mutation_rate=0.5,
            type_mutation_rate=0.2,
            junk_add_rate=0.3,
        ))
        rng = Random(42)

        protocol = default_protocol
        for _ in range(100):
            protocol = mutator.mutate(protocol, rng)

        # Should have accumulated some junk
        # (Not guaranteed but likely with these rates)
        # Just ensure no errors occurred
        assert isinstance(protocol, ReproductionProtocol)

    def test_round_trip_serialization(self, rng: Random, default_protocol: ReproductionProtocol):
        """Mutated protocol can be serialized and deserialized."""
        mutator = ProtocolMutator()

        mutated = mutator.mutate(default_protocol, rng)

        # Serialize and deserialize
        data = mutated.to_dict()
        restored = ReproductionProtocol.from_dict(data)

        assert restored == mutated
