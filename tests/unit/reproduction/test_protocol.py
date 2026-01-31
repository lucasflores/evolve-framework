"""
Unit tests for protocol dataclasses.

Tests cover:
- MatchabilityFunction serialization/deserialization
- ReproductionIntentPolicy serialization/deserialization
- CrossoverProtocolSpec serialization/deserialization
- ReproductionProtocol serialization/deserialization and default factory
- MateContext and IntentContext immutability
- ReproductionEvent creation
"""

import numpy as np
import pytest

from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    IntentContext,
    MatchabilityFunction,
    MateContext,
    ReproductionEvent,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)
from uuid import uuid4


class TestMatchabilityFunction:
    """Tests for MatchabilityFunction dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating with minimal args."""
        func = MatchabilityFunction(type="accept_all")
        assert func.type == "accept_all"
        assert func.params == {}
        assert func.active is True

    def test_creation_full(self) -> None:
        """Test creating with all args."""
        func = MatchabilityFunction(
            type="distance_threshold",
            params={"min_distance": 0.5},
            active=True,
        )
        assert func.type == "distance_threshold"
        assert func.params == {"min_distance": 0.5}
        assert func.active is True

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        func = MatchabilityFunction(
            type="fitness_ratio",
            params={"min_ratio": 0.8, "max_ratio": 1.2},
            active=False,
        )
        d = func.to_dict()
        assert d == {
            "type": "fitness_ratio",
            "params": {"min_ratio": 0.8, "max_ratio": 1.2},
            "active": False,
        }

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "type": "probabilistic",
            "params": {"base_prob": 0.7},
            "active": True,
        }
        func = MatchabilityFunction.from_dict(d)
        assert func.type == "probabilistic"
        assert func.params == {"base_prob": 0.7}
        assert func.active is True

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        original = MatchabilityFunction(
            type="different_niche",
            params={},
            active=True,
        )
        restored = MatchabilityFunction.from_dict(original.to_dict())
        assert restored == original

    def test_immutable(self) -> None:
        """Test that MatchabilityFunction is frozen."""
        func = MatchabilityFunction(type="accept_all")
        with pytest.raises(AttributeError):
            func.type = "reject_all"  # type: ignore


class TestReproductionIntentPolicy:
    """Tests for ReproductionIntentPolicy dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating with minimal args."""
        policy = ReproductionIntentPolicy(type="always_willing")
        assert policy.type == "always_willing"
        assert policy.params == {}
        assert policy.active is True

    def test_creation_full(self) -> None:
        """Test creating with all args."""
        policy = ReproductionIntentPolicy(
            type="fitness_threshold",
            params={"threshold": 0.5},
            active=True,
        )
        assert policy.type == "fitness_threshold"
        assert policy.params["threshold"] == 0.5

    def test_to_dict(self) -> None:
        """Test serialization."""
        policy = ReproductionIntentPolicy(
            type="resource_budget",
            params={"max_offspring": 3.0},
            active=True,
        )
        d = policy.to_dict()
        assert d["type"] == "resource_budget"
        assert d["params"]["max_offspring"] == 3.0

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = ReproductionIntentPolicy(
            type="age_dependent",
            params={"min_age": 2.0, "max_age": 10.0},
            active=False,
        )
        restored = ReproductionIntentPolicy.from_dict(original.to_dict())
        assert restored == original


class TestCrossoverProtocolSpec:
    """Tests for CrossoverProtocolSpec dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating with minimal args."""
        spec = CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT)
        assert spec.type == CrossoverType.SINGLE_POINT
        assert spec.params == {}
        assert spec.active is True

    def test_creation_full(self) -> None:
        """Test creating with all args."""
        spec = CrossoverProtocolSpec(
            type=CrossoverType.UNIFORM,
            params={"swap_prob": 0.5},
            active=True,
        )
        assert spec.type == CrossoverType.UNIFORM
        assert spec.params["swap_prob"] == 0.5

    def test_to_dict_serializes_enum(self) -> None:
        """Test that enum is serialized to string."""
        spec = CrossoverProtocolSpec(
            type=CrossoverType.BLEND,
            params={"alpha": 0.5},
        )
        d = spec.to_dict()
        assert d["type"] == "blend"  # String, not enum

    def test_from_dict_deserializes_enum(self) -> None:
        """Test that string is deserialized to enum."""
        d = {"type": "two_point", "params": {"point1_ratio": 0.3, "point2_ratio": 0.7}}
        spec = CrossoverProtocolSpec.from_dict(d)
        assert spec.type == CrossoverType.TWO_POINT

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = CrossoverProtocolSpec(
            type=CrossoverType.MODULE_EXCHANGE,
            params={"module_prob": 0.2},
            active=False,
        )
        restored = CrossoverProtocolSpec.from_dict(original.to_dict())
        assert restored == original

    def test_all_crossover_types(self) -> None:
        """Test all CrossoverType enum values."""
        for ctype in CrossoverType:
            spec = CrossoverProtocolSpec(type=ctype)
            restored = CrossoverProtocolSpec.from_dict(spec.to_dict())
            assert restored.type == ctype


class TestReproductionProtocol:
    """Tests for ReproductionProtocol dataclass."""

    def test_default_factory(self) -> None:
        """Test default() creates accept-all protocol."""
        protocol = ReproductionProtocol.default()
        assert protocol.matchability.type == "accept_all"
        assert protocol.matchability.active is True
        assert protocol.intent.type == "always_willing"
        assert protocol.intent.active is True
        assert protocol.crossover.type == CrossoverType.SINGLE_POINT
        assert protocol.crossover.active is True
        assert protocol.junk_data == {}

    def test_creation_full(self) -> None:
        """Test creating with all components."""
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="distance_threshold", params={"min_distance": 0.3}),
            intent=ReproductionIntentPolicy(type="fitness_threshold", params={"threshold": 0.5}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM, params={"swap_prob": 0.5}),
            junk_data={"dormant_param": 42},
        )
        assert protocol.matchability.type == "distance_threshold"
        assert protocol.intent.type == "fitness_threshold"
        assert protocol.crossover.type == CrossoverType.UNIFORM
        assert protocol.junk_data["dormant_param"] == 42

    def test_to_dict(self) -> None:
        """Test serialization."""
        protocol = ReproductionProtocol.default()
        d = protocol.to_dict()
        assert "matchability" in d
        assert "intent" in d
        assert "crossover" in d
        assert "junk_data" in d
        assert d["matchability"]["type"] == "accept_all"

    def test_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        original = ReproductionProtocol(
            matchability=MatchabilityFunction(type="probabilistic", params={"base_prob": 0.8}),
            intent=ReproductionIntentPolicy(type="never_willing", params={}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.CLONE, params={}),
            junk_data={"hidden": "value", "number": 123},
        )
        restored = ReproductionProtocol.from_dict(original.to_dict())
        assert restored == original

    def test_immutable(self) -> None:
        """Test that ReproductionProtocol is frozen."""
        protocol = ReproductionProtocol.default()
        with pytest.raises(AttributeError):
            protocol.matchability = MatchabilityFunction(type="reject_all")  # type: ignore

    def test_junk_data_preserved(self) -> None:
        """Test that junk_data is preserved through serialization."""
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            junk_data={
                "dormant_matchability": {"type": "fitness_ratio", "params": {"min_ratio": 0.9}},
                "dormant_intent": {"type": "resource_budget", "params": {"max_offspring": 5}},
                "random_data": [1, 2, 3],
            },
        )
        restored = ReproductionProtocol.from_dict(protocol.to_dict())
        assert restored.junk_data == protocol.junk_data


class TestMateContext:
    """Tests for MateContext dataclass."""

    def test_creation(self) -> None:
        """Test creating MateContext."""
        ctx = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=3,
            partner_fitness_ratio=1.2,
            partner_niche_id=None,
            population_diversity=0.7,
            crowding_distance=0.3,
            self_fitness=np.array([0.8]),
            partner_fitness=np.array([0.96]),
        )
        assert ctx.partner_distance == 0.5
        assert ctx.partner_fitness_rank == 3
        assert ctx.crowding_distance == 0.3

    def test_numpy_arrays_immutable(self) -> None:
        """Test that numpy arrays are made immutable."""
        ctx = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=1,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        with pytest.raises((ValueError, TypeError)):
            ctx.self_fitness[0] = 999.0

    def test_frozen(self) -> None:
        """Test that MateContext is frozen."""
        ctx = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([0.5]),
            partner_fitness=np.array([0.5]),
        )
        with pytest.raises(AttributeError):
            ctx.partner_distance = 0.9  # type: ignore


class TestIntentContext:
    """Tests for IntentContext dataclass."""

    def test_creation(self) -> None:
        """Test creating IntentContext."""
        ctx = IntentContext(
            fitness=np.array([0.7, 0.8]),
            fitness_rank=2,
            age=5,
            offspring_count=1,
            generation=10,
            population_size=100,
        )
        assert ctx.fitness_rank == 2
        assert ctx.age == 5
        assert ctx.offspring_count == 1
        assert ctx.generation == 10

    def test_numpy_arrays_immutable(self) -> None:
        """Test that numpy arrays are made immutable."""
        ctx = IntentContext(
            fitness=np.array([0.5]),
            fitness_rank=0,
            age=0,
            offspring_count=0,
            generation=0,
            population_size=50,
        )
        with pytest.raises((ValueError, TypeError)):
            ctx.fitness[0] = 999.0


class TestReproductionEvent:
    """Tests for ReproductionEvent dataclass."""

    def test_creation_success(self) -> None:
        """Test creating successful reproduction event."""
        parent1 = uuid4()
        parent2 = uuid4()
        offspring1 = uuid4()
        offspring2 = uuid4()
        
        event = ReproductionEvent(
            generation=10,
            parent1_id=parent1,
            parent2_id=parent2,
            success=True,
            failure_reason=None,
            offspring_ids=(offspring1, offspring2),
            matchability_result=(True, True),
            intent_result=(True, True),
        )
        assert event.success is True
        assert event.failure_reason is None
        assert len(event.offspring_ids) == 2

    def test_creation_failure(self) -> None:
        """Test creating failed reproduction event."""
        event = ReproductionEvent(
            generation=5,
            parent1_id=uuid4(),
            parent2_id=uuid4(),
            success=False,
            failure_reason="matchability_rejected",
            offspring_ids=None,
            matchability_result=(True, False),
            intent_result=(True, True),
        )
        assert event.success is False
        assert event.failure_reason == "matchability_rejected"
        assert event.offspring_ids is None
        # Parent 1 accepted, Parent 2 rejected
        assert event.matchability_result == (True, False)
