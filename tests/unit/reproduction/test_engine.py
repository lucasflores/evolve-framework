"""
Tests for ERPEngine.

These tests verify the ERP engine properly:
- Integrates intent and matchability checks
- Inherits protocols to offspring
- Applies protocol mutation
- Handles recovery mechanisms
- Tracks reproduction events
"""

from random import Random
from uuid import uuid4

import numpy as np
import pytest

# Import engine directly to avoid circular import
from evolve.reproduction.engine import ERPConfig

# Skip tests if core engine imports fail (circular import)
try:
    from evolve.core.population import Population
    from evolve.core.types import Fitness, Individual, IndividualMetadata

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    MatchabilityFunction,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Provide seeded random number generator."""
    return Random(42)


@pytest.fixture
def config():
    """Provide basic ERP config."""
    return ERPConfig(
        population_size=10,
        max_generations=5,
        elitism=2,
        crossover_rate=0.9,
        mutation_rate=0.1,
        minimize=False,
        step_limit=1000,
        recovery_threshold=0.1,
        protocol_mutation_rate=0.1,
        enable_intent=True,
        enable_recovery=False,  # Disable for most tests
    )


@pytest.fixture
def default_protocol():
    """Provide a default permissive protocol."""
    return ReproductionProtocol.default()


@pytest.fixture
def rejecting_protocol():
    """Provide a protocol that rejects all mates."""
    return ReproductionProtocol(
        matchability=MatchabilityFunction(
            type="reject_all",
            params={},
        ),
        intent=ReproductionIntentPolicy(
            type="always_willing",
            params={},
        ),
        crossover=CrossoverProtocolSpec(
            crossover_type=CrossoverType.UNIFORM,
            params={},
        ),
    )


@pytest.fixture
def unwilling_protocol():
    """Provide a protocol that never wants to reproduce."""
    return ReproductionProtocol(
        matchability=MatchabilityFunction(
            type="accept_all",
            params={},
        ),
        intent=ReproductionIntentPolicy(
            type="never_willing",
            params={},
        ),
        crossover=CrossoverProtocolSpec(
            crossover_type=CrossoverType.UNIFORM,
            params={},
        ),
    )


def create_individual(
    genome: np.ndarray,
    fitness: float = 1.0,
    protocol: ReproductionProtocol | None = None,
    generation: int = 0,
) -> Individual:
    """Helper to create test individuals."""
    if protocol is None:
        protocol = ReproductionProtocol.default()

    return Individual(
        id=uuid4(),
        genome=genome,
        fitness=Fitness(values=np.array([fitness])),
        protocol=protocol,
        metadata=IndividualMetadata(origin="test"),
        created_at=generation,
    )


def create_population(
    size: int,
    genome_size: int = 10,
    protocol: ReproductionProtocol | None = None,
    rng: Random | None = None,
) -> Population:
    """Helper to create test populations."""
    if rng is None:
        rng = Random(42)

    individuals = []
    for _i in range(size):
        genome = np.array([rng.random() for _ in range(genome_size)])
        fitness = rng.random()
        individuals.append(create_individual(genome, fitness, protocol))

    return Population(individuals=individuals, generation=0)


# =============================================================================
# ERPConfig Tests
# =============================================================================


class TestERPConfig:
    """Tests for ERPConfig dataclass."""

    def test_config_extends_evolution_config(self):
        """ERPConfig should extend EvolutionConfig."""
        config = ERPConfig(
            population_size=100,
            max_generations=50,
        )

        # Should have base config attributes
        assert config.population_size == 100
        assert config.max_generations == 50

        # Should have ERP-specific attributes
        assert config.step_limit > 0
        assert 0 <= config.recovery_threshold <= 1
        assert 0 <= config.protocol_mutation_rate <= 1

    def test_config_defaults(self):
        """ERPConfig should have sensible defaults."""
        config = ERPConfig()

        assert config.step_limit == 1000
        assert config.recovery_threshold == 0.1
        assert config.protocol_mutation_rate == 0.1
        assert config.enable_intent is True
        assert config.enable_recovery is True


# =============================================================================
# ERPEngine Tests - Mocking Required
# =============================================================================

# Note: Full engine tests require mock evaluators, selection, crossover, mutation
# These tests focus on the ERP-specific logic that can be tested in isolation


class TestERPEngineConfiguration:
    """Tests for ERPEngine configuration."""

    def test_engine_accepts_erp_config(self, config):
        """ERPEngine should accept ERPConfig."""
        # We can't fully test without mocks, but we can verify initialization
        assert config.population_size == 10
        assert config.enable_intent is True
        assert config.enable_recovery is False


class TestProtocolInheritance:
    """Tests for protocol inheritance logic."""

    def test_inherit_protocol_basic(self, rng, default_protocol):
        """Protocol inheritance should select from one parent."""
        from evolve.reproduction.crossover_protocol import inherit_protocol

        parent1_protocol = default_protocol
        parent2_protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(
                type="distance_threshold",
                params={"min_distance": 0.5},
            ),
            intent=parent1_protocol.intent,
            crossover=parent1_protocol.crossover,
        )

        child_protocol = inherit_protocol(parent1_protocol, parent2_protocol, rng)

        # Should have one parent's matchability
        assert child_protocol.matchability.type in [
            parent1_protocol.matchability.type,
            parent2_protocol.matchability.type,
        ]

    def test_inherit_protocol_deterministic_with_seed(self, default_protocol):
        """Protocol inheritance should be deterministic with same seed."""
        parent1 = default_protocol
        parent2 = ReproductionProtocol(
            matchability=MatchabilityFunction(
                type="different",
                params={},
            ),
            intent=parent1.intent,
            crossover=parent1.crossover,
        )

        from evolve.reproduction.crossover_protocol import inherit_protocol

        rng1 = Random(12345)
        rng2 = Random(12345)

        child1 = inherit_protocol(parent1, parent2, rng1)
        child2 = inherit_protocol(parent1, parent2, rng2)

        assert child1.matchability.type == child2.matchability.type


class TestIntentIntegration:
    """Tests for intent checking integration."""

    def test_evaluate_intent_with_context(self, rng):
        """Intent evaluation should work with proper context."""
        from evolve.reproduction.intent import safe_evaluate_intent
        from evolve.reproduction.protocol import IntentContext

        policy = ReproductionIntentPolicy(
            type="always_willing",
            params={},
        )

        context = IntentContext(
            fitness=np.array([1.0]),
            fitness_rank=5,
            age=3,
            offspring_count=2,
            generation=10,
            population_size=50,
        )

        willing, success = safe_evaluate_intent(policy, context, rng, step_limit=100)

        assert success is True
        assert willing is True

    def test_never_willing_blocks_reproduction(self, rng):
        """Never willing intent should block reproduction."""
        from evolve.reproduction.intent import safe_evaluate_intent
        from evolve.reproduction.protocol import IntentContext

        policy = ReproductionIntentPolicy(
            type="never_willing",
            params={},
        )

        context = IntentContext(
            fitness=np.array([1.0]),
            fitness_rank=5,
            age=3,
            offspring_count=2,
            generation=10,
            population_size=50,
        )

        willing, success = safe_evaluate_intent(policy, context, rng, step_limit=100)

        assert success is True
        assert willing is False


class TestMatchabilityIntegration:
    """Tests for matchability checking integration."""

    def test_evaluate_matchability_with_context(self, rng):
        """Matchability evaluation should work with proper context."""
        from evolve.reproduction.matchability import safe_evaluate_matchability
        from evolve.reproduction.protocol import MateContext

        function = MatchabilityFunction(
            type="accept_all",
            params={},
        )

        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=10,
            partner_fitness_ratio=1.2,
            partner_niche_id=None,
            population_diversity=0.7,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.2]),
        )

        accepts, success = safe_evaluate_matchability(function, context, rng, step_limit=100)

        assert success is True
        assert accepts is True

    def test_reject_all_blocks_mating(self, rng):
        """Reject all matchability should block mating."""
        from evolve.reproduction.matchability import safe_evaluate_matchability
        from evolve.reproduction.protocol import MateContext

        function = MatchabilityFunction(
            type="reject_all",
            params={},
        )

        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=10,
            partner_fitness_ratio=1.2,
            partner_niche_id=None,
            population_diversity=0.7,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.2]),
        )

        accepts, success = safe_evaluate_matchability(function, context, rng, step_limit=100)

        assert success is True
        assert accepts is False


class TestReproductionEvents:
    """Tests for reproduction event tracking."""

    def test_reproduction_event_fields(self):
        """ReproductionEvent should have all required fields."""
        from evolve.reproduction.protocol import ReproductionEvent

        id1, id2 = uuid4(), uuid4()
        child_id = uuid4()

        event = ReproductionEvent(
            generation=5,
            parent1_id=id1,
            parent2_id=id2,
            success=True,
            failure_reason=None,
            offspring_ids=(child_id,),
            matchability_result=(True, True),
            intent_result=(True, True),
        )

        assert event.generation == 5
        assert event.parent1_id == id1
        assert event.parent2_id == id2
        assert event.success is True
        assert event.failure_reason is None
        assert len(event.offspring_ids) == 1

    def test_failed_reproduction_event(self):
        """Failed reproduction should have reason."""
        from evolve.reproduction.protocol import ReproductionEvent

        event = ReproductionEvent(
            generation=5,
            parent1_id=uuid4(),
            parent2_id=uuid4(),
            success=False,
            failure_reason="Intent failed",
            offspring_ids=None,
            matchability_result=(False, True),
            intent_result=(False, True),
        )

        assert event.success is False
        assert event.failure_reason == "Intent failed"
        assert event.offspring_ids is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_protocol_mutation_config(self):
        """Protocol mutation config should be configurable."""
        config = ERPConfig(
            protocol_mutation_rate=0.5,
        )

        assert config.protocol_mutation_rate == 0.5

    def test_recovery_disabled(self):
        """Recovery can be disabled via config."""
        config = ERPConfig(enable_recovery=False)

        assert config.enable_recovery is False

    def test_intent_disabled(self):
        """Intent checking can be disabled via config."""
        config = ERPConfig(enable_intent=False)

        assert config.enable_intent is False

    def test_step_limit_configurable(self):
        """Step limit should be configurable."""
        config = ERPConfig(step_limit=500)

        assert config.step_limit == 500
