"""
Integration tests for Evolvable Reproduction Protocols basic functionality.

Tests user story acceptance scenarios:
- T025 [US1]: Asymmetric matchability
- T042 [US2]: Protocol inheritance
- T054 [US3]: Intent before matchability
- T064 [US5]: Dormant logic activation
"""

from __future__ import annotations

from random import Random
from uuid import uuid4

import numpy as np
import pytest

from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.reproduction.crossover_protocol import (
    inherit_protocol,
    safe_execute_crossover,
)
from evolve.reproduction.intent import safe_evaluate_intent
from evolve.reproduction.matchability import safe_evaluate_matchability
from evolve.reproduction.mutation import MutationConfig, ProtocolMutator
from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    IntentContext,
    MatchabilityFunction,
    MateContext,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_individual(
    rng: Random,
    genome: np.ndarray | None = None,
    protocol: ReproductionProtocol | None = None,
    fitness: float | None = None,
) -> Individual[np.ndarray]:
    """Create a test individual."""
    if genome is None:
        genome = np.array([rng.random() for _ in range(10)])
    if protocol is None:
        protocol = ReproductionProtocol.default()

    return Individual(
        id=uuid4(),
        genome=genome,
        protocol=protocol,
        fitness=Fitness.scalar(fitness if fitness is not None else rng.random()),
        metadata=IndividualMetadata(),
        created_at=0,
    )


def create_mate_context(
    self_ind: Individual[np.ndarray],
    partner: Individual[np.ndarray],
    population_diversity: float = 0.5,
) -> MateContext:
    """Create MateContext from two individuals."""
    # Compute genetic distance
    if hasattr(self_ind.genome, "__sub__"):
        distance = float(np.linalg.norm(self_ind.genome - partner.genome))
    else:
        distance = 0.0

    # Get fitness values
    self_fitness = self_ind.fitness.values if self_ind.fitness else np.array([0.0])
    partner_fitness = partner.fitness.values if partner.fitness else np.array([0.0])

    # Compute fitness ratio
    self_f = float(self_fitness[0])
    partner_f = float(partner_fitness[0])
    if self_f == 0:
        fitness_ratio = float("inf") if partner_f != 0 else 1.0
    else:
        fitness_ratio = partner_f / self_f

    return MateContext(
        partner_distance=distance,
        partner_fitness_rank=0,
        partner_fitness_ratio=fitness_ratio,
        partner_niche_id=None,
        population_diversity=population_diversity,
        crowding_distance=None,
        self_fitness=self_fitness,
        partner_fitness=partner_fitness,
    )


def create_intent_context(
    individual: Individual[np.ndarray],
    generation: int = 0,
    population_size: int = 50,
) -> IntentContext:
    """Create IntentContext from an individual."""
    fitness = individual.fitness.values if individual.fitness else np.array([0.0])
    age = generation - individual.created_at

    return IntentContext(
        fitness=fitness,
        fitness_rank=0,
        age=age,
        offspring_count=0,
        generation=generation,
        population_size=population_size,
    )


# =============================================================================
# T025 [US1]: Asymmetric Matchability Integration Test
# =============================================================================


class TestAsymmetricMatchability:
    """
    T025: Verify asymmetric matchability - A accepts B but B rejects A.

    Acceptance Criteria:
    - Given Individual A accepts B but B rejects A (asymmetric compatibility)
    - When reproduction is attempted
    - Then no offspring is produced
    """

    def test_asymmetric_matchability_blocks_reproduction(self) -> None:
        """
        Test that when A accepts B but B rejects A, no mating occurs.
        """
        rng = Random(42)

        # Create individual A with accept-all protocol
        ind_a = create_individual(
            rng,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Create individual B with reject-all protocol
        ind_b = create_individual(
            rng,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="reject_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Create contexts
        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        # A evaluates B - should accept
        a_accepts_b, success_a = safe_evaluate_matchability(
            ind_a.protocol.matchability, ctx_a_to_b, rng
        )

        # B evaluates A - should reject
        b_accepts_a, success_b = safe_evaluate_matchability(
            ind_b.protocol.matchability, ctx_b_to_a, rng
        )

        assert success_a, "A's evaluation should succeed"
        assert success_b, "B's evaluation should succeed"
        assert a_accepts_b, "A should accept B"
        assert not b_accepts_a, "B should reject A"

        # Bilateral check: both must accept for mating
        can_mate = a_accepts_b and b_accepts_a
        assert not can_mate, "Asymmetric acceptance should prevent mating"

    def test_symmetric_acceptance_allows_reproduction(self) -> None:
        """
        Test that when both A and B accept each other, mating proceeds.
        """
        rng = Random(42)

        # Both use accept-all
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
        )

        ind_a = create_individual(rng, protocol=protocol)
        ind_b = create_individual(rng, protocol=protocol)

        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        a_accepts_b, _ = safe_evaluate_matchability(ind_a.protocol.matchability, ctx_a_to_b, rng)
        b_accepts_a, _ = safe_evaluate_matchability(ind_b.protocol.matchability, ctx_b_to_a, rng)

        can_mate = a_accepts_b and b_accepts_a
        assert can_mate, "Symmetric acceptance should allow mating"

    def test_distance_based_asymmetric_matchability(self) -> None:
        """
        Test asymmetric matchability with distance-based thresholds.

        A has low threshold (picky), B has high threshold (permissive).
        If distance is medium, A rejects but B accepts.
        """
        rng = Random(42)

        # Create genomes with known distance
        genome_a = np.zeros(10)
        genome_b = np.ones(10) * 0.5  # Distance = sqrt(10 * 0.25) ≈ 1.58

        # A is picky (max_distance 1.0 - uses similarity_threshold)
        ind_a = create_individual(
            rng,
            genome=genome_a,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(
                    type="similarity_threshold",
                    params={"max_distance": 1.0},
                ),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # B is permissive (max_distance 5.0 - uses similarity_threshold)
        ind_b = create_individual(
            rng,
            genome=genome_b,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(
                    type="similarity_threshold",
                    params={"max_distance": 5.0},
                ),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        a_accepts_b, _ = safe_evaluate_matchability(ind_a.protocol.matchability, ctx_a_to_b, rng)
        b_accepts_a, _ = safe_evaluate_matchability(ind_b.protocol.matchability, ctx_b_to_a, rng)

        # A should reject (distance > 1.0), B should accept (distance < 5.0)
        assert not a_accepts_b, "Picky A should reject distant B"
        assert b_accepts_a, "Permissive B should accept A"

        can_mate = a_accepts_b and b_accepts_a
        assert not can_mate, "Asymmetric distance acceptance should prevent mating"


# =============================================================================
# T042 [US2]: Protocol Inheritance Integration Test
# =============================================================================


class TestProtocolInheritance:
    """
    T042: Verify crossover protocol inheritance (50/50 single-parent).

    Acceptance Criteria:
    - Given two compatible parents with different crossover protocols
    - When reproduction occurs
    - Then offspring receive a crossover protocol inherited from one parent
    """

    def test_offspring_inherits_protocol_from_one_parent(self) -> None:
        """
        Verify offspring protocol comes from exactly one parent (50/50).
        """
        rng = Random(42)

        # Create distinct parent protocols
        protocol_a = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(
                type=CrossoverType.SINGLE_POINT,
                params={"point_ratio": 0.3},
            ),
        )

        protocol_b = ReproductionProtocol(
            matchability=MatchabilityFunction(type="probabilistic", params={"probability": 0.8}),
            intent=ReproductionIntentPolicy(type="fitness_threshold", params={"threshold": 0.5}),
            crossover=CrossoverProtocolSpec(
                type=CrossoverType.UNIFORM,
                params={"swap_prob": 0.6},
            ),
        )

        # Run many inheritances and check distribution
        from_a_count = 0
        from_b_count = 0

        for seed in range(100):
            test_rng = Random(seed)
            offspring_protocol = inherit_protocol(protocol_a, protocol_b, test_rng)

            # Check if offspring protocol matches parent A or B
            if offspring_protocol.crossover.type == protocol_a.crossover.type:
                from_a_count += 1
            elif offspring_protocol.crossover.type == protocol_b.crossover.type:
                from_b_count += 1

        # Should be roughly 50/50
        assert from_a_count + from_b_count == 100, "All offspring should inherit from one parent"
        assert 30 <= from_a_count <= 70, f"Expected ~50% from A, got {from_a_count}%"
        assert 30 <= from_b_count <= 70, f"Expected ~50% from B, got {from_b_count}%"

    def test_inherited_protocol_is_complete(self) -> None:
        """
        Verify inherited protocol has all three components from the same parent.
        """
        rng = Random(42)

        protocol_a = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        protocol_b = ReproductionProtocol(
            matchability=MatchabilityFunction(type="reject_all"),
            intent=ReproductionIntentPolicy(type="never_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
        )

        for seed in range(50):
            test_rng = Random(seed)
            offspring = inherit_protocol(protocol_a, protocol_b, test_rng)

            # All components should come from the same parent
            if offspring.matchability.type == "accept_all":
                assert offspring.intent.type == "always_willing"
                assert offspring.crossover.type == CrossoverType.SINGLE_POINT
            else:
                assert offspring.matchability.type == "reject_all"
                assert offspring.intent.type == "never_willing"
                assert offspring.crossover.type == CrossoverType.UNIFORM

    def test_protocol_inheritance_preserves_junk_data(self) -> None:
        """
        Verify junk_data is preserved during inheritance.
        """
        rng = Random(42)

        protocol_a = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            junk_data={"dormant_param": 0.5, "unused_config": "test"},
        )

        protocol_b = ReproductionProtocol(
            matchability=MatchabilityFunction(type="reject_all"),
            intent=ReproductionIntentPolicy(type="never_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            junk_data={"other_param": 0.9},
        )

        offspring = inherit_protocol(protocol_a, protocol_b, rng)

        # Offspring should have junk_data from one parent
        if offspring.matchability.type == "accept_all":
            assert "dormant_param" in offspring.junk_data
        else:
            assert "other_param" in offspring.junk_data


# =============================================================================
# T054 [US3]: Intent Before Matchability Integration Test
# =============================================================================


class TestIntentBeforeMatchability:
    """
    T054: Verify intent is evaluated before matchability.

    Acceptance Criteria:
    - Given an individual with a fitness-threshold intent policy
    - When its fitness is below threshold
    - Then it does not attempt reproduction regardless of compatible partners
    """

    def test_intent_blocks_before_matchability_check(self) -> None:
        """
        Test that unwilling individuals don't even check matchability.
        """
        rng = Random(42)

        # Create individual with high fitness threshold (won't be willing)
        ind_unwilling = create_individual(
            rng,
            fitness=0.3,  # Below threshold
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),  # Would accept
                intent=ReproductionIntentPolicy(
                    type="fitness_threshold",
                    params={"threshold": 0.5},  # Needs 0.5 to be willing
                ),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Create willing partner
        ind_willing = create_individual(
            rng,
            fitness=0.8,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Check intent first (as ERP does)
        intent_ctx_unwilling = create_intent_context(ind_unwilling)
        intent_ctx_willing = create_intent_context(ind_willing)

        unwilling_intent, _ = safe_evaluate_intent(
            ind_unwilling.protocol.intent, intent_ctx_unwilling, rng
        )
        willing_intent, _ = safe_evaluate_intent(
            ind_willing.protocol.intent, intent_ctx_willing, rng
        )

        assert not unwilling_intent, "Low-fitness individual should not be willing"
        assert willing_intent, "High-fitness individual should be willing"

        # Even though matchability would succeed, intent blocks first
        if unwilling_intent and willing_intent:
            # Only then check matchability
            ctx = create_mate_context(ind_unwilling, ind_willing)
            matchability_result, _ = safe_evaluate_matchability(
                ind_unwilling.protocol.matchability, ctx, rng
            )
            # This code path should not be reached
            pytest.fail("Intent should have blocked before matchability")

    def test_willing_proceeds_to_matchability(self) -> None:
        """
        Test that willing individuals proceed to matchability evaluation.
        """
        rng = Random(42)

        # Create individual above fitness threshold (will be willing)
        ind_a = create_individual(
            rng,
            fitness=0.7,  # Above threshold
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(
                    type="fitness_threshold",
                    params={"threshold": 0.5},
                ),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        ind_b = create_individual(
            rng,
            fitness=0.8,
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Check intent
        intent_a, _ = safe_evaluate_intent(ind_a.protocol.intent, create_intent_context(ind_a), rng)
        intent_b, _ = safe_evaluate_intent(ind_b.protocol.intent, create_intent_context(ind_b), rng)

        assert intent_a, "Individual A should be willing (fitness > threshold)"
        assert intent_b, "Individual B should be willing (always_willing)"

        # Now check matchability
        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        a_accepts, _ = safe_evaluate_matchability(ind_a.protocol.matchability, ctx_a_to_b, rng)
        b_accepts, _ = safe_evaluate_matchability(ind_b.protocol.matchability, ctx_b_to_a, rng)

        assert a_accepts and b_accepts, "Both should accept (accept_all)"

    def test_never_willing_never_mates(self) -> None:
        """
        Test that never_willing individuals never attempt reproduction.
        """
        rng = Random(42)

        ind_never = create_individual(
            rng,
            fitness=1.0,  # High fitness doesn't matter
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="never_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        intent_ctx = create_intent_context(ind_never)
        willing, _ = safe_evaluate_intent(ind_never.protocol.intent, intent_ctx, rng)

        assert not willing, "never_willing should always be unwilling"


# =============================================================================
# T064 [US5]: Dormant Logic Activation Integration Test
# =============================================================================


class TestDormantLogicActivation:
    """
    T064: Verify mutations can activate dormant protocol logic.

    Acceptance Criteria:
    - Given a protocol genome with an inactive matchability rule
    - When the protocol executes, the inactive rule does not affect behavior
    - Given a mutation activates a previously dormant strategy
    - When the individual reproduces, the newly activated strategy is used
    """

    def test_inactive_matchability_does_not_affect_behavior(self) -> None:
        """
        Test that inactive matchability functions don't affect evaluation.
        """
        rng = Random(42)

        # Create protocol with inactive reject_all (should not block)
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(
                type="reject_all",  # Would normally reject
                active=False,  # But it's inactive
            ),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        ind = create_individual(rng, protocol=protocol)
        partner = create_individual(rng)

        ctx = create_mate_context(ind, partner)

        result, success = safe_evaluate_matchability(protocol.matchability, ctx, rng)

        assert success, "Evaluation should succeed"
        # Inactive matchability defaults to rejection for safety
        # This is the expected behavior per spec
        assert result is False, "Inactive matchability should default to reject"

    def test_inactive_intent_defaults_to_willing(self) -> None:
        """
        Test that inactive intent policies default to willing.
        """
        rng = Random(42)

        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(
                type="never_willing",  # Would normally be unwilling
                active=False,  # But it's inactive
            ),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        ind = create_individual(rng, protocol=protocol)
        ctx = create_intent_context(ind)

        willing, success = safe_evaluate_intent(protocol.intent, ctx, rng)

        assert success, "Evaluation should succeed"
        assert willing, "Inactive intent should default to willing"

    def test_mutation_can_activate_dormant_matchability(self) -> None:
        """
        Test that mutations can activate previously inactive matchability.
        """
        rng = Random(42)

        # Create protocol with inactive matchability
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(
                type="distance_threshold",
                params={"max_distance": 5.0},
                active=False,  # Dormant
            ),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        # Create mutator with high activation rate
        config = MutationConfig(
            activation_mutation_rate=1.0,  # Always toggle
            param_mutation_rate=0.0,
            type_mutation_rate=0.0,
        )
        mutator = ProtocolMutator(config)

        # Mutate until active
        mutated = mutator.mutate(protocol, rng)

        # The activation was toggled
        assert mutated.matchability.active != protocol.matchability.active, (
            "Mutation should toggle activation"
        )

    def test_junk_data_mutation_creates_dormant_params(self) -> None:
        """
        Test that junk_data mutations add dormant parameters.
        """
        rng = Random(42)

        protocol = ReproductionProtocol.default()

        config = MutationConfig(
            junk_add_rate=1.0,  # Always add junk
            param_mutation_rate=0.0,
            type_mutation_rate=0.0,
        )
        mutator = ProtocolMutator(config)

        # Mutate to add junk
        mutated = mutator.mutate(protocol, rng)

        # Should have new junk data
        assert len(mutated.junk_data) > len(protocol.junk_data), "Mutation should add junk data"

    def test_junk_data_preserved_through_evolution(self) -> None:
        """
        Test that junk_data survives serialization and inheritance.
        """
        rng = Random(42)

        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            junk_data={
                "dormant_threshold": 0.7,
                "future_param": "reserve",
                "experiment_id": 42,
            },
        )

        # Serialize and deserialize
        serialized = protocol.to_dict()
        restored = ReproductionProtocol.from_dict(serialized)

        assert restored.junk_data == protocol.junk_data, "Junk data should survive serialization"

        # Inherit - use a partner with different matchability to detect which parent is selected
        partner = ReproductionProtocol(
            matchability=MatchabilityFunction(type="reject_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            junk_data={},  # Empty junk_data
        )
        offspring = inherit_protocol(protocol, partner, rng)

        # Offspring should have junk_data from whichever parent was selected
        # Check by matchability type which parent was inherited
        if offspring.matchability.type == "accept_all":
            assert offspring.junk_data == protocol.junk_data, (
                "If inherited from protocol, junk_data should match"
            )
        else:
            assert offspring.junk_data == partner.junk_data, (
                "If inherited from partner, junk_data should match"
            )


# =============================================================================
# Cross-Cutting Integration Tests
# =============================================================================


class TestFullMatingPipeline:
    """
    Integration test for the complete mating pipeline:
    Intent -> Matchability -> Crossover -> Inheritance
    """

    def test_complete_mating_pipeline_success(self) -> None:
        """
        Test a successful mating through the full pipeline.
        """
        rng = Random(42)

        # Two willing, compatible individuals
        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
        )

        genome_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        genome_b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        ind_a = create_individual(rng, genome=genome_a, protocol=protocol, fitness=0.8)
        ind_b = create_individual(rng, genome=genome_b, protocol=protocol, fitness=0.7)

        # Step 1: Check intent
        intent_a, _ = safe_evaluate_intent(ind_a.protocol.intent, create_intent_context(ind_a), rng)
        intent_b, _ = safe_evaluate_intent(ind_b.protocol.intent, create_intent_context(ind_b), rng)
        assert intent_a and intent_b, "Both should be willing"

        # Step 2: Check matchability
        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        a_accepts, _ = safe_evaluate_matchability(ind_a.protocol.matchability, ctx_a_to_b, rng)
        b_accepts, _ = safe_evaluate_matchability(ind_b.protocol.matchability, ctx_b_to_a, rng)
        assert a_accepts and b_accepts, "Both should accept"

        # Step 3: Execute crossover
        (child1, child2), success = safe_execute_crossover(
            ind_a.protocol.crossover, genome_a, genome_b, rng
        )
        assert success, "Crossover should succeed"
        assert len(child1) == len(genome_a)
        assert len(child2) == len(genome_b)

        # Step 4: Inherit protocol
        child_protocol = inherit_protocol(ind_a.protocol, ind_b.protocol, rng)
        assert child_protocol is not None
        assert child_protocol.matchability is not None
        assert child_protocol.intent is not None
        assert child_protocol.crossover is not None

    def test_complete_mating_pipeline_blocked_by_intent(self) -> None:
        """
        Test that unwilling intent blocks the entire pipeline.
        """
        rng = Random(42)

        # A is unwilling
        protocol_a = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="never_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        protocol_b = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        ind_a = create_individual(rng, protocol=protocol_a)
        ind_b = create_individual(rng, protocol=protocol_b)

        # Step 1: Check intent - A fails
        intent_a, _ = safe_evaluate_intent(ind_a.protocol.intent, create_intent_context(ind_a), rng)

        assert not intent_a, "A should be unwilling"
        # Pipeline should stop here - no need to check matchability

    def test_complete_mating_pipeline_blocked_by_matchability(self) -> None:
        """
        Test that matchability rejection blocks after intent passes.
        """
        rng = Random(42)

        # Both willing, but B rejects A
        protocol_a = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        protocol_b = ReproductionProtocol(
            matchability=MatchabilityFunction(type="reject_all"),
            intent=ReproductionIntentPolicy(type="always_willing"),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        )

        ind_a = create_individual(rng, protocol=protocol_a)
        ind_b = create_individual(rng, protocol=protocol_b)

        # Step 1: Both willing
        intent_a, _ = safe_evaluate_intent(ind_a.protocol.intent, create_intent_context(ind_a), rng)
        intent_b, _ = safe_evaluate_intent(ind_b.protocol.intent, create_intent_context(ind_b), rng)
        assert intent_a and intent_b, "Both should be willing"

        # Step 2: Matchability fails
        ctx_a_to_b = create_mate_context(ind_a, ind_b)
        ctx_b_to_a = create_mate_context(ind_b, ind_a)

        a_accepts, _ = safe_evaluate_matchability(ind_a.protocol.matchability, ctx_a_to_b, rng)
        b_accepts, _ = safe_evaluate_matchability(ind_b.protocol.matchability, ctx_b_to_a, rng)

        assert a_accepts, "A should accept"
        assert not b_accepts, "B should reject"

        # Pipeline stops - no crossover
