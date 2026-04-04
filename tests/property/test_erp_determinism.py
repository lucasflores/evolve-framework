"""
Property tests for ERP determinism.

These tests verify that ERP operations are deterministic when
using the same random seed.
"""

from random import Random

import numpy as np
import pytest

from evolve.reproduction.crossover_protocol import execute_crossover, inherit_protocol
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
from evolve.reproduction.recovery import ImmigrationRecovery
from evolve.reproduction.sandbox import StepCounter

# =============================================================================
# Property Test: Matchability Determinism
# =============================================================================


class TestMatchabilityDeterminism:
    """Test that matchability evaluation is deterministic with same seed."""

    @pytest.mark.parametrize(
        "matchability_type",
        [
            "accept_all",
            "reject_all",
            "distance_threshold",
            "similarity_threshold",
            "fitness_ratio",
            "different_niche",
            "probabilistic",
        ],
    )
    def test_matchability_deterministic(self, matchability_type):
        """Matchability should be deterministic with same seed."""
        params = {
            "min_distance": 0.3,
            "max_distance": 0.7,
            "min_ratio": 0.5,
            "max_ratio": 2.0,
            "probability": 0.5,
        }

        function = MatchabilityFunction(type=matchability_type, params=params)

        context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=10,
            partner_fitness_ratio=1.2,
            partner_niche_id=1,
            population_diversity=0.7,
            crowding_distance=0.5,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.2]),
        )

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            result, success = safe_evaluate_matchability(function, context, rng, step_limit=100)
            results.append((result, success))

        assert results[0] == results[1], "Matchability not deterministic"

    def test_matchability_deterministic_multiple_calls(self):
        """Multiple matchability calls with same sequence should be deterministic."""
        function = MatchabilityFunction(type="probabilistic", params={"probability": 0.5})

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

        # Run 10 calls with same seed
        sequences = []
        for _ in range(2):
            rng = Random(99999)
            sequence = []
            for _ in range(10):
                result, _ = safe_evaluate_matchability(function, context, rng, step_limit=100)
                sequence.append(result)
            sequences.append(sequence)

        assert sequences[0] == sequences[1], "Sequence of calls not deterministic"


# =============================================================================
# Property Test: Intent Determinism
# =============================================================================


class TestIntentDeterminism:
    """Test that intent evaluation is deterministic with same seed."""

    @pytest.mark.parametrize(
        "intent_type",
        [
            "always_willing",
            "never_willing",
            "fitness_threshold",
            "fitness_rank_threshold",
            "resource_budget",
            "age_dependent",
            "probabilistic",
        ],
    )
    def test_intent_deterministic(self, intent_type):
        """Intent should be deterministic with same seed."""
        params = {
            "threshold": 0.5,
            "max_rank": 10,
            "budget": 5,
            "min_age": 1,
            "max_age": 10,
            "probability": 0.5,
        }

        policy = ReproductionIntentPolicy(type=intent_type, params=params)

        context = IntentContext(
            fitness=np.array([1.0]),
            fitness_rank=5,
            age=3,
            offspring_count=2,
            generation=10,
            population_size=50,
        )

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            result, success = safe_evaluate_intent(policy, context, rng, step_limit=100)
            results.append((result, success))

        assert results[0] == results[1], "Intent not deterministic"


# =============================================================================
# Property Test: Crossover Determinism
# =============================================================================


class TestCrossoverDeterminism:
    """Test that crossover is deterministic with same seed."""

    @pytest.mark.parametrize(
        "crossover_type",
        [
            CrossoverType.SINGLE_POINT,
            CrossoverType.TWO_POINT,
            CrossoverType.UNIFORM,
            CrossoverType.BLEND,
            CrossoverType.CLONE,
        ],
    )
    def test_crossover_deterministic(self, crossover_type):
        """Crossover should be deterministic with same seed."""
        spec = CrossoverProtocolSpec(
            type=crossover_type,
            params={"alpha": 0.5},  # For blend
        )

        parent1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        parent2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            counter = StepCounter(limit=100)
            child1, child2 = execute_crossover(spec, parent1, parent2, rng, counter)
            results.append((child1.tolist(), child2.tolist()))

        assert results[0] == results[1], "Crossover not deterministic"


# =============================================================================
# Property Test: Protocol Inheritance Determinism
# =============================================================================


class TestProtocolInheritanceDeterminism:
    """Test that protocol inheritance is deterministic."""

    def test_inherit_protocol_deterministic(self):
        """Protocol inheritance should be deterministic with same seed."""
        parent1 = ReproductionProtocol(
            matchability=MatchabilityFunction(type="accept_all", params={}),
            intent=ReproductionIntentPolicy(type="always_willing", params={}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM, params={}),
        )

        parent2 = ReproductionProtocol(
            matchability=MatchabilityFunction(type="reject_all", params={}),
            intent=ReproductionIntentPolicy(type="never_willing", params={}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT, params={}),
        )

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            child = inherit_protocol(parent1, parent2, rng)
            results.append(
                (
                    child.matchability.type,
                    child.intent.type,
                    child.crossover.type.value,
                )
            )

        assert results[0] == results[1], "Protocol inheritance not deterministic"


# =============================================================================
# Property Test: Mutation Determinism
# =============================================================================


class TestMutationDeterminism:
    """Test that protocol mutation is deterministic."""

    def test_mutator_deterministic(self):
        """ProtocolMutator should be deterministic with same seed."""
        config = MutationConfig(
            param_mutation_rate=0.5,
            param_mutation_strength=0.1,
            type_mutation_rate=0.5,
            activation_mutation_rate=0.5,
            junk_add_rate=0.5,
            junk_remove_rate=0.5,
            junk_modify_rate=0.5,
            junk_activate_rate=0.5,
        )
        mutator = ProtocolMutator(config)

        protocol = ReproductionProtocol(
            matchability=MatchabilityFunction(
                type="distance_threshold", params={"min_distance": 0.5}
            ),
            intent=ReproductionIntentPolicy(type="fitness_threshold", params={"threshold": 0.5}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM, params={}),
            junk_data={"dormant_param": 0.3},
        )

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            mutated = mutator.mutate(protocol, rng)
            results.append(
                (
                    mutated.matchability.type,
                    dict(mutated.matchability.params),
                    mutated.intent.type,
                    dict(mutated.intent.params),
                    mutated.crossover.type.value,
                    dict(mutated.junk_data) if mutated.junk_data else {},
                )
            )

        assert results[0] == results[1], "Mutation not deterministic"


# =============================================================================
# Property Test: Recovery Determinism
# =============================================================================


class TestRecoveryDeterminism:
    """Test that recovery is deterministic."""

    def test_immigration_recovery_deterministic(self):
        """Immigration recovery should be deterministic with same seed."""
        recovery = ImmigrationRecovery(immigration_rate=0.2)

        survivors = [np.array([i] * 10) for i in range(10)]

        def genome_factory(rng: Random) -> np.ndarray:
            return np.array([rng.random() for _ in range(10)])

        def protocol_factory(rng: Random) -> ReproductionProtocol:
            return ReproductionProtocol.default()

        # Run with same seed twice
        results = []
        for _ in range(2):
            rng = Random(12345)
            survivors_out, immigrants = recovery.recover(
                survivors, genome_factory, protocol_factory, rng
            )
            # Reset for next iteration
            recovery._last_trigger = -100

            # Capture immigrant genomes
            immigrant_genomes = [g.tolist() for g, p in immigrants]
            results.append(immigrant_genomes)

        assert results[0] == results[1], "Immigration recovery not deterministic"


# =============================================================================
# Property Test: End-to-End Determinism
# =============================================================================


class TestEndToEndDeterminism:
    """Test end-to-end ERP determinism."""

    def test_full_mating_cycle_deterministic(self):
        """Full mating cycle (intent + matchability + crossover) should be deterministic."""
        protocol1 = ReproductionProtocol(
            matchability=MatchabilityFunction(type="probabilistic", params={"probability": 0.8}),
            intent=ReproductionIntentPolicy(type="probabilistic", params={"probability": 0.9}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM, params={}),
        )
        protocol2 = ReproductionProtocol(
            matchability=MatchabilityFunction(type="probabilistic", params={"probability": 0.7}),
            intent=ReproductionIntentPolicy(type="probabilistic", params={"probability": 0.8}),
            crossover=CrossoverProtocolSpec(type=CrossoverType.BLEND, params={"alpha": 0.5}),
        )

        genome1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        genome2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        intent_context = IntentContext(
            fitness=np.array([1.0]),
            fitness_rank=5,
            age=3,
            offspring_count=2,
            generation=10,
            population_size=50,
        )

        mate_context = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=10,
            partner_fitness_ratio=1.2,
            partner_niche_id=None,
            population_diversity=0.7,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.2]),
        )

        def run_mating_cycle(seed: int):
            rng = Random(seed)

            # Intent checks
            intent1, _ = safe_evaluate_intent(protocol1.intent, intent_context, rng, 100)
            intent2, _ = safe_evaluate_intent(protocol2.intent, intent_context, rng, 100)

            # Matchability checks
            match1, _ = safe_evaluate_matchability(protocol1.matchability, mate_context, rng, 100)
            match2, _ = safe_evaluate_matchability(protocol2.matchability, mate_context, rng, 100)

            # Crossover if both passed
            if intent1 and intent2 and match1 and match2:
                counter = StepCounter(limit=100)
                child1, child2 = execute_crossover(
                    protocol1.crossover, genome1, genome2, rng, counter
                )
                return (True, child1.tolist(), child2.tolist())
            else:
                return (False, intent1, intent2, match1, match2)

        # Run with same seed twice
        result1 = run_mating_cycle(54321)
        result2 = run_mating_cycle(54321)

        assert result1 == result2, "Full mating cycle not deterministic"
