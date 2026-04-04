"""
Integration tests for ERP with NSGA-II multi-objective evolution.

Tests T057 and T058 for User Story 6:
- T057: ERPEngine respects selection authority over survival
- T058: ERP + NSGA-II integration test
"""

from __future__ import annotations

from random import Random
from uuid import uuid4

import numpy as np

from evolve.core.population import Population
from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.multiobjective.dominance import dominates
from evolve.multiobjective.fitness import MultiObjectiveFitness
from evolve.multiobjective.ranking import fast_non_dominated_sort
from evolve.reproduction.crossover_protocol import safe_execute_crossover
from evolve.reproduction.intent import safe_evaluate_intent
from evolve.reproduction.matchability import safe_evaluate_matchability
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


def create_multiobjective_individual(
    rng: Random,
    objectives: list[float],
    protocol: ReproductionProtocol | None = None,
) -> Individual[np.ndarray]:
    """Create an individual with multi-objective fitness."""
    genome = np.array([rng.random() for _ in range(10)])

    if protocol is None:
        protocol = ReproductionProtocol.default()

    return Individual(
        id=uuid4(),
        genome=genome,
        protocol=protocol,
        fitness=Fitness(values=np.array(objectives)),
        metadata=IndividualMetadata(),
        created_at=0,
    )


def create_pareto_population(
    rng: Random,
    size: int = 20,
) -> Population[np.ndarray]:
    """
    Create a population with a mix of Pareto-optimal and dominated individuals.
    """
    individuals = []

    # Create some Pareto-optimal individuals (trade-offs)
    for i in range(size // 2):
        # Trade-off between objectives
        obj1 = i / (size // 2)
        obj2 = 1.0 - obj1
        individuals.append(create_multiobjective_individual(rng, [obj1, obj2]))

    # Create some dominated individuals
    for i in range(size - size // 2):
        # Worse on both objectives
        obj1 = rng.uniform(0.2, 0.8)
        obj2 = rng.uniform(0.2, 0.8) - 0.3  # Shifted down
        individuals.append(create_multiobjective_individual(rng, [obj1, max(0, obj2)]))

    return Population(individuals=individuals, generation=0)


def create_mate_context(
    self_ind: Individual[np.ndarray],
    partner: Individual[np.ndarray],
    crowding_distance: float | None = None,
) -> MateContext:
    """Create MateContext for multi-objective individuals."""
    distance = float(np.linalg.norm(self_ind.genome - partner.genome))

    self_fitness = self_ind.fitness.values if self_ind.fitness else np.array([0.0])
    partner_fitness = partner.fitness.values if partner.fitness else np.array([0.0])

    # Use first objective for ratio
    self_f = float(self_fitness[0])
    partner_f = float(partner_fitness[0])
    ratio = partner_f / self_f if self_f != 0 else 1.0

    return MateContext(
        partner_distance=distance,
        partner_fitness_rank=0,
        partner_fitness_ratio=ratio,
        partner_niche_id=None,
        population_diversity=0.5,
        crowding_distance=crowding_distance,
        self_fitness=self_fitness,
        partner_fitness=partner_fitness,
    )


def create_intent_context(
    individual: Individual[np.ndarray],
    fitness_rank: int = 0,
) -> IntentContext:
    """Create IntentContext for multi-objective individuals."""
    fitness = individual.fitness.values if individual.fitness else np.array([0.0])

    return IntentContext(
        fitness=fitness,
        fitness_rank=fitness_rank,
        age=0,
        offspring_count=0,
        generation=0,
        population_size=50,
    )


# =============================================================================
# T057: ERPEngine Respects Selection Authority
# =============================================================================


class TestSelectionAuthority:
    """
    T057: Verify ERP respects system-level selection authority.

    FR-023: System-level selection (Pareto ranking, elitism) MUST remain
            authoritative over survival decisions
    FR-024: ERP MUST only influence mating pair formation, not survival selection
    """

    def test_erp_does_not_override_pareto_ranking(self) -> None:
        """
        Verify that ERP matchability doesn't affect Pareto ranking.

        Pareto ranking is determined by dominance, not by whether
        individuals want to mate with each other.
        """
        rng = Random(42)

        # Create individuals with different protocols but same fitness
        ind_a = create_multiobjective_individual(
            rng,
            objectives=[0.8, 0.2],  # Pareto-optimal
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="reject_all"),
                intent=ReproductionIntentPolicy(type="never_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        ind_b = create_multiobjective_individual(
            rng,
            objectives=[0.2, 0.8],  # Pareto-optimal
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        ind_c = create_multiobjective_individual(
            rng,
            objectives=[0.3, 0.3],  # Dominated by A and B
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # Pareto ranking should be based on fitness only
        population = Population(individuals=[ind_a, ind_b, ind_c], generation=0)

        # Use fast non-dominated sort with MultiObjectiveFitness objects
        mo_fitnesses = [MultiObjectiveFitness(ind.fitness.values) for ind in population.individuals]
        fronts = fast_non_dominated_sort(mo_fitnesses)

        # A and B should be in front 0 (non-dominated) - fronts contain indices
        front_0_indices = set(fronts[0])
        assert 0 in front_0_indices, "A (index 0) should be in Pareto front 0"
        assert 1 in front_0_indices, "B (index 1) should be in Pareto front 0"

        # C should be dominated (front 1 or later) - fronts contain indices
        if len(fronts) > 1:
            front_1_indices = set(fronts[1])
            assert 2 in front_1_indices, "C (index 2) should be dominated"

        # ERP protocols (reject_all, never_willing) don't affect ranking

    def test_erp_only_affects_mating_not_survival(self) -> None:
        """
        Verify that ERP matchability affects mating, not survival.

        An individual with reject_all matchability can still survive
        (be selected for next generation), it just can't mate.
        """
        rng = Random(42)

        # Create a high-fitness individual that rejects everyone
        elite = create_multiobjective_individual(
            rng,
            objectives=[1.0, 1.0],  # Best possible
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="reject_all"),
                intent=ReproductionIntentPolicy(type="never_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
            ),
        )

        # This individual should:
        # 1. Be selected for survival (high fitness)
        # 2. NOT be able to mate (reject_all + never_willing)

        # Check survival: dominates others
        other = create_multiobjective_individual(rng, objectives=[0.5, 0.5])
        elite_mo = MultiObjectiveFitness(elite.fitness.values)
        other_mo = MultiObjectiveFitness(other.fitness.values)
        assert dominates(elite_mo, other_mo), "Elite should dominate others"

        # Check mating: cannot mate
        ctx = create_intent_context(elite)
        willing, _ = safe_evaluate_intent(elite.protocol.intent, ctx, rng)
        assert not willing, "Elite should not be willing to mate"

    def test_dominated_individual_can_still_mate(self) -> None:
        """
        Verify that dominated individuals can still participate in mating.

        ERP governs MATING, not survival selection.
        """
        rng = Random(42)

        # Create a dominated individual with permissive mating
        dominated = create_multiobjective_individual(
            rng,
            objectives=[0.1, 0.1],  # Low fitness, likely dominated
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            ),
        )

        # Create a dominating individual
        dominator = create_multiobjective_individual(
            rng,
            objectives=[0.9, 0.9],  # High fitness
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(type="accept_all"),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            ),
        )

        # Verify dominance relationship
        dominator_mo = MultiObjectiveFitness(dominator.fitness.values)
        dominated_mo = MultiObjectiveFitness(dominated.fitness.values)
        assert dominates(dominator_mo, dominated_mo), "Dominator should dominate"

        # But dominated individual can still mate
        intent_ctx = create_intent_context(dominated)
        willing, _ = safe_evaluate_intent(dominated.protocol.intent, intent_ctx, rng)
        assert willing, "Dominated individual should be willing to mate"

        mate_ctx = create_mate_context(dominated, dominator)
        accepts, _ = safe_evaluate_matchability(dominated.protocol.matchability, mate_ctx, rng)
        assert accepts, "Dominated individual should accept mate"


# =============================================================================
# T058: ERP + NSGA-II Integration Test
# =============================================================================


class TestERPWithNSGAII:
    """
    T058: Verify ERP works with NSGA-II style evolution.

    FR-025: Matchability functions MUST be able to access crowding
            distance and diversity metrics
    """

    def test_matchability_can_access_crowding_distance(self) -> None:
        """
        Verify that crowding distance is available to matchability functions.
        """
        rng = Random(42)

        # Create a diversity-seeking individual
        ind_a = create_multiobjective_individual(
            rng,
            objectives=[0.5, 0.5],
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(
                    type="diversity_seeking",
                    params={"diversity_weight": 0.8},
                ),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            ),
        )

        ind_b = create_multiobjective_individual(rng, objectives=[0.6, 0.4])

        # Create context with crowding distance
        ctx = create_mate_context(ind_a, ind_b, crowding_distance=0.75)

        # Verify crowding distance is in context
        assert ctx.crowding_distance == 0.75, "Crowding distance should be in context"

        # Evaluate matchability (should have access to crowding distance)
        result, success = safe_evaluate_matchability(ind_a.protocol.matchability, ctx, rng)
        assert success, "Matchability evaluation should succeed"

    def test_matchability_can_access_population_diversity(self) -> None:
        """
        Verify that population diversity is available to matchability.
        """
        rng = Random(42)

        ind_a = create_multiobjective_individual(
            rng,
            objectives=[0.5, 0.5],
            protocol=ReproductionProtocol(
                matchability=MatchabilityFunction(
                    type="diversity_seeking",
                    params={"diversity_weight": 0.5},
                ),
                intent=ReproductionIntentPolicy(type="always_willing"),
                crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
            ),
        )

        ind_b = create_multiobjective_individual(rng, objectives=[0.7, 0.3])

        # Create context with specific diversity
        ctx = MateContext(
            partner_distance=0.5,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.9,  # High diversity
            crowding_distance=None,
            self_fitness=ind_a.fitness.values,
            partner_fitness=ind_b.fitness.values,
        )

        assert ctx.population_diversity == 0.9, "Population diversity should be in context"

        result, success = safe_evaluate_matchability(ind_a.protocol.matchability, ctx, rng)
        assert success, "Should succeed with diversity access"

    def test_nsga2_workflow_with_erp_mating(self) -> None:
        """
        Test complete NSGA-II workflow with ERP-based mating.

        1. Non-dominated sorting for ranking
        2. ERP for mating pair selection
        3. Crossover via ERP protocols
        """
        rng = Random(42)
        population = create_pareto_population(rng, size=20)

        # Step 1: Non-dominated sorting (selection authority)
        mo_fitnesses = [MultiObjectiveFitness(ind.fitness.values) for ind in population.individuals]
        fronts = fast_non_dominated_sort(mo_fitnesses)

        assert len(fronts) > 0, "Should have at least one front"
        assert len(fronts[0]) > 0, "First front should have individuals"

        # Step 2: Select parents from front 0 (best individuals)
        # fronts contain indices, so get actual individuals
        parents = [population.individuals[i] for i in fronts[0][:6]]  # Take first 6

        # Step 3: ERP-based mating
        offspring_genomes = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            # Check intent
            intent1, _ = safe_evaluate_intent(
                parent1.protocol.intent, create_intent_context(parent1), rng
            )
            intent2, _ = safe_evaluate_intent(
                parent2.protocol.intent, create_intent_context(parent2), rng
            )

            if not (intent1 and intent2):
                continue

            # Check matchability
            ctx1 = create_mate_context(parent1, parent2)
            ctx2 = create_mate_context(parent2, parent1)

            match1, _ = safe_evaluate_matchability(parent1.protocol.matchability, ctx1, rng)
            match2, _ = safe_evaluate_matchability(parent2.protocol.matchability, ctx2, rng)

            if not (match1 and match2):
                continue

            # Execute crossover
            (child1, child2), success = safe_execute_crossover(
                parent1.protocol.crossover,
                parent1.genome,
                parent2.genome,
                rng,
            )

            if success:
                offspring_genomes.extend([child1, child2])

        # Should have produced some offspring
        assert len(offspring_genomes) > 0, "Should produce offspring"

    def test_erp_with_mixed_protocols_in_pareto_front(self) -> None:
        """
        Test that individuals with different protocols can coexist in Pareto front.
        """
        rng = Random(42)

        # Create Pareto-optimal individuals with different protocols
        individuals = [
            create_multiobjective_individual(
                rng,
                objectives=[0.9, 0.1],
                protocol=ReproductionProtocol(
                    matchability=MatchabilityFunction(type="accept_all"),
                    intent=ReproductionIntentPolicy(type="always_willing"),
                    crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
                ),
            ),
            create_multiobjective_individual(
                rng,
                objectives=[0.5, 0.5],
                protocol=ReproductionProtocol(
                    matchability=MatchabilityFunction(
                        type="distance_threshold",
                        params={"max_distance": 3.0},
                    ),
                    intent=ReproductionIntentPolicy(
                        type="fitness_threshold",
                        params={"threshold": 0.4},
                    ),
                    crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
                ),
            ),
            create_multiobjective_individual(
                rng,
                objectives=[0.1, 0.9],
                protocol=ReproductionProtocol(
                    matchability=MatchabilityFunction(type="reject_all"),
                    intent=ReproductionIntentPolicy(type="never_willing"),
                    crossover=CrossoverProtocolSpec(type=CrossoverType.CLONE),
                ),
            ),
        ]

        population = Population(individuals=individuals, generation=0)
        mo_fitnesses = [MultiObjectiveFitness(ind.fitness.values) for ind in population.individuals]
        fronts = fast_non_dominated_sort(mo_fitnesses)

        # All three should be in front 0 (non-dominated by each other)
        # fronts contain indices, not individuals
        front_0_indices = set(fronts[0])

        assert len(front_0_indices) == 3, "All individuals should be non-dominated"
        for i, ind in enumerate(individuals):
            assert i in front_0_indices, f"Individual {i} should be in front 0"

    def test_erp_preserves_pareto_optimality(self) -> None:
        """
        Test that ERP mating doesn't accidentally affect Pareto calculations.

        The act of checking matchability/intent should not modify fitness.
        """
        rng = Random(42)

        ind_a = create_multiobjective_individual(
            rng,
            objectives=[0.7, 0.3],
            protocol=ReproductionProtocol.default(),
        )

        original_fitness = ind_a.fitness.values.copy()

        # Perform many ERP evaluations
        for _ in range(100):
            ctx = create_intent_context(ind_a)
            safe_evaluate_intent(ind_a.protocol.intent, ctx, rng)

            mate_ctx = create_mate_context(ind_a, ind_a)
            safe_evaluate_matchability(ind_a.protocol.matchability, mate_ctx, rng)

        # Fitness should be unchanged
        np.testing.assert_array_equal(
            ind_a.fitness.values,
            original_fitness,
            err_msg="ERP evaluation should not modify fitness",
        )
