"""
ERPEngine - Evolution engine with Evolvable Reproduction Protocols.

This engine extends the base EvolutionEngine to support individual-level
control over reproduction through encoded protocols. Each individual
can encode:
- Who they will mate with (matchability)
- When they attempt reproduction (intent)
- How offspring are constructed (crossover)

Key differences from base engine:
- Mating requires mutual consent (both partners must accept)
- Intent is evaluated before matchability
- Protocol inheritance follows 50/50 single-parent rule
- Recovery mechanisms handle population collapse

Note: Import this module directly (not via __init__) to avoid circular imports:
    from evolve.reproduction.engine import ERPConfig, ERPEngine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, TypeVar
from uuid import UUID, uuid4

import numpy as np

from evolve.reproduction.crossover_protocol import (
    execute_crossover,
    inherit_protocol,
    safe_execute_crossover,
    validate_offspring,
)
from evolve.reproduction.intent import evaluate_intent, safe_evaluate_intent
from evolve.reproduction.matchability import evaluate_matchability, safe_evaluate_matchability
from evolve.reproduction.mutation import MutationConfig, ProtocolMutator
from evolve.reproduction.protocol import (
    IntentContext,
    MateContext,
    ReproductionEvent,
    ReproductionProtocol,
)
from evolve.reproduction.recovery import ImmigrationRecovery, RecoveryStrategy
from evolve.reproduction.sandbox import StepCounter

# Import core modules here (not in __init__ to avoid circular imports)
from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.population import Population
from evolve.core.types import Individual, IndividualMetadata

G = TypeVar("G")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ERPConfig(EvolutionConfig):
    """
    Configuration for ERP evolution.

    Extends EvolutionConfig with ERP-specific parameters.

    Attributes:
        step_limit: Max computation steps per protocol evaluation
        recovery_threshold: Success rate below which recovery triggers
        protocol_mutation_rate: Probability of mutating the protocol
        enable_intent: Whether to evaluate intent policies
        enable_recovery: Whether to use recovery mechanisms
    """

    step_limit: int = 1000
    recovery_threshold: float = 0.1
    protocol_mutation_rate: float = 0.1
    enable_intent: bool = True
    enable_recovery: bool = True


# =============================================================================
# ERP Engine
# =============================================================================


class ERPEngine(EvolutionEngine[G]):
    """
    Evolution engine with Evolvable Reproduction Protocols.

    This engine extends base evolution with individual-level reproductive
    control. Mating requires:
    1. Both parents are willing to reproduce (intent)
    2. Both parents accept each other (matchability)

    If reproduction fails too often, recovery mechanisms inject immigrants
    or relax constraints to prevent population collapse.

    Example:
        >>> from evolve.reproduction.engine import ERPEngine, ERPConfig
        >>> from evolve.reproduction.protocol import ReproductionProtocol
        >>>
        >>> config = ERPConfig(population_size=50, max_generations=100)
        >>> engine = ERPEngine(
        ...     config=config,
        ...     evaluator=evaluator,
        ...     selection=TournamentSelection(),
        ...     crossover=UniformCrossover(),
        ...     mutation=GaussianMutation(),
        ...     seed=42
        ... )
        >>> # Individuals must have protocol field set
        >>> result = engine.run(initial_population=pop)
    """

    def __init__(
        self,
        config: ERPConfig,
        evaluator: Evaluator[G],
        selection: Any,
        crossover: Any,
        mutation: Any,
        seed: int = 42,
        stopping: Any | None = None,
        recovery: RecoveryStrategy | None = None,
        protocol_mutator: ProtocolMutator | None = None,
        default_protocol_factory: Callable[[Random], ReproductionProtocol] | None = None,
    ) -> None:
        """
        Initialize ERP engine.

        Args:
            config: Evolution and ERP parameters
            evaluator: Fitness evaluator
            selection: Selection operator
            crossover: Crossover operator (used as fallback)
            mutation: Genome mutation operator
            seed: Master random seed
            stopping: Optional stopping criterion
            recovery: Optional recovery strategy (default: immigration)
            protocol_mutator: Optional protocol mutator
            default_protocol_factory: Factory for default protocols
        """
        super().__init__(config, evaluator, selection, crossover, mutation, seed, stopping)

        self.erp_config = config

        # Set up recovery
        if recovery is None and config.enable_recovery:
            self.recovery = ImmigrationRecovery(
                trigger_threshold=config.recovery_threshold,
                immigration_rate=0.1,
            )
        else:
            self.recovery = recovery

        # Protocol mutator
        if protocol_mutator is None:
            self.protocol_mutator = ProtocolMutator(MutationConfig())
        else:
            self.protocol_mutator = protocol_mutator

        # Protocol factory for immigrants/defaults
        if default_protocol_factory is None:
            self.default_protocol_factory = lambda rng: ReproductionProtocol.default()
        else:
            self.default_protocol_factory = default_protocol_factory

        # ERP tracking
        self._events: list[ReproductionEvent] = []
        self._successful_matings = 0
        self._attempted_matings = 0

    def _step(self, population: Population[G]) -> Population[G]:
        """
        Perform one ERP evolution step.

        1. Select candidate parents
        2. Attempt matings with intent/matchability checks
        3. Create offspring with protocol inheritance
        4. Apply recovery if needed
        5. Evaluate new population
        """
        pop_size = self.config.population_size
        n_elites = self.config.elitism
        n_offspring = pop_size - n_elites

        # Reset generation tracking
        self._events = []
        self._successful_matings = 0
        self._attempted_matings = 0

        # Get elites
        elites = list(population.best(n_elites, minimize=self.config.minimize))

        # Select candidate parents
        n_parents = n_offspring * 2
        parents = list(self.selection.select(population, n_parents, self.rng))

        # Attempt matings with ERP
        offspring: list[Individual[G]] = []
        max_attempts = n_offspring * 3  # Allow extra attempts

        for attempt in range(0, min(len(parents) - 1, max_attempts), 2):
            if len(offspring) >= n_offspring:
                break

            parent1 = parents[attempt % len(parents)]
            parent2 = parents[(attempt + 1) % len(parents)]

            # Skip if parents are the same
            if parent1.id == parent2.id:
                continue

            # Attempt mating
            children = self._attempt_mating(parent1, parent2, population)
            offspring.extend(children)

        # Trim to exact size
        offspring = offspring[:n_offspring]

        # Check if recovery needed
        if self.recovery and self.erp_config.enable_recovery:
            if self.recovery.should_trigger(
                self._successful_matings,
                self._attempted_matings,
                len(population),
                self._generation,
            ):
                # Recovery may add immigrants
                result = self.recovery.recover(
                    offspring,
                    lambda rng: self._create_random_genome(rng),
                    self.default_protocol_factory,
                    self.rng,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    survivors, immigrants = result
                    # Convert immigrants to individuals
                    for genome, protocol in immigrants:
                        immigrant = Individual(
                            id=uuid4(),
                            genome=genome,
                            protocol=protocol,
                            metadata=IndividualMetadata(origin="immigration"),
                            created_at=self._generation + 1,
                        )
                        offspring.append(immigrant)
                    offspring = offspring[:n_offspring]

        # If still not enough offspring, fallback to cloning
        while len(offspring) < n_offspring:
            # Clone a random parent
            parent = parents[self.rng.randint(0, len(parents) - 1)]
            clone = Individual(
                id=uuid4(),
                genome=parent.genome.copy() if hasattr(parent.genome, 'copy') else parent.genome,
                protocol=parent.protocol,
                metadata=IndividualMetadata(
                    parent_ids=(parent.id,),
                    origin="clone",
                ),
                created_at=self._generation + 1,
            )
            offspring.append(clone)

        # Combine elites and offspring
        new_individuals = elites + offspring[:n_offspring]

        # Create and evaluate new population
        new_population = Population(
            individuals=new_individuals,
            generation=self._generation + 1,
        )

        return self._evaluate_population(new_population)

    def _attempt_mating(
        self,
        parent1: Individual[G],
        parent2: Individual[G],
        population: Population[G],
    ) -> list[Individual[G]]:
        """
        Attempt mating between two individuals.

        Checks:
        1. Both have intent to reproduce
        2. Both accept each other (matchability)
        3. Crossover succeeds
        4. Offspring are valid

        Args:
            parent1: First parent
            parent2: Second parent
            population: Current population (for context)

        Returns:
            List of offspring (empty if mating failed)
        """
        self._attempted_matings += 1

        # Get protocols (use default if missing)
        protocol1 = parent1.protocol or ReproductionProtocol.default()
        protocol2 = parent2.protocol or ReproductionProtocol.default()

        # Check intent
        if self.erp_config.enable_intent:
            intent1, _ = self._check_intent(parent1, population)
            intent2, _ = self._check_intent(parent2, population)

            if not intent1 or not intent2:
                self._emit_event(
                    parent1, parent2, False,
                    "Intent failed",
                    matchability_result=(False, False),
                    intent_result=(intent1, intent2),
                )
                return []
        else:
            intent1, intent2 = True, True

        # Check matchability (bidirectional)
        match1, _ = self._check_matchability(parent1, parent2, population)
        match2, _ = self._check_matchability(parent2, parent1, population)

        if not match1 or not match2:
            self._emit_event(
                parent1, parent2, False,
                "Matchability failed",
                matchability_result=(match1, match2),
                intent_result=(intent1, intent2),
            )
            return []

        # Perform crossover
        crossover_spec = protocol1.crossover  # Use parent1's crossover spec

        (child1_genome, child2_genome), success = safe_execute_crossover(
            crossover_spec,
            parent1.genome,
            parent2.genome,
            self.rng,
            step_limit=self.erp_config.step_limit,
        )

        if not success:
            # Fallback to standard crossover
            child1_genome, child2_genome = self.crossover.crossover(
                parent1.genome, parent2.genome, self.rng
            )

        # Validate offspring
        valid1, _ = validate_offspring(child1_genome, parent1.genome, parent2.genome)
        valid2, _ = validate_offspring(child2_genome, parent1.genome, parent2.genome)

        if not valid1:
            child1_genome = parent1.genome.copy() if hasattr(parent1.genome, 'copy') else parent1.genome
        if not valid2:
            child2_genome = parent2.genome.copy() if hasattr(parent2.genome, 'copy') else parent2.genome

        # Apply genome mutation
        if self.rng.random() < self.config.mutation_rate:
            child1_genome = self.mutation.mutate(child1_genome, self.rng)
        if self.rng.random() < self.config.mutation_rate:
            child2_genome = self.mutation.mutate(child2_genome, self.rng)

        # Inherit protocols
        child1_protocol = inherit_protocol(protocol1, protocol2, self.rng)
        child2_protocol = inherit_protocol(protocol2, protocol1, self.rng)

        # Apply protocol mutation
        if self.rng.random() < self.erp_config.protocol_mutation_rate:
            child1_protocol = self.protocol_mutator.mutate(child1_protocol, self.rng)
        if self.rng.random() < self.erp_config.protocol_mutation_rate:
            child2_protocol = self.protocol_mutator.mutate(child2_protocol, self.rng)

        # Create offspring individuals
        child1 = Individual(
            id=uuid4(),
            genome=child1_genome,
            protocol=child1_protocol,
            metadata=IndividualMetadata(
                parent_ids=(parent1.id, parent2.id),
                origin="erp_crossover",
            ),
            created_at=self._generation + 1,
        )
        child2 = Individual(
            id=uuid4(),
            genome=child2_genome,
            protocol=child2_protocol,
            metadata=IndividualMetadata(
                parent_ids=(parent1.id, parent2.id),
                origin="erp_crossover",
            ),
            created_at=self._generation + 1,
        )

        self._successful_matings += 1
        self._emit_event(
            parent1, parent2, True,
            None,
            matchability_result=(match1, match2),
            intent_result=(intent1, intent2),
            offspring_ids=(child1.id, child2.id),
        )

        return [child1, child2]

    def _check_intent(
        self,
        individual: Individual[G],
        population: Population[G],
    ) -> tuple[bool, bool]:
        """
        Check if individual is willing to reproduce.

        Args:
            individual: The individual to check
            population: Current population

        Returns:
            Tuple of (willing, success)
        """
        protocol = individual.protocol or ReproductionProtocol.default()

        # Build intent context
        fitness = individual.fitness
        if fitness is None:
            fitness_values = np.array([0.0])
        else:
            fitness_values = fitness.values if hasattr(fitness, 'values') else np.array([float(fitness.value)])

        # Calculate fitness rank
        fitness_ranks = self._compute_fitness_ranks(population)
        fitness_rank = fitness_ranks.get(individual.id, len(population))

        # Calculate age
        age = self._generation - individual.created_at

        context = IntentContext(
            fitness=fitness_values,
            fitness_rank=fitness_rank,
            age=age,
            offspring_count=0,  # Not tracked per-generation yet
            generation=self._generation,
            population_size=len(population),
        )

        return safe_evaluate_intent(
            protocol.intent,
            context,
            self.rng,
            step_limit=self.erp_config.step_limit,
        )

    def _check_matchability(
        self,
        self_ind: Individual[G],
        partner: Individual[G],
        population: Population[G],
    ) -> tuple[bool, bool]:
        """
        Check if self_ind accepts partner as a mate.

        Args:
            self_ind: Individual making the decision
            partner: Potential mate
            population: Current population

        Returns:
            Tuple of (accepts, success)
        """
        protocol = self_ind.protocol or ReproductionProtocol.default()

        # Build mate context
        # Compute genetic distance
        if hasattr(self_ind.genome, '__sub__'):
            distance = float(np.linalg.norm(self_ind.genome - partner.genome))
        else:
            distance = 0.0

        # Get fitness values
        self_fitness = self_ind.fitness
        partner_fitness = partner.fitness

        if self_fitness is None:
            self_fitness_values = np.array([0.0])
        else:
            self_fitness_values = self_fitness.values if hasattr(self_fitness, 'values') else np.array([float(self_fitness.value)])

        if partner_fitness is None:
            partner_fitness_values = np.array([0.0])
        else:
            partner_fitness_values = partner_fitness.values if hasattr(partner_fitness, 'values') else np.array([float(partner_fitness.value)])

        # Compute fitness ratio
        self_f = float(self_fitness_values[0]) if len(self_fitness_values) > 0 else 1.0
        partner_f = float(partner_fitness_values[0]) if len(partner_fitness_values) > 0 else 1.0
        if self_f == 0:
            fitness_ratio = float('inf') if partner_f != 0 else 1.0
        else:
            fitness_ratio = partner_f / self_f

        # Compute fitness rank
        fitness_ranks = self._compute_fitness_ranks(population)
        partner_rank = fitness_ranks.get(partner.id, len(population))

        # Population diversity (simple std-based measure)
        diversity = self._compute_population_diversity(population)

        context = MateContext(
            partner_distance=distance,
            partner_fitness_rank=partner_rank,
            partner_fitness_ratio=fitness_ratio,
            partner_niche_id=None,  # Not implemented yet
            population_diversity=diversity,
            crowding_distance=None,  # Not implemented yet
            self_fitness=self_fitness_values,
            partner_fitness=partner_fitness_values,
        )

        return safe_evaluate_matchability(
            protocol.matchability,
            context,
            self.rng,
            step_limit=self.erp_config.step_limit,
        )

    def _compute_fitness_ranks(self, population: Population[G]) -> dict[UUID, int]:
        """Compute fitness rankings for all individuals."""
        # Sort by fitness (handle None fitness)
        individuals = list(population.individuals)
        
        def get_fitness_value(ind: Individual[G]) -> float:
            if ind.fitness is None:
                return float('inf') if self.config.minimize else float('-inf')
            if hasattr(ind.fitness, 'values'):
                return float(ind.fitness.values[0])
            return float(ind.fitness.value)

        individuals.sort(key=get_fitness_value, reverse=not self.config.minimize)
        
        return {ind.id: rank for rank, ind in enumerate(individuals)}

    def _compute_population_diversity(self, population: Population[G]) -> float:
        """Compute population diversity metric (0-1)."""
        if len(population) < 2:
            return 0.0

        # Use fitness standard deviation as diversity proxy
        fitness_values = []
        for ind in population.individuals:
            if ind.fitness is not None:
                if hasattr(ind.fitness, 'values'):
                    fitness_values.append(float(ind.fitness.values[0]))
                else:
                    fitness_values.append(float(ind.fitness.value))

        if len(fitness_values) < 2:
            return 0.0

        std = float(np.std(fitness_values))
        mean = float(np.mean(fitness_values))
        
        # Coefficient of variation, clamped to [0, 1]
        if mean == 0:
            return min(1.0, std)
        return min(1.0, std / abs(mean))

    def _create_random_genome(self, rng: Random) -> G:
        """Create a random genome for immigrants."""
        # This is a stub - should be customized by subclasses
        # For now, return None and let caller handle it
        raise NotImplementedError(
            "ERPEngine requires a genome_factory for immigration recovery. "
            "Either disable recovery or provide a genome factory."
        )

    def _emit_event(
        self,
        parent1: Individual[G],
        parent2: Individual[G],
        success: bool,
        failure_reason: str | None,
        matchability_result: tuple[bool, bool],
        intent_result: tuple[bool, bool],
        offspring_ids: tuple[UUID, ...] | None = None,
    ) -> None:
        """Emit a reproduction event for observability."""
        event = ReproductionEvent(
            generation=self._generation,
            parent1_id=parent1.id,
            parent2_id=parent2.id,
            success=success,
            failure_reason=failure_reason,
            offspring_ids=offspring_ids,
            matchability_result=matchability_result,
            intent_result=intent_result,
        )
        self._events.append(event)

        # Notify callbacks
        for cb in self._callbacks:
            if hasattr(cb, 'on_reproduction_event'):
                cb.on_reproduction_event(event)

    @property
    def reproduction_events(self) -> list[ReproductionEvent]:
        """Get reproduction events from current generation."""
        return self._events.copy()

    @property
    def success_rate(self) -> float:
        """Get mating success rate for current generation."""
        if self._attempted_matings == 0:
            return 0.0
        return self._successful_matings / self._attempted_matings

    def _compute_metrics(self, population: Population[G]) -> dict[str, Any]:
        """Compute generation metrics including ERP mating statistics."""
        metrics = super()._compute_metrics(population)
        
        # Add ERP mating statistics
        metrics["attempted_matings"] = self._attempted_matings
        metrics["successful_matings"] = self._successful_matings
        metrics["mating_success_rate"] = self.success_rate
        
        return metrics
