"""
Recovery strategies for Evolvable Reproduction Protocols.

When ERP causes reproduction rates to drop (e.g., highly selective
matchability functions), recovery mechanisms kick in to ensure
population survival. This module implements various recovery strategies.

Key Components:
- RecoveryStrategy: Protocol interface for recovery mechanisms
- ImmigrationRecovery: Injects random individuals when population stalls
- MutationBoostRecovery: Temporarily increases mutation rates
- RelaxedMatchingRecovery: Temporarily relaxes matchability thresholds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from evolve.reproduction.protocol import ReproductionProtocol

G = TypeVar("G")


# =============================================================================
# Recovery Strategy Interface
# =============================================================================


@runtime_checkable
class RecoveryStrategy(Protocol[G]):
    """
    Protocol interface for recovery strategies.

    Recovery strategies are invoked when the ERP engine detects that
    reproduction rates have dropped below a threshold.
    """

    def should_trigger(
        self,
        successful_matings: int,
        attempted_matings: int,
        population_size: int,
        generation: int,
    ) -> bool:
        """
        Check if recovery should be triggered.

        Args:
            successful_matings: Number of successful matings this generation
            attempted_matings: Total mating attempts
            population_size: Current population size
            generation: Current generation number

        Returns:
            True if recovery should be triggered
        """
        ...

    def recover(
        self,
        population: list[Any],
        genome_factory: Callable[[Random], G],
        protocol_factory: Callable[[Random], ReproductionProtocol],
        rng: Random,
    ) -> list[Any]:
        """
        Apply recovery mechanism to population.

        Args:
            population: Current population (list of individuals)
            genome_factory: Function to create new random genomes
            protocol_factory: Function to create new random protocols
            rng: Random number generator

        Returns:
            Modified population with recovery applied
        """
        ...


# =============================================================================
# Immigration Recovery
# =============================================================================


@dataclass
class ImmigrationRecovery:
    """
    Recovery strategy that injects random immigrants.

    When triggered, replaces a portion of the population with new
    random individuals. These immigrants have fresh protocols that
    may be more permissive.

    Attributes:
        trigger_threshold: Min success rate to avoid triggering (0-1)
        immigration_rate: Fraction of population to replace (0-1)
        min_generations: Minimum generations before recovery can trigger
        cooldown_generations: Generations to wait between triggers
    """

    trigger_threshold: float = 0.1
    immigration_rate: float = 0.1
    min_generations: int = 5
    cooldown_generations: int = 3
    _last_trigger: int = field(default=-100, init=False)

    def should_trigger(
        self,
        successful_matings: int,
        attempted_matings: int,
        population_size: int,
        generation: int,
    ) -> bool:
        """Check if immigration should be triggered."""
        if generation < self.min_generations:
            return False

        if generation - self._last_trigger < self.cooldown_generations:
            return False

        if attempted_matings == 0:
            return True

        success_rate = successful_matings / attempted_matings
        return success_rate < self.trigger_threshold

    def recover(
        self,
        population: list[Any],
        genome_factory: Callable[[Random], G],
        protocol_factory: Callable[[Random], ReproductionProtocol],
        rng: Random,
        generation: int = 0,
    ) -> list[Any]:
        """
        Add immigrants to the population.

        Args:
            population: Current population
            genome_factory: Creates new random genomes
            protocol_factory: Creates new random protocols
            rng: Random number generator
            generation: Current generation (for cooldown tracking)

        Returns:
            Population with immigrants added
        """
        self._last_trigger = generation

        num_immigrants = max(1, int(len(population) * self.immigration_rate))

        # Remove worst individuals (assuming sorted by fitness)
        survivors = population[:-num_immigrants] if num_immigrants < len(population) else []

        # Create immigrants
        immigrants = []
        for _ in range(num_immigrants):
            genome = genome_factory(rng)
            protocol = protocol_factory(rng)
            # Note: Caller must wrap these in Individual objects
            immigrants.append((genome, protocol))

        return (survivors, immigrants)


# =============================================================================
# Mutation Boost Recovery
# =============================================================================


@dataclass
class MutationBoostRecovery:
    """
    Recovery strategy that boosts mutation rates temporarily.

    When triggered, returns a multiplier for mutation rates that
    should be applied for the next few generations.

    Attributes:
        trigger_threshold: Min success rate to avoid triggering
        boost_multiplier: How much to multiply mutation rates
        boost_duration: Generations to maintain boost
    """

    trigger_threshold: float = 0.1
    boost_multiplier: float = 3.0
    boost_duration: int = 5
    _boost_remaining: int = field(default=0, init=False)

    def should_trigger(
        self,
        successful_matings: int,
        attempted_matings: int,
        population_size: int,
        generation: int,
    ) -> bool:
        """Check if mutation boost should be triggered."""
        if self._boost_remaining > 0:
            return False  # Already boosted

        if attempted_matings == 0:
            return True

        success_rate = successful_matings / attempted_matings
        return success_rate < self.trigger_threshold

    def recover(
        self,
        population: list[Any],
        genome_factory: Callable[[Random], G],
        protocol_factory: Callable[[Random], ReproductionProtocol],
        rng: Random,
    ) -> list[Any]:
        """
        Activate mutation boost.

        Returns the population unchanged but activates internal boost state.
        """
        self._boost_remaining = self.boost_duration
        return population

    def get_mutation_multiplier(self) -> float:
        """
        Get current mutation rate multiplier.

        Returns:
            Multiplier to apply to mutation rates (1.0 if no boost)
        """
        if self._boost_remaining > 0:
            self._boost_remaining -= 1
            return self.boost_multiplier
        return 1.0


# =============================================================================
# Relaxed Matching Recovery
# =============================================================================


@dataclass
class RelaxedMatchingRecovery:
    """
    Recovery strategy that temporarily relaxes matchability.

    When triggered, signals that matchability evaluation should
    use more permissive fallbacks for a period.

    Attributes:
        trigger_threshold: Min success rate to avoid triggering
        relaxation_duration: Generations to maintain relaxation
    """

    trigger_threshold: float = 0.1
    relaxation_duration: int = 3
    _relaxation_remaining: int = field(default=0, init=False)

    def should_trigger(
        self,
        successful_matings: int,
        attempted_matings: int,
        population_size: int,
        generation: int,
    ) -> bool:
        """Check if matching relaxation should be triggered."""
        if self._relaxation_remaining > 0:
            return False

        if attempted_matings == 0:
            return True

        success_rate = successful_matings / attempted_matings
        return success_rate < self.trigger_threshold

    def recover(
        self,
        population: list[Any],
        genome_factory: Callable[[Random], G],
        protocol_factory: Callable[[Random], ReproductionProtocol],
        rng: Random,
    ) -> list[Any]:
        """Activate relaxation mode."""
        self._relaxation_remaining = self.relaxation_duration
        return population

    def is_relaxed(self) -> bool:
        """
        Check if matching should be relaxed.

        Returns:
            True if currently in relaxation mode
        """
        if self._relaxation_remaining > 0:
            self._relaxation_remaining -= 1
            return True
        return False


# =============================================================================
# Composite Recovery
# =============================================================================


@dataclass
class CompositeRecovery:
    """
    Combines multiple recovery strategies.

    Applies strategies in order until one triggers. Only one
    strategy triggers per generation.

    Attributes:
        strategies: List of recovery strategies to try
    """

    strategies: list[RecoveryStrategy] = field(default_factory=list)

    def should_trigger(
        self,
        successful_matings: int,
        attempted_matings: int,
        population_size: int,
        generation: int,
    ) -> bool:
        """Check if any strategy should trigger."""
        for strategy in self.strategies:
            if strategy.should_trigger(
                successful_matings, attempted_matings, population_size, generation
            ):
                return True
        return False

    def recover(
        self,
        population: list[Any],
        genome_factory: Callable[[Random], G],
        protocol_factory: Callable[[Random], ReproductionProtocol],
        rng: Random,
        successful_matings: int = 0,
        attempted_matings: int = 0,
        population_size: int = 0,
        generation: int = 0,
    ) -> list[Any]:
        """Apply first triggered strategy."""
        for strategy in self.strategies:
            if strategy.should_trigger(
                successful_matings, attempted_matings, population_size, generation
            ):
                return strategy.recover(population, genome_factory, protocol_factory, rng)
        return population
