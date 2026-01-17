"""
Selection operators - Choose individuals for reproduction.

Selection operators MUST:
- Accept explicit RNG for determinism
- Support elitism via separate mechanism
- Handle evaluated populations (individuals with fitness)
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

from evolve.core.population import Population
from evolve.core.types import Individual

G = TypeVar("G")


@runtime_checkable
class SelectionOperator(Protocol[G]):
    """
    Selects individuals from population for reproduction.
    
    Selection operators MUST:
    - Accept explicit RNG for determinism
    - Support elitism via separate preserve_elites() call
    - Handle multi-objective populations (Pareto ranking)
    """

    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random,
    ) -> Sequence[Individual[G]]:
        """
        Select n individuals for reproduction.
        
        Args:
            population: Source population
            n: Number to select (may include duplicates)
            rng: Random number generator
            
        Returns:
            Selected individuals (references, not copies)
        """
        ...


@runtime_checkable
class ElitistSelection(Protocol[G]):
    """Selection with explicit elitism support."""

    def select_with_elites(
        self,
        population: Population[G],
        n_select: int,
        n_elites: int,
        rng: Random,
    ) -> tuple[Sequence[Individual[G]], Sequence[Individual[G]]]:
        """
        Select individuals and preserve elites.
        
        Args:
            population: Source population
            n_select: Number to select for variation
            n_elites: Number of elites to preserve unchanged
            rng: Random number generator
            
        Returns:
            (selected_for_variation, elites_to_preserve)
        """
        ...


@dataclass
class TournamentSelection(Generic[G]):
    """
    Tournament selection with configurable size.
    
    Selects k random individuals, returns best.
    Larger k = higher selection pressure.
    
    Attributes:
        tournament_size: Number of individuals in each tournament (default: 3)
        minimize: If True, lower fitness is better (default: True)
    """

    tournament_size: int = 3
    minimize: bool = True

    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random,
    ) -> Sequence[Individual[G]]:
        """
        Select n individuals via tournament selection.
        
        For each selection:
        1. Pick tournament_size random individuals
        2. Return the one with best fitness
        """
        selected: list[Individual[G]] = []
        individuals = list(population.individuals)
        
        # Filter to evaluated individuals
        evaluated = [ind for ind in individuals if ind.fitness is not None]
        if not evaluated:
            raise ValueError("Cannot select from unevaluated population")
        
        for _ in range(n):
            # Random tournament
            tournament = rng.sample(evaluated, min(self.tournament_size, len(evaluated)))
            
            # Find best in tournament
            if self.minimize:
                winner = min(
                    tournament,
                    key=lambda ind: float(ind.fitness.values[0]) if ind.fitness else float("inf"),
                )
            else:
                winner = max(
                    tournament,
                    key=lambda ind: float(ind.fitness.values[0]) if ind.fitness else float("-inf"),
                )
            
            selected.append(winner)
        
        return selected

    def select_with_elites(
        self,
        population: Population[G],
        n_select: int,
        n_elites: int,
        rng: Random,
    ) -> tuple[Sequence[Individual[G]], Sequence[Individual[G]]]:
        """Select with elitism."""
        elites = list(population.best(n_elites, minimize=self.minimize))
        selected = self.select(population, n_select, rng)
        return selected, elites


@dataclass
class RouletteSelection(Generic[G]):
    """
    Fitness-proportionate selection.
    
    Probability of selection proportional to fitness.
    Only valid for positive fitness values.
    
    Attributes:
        minimize: If True, inverts fitness for selection
    """

    minimize: bool = True

    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random,
    ) -> Sequence[Individual[G]]:
        """
        Select n individuals via roulette wheel.
        
        Probability proportional to fitness (or inverse if minimizing).
        """
        import numpy as np
        
        evaluated = [ind for ind in population.individuals if ind.fitness is not None]
        if not evaluated:
            raise ValueError("Cannot select from unevaluated population")
        
        # Get fitness values
        fitness_vals = np.array([
            float(ind.fitness.values[0]) if ind.fitness else 0.0
            for ind in evaluated
        ])
        
        # Handle minimization by inverting
        if self.minimize:
            # Shift to positive and invert
            max_fit = np.max(fitness_vals)
            fitness_vals = max_fit - fitness_vals + 1e-10
        
        # Normalize to probabilities
        total = np.sum(fitness_vals)
        if total <= 0:
            probs = np.ones(len(evaluated)) / len(evaluated)
        else:
            probs = fitness_vals / total
        
        # Select
        indices = rng.choices(range(len(evaluated)), weights=probs.tolist(), k=n)
        return [evaluated[i] for i in indices]


@dataclass
class RankSelection(Generic[G]):
    """
    Rank-based selection.
    
    Selection probability based on rank, not raw fitness.
    More robust to fitness scaling issues.
    
    Attributes:
        selection_pressure: 1.0 = uniform, 2.0 = strong pressure
        minimize: If True, lower fitness = better rank
    """

    selection_pressure: float = 1.5
    minimize: bool = True

    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random,
    ) -> Sequence[Individual[G]]:
        """
        Select n individuals via rank-based selection.
        
        Better-ranked individuals have higher selection probability.
        """
        evaluated = [ind for ind in population.individuals if ind.fitness is not None]
        if not evaluated:
            raise ValueError("Cannot select from unevaluated population")
        
        # Sort by fitness
        sorted_inds = sorted(
            evaluated,
            key=lambda ind: float(ind.fitness.values[0]) if ind.fitness else float("inf"),
            reverse=not self.minimize,
        )
        
        # Compute rank-based probabilities (linear ranking)
        # P(rank i) = (2 - sp) / N + 2 * (sp - 1) * (N - i) / (N * (N - 1))
        N = len(sorted_inds)
        sp = self.selection_pressure
        
        probs = []
        for i in range(N):
            rank = N - i  # Best has rank N, worst has rank 1
            prob = (2 - sp) / N + 2 * (sp - 1) * (rank - 1) / (N * (N - 1) + 1e-10)
            probs.append(max(0, prob))
        
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Select
        indices = rng.choices(range(N), weights=probs, k=n)
        return [sorted_inds[i] for i in indices]
