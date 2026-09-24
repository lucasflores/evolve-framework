"""
Selection operators - Choose individuals for reproduction.

Registry names (for ``UnifiedConfig(selection=...)``)::

    "tournament", "roulette", "rank", "crowded_tournament"

``"crowded_tournament"`` (evolve.multiobjective.selection) is for
multi-objective configs only. Elitism is not a selection operator; it is
set with ``UnifiedConfig.elitism``.

Selection operators MUST:
- Accept explicit RNG for determinism
- Support elitism via separate mechanism
- Handle evaluated populations (individuals with fitness)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from evolve.core.population import Population
from evolve.core.types import Individual, fitness_sort_key

G = TypeVar("G")


def check_selection_direction(selection: Any, minimize: bool) -> None:
    """
    Refuse a selection operator that ranks in the other direction from its engine.

    Operators with a boolean ``minimize`` (tournament, rank, roulette) must
    agree with the engine's ``minimize``, or selection would favour the
    individuals elitism discards. Operators without one are accepted.
    Every engine calls this on construction (single-objective mode).

    Raises:
        ValueError: If ``selection.minimize`` differs from ``minimize``.
    """
    selection_minimize = getattr(selection, "minimize", None)
    if isinstance(selection_minimize, bool) and selection_minimize != minimize:
        raise ValueError(
            f"{type(selection).__name__} has minimize={selection_minimize} but the engine "
            f"has minimize={minimize}; selection and elitism must rank in one direction."
        )


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

    "Best" is feasibility-first (Deb's rules): feasible beats infeasible;
    between infeasible, lower total constraint violation wins; otherwise
    ``values[0]`` decides.

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

            # Find best in tournament (feasibility first)
            winner = min(tournament, key=lambda ind: fitness_sort_key(ind.fitness, self.minimize))

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

    Probability of selection proportional to fitness, shifted so the worst
    individual gets (almost) zero weight: ``f - min(f)`` when maximizing,
    ``max(f) - f`` when minimizing. Any sign of fitness is valid.

    Constraints are ignored: probabilities come from raw ``values[0]`` only.
    Feasibility-first ranking has no proportional analogue, and giving
    infeasible individuals zero weight would collapse the wheel onto the
    few feasible ones early in a run. Elitism (``Population.best``) still
    ranks feasibility first; use tournament or rank selection when
    constraints should steer parent choice.

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

        Probability proportional to the shifted fitness (see class docstring).
        """
        import numpy as np

        evaluated = [ind for ind in population.individuals if ind.fitness is not None]
        if not evaluated:
            raise ValueError("Cannot select from unevaluated population")

        # Get fitness values
        fitness_vals = np.array(
            [float(ind.fitness.values[0]) if ind.fitness else 0.0 for ind in evaluated]
        )

        # Shift weights positive in both directions (worst gets ~0), so zero,
        # negative and mixed-sign fitness make a valid wheel
        if self.minimize:
            fitness_vals = np.max(fitness_vals) - fitness_vals + 1e-10
        else:
            fitness_vals = fitness_vals - np.min(fitness_vals) + 1e-10

        # Normalize to probabilities
        total = np.sum(fitness_vals)
        probs = np.ones(len(evaluated)) / len(evaluated) if total <= 0 else fitness_vals / total

        # Select
        indices = rng.choices(range(len(evaluated)), weights=probs.tolist(), k=n)
        return [evaluated[i] for i in indices]


@dataclass
class RankSelection(Generic[G]):
    """
    Rank-based selection.

    Selection probability based on rank, not raw fitness.
    More robust to fitness scaling issues. Ranks are feasibility-first
    (Deb's rules), as in tournament selection.

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

        # Sort by fitness (feasibility first)
        sorted_inds = sorted(
            evaluated,
            key=lambda ind: fitness_sort_key(ind.fitness, self.minimize),
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
