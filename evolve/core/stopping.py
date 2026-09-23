"""
Stopping criteria - Determine when evolution should terminate.

Multiple stopping criteria can be combined (any triggers stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from evolve.core.population import Population
from evolve.core.types import fitness_sort_key


@dataclass
class GenerationLimitStopping:
    """
    Stop after a fixed number of generations.

    Attributes:
        max_generations: Maximum generations to run
    """

    max_generations: int = 100
    _reason: str = field(default="", init=False)

    def should_stop(
        self,
        generation: int,
        _population: Population[Any],
        _history: list[dict[str, Any]],
    ) -> bool:
        """Check if generation limit reached."""
        if generation >= self.max_generations:
            self._reason = f"Reached {self.max_generations} generations"
            return True
        return False

    @property
    def reason(self) -> str:
        """Reason for stopping."""
        return self._reason or f"Generation limit ({self.max_generations})"


@dataclass
class FitnessThresholdStopping:
    """
    Stop when fitness reaches a target threshold.

    Only a feasible fitness can meet the threshold.

    Attributes:
        threshold: Target fitness value
        minimize: If True, stop when fitness <= threshold
    """

    threshold: float
    minimize: bool = True
    _reason: str = field(default="", init=False)

    def should_stop(
        self,
        _generation: int,
        population: Population[Any],
        _history: list[dict[str, Any]],
    ) -> bool:
        """Check if fitness threshold reached."""
        stats = population.statistics
        # Best is feasibility-first, so an infeasible best means nothing is feasible
        if stats.best_fitness is None or not stats.best_fitness.is_feasible:
            return False

        best = float(stats.best_fitness.values[0])

        if self.minimize:
            if best <= self.threshold:
                self._reason = f"Fitness {best:.6f} <= threshold {self.threshold}"
                return True
        else:
            if best >= self.threshold:
                self._reason = f"Fitness {best:.6f} >= threshold {self.threshold}"
                return True

        return False

    @property
    def reason(self) -> str:
        """Reason for stopping."""
        return self._reason or f"Fitness threshold ({self.threshold})"


@dataclass
class StagnationStopping:
    """
    Stop when fitness doesn't improve for N generations.

    Detects convergence/stagnation. Progress is measured on the best
    individual's feasibility-first key: becoming feasible, or reducing the
    total constraint violation, is an improvement even when the raw value
    gets worse.

    Attributes:
        patience: Number of generations without improvement
        min_delta: Minimum improvement to count as progress (on the value, or
            on the violation while infeasible)
        minimize: If True, improvement means fitness decreased
    """

    patience: int = 20
    min_delta: float = 1e-6
    minimize: bool = True
    _reason: str = field(default="", init=False)
    _best_key: tuple[int, float, float] | None = field(default=None, init=False)
    _stagnant_gens: int = field(default=0, init=False)

    def should_stop(
        self,
        _generation: int,
        population: Population[Any],
        _history: list[dict[str, Any]],
    ) -> bool:
        """Check if evolution has stagnated."""
        stats = population.statistics
        if stats.best_fitness is None:
            return False

        # Compare feasibility-first keys, so feasibility arriving (or less
        # violation) counts as progress even if the raw value gets worse
        current = fitness_sort_key(stats.best_fitness, self.minimize)
        if self._best_key is None:
            improved = True
        else:
            infeasible, violation, value = self._best_key
            threshold = (
                infeasible,
                violation - self.min_delta if infeasible else violation,
                value - self.min_delta,
            )
            improved = current < threshold

        if improved:
            self._best_key = current
            self._stagnant_gens = 0
        else:
            self._stagnant_gens += 1

        if self._stagnant_gens >= self.patience:
            self._reason = f"No improvement for {self.patience} generations"
            return True

        return False

    @property
    def reason(self) -> str:
        """Reason for stopping."""
        return self._reason or f"Stagnation ({self.patience} generations)"

    def reset(self) -> None:
        """Reset stagnation counter for new run."""
        self._best_key = None
        self._stagnant_gens = 0


@dataclass
class TimeLimitStopping:
    """
    Stop after a time limit.

    Attributes:
        max_seconds: Maximum runtime in seconds
    """

    max_seconds: float
    _start_time: float | None = field(default=None, init=False)
    _reason: str = field(default="", init=False)

    def should_stop(
        self,
        _generation: int,
        _population: Population[Any],
        _history: list[dict[str, Any]],
    ) -> bool:
        """Check if time limit exceeded."""
        import time

        if self._start_time is None:
            self._start_time = time.time()

        elapsed = time.time() - self._start_time
        if elapsed >= self.max_seconds:
            self._reason = f"Time limit ({self.max_seconds}s) exceeded"
            return True

        return False

    @property
    def reason(self) -> str:
        """Reason for stopping."""
        return self._reason or f"Time limit ({self.max_seconds}s)"

    def reset(self) -> None:
        """Reset timer for new run."""
        self._start_time = None


@dataclass
class CompositeStoppingCriterion:
    """
    Combines multiple stopping criteria (any triggers stop).

    Attributes:
        criteria: List of stopping criteria
    """

    criteria: list[Any] = field(default_factory=list)
    _triggered: Any = field(default=None, init=False)

    def should_stop(
        self,
        generation: int,
        population: Population[Any],
        history: list[dict[str, Any]],
    ) -> bool:
        """Check all criteria."""
        for criterion in self.criteria:
            if criterion.should_stop(generation, population, history):
                self._triggered = criterion
                return True
        return False

    @property
    def reason(self) -> str:
        """Reason from triggered criterion."""
        if self._triggered is not None:
            return cast(str, self._triggered.reason)
        return "No criterion triggered"

    def add(self, criterion: Any) -> CompositeStoppingCriterion:
        """Add a criterion and return self."""
        self.criteria.append(criterion)
        return self
