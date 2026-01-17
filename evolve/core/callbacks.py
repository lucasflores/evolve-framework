"""
Callback protocols - Event hooks for evolution monitoring.

Callbacks allow users to:
- Monitor evolution progress
- Log metrics
- Implement custom stopping conditions
- Visualize population dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from evolve.core.population import Population

G = TypeVar("G")


@runtime_checkable
class Callback(Protocol[G]):
    """
    Event hook for evolution monitoring.
    
    Callbacks are invoked at key points during evolution:
    - on_generation_start: Before evaluation
    - on_generation_end: After selection/variation
    - on_run_start: Before first generation
    - on_run_end: After termination
    
    Callbacks can access but should not modify the population.
    """

    def on_generation_start(
        self,
        generation: int,
        population: Population[G],
    ) -> None:
        """Called before generation evaluation."""
        ...

    def on_generation_end(
        self,
        generation: int,
        population: Population[G],
        metrics: dict[str, Any],
    ) -> None:
        """
        Called after generation completes.
        
        Args:
            generation: Current generation number
            population: Population after selection/variation
            metrics: Dict with 'best_fitness', 'mean_fitness', etc.
        """
        ...

    def on_run_start(self, config: Any) -> None:
        """Called before evolution begins."""
        ...

    def on_run_end(
        self,
        population: Population[G],
        reason: str,
    ) -> None:
        """
        Called when evolution terminates.
        
        Args:
            population: Final population
            reason: Why evolution stopped (e.g., "max_generations", "converged")
        """
        ...


@runtime_checkable
class StoppingCriterion(Protocol):
    """
    Determines when evolution should terminate.
    
    Stopping criteria are checked after each generation.
    Multiple criteria can be combined (any triggers stop).
    """

    def should_stop(
        self,
        generation: int,
        population: "Population[Any]",
        history: list[dict[str, Any]],
    ) -> bool:
        """
        Check if evolution should stop.
        
        Args:
            generation: Current generation number
            population: Current population
            history: List of metrics from previous generations
            
        Returns:
            True if evolution should terminate
        """
        ...

    @property
    def reason(self) -> str:
        """Human-readable reason for stopping."""
        ...


@dataclass
class SimpleCallback:
    """
    Base callback with no-op implementations.
    
    Subclass and override only the methods you need.
    """

    def on_generation_start(
        self,
        generation: int,
        population: Population[Any],
    ) -> None:
        """No-op by default."""
        pass

    def on_generation_end(
        self,
        generation: int,
        population: Population[Any],
        metrics: dict[str, Any],
    ) -> None:
        """No-op by default."""
        pass

    def on_run_start(self, config: Any) -> None:
        """No-op by default."""
        pass

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """No-op by default."""
        pass


@dataclass
class PrintCallback:
    """
    Simple callback that prints progress.
    
    Attributes:
        print_every: Print every N generations (default: 1)
        show_best: Also print best fitness (default: True)
    """

    print_every: int = 1
    show_best: bool = True

    def on_generation_start(
        self,
        generation: int,
        population: Population[Any],
    ) -> None:
        """No-op."""
        pass

    def on_generation_end(
        self,
        generation: int,
        population: Population[Any],
        metrics: dict[str, Any],
    ) -> None:
        """Print generation summary."""
        if generation % self.print_every == 0:
            msg = f"Generation {generation}"
            if self.show_best and "best_fitness" in metrics:
                msg += f" | Best: {metrics['best_fitness']:.6f}"
            if "mean_fitness" in metrics:
                msg += f" | Mean: {metrics['mean_fitness']:.6f}"
            print(msg)

    def on_run_start(self, config: Any) -> None:
        """Print start message."""
        print("Evolution started")

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """Print termination message."""
        print(f"Evolution finished: {reason}")
        stats = population.statistics
        if stats.best_fitness is not None:
            print(f"Final best fitness: {stats.best_fitness.values[0]:.6f}")


@dataclass
class HistoryCallback:
    """
    Callback that records evolution history.
    
    Records metrics from each generation for later analysis.
    
    Attributes:
        history: List of generation metrics (populated during run)
    """

    def __post_init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.config: Any = None

    def on_generation_start(
        self,
        generation: int,
        population: Population[Any],
    ) -> None:
        """No-op."""
        pass

    def on_generation_end(
        self,
        generation: int,
        population: Population[Any],
        metrics: dict[str, Any],
    ) -> None:
        """Record generation metrics."""
        record = {
            "generation": generation,
            "population_size": len(population),
            **metrics,
        }
        self.history.append(record)

    def on_run_start(self, config: Any) -> None:
        """Store config and clear history."""
        self.config = config
        self.history = []

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """Record final state."""
        pass

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Convert history to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history)
