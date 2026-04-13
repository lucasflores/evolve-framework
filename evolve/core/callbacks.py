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
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from evolve.core.population import Population

if TYPE_CHECKING:
    import pandas as pd

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

    @property
    def priority(self) -> int:
        """Execution priority (lower runs first, default 0)."""
        ...

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
        population: Population[Any],
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

    @property
    def priority(self) -> int:
        """Default priority (0 = highest)."""
        return 0

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

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0

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
        _population: Population[Any],
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

    def on_run_start(self, _config: Any) -> None:
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

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0

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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.history)


@dataclass
class LoggingCallback:
    """
    Callback that logs evolution progress using Python's logging module.

    Supports configurable log levels and destinations.

    Attributes:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_destination: Where to log - 'console', 'file', or path to log file
        logger: The configured logger instance
    """

    log_level: str = "INFO"
    log_destination: str = "console"

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0

    def __post_init__(self) -> None:
        import logging

        self.logger = logging.getLogger("evolve.evolution")
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Configure handler based on destination
        if self.log_destination == "console":
            handler: logging.Handler = logging.StreamHandler()
        else:
            # Treat as file path
            import os

            log_dir = os.path.dirname(self.log_destination)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(self.log_destination)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        self.logger.addHandler(handler)

    def on_generation_start(
        self,
        generation: int,
        population: Population[Any],
    ) -> None:
        """Log generation start."""
        self.logger.debug(f"Generation {generation} starting with {len(population)} individuals")

    def on_generation_end(
        self,
        generation: int,
        _population: Population[Any],
        metrics: dict[str, Any],
    ) -> None:
        """Log generation metrics."""
        best = metrics.get("best_fitness", "N/A")
        mean = metrics.get("mean_fitness", "N/A")
        self.logger.info(f"Generation {generation} | Best: {best} | Mean: {mean}")

    def on_run_start(self, _config: Any) -> None:
        """Log evolution start."""
        self.logger.info("Evolution run started")

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """Log evolution termination."""
        self.logger.info(f"Evolution terminated: {reason}")
        stats = population.statistics
        if stats.best_fitness is not None:
            self.logger.info(f"Final best fitness: {stats.best_fitness.values[0]}")


@dataclass
class CheckpointCallback:
    """
    Callback that saves periodic checkpoints during evolution.

    Checkpoints include population state and metrics, allowing
    evolution to be resumed from intermediate states.

    Attributes:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_frequency: Save every N generations
    """

    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 10

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0

    def __post_init__(self) -> None:
        import os

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._run_id: str | None = None

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
        """Save checkpoint if frequency matches."""
        if self.checkpoint_frequency <= 0:
            return

        if generation > 0 and generation % self.checkpoint_frequency == 0:
            self._save_checkpoint(generation, population, metrics)

    def on_run_start(self, _config: Any) -> None:
        """Initialize run ID for checkpoint naming."""
        import time

        self._run_id = f"run_{int(time.time())}"

    def on_run_end(
        self,
        population: Population[Any],
        reason: str,
    ) -> None:
        """Save final checkpoint."""
        self._save_checkpoint("final", population, {"termination_reason": reason})

    def _save_checkpoint(
        self,
        generation: int | str,
        population: Population[Any],
        metrics: dict[str, Any],
    ) -> None:
        """Save checkpoint to file."""
        import json
        import os

        checkpoint_data = {
            "generation": generation,
            "population_size": len(population),
            "metrics": {
                k: float(v) if isinstance(v, int | float) else str(v) for k, v in metrics.items()
            },
            "run_id": self._run_id,
        }

        filename = f"checkpoint_{self._run_id}_gen{generation}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
