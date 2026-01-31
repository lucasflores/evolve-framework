"""
Callbacks for Evolvable Reproduction Protocols.

Provides observability into ERP-specific metrics like:
- Protocol composition distribution
- Mating success rates
- Recovery trigger frequency
- Protocol evolution dynamics

Note: Import this module directly to avoid circular imports:
    from evolve.reproduction.callbacks import ERPMetricsCallback
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.reproduction.protocol import ReproductionEvent, ReproductionProtocol


# Define a minimal callback protocol to avoid importing core.callbacks
@runtime_checkable
class Callback(Protocol):
    """Protocol for evolution callbacks."""

    def on_generation_start(
        self, generation: int, population: Any
    ) -> None:
        """Called at the start of each generation."""
        ...

    def on_generation_end(
        self, generation: int, population: Any, best_fitness: float
    ) -> None:
        """Called at the end of each generation."""
        ...


@dataclass
class ERPMetrics:
    """
    Aggregated ERP metrics for a generation.

    Attributes:
        generation: Generation number
        attempted_matings: Total mating attempts
        successful_matings: Successful matings (offspring produced)
        success_rate: Fraction of successful matings
        intent_rejections: Matings blocked by intent
        matchability_rejections: Matings blocked by matchability
        recovery_triggered: Whether recovery was triggered
        matchability_distribution: Count of each matchability type
        intent_distribution: Count of each intent type
        crossover_distribution: Count of each crossover type
    """

    generation: int = 0
    attempted_matings: int = 0
    successful_matings: int = 0
    success_rate: float = 0.0
    intent_rejections: int = 0
    matchability_rejections: int = 0
    recovery_triggered: bool = False
    matchability_distribution: dict[str, int] = field(default_factory=dict)
    intent_distribution: dict[str, int] = field(default_factory=dict)
    crossover_distribution: dict[str, int] = field(default_factory=dict)


class ERPMetricsCallback(Callback):
    """
    Callback that tracks ERP-specific metrics.

    Records protocol composition, mating success rates, and
    recovery events for each generation.

    Example:
        >>> from evolve.reproduction.callbacks import ERPMetricsCallback
        >>> callback = ERPMetricsCallback()
        >>> engine = ERPEngine(..., callbacks=[callback])
        >>> result = engine.run(population)
        >>> print(callback.history[-1].success_rate)
    """

    def __init__(self) -> None:
        """Initialize the metrics callback."""
        self.history: list[ERPMetrics] = []
        self._current_metrics: ERPMetrics = ERPMetrics()
        self._events: list[ReproductionEvent] = []

    def on_generation_start(
        self, generation: int, population: Population[Any]
    ) -> None:
        """Reset metrics at the start of each generation."""
        self._current_metrics = ERPMetrics(generation=generation)
        self._events = []

        # Count protocol distribution
        self._count_protocol_distribution(population)

    def on_generation_end(
        self, generation: int, population: Population[Any], best_fitness: float
    ) -> None:
        """Finalize and store metrics at generation end."""
        self._compute_event_metrics()
        self.history.append(self._current_metrics)

    def on_reproduction_event(self, event: ReproductionEvent) -> None:
        """
        Record a reproduction event.

        Called by ERPEngine for each mating attempt.
        """
        self._events.append(event)

    def on_recovery_triggered(self, recovery_type: str) -> None:
        """Record that recovery was triggered."""
        self._current_metrics.recovery_triggered = True

    def _count_protocol_distribution(self, population: Population[Any]) -> None:
        """Count protocol types in the population."""
        matchability_counts: dict[str, int] = defaultdict(int)
        intent_counts: dict[str, int] = defaultdict(int)
        crossover_counts: dict[str, int] = defaultdict(int)

        for individual in population.individuals:
            if individual.protocol is not None:
                protocol = individual.protocol
                matchability_counts[protocol.matchability.type] += 1
                intent_counts[protocol.intent.type] += 1
                crossover_counts[protocol.crossover.type.value] += 1

        self._current_metrics.matchability_distribution = dict(matchability_counts)
        self._current_metrics.intent_distribution = dict(intent_counts)
        self._current_metrics.crossover_distribution = dict(crossover_counts)

    def _compute_event_metrics(self) -> None:
        """Compute aggregate metrics from reproduction events."""
        if not self._events:
            return

        attempted = len(self._events)
        successful = sum(1 for e in self._events if e.success)
        intent_blocked = 0
        matchability_blocked = 0

        for event in self._events:
            if not event.success:
                # Check what blocked the mating
                if event.intent_result and not all(event.intent_result):
                    intent_blocked += 1
                elif event.matchability_result and not all(event.matchability_result):
                    matchability_blocked += 1

        self._current_metrics.attempted_matings = attempted
        self._current_metrics.successful_matings = successful
        self._current_metrics.success_rate = successful / attempted if attempted > 0 else 0.0
        self._current_metrics.intent_rejections = intent_blocked
        self._current_metrics.matchability_rejections = matchability_blocked

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics across all generations.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.history:
            return {"generations": 0}

        success_rates = [m.success_rate for m in self.history]
        recovery_count = sum(1 for m in self.history if m.recovery_triggered)

        return {
            "generations": len(self.history),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "min_success_rate": min(success_rates),
            "max_success_rate": max(success_rates),
            "recovery_triggers": recovery_count,
            "total_attempted": sum(m.attempted_matings for m in self.history),
            "total_successful": sum(m.successful_matings for m in self.history),
        }

    def get_protocol_evolution(self) -> dict[str, list[dict[str, int]]]:
        """
        Get protocol distribution over time.

        Returns:
            Dictionary mapping protocol component to list of distributions
        """
        return {
            "matchability": [m.matchability_distribution for m in self.history],
            "intent": [m.intent_distribution for m in self.history],
            "crossover": [m.crossover_distribution for m in self.history],
        }


class ERPLoggerCallback(Callback):
    """
    Callback that logs ERP events to stdout or a file.

    Example:
        >>> callback = ERPLoggerCallback(verbose=True)
        >>> engine = ERPEngine(..., callbacks=[callback])
    """

    def __init__(self, verbose: bool = False, log_every: int = 10) -> None:
        """
        Initialize the logger callback.

        Args:
            verbose: If True, log every event. If False, only log summaries.
            log_every: Log summary every N generations.
        """
        self.verbose = verbose
        self.log_every = log_every
        self._attempted = 0
        self._successful = 0

    def on_generation_start(
        self, generation: int, population: Population[Any]
    ) -> None:
        """Reset counters at generation start."""
        self._attempted = 0
        self._successful = 0

    def on_generation_end(
        self, generation: int, population: Population[Any], best_fitness: float
    ) -> None:
        """Log summary at generation end."""
        if generation % self.log_every == 0:
            rate = self._successful / self._attempted if self._attempted > 0 else 0.0
            print(
                f"[ERP] Gen {generation}: "
                f"mating success {self._successful}/{self._attempted} ({rate:.1%})"
            )

    def on_reproduction_event(self, event: ReproductionEvent) -> None:
        """Log reproduction event."""
        self._attempted += 1
        if event.success:
            self._successful += 1

        if self.verbose and not event.success:
            print(
                f"  Mating failed: {event.failure_reason} "
                f"(intent={event.intent_result}, match={event.matchability_result})"
            )

    def on_recovery_triggered(self, recovery_type: str) -> None:
        """Log recovery event."""
        print(f"[ERP] Recovery triggered: {recovery_type}")
