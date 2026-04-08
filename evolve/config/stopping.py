"""
Stopping Criteria Configuration.

Provides configuration for evolution stopping conditions.
Multiple criteria can be specified; evolution stops when ANY is met.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class StoppingConfig:
    """
    Stopping criteria specification.

    Multiple criteria can be specified; evolution stops when ANY is met.
    All fields are optional; omit to disable that criterion.

    Attributes:
        max_generations: Stop after specified number of generations.
        fitness_threshold: Stop when best fitness reaches this value.
        stagnation_generations: Stop after N generations with no improvement.
        time_limit_seconds: Stop after specified wall-clock duration.

    Example:
        >>> config = StoppingConfig(
        ...     max_generations=100,
        ...     fitness_threshold=0.001,
        ...     stagnation_generations=20,
        ... )
        >>> # Evolution stops when ANY condition is met
    """

    max_generations: int | None = None
    """Stop after specified number of generations (FR-009)."""

    fitness_threshold: float | None = None
    """Stop when best fitness reaches this value (FR-010)."""

    stagnation_generations: int | None = None
    """Stop after N generations with no fitness improvement (FR-011)."""

    time_limit_seconds: float | None = None
    """Stop after specified wall-clock duration in seconds (FR-012)."""

    def __post_init__(self) -> None:
        """Validate stopping configuration."""
        if self.max_generations is not None and self.max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if self.stagnation_generations is not None and self.stagnation_generations <= 0:
            raise ValueError("stagnation_generations must be positive")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")

    def is_empty(self) -> bool:
        """Check if no stopping criteria are specified."""
        return all(
            getattr(self, f) is None
            for f in (
                "max_generations",
                "fitness_threshold",
                "stagnation_generations",
                "time_limit_seconds",
            )
        )

    def active_criteria(self) -> Iterator[str]:
        """Yield names of active (non-None) stopping criteria."""
        if self.max_generations is not None:
            yield "max_generations"
        if self.fitness_threshold is not None:
            yield "fitness_threshold"
        if self.stagnation_generations is not None:
            yield "stagnation_generations"
        if self.time_limit_seconds is not None:
            yield "time_limit_seconds"

    def to_dict(self) -> dict[str, int | float | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_generations": self.max_generations,
            "fitness_threshold": self.fitness_threshold,
            "stagnation_generations": self.stagnation_generations,
            "time_limit_seconds": self.time_limit_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int | float | None]) -> StoppingConfig:
        """Create from dictionary."""
        max_gen = data.get("max_generations")
        stag_gen = data.get("stagnation_generations")
        return cls(
            max_generations=int(max_gen) if max_gen is not None else None,
            fitness_threshold=data.get("fitness_threshold"),
            stagnation_generations=int(stag_gen) if stag_gen is not None else None,
            time_limit_seconds=data.get("time_limit_seconds"),
        )
