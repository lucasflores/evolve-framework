"""
Meta-Evolution Configuration.

Provides configuration for meta-evolution (hyperparameter optimization),
including parameter specifications and outer loop settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for an evolvable parameter in meta-evolution.

    Defines how a configuration parameter should be encoded into
    a vector genome dimension for meta-evolution.

    Attributes:
        path: Dot-notation path to parameter (e.g., 'mutation_params.sigma').
        param_type: Type of parameter for encoding strategy.
        bounds: Min/max bounds for continuous/integer parameters.
        choices: Valid choices for categorical parameters.
        log_scale: Whether to use logarithmic scaling.

    Example:
        >>> # Continuous parameter
        >>> param1 = ParameterSpec(
        ...     path="mutation_rate",
        ...     param_type="continuous",
        ...     bounds=(0.01, 0.3),
        ... )
        >>> # Categorical parameter
        >>> param2 = ParameterSpec(
        ...     path="selection",
        ...     param_type="categorical",
        ...     choices=("tournament", "roulette", "rank"),
        ... )
    """

    path: str
    """Dot-notation path to parameter (e.g., 'mutation_params.sigma')."""

    param_type: Literal["continuous", "integer", "categorical"] = "continuous"
    """Type of parameter for encoding strategy."""

    bounds: tuple[float, float] | None = None
    """Min/max bounds for continuous/integer parameters."""

    choices: tuple[Any, ...] | None = None
    """Valid choices for categorical parameters."""

    log_scale: bool = False
    """Whether to use logarithmic scaling for continuous parameters."""

    def __post_init__(self) -> None:
        """Validate parameter specification."""
        if not self.path:
            raise ValueError("Parameter path cannot be empty")

        if self.param_type in ("continuous", "integer"):
            if self.bounds is None:
                raise ValueError(f"bounds required for {self.param_type} parameter")
            if len(self.bounds) != 2:
                raise ValueError("bounds must be a tuple of (min, max)")
            if self.bounds[0] > self.bounds[1]:
                raise ValueError("bounds[0] must be <= bounds[1]")
            if self.param_type == "continuous" and self.log_scale:
                if self.bounds[0] <= 0:
                    raise ValueError("log_scale requires positive lower bound")
        elif self.param_type == "categorical":
            if self.choices is None or len(self.choices) == 0:
                raise ValueError("choices required for categorical parameter")
        else:
            raise ValueError(
                f"param_type must be 'continuous', 'integer', or 'categorical', "
                f"got {self.param_type}"
            )

    @property
    def num_dimensions(self) -> int:
        """Get number of genome dimensions needed for this parameter."""
        # All parameter types use 1 dimension
        return 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "path": self.path,
            "param_type": self.param_type,
        }
        if self.bounds is not None:
            result["bounds"] = list(self.bounds)
        if self.choices is not None:
            result["choices"] = list(self.choices)
        if self.log_scale:
            result["log_scale"] = self.log_scale
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSpec:
        """Create from dictionary."""
        bounds = data.get("bounds")
        choices = data.get("choices")
        return cls(
            path=data["path"],
            param_type=data.get("param_type", "continuous"),
            bounds=tuple(bounds) if bounds else None,
            choices=tuple(choices) if choices else None,
            log_scale=data.get("log_scale", False),
        )


@dataclass(frozen=True)
class MetaEvolutionConfig:
    """
    Configuration for meta-evolution (hyperparameter optimization).

    When present in UnifiedConfig, enables meta-evolution outer loop
    that evolves configuration parameters.

    Attributes:
        evolvable_params: Parameters to evolve with their bounds.
        outer_population_size: Population size for outer evolutionary loop.
        outer_generations: Number of generations for outer loop.
        trials_per_config: Number of inner runs per configuration.
        aggregation: How to aggregate fitness across trials.
        inner_generations: Override inner loop generations for speed.

    Example:
        >>> config = MetaEvolutionConfig(
        ...     evolvable_params=(
        ...         ParameterSpec(path="mutation_rate", bounds=(0.01, 0.3)),
        ...         ParameterSpec(path="population_size", param_type="integer", bounds=(50, 500)),
        ...     ),
        ...     outer_population_size=20,
        ...     outer_generations=10,
        ...     trials_per_config=3,
        ... )
    """

    evolvable_params: tuple[ParameterSpec, ...] = ()
    """Parameters to evolve with their bounds."""

    outer_population_size: int = 20
    """Population size for outer evolutionary loop."""

    outer_generations: int = 10
    """Number of generations for outer loop."""

    trials_per_config: int = 1
    """Number of inner runs per configuration for robustness."""

    aggregation: Literal["mean", "median", "best"] = "mean"
    """How to aggregate fitness across trials."""

    inner_generations: int | None = None
    """Override inner loop generations for speed (None = use config's max_generations)."""

    def __post_init__(self) -> None:
        """Validate meta-evolution configuration."""
        if len(self.evolvable_params) == 0:
            raise ValueError("At least one evolvable parameter required")
        if self.outer_population_size <= 0:
            raise ValueError("outer_population_size must be positive")
        if self.outer_generations <= 0:
            raise ValueError("outer_generations must be positive")
        if self.trials_per_config < 1:
            raise ValueError("trials_per_config must be at least 1")
        if self.aggregation not in ("mean", "median", "best"):
            raise ValueError(
                f"aggregation must be 'mean', 'median', or 'best', got {self.aggregation}"
            )
        if self.inner_generations is not None and self.inner_generations <= 0:
            raise ValueError("inner_generations must be positive when specified")

    @property
    def num_dimensions(self) -> int:
        """Get total genome dimensions for all evolvable parameters."""
        return sum(p.num_dimensions for p in self.evolvable_params)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evolvable_params": [p.to_dict() for p in self.evolvable_params],
            "outer_population_size": self.outer_population_size,
            "outer_generations": self.outer_generations,
            "trials_per_config": self.trials_per_config,
            "aggregation": self.aggregation,
            "inner_generations": self.inner_generations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaEvolutionConfig:
        """Create from dictionary."""
        params = tuple(ParameterSpec.from_dict(p) for p in data.get("evolvable_params", []))
        return cls(
            evolvable_params=params,
            outer_population_size=data.get("outer_population_size", 20),
            outer_generations=data.get("outer_generations", 10),
            trials_per_config=data.get("trials_per_config", 1),
            aggregation=data.get("aggregation", "mean"),
            inner_generations=data.get("inner_generations"),
        )
