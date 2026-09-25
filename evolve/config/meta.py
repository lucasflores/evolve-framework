"""
Meta-Evolution Configuration.

Provides configuration for meta-evolution (hyperparameter optimization),
including parameter specifications and outer loop settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for an evolvable parameter in meta-evolution.

    Defines how a configuration parameter should be encoded into
    vector genome positions on [0, 1] for meta-evolution, or for
    ``decode_parameters()``.

    A spec may name a ``parent`` spec whose decoded value it depends on
    (``decode_parameters()`` only; ConfigCodec refuses it):

    - ``active_values``: active only while the parent's value is one of them.
    - ``choices_by_parent``: options that depend on a categorical parent's
      value; inactive when the value has no entry. For a categorical the
      entry replaces ``choices``; for a subset ``choices`` still lays out the
      positions and the entry is the allowed part of it.
    - A subset with a subset parent and neither of the above keeps only the
      choices the parent's value also holds.

    Attributes:
        path: Dot-notation path to parameter (e.g., 'mutation_params.sigma').
        param_type: Type of parameter for encoding strategy.
        bounds: Min/max bounds for continuous/integer parameters.
        choices: Valid choices for categorical parameters; the universe for subsets.
        log_scale: Whether to use logarithmic scaling.
        parent: Path of the spec this one depends on.
        active_values: Parent values for which this spec is active.
        choices_by_parent: (parent value, options) pairs: categorical options,
            or allowed subset choices, for each parent value. A mapping is
            accepted and stored as pairs, so the spec stays hashable and
            non-string parent values survive JSON.

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

    param_type: Literal["continuous", "integer", "categorical", "subset"] = "continuous"
    """Type of parameter for encoding strategy."""

    bounds: tuple[float, float] | None = None
    """Min/max bounds for continuous/integer parameters."""

    choices: tuple[Any, ...] | None = None
    """Valid choices for categorical parameters; the universe for subset parameters."""

    log_scale: bool = False
    """Whether to use logarithmic scaling for continuous parameters."""

    parent: str | None = None
    """Path of the spec whose decoded value this one depends on."""

    active_values: tuple[Any, ...] | None = None
    """Parent values for which this spec is active (inactive otherwise)."""

    choices_by_parent: tuple[tuple[Any, tuple[Any, ...]], ...] | None = None
    """(parent value, options) pairs; inactive for a parent value not listed."""

    def __post_init__(self) -> None:
        """Validate parameter specification."""
        if not self.path:
            raise ValueError("Parameter path cannot be empty")

        if self.choices_by_parent is not None:
            items = (
                self.choices_by_parent.items()
                if isinstance(self.choices_by_parent, Mapping)
                else self.choices_by_parent
            )
            pairs = tuple((value, tuple(options)) for value, options in items)
            object.__setattr__(self, "choices_by_parent", pairs)
            values = [value for value, _ in pairs]
            repeated = [v for i, v in enumerate(values) if v in values[:i]]
            if repeated:
                raise ValueError(
                    f"choices_by_parent lists parent values more than once: {repeated}"
                )
            if not pairs:
                raise ValueError("choices_by_parent is empty, so the parameter is never active")
        if self.active_values is not None and len(self.active_values) == 0:
            raise ValueError("active_values is empty, so the parameter is never active")

        if self.param_type in ("continuous", "integer"):
            if self.bounds is None:
                raise ValueError(f"bounds required for {self.param_type} parameter")
            if len(self.bounds) != 2:
                raise ValueError("bounds must be a tuple of (min, max)")
            if self.bounds[0] > self.bounds[1]:
                raise ValueError("bounds[0] must be <= bounds[1]")
            if self.param_type == "continuous" and self.log_scale and self.bounds[0] <= 0:
                raise ValueError("log_scale requires positive lower bound")
        elif self.param_type == "categorical":
            if self.choices_by_parent is not None:
                if self.choices is not None:
                    raise ValueError("give choices or choices_by_parent, not both")
                if not all(options for _, options in self.choices_by_parent):
                    raise ValueError("choices_by_parent needs options for every parent value")
            elif self.choices is None or len(self.choices) == 0:
                raise ValueError("choices required for categorical parameter")
        elif self.param_type == "subset":
            if self.choices is None or len(self.choices) == 0:
                raise ValueError("choices required for subset parameter")
            for _, allowed in self.choices_by_parent or ():
                outside = [c for c in allowed if c not in self.choices]
                if outside:
                    raise ValueError(f"choices_by_parent lists choices not in choices: {outside}")
        else:
            raise ValueError(
                f"param_type must be 'continuous', 'integer', 'categorical', or 'subset', "
                f"got {self.param_type}"
            )

        if self.choices_by_parent is not None and self.param_type not in ("categorical", "subset"):
            raise ValueError("choices_by_parent is only for categorical and subset parameters")
        if self.parent is None:
            if self.active_values is not None or self.choices_by_parent is not None:
                raise ValueError("active_values and choices_by_parent require a parent")
        elif self.active_values is None and self.choices_by_parent is None and not self.is_relative:
            raise ValueError(
                f"parent of '{self.path}' is unused: give active_values or "
                "choices_by_parent (only a subset can be relative to its parent)"
            )

    @property
    def is_relative(self) -> bool:
        """Whether this is a subset limited to its subset parent's value."""
        return (
            self.param_type == "subset"
            and self.parent is not None
            and self.active_values is None
            and self.choices_by_parent is None
        )

    @property
    def num_dimensions(self) -> int:
        """Get number of genome dimensions needed for this parameter."""
        # A subset uses one position per choice; the other types use 1
        if self.param_type == "subset":
            assert self.choices is not None
            return len(self.choices)
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
        if self.parent is not None:
            result["parent"] = self.parent
        if self.active_values is not None:
            result["active_values"] = list(self.active_values)
        if self.choices_by_parent is not None:
            result["choices_by_parent"] = [[v, list(o)] for v, o in self.choices_by_parent]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSpec:
        """Create from dictionary."""
        bounds = data.get("bounds")
        choices = data.get("choices")
        active_values = data.get("active_values")
        return cls(
            path=data["path"],
            param_type=data.get("param_type", "continuous"),
            bounds=tuple(bounds) if bounds else None,
            choices=tuple(choices) if choices else None,
            log_scale=data.get("log_scale", False),
            parent=data.get("parent"),
            active_values=tuple(active_values) if active_values is not None else None,
            # A list of pairs, or a mapping in a hand-written config
            choices_by_parent=data.get("choices_by_parent"),
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
        for spec in self.evolvable_params:
            if spec.param_type == "subset" or spec.parent is not None:
                raise ValueError(
                    f"MetaEvolutionConfig does not support subset or dependent parameters "
                    f"('{spec.path}'); decode them with decode_parameters()"
                )
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
