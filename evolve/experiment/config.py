"""
Experiment configuration management.

Provides dataclasses for complete experiment configuration
with validation, serialization, and hashing.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass
class ExperimentConfig:
    """
    Complete configuration for an evolutionary experiment.

    Designed for:
    - Reproducibility (all parameters explicit)
    - Serialization (JSON-compatible)
    - Hashing (for deduplication)

    Example:
        >>> config = ExperimentConfig(
        ...     name="sphere_optimization",
        ...     population_size=100,
        ...     n_generations=50,
        ... )
        >>> config.validate()
        []
        >>> config.to_json("config.json")
    """

    # Identification
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Random seed
    seed: int = 42

    # Population
    population_size: int = 100
    n_generations: int = 100

    # Selection
    selection_method: str = "tournament"
    selection_params: dict[str, Any] = field(default_factory=dict)

    # Operators
    crossover_method: str = "uniform"
    crossover_rate: float = 0.9
    crossover_params: dict[str, Any] = field(default_factory=dict)

    mutation_method: str = "gaussian"
    mutation_rate: float = 0.1
    mutation_params: dict[str, Any] = field(default_factory=dict)

    # Representation
    genome_type: str = "vector"
    genome_params: dict[str, Any] = field(default_factory=dict)

    # Evaluation
    evaluator_type: str = "function"
    evaluator_params: dict[str, Any] = field(default_factory=dict)
    minimize: bool = True

    # Multi-objective (optional)
    multi_objective: bool = False
    n_objectives: int = 1

    # Island model (optional)
    islands: int = 1
    migration_rate: float = 0.1
    migration_interval: int = 10
    topology: str = "ring"

    # Diversity (optional)
    speciation: bool = False
    speciation_params: dict[str, Any] = field(default_factory=dict)
    novelty_search: bool = False
    novelty_params: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    callbacks: list[str] = field(default_factory=list)

    # Stopping criteria
    max_evaluations: int | None = None
    target_fitness: float | None = None
    stagnation_limit: int | None = None

    # Output
    output_dir: str = "./experiments"
    checkpoint_interval: int = 10
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dict.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create from dict.

        Args:
            data: Dictionary with config values

        Returns:
            ExperimentConfig instance
        """
        return cls(**data)

    def to_json(self, path: Path | str) -> None:
        """
        Save to JSON file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        """
        Load from JSON file.

        Args:
            path: Input file path

        Returns:
            ExperimentConfig instance
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def hash(self) -> str:
        """
        Deterministic hash of configuration.

        Useful for detecting duplicate experiments.

        Returns:
            16-character hex hash
        """
        # Sort dict keys for deterministic serialization
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.name:
            errors.append("name is required")

        if self.population_size < 2:
            errors.append("population_size must be >= 2")

        if self.n_generations < 1:
            errors.append("n_generations must be >= 1")

        if not 0 <= self.crossover_rate <= 1:
            errors.append("crossover_rate must be in [0, 1]")

        if not 0 <= self.mutation_rate <= 1:
            errors.append("mutation_rate must be in [0, 1]")

        if self.multi_objective and self.n_objectives < 2:
            errors.append("multi_objective requires n_objectives >= 2")

        if self.islands > 1:
            if not 0 < self.migration_rate <= 1:
                errors.append("migration_rate must be in (0, 1] for island model")
            if self.migration_interval < 1:
                errors.append("migration_interval must be >= 1")

        if self.checkpoint_interval < 1:
            errors.append("checkpoint_interval must be >= 1")

        return errors

    def is_valid(self) -> bool:
        """
        Check if configuration is valid.

        Returns:
            True if no validation errors
        """
        return len(self.validate()) == 0

    def copy(self, **overrides: Any) -> Self:
        """
        Create a copy with optional overrides.

        Args:
            **overrides: Values to override in the copy

        Returns:
            New config with overrides applied
        """
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)

    def __str__(self) -> str:
        return (
            f"ExperimentConfig(name={self.name!r}, "
            f"pop={self.population_size}, gen={self.n_generations}, "
            f"seed={self.seed})"
        )


@dataclass
class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    errors: list[str]

    def __str__(self) -> str:
        return f"Config validation failed: {self.errors}"


def validate_config(config: ExperimentConfig) -> None:
    """
    Validate config and raise if invalid.

    Args:
        config: Config to validate

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = config.validate()
    if errors:
        raise ConfigValidationError(errors)


__all__ = [
    "ExperimentConfig",
    "ConfigValidationError",
    "validate_config",
]
