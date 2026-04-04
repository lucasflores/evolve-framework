"""
Unified Configuration Contracts

Defines the interfaces for the unified configuration system.
These are API contracts that implementations must satisfy.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

# =============================================================================
# Stopping Criteria Configuration
# =============================================================================


@dataclass(frozen=True)
class StoppingConfig:
    """
    Stopping criteria specification.

    Multiple criteria can be specified; evolution stops when ANY is met.
    All fields are optional; omit to disable that criterion.
    """

    max_generations: int | None = None
    """Stop after specified number of generations."""

    fitness_threshold: float | None = None
    """Stop when best fitness reaches this value."""

    stagnation_generations: int | None = None
    """Stop after N generations with no fitness improvement."""

    time_limit_seconds: float | None = None
    """Stop after specified wall-clock duration."""

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


# =============================================================================
# Callback Configuration
# =============================================================================


@dataclass(frozen=True)
class CallbackConfig:
    """
    Built-in callback configuration.

    Custom callbacks must be passed to factory separately.
    """

    # Logging
    enable_logging: bool = True
    """Whether to enable progress logging callback."""

    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    """Log verbosity level."""

    log_destination: str = "console"
    """Where to log: 'console', 'file', or a file path."""

    # Checkpointing
    enable_checkpointing: bool = False
    """Whether to enable checkpoint saving."""

    checkpoint_dir: str | None = None
    """Directory for checkpoint files."""

    checkpoint_frequency: int = 10
    """Save checkpoint every N generations."""

    def __post_init__(self) -> None:
        """Validate callback configuration."""
        if self.enable_checkpointing and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir required when checkpointing enabled")
        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive")


# =============================================================================
# ERP Configuration
# =============================================================================


@dataclass(frozen=True)
class ERPSettings:
    """
    Evolvable Reproduction Protocol settings.

    When present in UnifiedConfig, factory produces ERPEngine.
    """

    step_limit: int = 1000
    """Maximum computation steps per protocol evaluation."""

    recovery_threshold: float = 0.1
    """Success rate below which recovery triggers."""

    protocol_mutation_rate: float = 0.1
    """Probability of mutating reproduction protocol."""

    enable_intent: bool = True
    """Whether to evaluate intent policies."""

    enable_recovery: bool = True
    """Whether to use recovery mechanisms."""

    def __post_init__(self) -> None:
        """Validate ERP settings."""
        if self.step_limit <= 0:
            raise ValueError("step_limit must be positive")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery_threshold must be in [0, 1]")
        if not 0.0 <= self.protocol_mutation_rate <= 1.0:
            raise ValueError("protocol_mutation_rate must be in [0, 1]")


# =============================================================================
# Multi-Objective Configuration
# =============================================================================


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single optimization objective."""

    name: str
    """Objective identifier used in fitness dictionaries."""

    direction: Literal["minimize", "maximize"] = "minimize"
    """Whether to minimize or maximize this objective."""

    weight: float = 1.0
    """Weight for weighted-sum scalarization."""


@dataclass(frozen=True)
class ConstraintSpec:
    """Specification for a constraint function."""

    name: str
    """Constraint identifier."""

    penalty_weight: float = 1.0
    """Weight for penalty function approach."""


@dataclass(frozen=True)
class MultiObjectiveConfig:
    """
    Multi-objective optimization settings.

    When present, factory configures NSGA-II selection.
    """

    objectives: tuple[ObjectiveSpec, ...] = ()
    """Tuple of objective specifications."""

    reference_point: tuple[float, ...] | None = None
    """Reference point for hypervolume calculation."""

    constraints: tuple[ConstraintSpec, ...] = ()
    """Constraint specifications for constraint dominance."""

    constraint_handling: Literal["dominance", "penalty"] = "dominance"
    """How to handle constraints in Pareto ranking."""

    def __post_init__(self) -> None:
        """Validate multi-objective configuration."""
        if len(self.objectives) < 2:
            raise ValueError("Multi-objective requires at least 2 objectives")
        if self.reference_point is not None and len(self.reference_point) != len(self.objectives):
            raise ValueError("reference_point length must match objectives")


# =============================================================================
# Meta-Evolution Configuration
# =============================================================================


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for an evolvable parameter in meta-evolution."""

    path: str
    """Dot-notation path to parameter (e.g., 'mutation_params.sigma')."""

    param_type: Literal["continuous", "integer", "categorical"] = "continuous"
    """Type of parameter for encoding strategy."""

    bounds: tuple[float, float] | None = None
    """Min/max bounds for continuous/integer parameters."""

    choices: tuple[Any, ...] | None = None
    """Valid choices for categorical parameters."""

    log_scale: bool = False
    """Whether to use logarithmic scaling."""

    def __post_init__(self) -> None:
        """Validate parameter specification."""
        if self.param_type in ("continuous", "integer"):
            if self.bounds is None:
                raise ValueError(f"bounds required for {self.param_type} parameter")
            if self.bounds[0] > self.bounds[1]:
                raise ValueError("bounds[0] must be <= bounds[1]")
        elif self.param_type == "categorical":
            if self.choices is None or len(self.choices) == 0:
                raise ValueError("choices required for categorical parameter")


@dataclass(frozen=True)
class MetaEvolutionConfig:
    """
    Configuration for meta-evolution (hyperparameter optimization).

    When present in UnifiedConfig, enables meta-evolution outer loop.
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
    """Override inner loop generations for speed."""

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
        if self.inner_generations is not None and self.inner_generations <= 0:
            raise ValueError("inner_generations must be positive when specified")


# =============================================================================
# Unified Configuration
# =============================================================================


@dataclass(frozen=True)
class UnifiedConfig:
    """
    Complete experiment specification.

    Covers standard evolution, ERP, multi-objective optimization,
    and meta-evolution in a single JSON-serializable configuration.

    Example:
        >>> config = UnifiedConfig(
        ...     name="sphere_opt",
        ...     population_size=100,
        ...     selection="tournament",
        ...     crossover="sbx",
        ...     mutation="gaussian",
        ...     genome_type="vector",
        ...     genome_params={"dimensions": 10, "bounds": [-5.12, 5.12]},
        ... )
        >>> engine = create_engine(config, fitness_fn)
        >>> result = engine.run()
    """

    # Schema identification
    schema_version: str = "1.0.0"
    """Schema version for forward/backward compatibility."""

    # Experiment identification
    name: str = ""
    """Human-readable experiment identifier."""

    description: str = ""
    """Optional description for documentation."""

    tags: tuple[str, ...] = ()
    """Tags for categorization and filtering."""

    # Random seed
    seed: int | None = None
    """Random seed for reproducibility (None = random)."""

    # Population settings
    population_size: int = 100
    """Number of individuals in population."""

    max_generations: int = 100
    """Maximum generations (basic stopping condition)."""

    elitism: int = 1
    """Number of elite individuals preserved."""

    # Selection
    selection: str = "tournament"
    """Selection operator name (resolved via OperatorRegistry)."""

    selection_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for selection operator."""

    # Crossover
    crossover: str = "uniform"
    """Crossover operator name."""

    crossover_rate: float = 0.9
    """Probability of applying crossover."""

    crossover_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for crossover operator."""

    # Mutation
    mutation: str = "gaussian"
    """Mutation operator name."""

    mutation_rate: float = 1.0
    """Probability of mutation per individual."""

    mutation_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for mutation operator."""

    # Representation
    genome_type: str = "vector"
    """Genome representation type (resolved via GenomeRegistry)."""

    genome_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for genome initialization."""

    minimize: bool = True
    """If True, lower fitness is better."""

    # Nested configurations (None = disabled)
    stopping: StoppingConfig | None = None
    """Stopping criteria (beyond max_generations)."""

    callbacks: CallbackConfig | None = None
    """Built-in callback configuration."""

    erp: ERPSettings | None = None
    """ERP settings (enables ERPEngine when present)."""

    multiobjective: MultiObjectiveConfig | None = None
    """Multi-objective settings (enables NSGA-II when present)."""

    meta: MetaEvolutionConfig | None = None
    """Meta-evolution settings (enables outer loop when present)."""

    def __post_init__(self) -> None:
        """Validate unified configuration."""
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if not 0 <= self.elitism <= self.population_size:
            raise ValueError("elitism must be in [0, population_size]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "seed": self.seed,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "elitism": self.elitism,
            "selection": self.selection,
            "selection_params": dict(self.selection_params),
            "crossover": self.crossover,
            "crossover_rate": self.crossover_rate,
            "crossover_params": dict(self.crossover_params),
            "mutation": self.mutation,
            "mutation_rate": self.mutation_rate,
            "mutation_params": dict(self.mutation_params),
            "genome_type": self.genome_type,
            "genome_params": dict(self.genome_params),
            "minimize": self.minimize,
        }

        # Nested configs (convert or None)
        result["stopping"] = self._nested_to_dict(self.stopping)
        result["callbacks"] = self._nested_to_dict(self.callbacks)
        result["erp"] = self._nested_to_dict(self.erp)
        result["multiobjective"] = self._multiobjective_to_dict()
        result["meta"] = self._meta_to_dict()

        return result

    def _nested_to_dict(self, obj: Any) -> dict[str, Any] | None:
        """Convert nested dataclass to dict or None."""
        if obj is None:
            return None
        from dataclasses import asdict

        return asdict(obj)

    def _multiobjective_to_dict(self) -> dict[str, Any] | None:
        """Convert multi-objective config with nested ObjectiveSpec."""
        if self.multiobjective is None:
            return None
        return {
            "objectives": [
                {"name": o.name, "direction": o.direction, "weight": o.weight}
                for o in self.multiobjective.objectives
            ],
            "reference_point": list(self.multiobjective.reference_point)
            if self.multiobjective.reference_point
            else None,
            "constraints": [
                {"name": c.name, "penalty_weight": c.penalty_weight}
                for c in self.multiobjective.constraints
            ],
            "constraint_handling": self.multiobjective.constraint_handling,
        }

    def _meta_to_dict(self) -> dict[str, Any] | None:
        """Convert meta-evolution config with nested ParameterSpec."""
        if self.meta is None:
            return None
        return {
            "evolvable_params": [
                {
                    "path": p.path,
                    "param_type": p.param_type,
                    "bounds": list(p.bounds) if p.bounds else None,
                    "choices": list(p.choices) if p.choices else None,
                    "log_scale": p.log_scale,
                }
                for p in self.meta.evolvable_params
            ],
            "outer_population_size": self.meta.outer_population_size,
            "outer_generations": self.meta.outer_generations,
            "trials_per_config": self.meta.trials_per_config,
            "aggregation": self.meta.aggregation,
            "inner_generations": self.meta.inner_generations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create from dictionary.

        Args:
            data: Dictionary from JSON deserialization.

        Returns:
            UnifiedConfig instance.

        Raises:
            IncompatibleSchemaError: If schema version incompatible.
        """
        # Version check
        version = data.get("schema_version", "1.0.0")
        # TODO: Implement version migration

        # Parse nested configs
        stopping = None
        if data.get("stopping"):
            stopping = StoppingConfig(**data["stopping"])

        callbacks = None
        if data.get("callbacks"):
            callbacks = CallbackConfig(**data["callbacks"])

        erp = None
        if data.get("erp"):
            erp = ERPSettings(**data["erp"])

        multiobjective = None
        if data.get("multiobjective"):
            mo_data = data["multiobjective"]
            objectives = tuple(ObjectiveSpec(**o) for o in mo_data.get("objectives", []))
            constraints = tuple(ConstraintSpec(**c) for c in mo_data.get("constraints", []))
            multiobjective = MultiObjectiveConfig(
                objectives=objectives,
                reference_point=tuple(mo_data["reference_point"])
                if mo_data.get("reference_point")
                else None,
                constraints=constraints,
                constraint_handling=mo_data.get("constraint_handling", "dominance"),
            )

        meta = None
        if data.get("meta"):
            meta_data = data["meta"]
            params = tuple(
                ParameterSpec(
                    path=p["path"],
                    param_type=p.get("param_type", "continuous"),
                    bounds=tuple(p["bounds"]) if p.get("bounds") else None,
                    choices=tuple(p["choices"]) if p.get("choices") else None,
                    log_scale=p.get("log_scale", False),
                )
                for p in meta_data.get("evolvable_params", [])
            )
            meta = MetaEvolutionConfig(
                evolvable_params=params,
                outer_population_size=meta_data.get("outer_population_size", 20),
                outer_generations=meta_data.get("outer_generations", 10),
                trials_per_config=meta_data.get("trials_per_config", 1),
                aggregation=meta_data.get("aggregation", "mean"),
                inner_generations=meta_data.get("inner_generations"),
            )

        return cls(
            schema_version=version,
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=tuple(data.get("tags", [])),
            seed=data.get("seed"),
            population_size=data.get("population_size", 100),
            max_generations=data.get("max_generations", 100),
            elitism=data.get("elitism", 1),
            selection=data.get("selection", "tournament"),
            selection_params=dict(data.get("selection_params", {})),
            crossover=data.get("crossover", "uniform"),
            crossover_rate=data.get("crossover_rate", 0.9),
            crossover_params=dict(data.get("crossover_params", {})),
            mutation=data.get("mutation", "gaussian"),
            mutation_rate=data.get("mutation_rate", 1.0),
            mutation_params=dict(data.get("mutation_params", {})),
            genome_type=data.get("genome_type", "vector"),
            genome_params=dict(data.get("genome_params", {})),
            minimize=data.get("minimize", True),
            stopping=stopping,
            callbacks=callbacks,
            erp=erp,
            multiobjective=multiobjective,
            meta=meta,
        )

    def to_json(self, path: str | None = None) -> str:
        """
        Serialize to JSON string.

        Args:
            path: If provided, write to file at this path.

        Returns:
            JSON string representation.
        """
        import json

        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        Create from JSON string.

        Args:
            json_str: JSON string or file path.

        Returns:
            UnifiedConfig instance.
        """
        import json
        from pathlib import Path

        # Check if it's a file path
        if not json_str.strip().startswith("{"):
            path = Path(json_str)
            if path.exists():
                json_str = path.read_text()

        data = json.loads(json_str)
        return cls.from_dict(data)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def with_params(self, **kwargs: Any) -> Self:
        """
        Create modified copy with updated parameters.

        Args:
            **kwargs: Parameters to update.

        Returns:
            New UnifiedConfig with updated values.
        """
        from dataclasses import replace

        return replace(self, **kwargs)

    def compute_hash(self) -> str:
        """
        Compute deterministic hash for tracking.

        Returns:
            Hex digest of configuration hash.
        """
        import hashlib
        import json

        # Exclude non-deterministic fields
        data = self.to_dict()
        data.pop("name", None)
        data.pop("description", None)
        data.pop("tags", None)

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @property
    def erp_enabled(self) -> bool:
        """Check if ERP is enabled."""
        return self.erp is not None

    @property
    def multiobjective_enabled(self) -> bool:
        """Check if multi-objective is enabled."""
        return self.multiobjective is not None

    @property
    def meta_enabled(self) -> bool:
        """Check if meta-evolution is enabled."""
        return self.meta is not None
