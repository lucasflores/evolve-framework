"""
Unified Configuration.

Provides the main UnifiedConfig class that encompasses all parameters
from EvolutionConfig, ERPConfig, and ExperimentConfig into a single
JSON-serializable specification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from evolve.config.callbacks import CallbackConfig
from evolve.config.erp import ERPSettings
from evolve.config.merge import MergeConfig
from evolve.config.meta import MetaEvolutionConfig
from evolve.config.multiobjective import (
    ConstraintSpec,
    MultiObjectiveConfig,
    ObjectiveSpec,
)
from evolve.config.schema import CURRENT_SCHEMA_VERSION, validate_schema_version
from evolve.config.stopping import StoppingConfig
from evolve.config.tracking import MetricCategory, TrackingConfig

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for a dataset reference (T036).

    Attributes:
        name: Human-readable dataset identifier.
        path: Optional filesystem path to dataset.
        data: Optional in-memory data (not serialized).
        context: Dataset context description (e.g., "training", "validation").
    """

    name: str
    path: str | None = None
    data: Any = field(default=None, repr=False)
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict (excludes in-memory data)."""
        result: dict[str, Any] = {"name": self.name}
        if self.path is not None:
            result["path"] = self.path
        if self.context:
            result["context"] = self.context
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Reconstruct from dict."""
        return cls(
            name=data["name"],
            path=data.get("path"),
            context=data.get("context", ""),
        )


@dataclass(frozen=True)
class UnifiedConfig:
    """
    Complete experiment specification.

    Covers standard evolution, ERP, multi-objective optimization,
    and meta-evolution in a single JSON-serializable configuration.

    This is the main entry point for defining experiments declaratively.
    Once created, use `create_engine(config, evaluator)` to produce a
    ready-to-run evolution engine.

    Attributes:
        schema_version: Schema version for forward/backward compatibility.
        name: Human-readable experiment identifier.
        description: Optional description for documentation.
        tags: Tags for categorization and filtering.
        seed: Random seed for reproducibility (None = random).
        population_size: Number of individuals in population.
        max_generations: Maximum generations (basic stopping condition).
        elitism: Number of elite individuals preserved.
        selection: Selection operator name (resolved via OperatorRegistry).
        selection_params: Parameters for selection operator.
        crossover: Crossover operator name.
        crossover_rate: Probability of applying crossover.
        crossover_params: Parameters for crossover operator.
        mutation: Mutation operator name.
        mutation_rate: Probability of mutation per individual.
        mutation_params: Parameters for mutation operator.
        genome_type: Genome representation type (resolved via GenomeRegistry).
        genome_params: Parameters for genome initialization.
        minimize: If True, lower fitness is better.
        evaluator: Evaluator registry name (resolved via EvaluatorRegistry).
        evaluator_params: Parameters passed to evaluator factory.
        decoder: Decoder registry name (resolved via DecoderRegistry).
        decoder_params: Parameters passed to decoder factory.
        custom_callbacks: Declarative callback entries resolved via CallbackRegistry.
        stopping: Additional stopping criteria.
        callbacks: Built-in callback configuration.
        erp: ERP settings (enables ERPEngine when present).
        multiobjective: Multi-objective settings (enables NSGA-II when present).
        meta: Meta-evolution settings (enables outer loop when present).
        tracking: Tracking configuration for experiment observability.
        merge: Symbiogenetic merge settings (enables merge phase when present).
        training_data: Training dataset reference for MLflow logging.
        validation_data: Validation dataset reference for MLflow logging.

    Example:
        >>> config = UnifiedConfig(
        ...     name="sphere_optimization",
        ...     population_size=100,
        ...     selection="tournament",
        ...     selection_params={"tournament_size": 5},
        ...     crossover="sbx",
        ...     mutation="gaussian",
        ...     mutation_params={"sigma": 0.1},
        ...     genome_type="vector",
        ...     genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
        ... )
        >>> engine = create_engine(config, fitness_fn)
        >>> result = engine.run()
    """

    # -------------------------------------------------------------------------
    # Schema identification
    # -------------------------------------------------------------------------

    schema_version: str = CURRENT_SCHEMA_VERSION
    """Schema version for forward/backward compatibility (FR-006)."""

    # -------------------------------------------------------------------------
    # Experiment identification
    # -------------------------------------------------------------------------

    name: str = ""
    """Human-readable experiment identifier."""

    description: str = ""
    """Optional description for documentation."""

    tags: tuple[str, ...] = ()
    """Tags for categorization and filtering."""

    # -------------------------------------------------------------------------
    # Random seed
    # -------------------------------------------------------------------------

    seed: int | None = None
    """Random seed for reproducibility (None = random)."""

    # -------------------------------------------------------------------------
    # Population settings
    # -------------------------------------------------------------------------

    population_size: int = 100
    """Number of individuals in population."""

    max_generations: int = 100
    """Maximum generations (basic stopping condition)."""

    elitism: int = 1
    """Number of elite individuals preserved."""

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    selection: str = "tournament"
    """Selection operator name (resolved via OperatorRegistry)."""

    selection_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for selection operator."""

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    crossover: str = "uniform"
    """Crossover operator name."""

    crossover_rate: float = 0.9
    """Probability of applying crossover."""

    crossover_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for crossover operator."""

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    mutation: str = "gaussian"
    """Mutation operator name."""

    mutation_rate: float = 1.0
    """Probability of mutation per individual."""

    mutation_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for mutation operator."""

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    genome_type: str = "vector"
    """Genome representation type (resolved via GenomeRegistry)."""

    genome_params: dict[str, Any] = field(default_factory=dict)
    """Parameters for genome initialization."""

    minimize: bool = True
    """If True, lower fitness is better."""

    # -------------------------------------------------------------------------
    # Evaluator & Callback declarative fields
    # -------------------------------------------------------------------------

    evaluator: str | None = None
    """Evaluator registry name (resolved via EvaluatorRegistry)."""

    evaluator_params: dict[str, Any] = field(default_factory=dict)
    """Parameters passed to evaluator factory."""

    decoder: str | None = None
    """Decoder registry name (resolved via DecoderRegistry)."""

    decoder_params: dict[str, Any] = field(default_factory=dict)
    """Parameters passed to decoder factory."""

    custom_callbacks: tuple[dict[str, Any], ...] = ()
    """Declarative callback entries: each dict has 'name' (str) and optional 'params' (dict)."""

    # -------------------------------------------------------------------------
    # Nested configurations (None = disabled)
    # -------------------------------------------------------------------------

    stopping: StoppingConfig | None = None
    """Additional stopping criteria beyond max_generations."""

    callbacks: CallbackConfig | None = None
    """Built-in callback configuration."""

    erp: ERPSettings | None = None
    """ERP settings (enables ERPEngine when present)."""

    multiobjective: MultiObjectiveConfig | None = None
    """Multi-objective settings (enables NSGA-II when present)."""

    meta: MetaEvolutionConfig | None = None
    """Meta-evolution settings (enables outer loop when present)."""

    tracking: TrackingConfig | None = None
    """Tracking configuration for experiment observability (FR-002)."""

    merge: MergeConfig | None = None
    """Symbiogenetic merge settings (enables merge phase when present)."""

    training_data: DatasetConfig | None = None
    """Training dataset configuration for MLflow logging."""

    validation_data: DatasetConfig | None = None
    """Validation dataset configuration for MLflow logging."""

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate unified configuration (FR-004)."""
        # Population settings
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive")
        if not 0 <= self.elitism <= self.population_size:
            raise ValueError("elitism must be in [0, population_size]")

        # Operator rates
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")

        # Operator names
        if not self.selection:
            raise ValueError("selection operator name required")
        if not self.crossover:
            raise ValueError("crossover operator name required")
        if not self.mutation:
            raise ValueError("mutation operator name required")

        # Genome type
        if not self.genome_type:
            raise ValueError("genome_type required")

        # Evaluator declarative fields
        if self.evaluator is not None and not self.evaluator:
            raise ValueError("evaluator must be a non-empty string when set")

        if self.evaluator_params and self.evaluator is None:
            import warnings

            warnings.warn(
                "evaluator_params set without evaluator name; "
                "params will be ignored unless an evaluator is specified",
                UserWarning,
                stacklevel=2,
            )

        # Decoder declarative fields
        if self.decoder is not None and not self.decoder:
            raise ValueError("decoder must be a non-empty string when set")

        if self.decoder_params and self.decoder is None:
            import warnings

            warnings.warn(
                "decoder_params set without decoder name; "
                "params will be ignored unless a decoder is specified",
                UserWarning,
                stacklevel=2,
            )

        # Validate custom_callbacks structure
        for i, entry in enumerate(self.custom_callbacks):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"custom_callbacks[{i}] must be a dict, got {type(entry).__name__}"
                )
            if "name" not in entry:
                raise ValueError(f"custom_callbacks[{i}] must have a 'name' key")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_erp_enabled(self) -> bool:
        """Check if ERP mode is enabled."""
        return self.erp is not None

    @property
    def is_multiobjective(self) -> bool:
        """Check if multi-objective mode is enabled."""
        return self.multiobjective is not None

    @property
    def is_meta_evolution(self) -> bool:
        """Check if meta-evolution is enabled."""
        return self.meta is not None

    @property
    def is_tracking_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self.tracking is not None and self.tracking.enabled

    @property
    def is_merge_enabled(self) -> bool:
        """Check if symbiogenetic merge phase is enabled."""
        return self.merge is not None and self.merge.merge_rate > 0.0

    # -------------------------------------------------------------------------
    # Serialization (FR-002)
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
            "evaluator": self.evaluator,
            "evaluator_params": dict(self.evaluator_params),
            "decoder": self.decoder,
            "decoder_params": dict(self.decoder_params),
            "custom_callbacks": [dict(cb) for cb in self.custom_callbacks],
        }

        # Nested configs
        result["stopping"] = self.stopping.to_dict() if self.stopping else None
        result["callbacks"] = self.callbacks.to_dict() if self.callbacks else None
        result["erp"] = self.erp.to_dict() if self.erp else None
        result["multiobjective"] = self.multiobjective.to_dict() if self.multiobjective else None
        result["meta"] = self.meta.to_dict() if self.meta else None
        result["tracking"] = self.tracking.to_dict() if self.tracking else None
        result["merge"] = self.merge.to_dict() if self.merge else None
        result["training_data"] = self.training_data.to_dict() if self.training_data else None
        result["validation_data"] = self.validation_data.to_dict() if self.validation_data else None

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedConfig:
        """
        Create from dictionary.

        Args:
            data: Dictionary from JSON deserialization.

        Returns:
            UnifiedConfig instance.

        Raises:
            SchemaVersionError: If schema version is incompatible (FR-007, FR-008).
            ValueError: If required fields are missing or invalid.
        """
        # Version validation (FR-007, FR-008)
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        validate_schema_version(version)

        # Parse nested configs
        stopping = None
        if data.get("stopping"):
            stopping = StoppingConfig.from_dict(data["stopping"])

        callbacks = None
        if data.get("callbacks"):
            callbacks = CallbackConfig.from_dict(data["callbacks"])

        erp = None
        if data.get("erp"):
            erp = ERPSettings.from_dict(data["erp"])

        multiobjective = None
        if data.get("multiobjective"):
            multiobjective = MultiObjectiveConfig.from_dict(data["multiobjective"])

        meta = None
        if data.get("meta"):
            meta = MetaEvolutionConfig.from_dict(data["meta"])

        tracking = None
        if data.get("tracking"):
            tracking = TrackingConfig.from_dict(data["tracking"])

        merge = None
        if data.get("merge"):
            merge = MergeConfig.from_dict(data["merge"])

        training_data = None
        if data.get("training_data"):
            training_data = DatasetConfig.from_dict(data["training_data"])

        validation_data = None
        if data.get("validation_data"):
            validation_data = DatasetConfig.from_dict(data["validation_data"])

        # Handle tags as list or tuple
        tags = data.get("tags", ())
        if isinstance(tags, list):
            tags = tuple(tags)

        # Handle custom_callbacks as list → tuple
        custom_callbacks_raw = data.get("custom_callbacks", ())
        if isinstance(custom_callbacks_raw, list):
            custom_callbacks = tuple(custom_callbacks_raw)
        else:
            custom_callbacks = custom_callbacks_raw

        return cls(
            schema_version=version,
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=tags,
            seed=data.get("seed"),
            population_size=data.get("population_size", 100),
            max_generations=data.get("max_generations", 100),
            elitism=data.get("elitism", 1),
            selection=data.get("selection", "tournament"),
            selection_params=data.get("selection_params", {}),
            crossover=data.get("crossover", "uniform"),
            crossover_rate=data.get("crossover_rate", 0.9),
            crossover_params=data.get("crossover_params", {}),
            mutation=data.get("mutation", "gaussian"),
            mutation_rate=data.get("mutation_rate", 1.0),
            mutation_params=data.get("mutation_params", {}),
            genome_type=data.get("genome_type", "vector"),
            genome_params=data.get("genome_params", {}),
            minimize=data.get("minimize", True),
            evaluator=data.get("evaluator"),
            evaluator_params=data.get("evaluator_params", {}),
            decoder=data.get("decoder"),
            decoder_params=data.get("decoder_params", {}),
            custom_callbacks=custom_callbacks,
            stopping=stopping,
            callbacks=callbacks,
            erp=erp,
            multiobjective=multiobjective,
            meta=meta,
            tracking=tracking,
            merge=merge,
            training_data=training_data,
            validation_data=validation_data,
        )

    def to_json(self, indent: int | None = 2) -> str:
        """
        Serialize to JSON string.

        Args:
            indent: JSON indentation level (None for compact).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_string: str) -> UnifiedConfig:
        """
        Deserialize from JSON string.

        Args:
            json_string: JSON string representation.

        Returns:
            UnifiedConfig instance.
        """
        data = json.loads(json_string)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> UnifiedConfig:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            UnifiedConfig instance.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_file(self, path: str, indent: int = 2) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to output JSON file.
            indent: JSON indentation level.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    # -------------------------------------------------------------------------
    # Hashing (FR-003)
    # -------------------------------------------------------------------------

    def compute_hash(self) -> str:
        """
        Compute deterministic hash for experiment tracking.

        Returns:
            Hex digest of configuration hash.

        Note:
            Hash is based on all configuration parameters except name,
            description, and tags (which don't affect experiment behavior).
            New fields (evaluator, evaluator_params, custom_callbacks) are
            only included when non-default to preserve backward compatibility.
        """
        # Create dict excluding non-behavioral fields
        data = self.to_dict()
        data.pop("name", None)
        data.pop("description", None)
        data.pop("tags", None)

        # Exclude new fields when at defaults (backward compat)
        if data.get("evaluator") is None:
            data.pop("evaluator", None)
        if not data.get("evaluator_params"):
            data.pop("evaluator_params", None)
        if not data.get("custom_callbacks"):
            data.pop("custom_callbacks", None)
        if data.get("decoder") is None:
            data.pop("decoder", None)
        if not data.get("decoder_params"):
            data.pop("decoder_params", None)

        # Sort for determinism
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    # -------------------------------------------------------------------------
    # Modification (FR-005)
    # -------------------------------------------------------------------------

    def with_params(self, **kwargs: Any) -> UnifiedConfig:
        """
        Create a modified copy with updated parameters.

        Args:
            **kwargs: Parameters to update.

        Returns:
            New UnifiedConfig with updated parameters.

        Example:
            >>> new_config = config.with_params(
            ...     population_size=200,
            ...     mutation_rate=0.5,
            ... )
        """
        return replace(self, **kwargs)

    def with_stopping(
        self,
        max_generations: int | None = None,
        fitness_threshold: float | None = None,
        stagnation_generations: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> UnifiedConfig:
        """
        Create a copy with updated stopping criteria.

        Args:
            max_generations: Stop after N generations.
            fitness_threshold: Stop when fitness reaches this value.
            stagnation_generations: Stop after N generations with no improvement.
            time_limit_seconds: Stop after this duration.

        Returns:
            New UnifiedConfig with updated stopping criteria.
        """
        stopping = StoppingConfig(
            max_generations=max_generations,
            fitness_threshold=fitness_threshold,
            stagnation_generations=stagnation_generations,
            time_limit_seconds=time_limit_seconds,
        )
        return replace(self, stopping=stopping)

    def with_erp(
        self,
        step_limit: int = 1000,
        recovery_threshold: float = 0.1,
        protocol_mutation_rate: float = 0.1,
        enable_intent: bool = True,
        enable_recovery: bool = True,
    ) -> UnifiedConfig:
        """
        Create a copy with ERP mode enabled.

        Automatically enables ERP tracking category if tracking is enabled.

        Args:
            step_limit: Maximum steps per protocol evaluation.
            recovery_threshold: Success rate for recovery trigger.
            protocol_mutation_rate: Protocol mutation probability.
            enable_intent: Whether to evaluate intent policies.
            enable_recovery: Whether to use recovery mechanisms.

        Returns:
            New UnifiedConfig with ERP enabled.
        """
        erp = ERPSettings(
            step_limit=step_limit,
            recovery_threshold=recovery_threshold,
            protocol_mutation_rate=protocol_mutation_rate,
            enable_intent=enable_intent,
            enable_recovery=enable_recovery,
        )

        # Auto-enable ERP tracking category
        tracking = self.tracking
        if (
            tracking is not None
            and tracking.enabled
            and not tracking.has_category(MetricCategory.ERP)
        ):
            tracking = tracking.with_category(MetricCategory.ERP)

        return replace(self, erp=erp, tracking=tracking)

    def with_multiobjective(
        self,
        objectives: tuple[ObjectiveSpec, ...],
        reference_point: tuple[float, ...] | None = None,
        constraints: tuple[ConstraintSpec, ...] = (),
        constraint_handling: str = "dominance",
    ) -> UnifiedConfig:
        """
        Create a copy with multi-objective mode enabled.

        Automatically enables MULTIOBJECTIVE tracking category if tracking
        is enabled.

        Args:
            objectives: Tuple of objective specifications.
            reference_point: Reference point for hypervolume.
            constraints: Constraint specifications.
            constraint_handling: How to handle constraints.

        Returns:
            New UnifiedConfig with multi-objective enabled.
        """
        mo = MultiObjectiveConfig(
            objectives=objectives,
            reference_point=reference_point,
            constraints=constraints,
            constraint_handling=constraint_handling,  # type: ignore[arg-type]
        )

        # Auto-enable MULTIOBJECTIVE tracking category (T057)
        tracking = self.tracking
        if (
            tracking is not None
            and tracking.enabled
            and not tracking.has_category(MetricCategory.MULTIOBJECTIVE)
        ):
            tracking = tracking.with_category(MetricCategory.MULTIOBJECTIVE)

        return replace(self, multiobjective=mo, tracking=tracking)

    def with_merge(
        self,
        merge_rate: float = 0.1,
        operator: str = "graph_symbiogenetic",
        **kwargs: Any,
    ) -> UnifiedConfig:
        """
        Create a copy with symbiogenetic merge enabled.

        Automatically enables SYMBIOGENESIS tracking category if tracking
        is enabled.

        Args:
            merge_rate: Per-individual merge probability.
            operator: Merge operator name.
            **kwargs: Additional MergeConfig parameters.

        Returns:
            New UnifiedConfig with merge enabled.
        """
        merge = MergeConfig(operator=operator, merge_rate=merge_rate, **kwargs)

        # Auto-enable SYMBIOGENESIS tracking category
        tracking = self.tracking
        if (
            tracking is not None
            and tracking.enabled
            and not tracking.has_category(MetricCategory.SYMBIOGENESIS)
        ):
            tracking = tracking.with_category(MetricCategory.SYMBIOGENESIS)

        return replace(self, merge=merge, tracking=tracking)
