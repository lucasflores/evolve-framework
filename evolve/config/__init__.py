"""
Unified Configuration Module.

Provides JSON-serializable configuration classes for defining
experiments across all supported modes: standard evolution, ERP,
multi-objective optimization, and meta-evolution.

Public API:
    UnifiedConfig: Complete experiment specification
    StoppingConfig: Stopping criteria configuration
    CallbackConfig: Callback settings
    ERPSettings: ERP-specific configuration
    MultiObjectiveConfig: Multi-objective settings
    MetaEvolutionConfig: Meta-evolution settings
"""

from evolve.config.stopping import StoppingConfig
from evolve.config.callbacks import CallbackConfig
from evolve.config.erp import ERPSettings
from evolve.config.multiobjective import (
    ObjectiveSpec,
    ConstraintSpec,
    MultiObjectiveConfig,
)
from evolve.config.meta import ParameterSpec, MetaEvolutionConfig
from evolve.config.schema import SchemaVersion, validate_schema_version
from evolve.config.unified import UnifiedConfig

__all__ = [
    # Core config
    "UnifiedConfig",
    # Stopping
    "StoppingConfig",
    # Callbacks
    "CallbackConfig",
    # ERP
    "ERPSettings",
    # Multi-objective
    "ObjectiveSpec",
    "ConstraintSpec",
    "MultiObjectiveConfig",
    # Meta-evolution
    "ParameterSpec",
    "MetaEvolutionConfig",
    # Schema
    "SchemaVersion",
    "validate_schema_version",
]
