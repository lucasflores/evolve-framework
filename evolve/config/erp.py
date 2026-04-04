"""
ERP (Evolvable Reproduction Protocol) Settings.

Provides configuration for ERP-specific evolution parameters.
When present in UnifiedConfig, the factory produces ERPEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ERPSettings:
    """
    Evolvable Reproduction Protocol settings.

    When present in UnifiedConfig, the factory produces ERPEngine
    instead of standard EvolutionEngine.

    Attributes:
        step_limit: Maximum computation steps per protocol evaluation.
        recovery_threshold: Success rate below which recovery triggers.
        protocol_mutation_rate: Probability of mutating reproduction protocol.
        enable_intent: Whether to evaluate intent policies.
        enable_recovery: Whether to use recovery mechanisms.

    Example:
        >>> settings = ERPSettings(
        ...     step_limit=1000,
        ...     recovery_threshold=0.1,
        ...     protocol_mutation_rate=0.1,
        ... )
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_limit": self.step_limit,
            "recovery_threshold": self.recovery_threshold,
            "protocol_mutation_rate": self.protocol_mutation_rate,
            "enable_intent": self.enable_intent,
            "enable_recovery": self.enable_recovery,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ERPSettings:
        """Create from dictionary."""
        return cls(
            step_limit=data.get("step_limit", 1000),
            recovery_threshold=data.get("recovery_threshold", 0.1),
            protocol_mutation_rate=data.get("protocol_mutation_rate", 0.1),
            enable_intent=data.get("enable_intent", True),
            enable_recovery=data.get("enable_recovery", True),
        )
