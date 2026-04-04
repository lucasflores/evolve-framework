"""
Callback Configuration.

Provides configuration for built-in callbacks including logging
and checkpointing. Custom callbacks must be passed to factory separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CallbackConfig:
    """
    Built-in callback configuration.

    Configures logging and checkpointing callbacks that can be
    instantiated declaratively. Custom callbacks must be passed
    to the factory function separately.

    Attributes:
        enable_logging: Whether to enable progress logging callback.
        log_level: Logging verbosity level (DEBUG, INFO, WARNING).
        log_destination: Where to log ('console', 'file', or a file path).
        enable_checkpointing: Whether to enable checkpoint saving.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_frequency: Save checkpoint every N generations.

    Example:
        >>> config = CallbackConfig(
        ...     enable_logging=True,
        ...     log_level="INFO",
        ...     enable_checkpointing=True,
        ...     checkpoint_dir="./checkpoints",
        ...     checkpoint_frequency=10,
        ... )
    """

    # Logging (FR-041)
    enable_logging: bool = True
    """Whether to enable progress logging callback."""

    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    """Log verbosity level (FR-041)."""

    log_destination: str = "console"
    """Where to log: 'console', 'file', or a file path (FR-041)."""

    # Checkpointing (FR-042)
    enable_checkpointing: bool = False
    """Whether to enable checkpoint saving."""

    checkpoint_dir: str | None = None
    """Directory for checkpoint files (FR-042)."""

    checkpoint_frequency: int = 10
    """Save checkpoint every N generations (FR-042)."""

    def __post_init__(self) -> None:
        """Validate callback configuration."""
        if self.enable_checkpointing and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir required when checkpointing enabled")
        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive")
        if self.log_level not in ("DEBUG", "INFO", "WARNING"):
            raise ValueError(f"log_level must be DEBUG, INFO, or WARNING, got {self.log_level}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_destination": self.log_destination,
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_frequency": self.checkpoint_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallbackConfig:
        """Create from dictionary."""
        return cls(
            enable_logging=data.get("enable_logging", True),
            log_level=data.get("log_level", "INFO"),
            log_destination=data.get("log_destination", "console"),
            enable_checkpointing=data.get("enable_checkpointing", False),
            checkpoint_dir=data.get("checkpoint_dir"),
            checkpoint_frequency=data.get("checkpoint_frequency", 10),
        )
