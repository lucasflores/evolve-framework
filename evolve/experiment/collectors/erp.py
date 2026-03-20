"""
ERP (Evolvable Reproduction Protocol) metrics collector.

Collects metrics related to mating dynamics when ERP reproduction is enabled.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from evolve.experiment.collectors.base import MetricCollector, CollectionContext, MatingStats

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@dataclass
class ERPMetricCollector(MetricCollector):
    """
    Collect ERP mating statistics (FR-013).
    
    Tracks mating success rates overall and per-protocol to help
    debug reproduction dynamics in ERP-based evolution.
    
    Attributes:
        warn_on_zero_success: Log warning when success rate drops to zero.
        _previous_success_rate: Track previous rate for trend detection.
    
    Example:
        >>> collector = ERPMetricCollector()
        >>> mating_stats = MatingStats(
        ...     attempted_matings=100,
        ...     successful_matings=85,
        ...     protocol_attempts={"symmetric": 50, "asymmetric": 50},
        ...     protocol_successes={"symmetric": 45, "asymmetric": 40},
        ... )
        >>> context = CollectionContext(
        ...     generation=10,
        ...     population=population,
        ...     mating_stats=mating_stats,
        ... )
        >>> metrics = collector.collect(context)
        >>> assert metrics["mating_success_rate"] == 0.85
    """
    
    warn_on_zero_success: bool = True
    
    _previous_success_rate: float | None = field(default=None, repr=False)
    _zero_success_warned: bool = field(default=False, repr=False)
    
    def collect(self, context: CollectionContext) -> dict[str, float]:
        """
        Collect ERP mating metrics.
        
        Args:
            context: Collection context with mating_stats.
            
        Returns:
            Dictionary of ERP metrics:
                - mating_success_rate: Fraction of successful matings (0.0-1.0)
                - attempted_matings: Total mating attempts this generation
                - successful_matings: Successful matings producing offspring
                - erp_protocol_{name}_success_rate: Per-protocol success rates
        """
        mating_stats = context.mating_stats
        
        if mating_stats is None:
            return {}
        
        metrics: dict[str, float] = {}
        
        # Core mating metrics
        metrics["attempted_matings"] = float(mating_stats.attempted_matings)
        metrics["successful_matings"] = float(mating_stats.successful_matings)
        metrics["mating_success_rate"] = mating_stats.success_rate
        
        # Check for zero success rate warning
        if self.warn_on_zero_success:
            self._check_zero_success(mating_stats.success_rate, context.generation)
        
        # Track for trend detection
        self._previous_success_rate = mating_stats.success_rate
        
        # Per-protocol success rates
        for protocol_name in mating_stats.protocol_attempts:
            rate = mating_stats.protocol_success_rate(protocol_name)
            # Sanitize protocol name for metric key (replace spaces, special chars)
            safe_name = self._sanitize_protocol_name(protocol_name)
            metrics[f"erp_protocol_{safe_name}_success_rate"] = rate
            metrics[f"erp_protocol_{safe_name}_attempts"] = float(
                mating_stats.protocol_attempts.get(protocol_name, 0)
            )
            metrics[f"erp_protocol_{safe_name}_successes"] = float(
                mating_stats.protocol_successes.get(protocol_name, 0)
            )
        
        return metrics
    
    def reset(self) -> None:
        """Reset internal state between runs."""
        self._previous_success_rate = None
        self._zero_success_warned = False
    
    def _check_zero_success(self, success_rate: float, generation: int) -> None:
        """
        Check for zero success rate and log warning.
        
        Only logs once per run to avoid spam.
        """
        if success_rate == 0.0 and not self._zero_success_warned:
            logger.warning(
                f"ERP mating success rate dropped to zero at generation {generation}. "
                "This may indicate incompatible protocols or overly restrictive "
                "matchability/intent rules. Consider enabling recovery mechanisms."
            )
            self._zero_success_warned = True
    
    @staticmethod
    def _sanitize_protocol_name(name: str) -> str:
        """
        Sanitize protocol name for use as metric key suffix.
        
        Replaces spaces and special characters with underscores.
        """
        # Replace common problematic characters
        sanitized = name.lower()
        for char in [" ", "-", ".", "/", "\\"]:
            sanitized = sanitized.replace(char, "_")
        # Remove any remaining non-alphanumeric characters except underscore
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
        return sanitized


__all__ = ["ERPMetricCollector"]
