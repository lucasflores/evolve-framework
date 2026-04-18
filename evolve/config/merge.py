"""
Merge Configuration.

Provides MergeConfig frozen dataclass for configuring the
symbiogenetic merge phase in the evolutionary engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MergeConfig:
    """
    Configuration for the symbiogenetic merge phase.

    When present in UnifiedConfig, the engine inserts a merge phase
    after crossover/mutation and before population replacement.

    Attributes:
        operator: Merge operator name (resolved via OperatorRegistry).
        merge_rate: Per-individual probability of being selected as merge host.
        symbiont_source: Where to source the symbiont from.
        symbiont_fate: What happens to the symbiont after merge.
        archive_size: Size of hall-of-fame archive (when symbiont_source="archive").
        interface_count: Number of interface connections for graph merges.
        interface_ratio: Fraction of interface connections that are host→symbiont.
        weight_method: How to initialise weights on interface connections.
        weight_mean: Mean for Gaussian weight init (weight_method="random").
        weight_std: Std dev for Gaussian weight init (weight_method="random").
        max_complexity: Optional upper bound on merged genome gene count.
        operator_params: Extra kwargs passed to merge operator constructor.
    """

    operator: str = "graph_symbiogenetic"
    merge_rate: float = 0.0
    symbiont_source: Literal["cross_species", "archive"] = "cross_species"
    symbiont_fate: Literal["consumed", "survives"] = "consumed"
    archive_size: int = 50
    interface_count: int = 4
    interface_ratio: float = 0.5
    weight_method: Literal["mean", "host_biased", "random"] = "mean"
    weight_mean: float = 0.0
    weight_std: float = 1.0
    max_complexity: int | None = None
    operator_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate merge configuration."""
        if not 0.0 <= self.merge_rate <= 1.0:
            raise ValueError("merge_rate must be in [0.0, 1.0]")
        if self.interface_count <= 0:
            raise ValueError("interface_count must be > 0")
        if not 0.0 <= self.interface_ratio <= 1.0:
            raise ValueError("interface_ratio must be in [0.0, 1.0]")
        if self.archive_size <= 0:
            raise ValueError("archive_size must be > 0")
        if self.weight_std <= 0.0:
            raise ValueError("weight_std must be > 0.0")
        if self.max_complexity is not None and self.max_complexity <= 0:
            raise ValueError("max_complexity must be > 0 when set")

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "operator": self.operator,
            "merge_rate": self.merge_rate,
            "symbiont_source": self.symbiont_source,
            "symbiont_fate": self.symbiont_fate,
            "archive_size": self.archive_size,
            "interface_count": self.interface_count,
            "interface_ratio": self.interface_ratio,
            "weight_method": self.weight_method,
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "max_complexity": self.max_complexity,
            "operator_params": dict(self.operator_params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConfig:
        """Create from dictionary."""
        return cls(
            operator=data.get("operator", "graph_symbiogenetic"),
            merge_rate=data.get("merge_rate", 0.0),
            symbiont_source=data.get("symbiont_source", "cross_species"),
            symbiont_fate=data.get("symbiont_fate", "consumed"),
            archive_size=data.get("archive_size", 50),
            interface_count=data.get("interface_count", 4),
            interface_ratio=data.get("interface_ratio", 0.5),
            weight_method=data.get("weight_method", "mean"),
            weight_mean=data.get("weight_mean", 0.0),
            weight_std=data.get("weight_std", 1.0),
            max_complexity=data.get("max_complexity"),
            operator_params=data.get("operator_params", {}),
        )


__all__ = ["MergeConfig"]
