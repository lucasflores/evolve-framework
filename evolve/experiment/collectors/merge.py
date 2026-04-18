"""
Merge Metric Collector.

Collects metrics for symbiogenetic merge events:
- merge/count: Number of merge operations per generation
- merge/mean_genome_complexity: Average gene count of merged offspring
- merge/complexity_delta: Mean increase in complexity from merge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from evolve.experiment.collectors.base import CollectionContext

_logger = logging.getLogger(__name__)


@dataclass
class MergeMetricCollector:
    """
    Collector for symbiogenetic merge metrics.

    Tracks merge event counts, genome complexity changes, and
    per-generation merge statistics.

    Attributes:
        _merge_count: Number of merges in the current generation.
        _complexity_deltas: Complexity increases from each merge.
        _merged_complexities: Complexity of each merged offspring.
    """

    _merge_count: int = field(default=0, init=False)
    _complexity_deltas: list[float] = field(default_factory=list, init=False)
    _merged_complexities: list[float] = field(default_factory=list, init=False)

    def record_merge(
        self,
        host_complexity: int,
        symbiont_complexity: int,  # noqa: ARG002
        merged_complexity: int,
    ) -> None:
        """
        Record a single merge event.

        Args:
            host_complexity: Gene count of host before merge.
            symbiont_complexity: Gene count of symbiont.
            merged_complexity: Gene count of merged offspring.
        """
        self._merge_count += 1
        self._complexity_deltas.append(float(merged_complexity - host_complexity))
        self._merged_complexities.append(float(merged_complexity))

    def collect(self, context: CollectionContext) -> dict[str, Any]:  # noqa: ARG002
        """
        Collect merge metrics for the current generation.

        Args:
            context: Collection context (unused — metrics come from record_merge).

        Returns:
            Dictionary of merge metrics.
        """
        metrics: dict[str, Any] = {
            "merge/count": self._merge_count,
        }

        if self._merged_complexities:
            metrics["merge/mean_genome_complexity"] = sum(self._merged_complexities) / len(
                self._merged_complexities
            )
        else:
            metrics["merge/mean_genome_complexity"] = 0.0

        if self._complexity_deltas:
            metrics["merge/complexity_delta"] = sum(self._complexity_deltas) / len(
                self._complexity_deltas
            )
        else:
            metrics["merge/complexity_delta"] = 0.0

        return metrics

    def reset(self) -> None:
        """Reset per-generation state."""
        self._merge_count = 0
        self._complexity_deltas.clear()
        self._merged_complexities.clear()

    def reset_generation(self) -> None:
        """Reset per-generation state (alias for reset)."""
        self.reset()


__all__ = ["MergeMetricCollector"]
