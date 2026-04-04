"""
Fitness Metadata Collector.

Automatically extracts and aggregates domain-specific data from Fitness.metadata
fields. Supports:
- Numeric field extraction with aggregation (best, mean, std)
- Configurable metadata prefix
- Majority threshold for sparse fields
- Nested metadata structure handling

Implements FR-018, FR-019, FR-020 from the tracking specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.experiment.collectors.base import CollectionContext

if TYPE_CHECKING:
    pass


_logger = logging.getLogger(__name__)


@dataclass
class FitnessMetadataCollector:
    """
    Collector for extracting metrics from Fitness.metadata fields.

    Automatically discovers numeric fields in Fitness.metadata across
    the population and aggregates them into trackable metrics.

    Attributes:
        prefix: Prefix for extracted metadata fields (default: "meta_").
        threshold: Minimum fraction of individuals with a field to include it.
        aggregations: List of aggregation functions ("best", "mean", "std", "min", "max").
        flatten_nested: Whether to flatten nested dicts with dot notation.
        max_depth: Maximum nesting depth for flattening.
        skip_fields: Field names to skip during extraction.

    Example:
        >>> from evolve.experiment.collectors.metadata import FitnessMetadataCollector
        >>>
        >>> collector = FitnessMetadataCollector(prefix="meta_", threshold=0.5)
        >>> context = CollectionContext(generation=10, population=population)
        >>> metrics = collector.collect(context)
        >>> # If evaluator populates fitness.metadata["latency"] = 0.5
        >>> metrics.get("meta_latency_mean")
        0.45
    """

    prefix: str = "meta_"
    threshold: float = 0.5
    aggregations: tuple[str, ...] = ("best", "mean", "std")
    flatten_nested: bool = True
    max_depth: int = 3
    skip_fields: frozenset[str] = field(default_factory=frozenset)

    # Track which fields we've warned about
    _warned_non_numeric: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._warned_non_numeric = set()

    def collect(self, context: CollectionContext) -> dict[str, Any]:
        """
        Collect metadata metrics from population.

        Args:
            context: Collection context with population.

        Returns:
            Dictionary of aggregated metadata metrics with configured prefix.
        """
        metrics: dict[str, Any] = {}

        # Extract all metadata fields from population
        field_values = self._extract_fields(context)

        if not field_values:
            return metrics

        # Aggregate each field that meets the threshold
        population_size = context.population_size

        for field_name, values in field_values.items():
            # Check threshold
            coverage = len(values) / population_size
            if coverage < self.threshold:
                _logger.debug(
                    f"Skipping field '{field_name}' - only {coverage:.1%} coverage "
                    f"(threshold: {self.threshold:.1%})"
                )
                continue

            # Aggregate values
            field_metrics = self._aggregate_field(field_name, values, context)
            metrics.update(field_metrics)

        return metrics

    def reset(self) -> None:
        """Reset internal state between runs."""
        self._warned_non_numeric = set()

    def _extract_fields(self, context: CollectionContext) -> dict[str, list[float]]:
        """
        Extract numeric metadata fields from all individuals.

        Args:
            context: Collection context.

        Returns:
            Dict mapping field names to lists of values.
        """
        field_values: dict[str, list[float]] = {}

        for ind in context.population.individuals:
            if ind.fitness is None:
                continue

            metadata = getattr(ind.fitness, "metadata", None)
            if not metadata:
                continue

            # Extract fields (potentially flattened)
            flat_metadata = self._flatten_metadata(metadata) if self.flatten_nested else metadata

            for key, value in flat_metadata.items():
                # Skip configured fields
                if key in self.skip_fields:
                    continue

                # Try to convert to numeric
                numeric_value = self._try_numeric(key, value)
                if numeric_value is not None:
                    if key not in field_values:
                        field_values[key] = []
                    field_values[key].append(numeric_value)

        return field_values

    def _flatten_metadata(
        self,
        metadata: dict[str, Any],
        prefix: str = "",
        depth: int = 0,
    ) -> dict[str, Any]:
        """
        Flatten nested metadata dict using dot notation.

        Args:
            metadata: Metadata dict to flatten.
            prefix: Current key prefix.
            depth: Current nesting depth.

        Returns:
            Flattened metadata dict.
        """
        if depth >= self.max_depth:
            return {}

        result: dict[str, Any] = {}

        for key, value in metadata.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dict
                nested = self._flatten_metadata(value, f"{full_key}.", depth + 1)
                result.update(nested)
            else:
                result[full_key] = value

        return result

    def _try_numeric(self, key: str, value: Any) -> float | None:
        """
        Try to convert value to numeric.

        Args:
            key: Field name (for logging).
            value: Value to convert.

        Returns:
            Float value or None if not numeric.
        """
        if isinstance(value, (int, float)):
            if np.isfinite(value):
                return float(value)
            return None

        if isinstance(value, np.ndarray):
            if value.size == 1:
                val = float(value.flat[0])
                if np.isfinite(val):
                    return val
            return None

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._try_numeric(key, value[0])
            return None

        # Try casting string to float
        if isinstance(value, str):
            try:
                val = float(value)
                if np.isfinite(val):
                    return val
            except (ValueError, TypeError):
                pass

        # Log warning for non-numeric fields (once per field)
        if key not in self._warned_non_numeric:
            _logger.debug(
                f"Skipping non-numeric metadata field '{key}' with type {type(value).__name__}"
            )
            self._warned_non_numeric.add(key)

        return None

    def _aggregate_field(
        self,
        field_name: str,
        values: list[float],
        context: CollectionContext,
    ) -> dict[str, float]:
        """
        Aggregate values for a single field.

        Args:
            field_name: Metadata field name.
            values: List of numeric values.
            context: Collection context (for best_index, etc.).

        Returns:
            Dict of aggregated metrics.
        """
        metrics: dict[str, float] = {}
        arr = np.array(values)

        for agg in self.aggregations:
            metric_name = f"{self.prefix}{field_name}_{agg}"

            if agg == "mean":
                metrics[metric_name] = float(np.mean(arr))

            elif agg == "std":
                metrics[metric_name] = float(np.std(arr))

            elif agg == "min":
                metrics[metric_name] = float(np.min(arr))

            elif agg == "max":
                metrics[metric_name] = float(np.max(arr))

            elif agg == "best":
                # "best" uses the value from the best individual
                # We need to find it based on context or population
                best_value = self._get_best_value(field_name, context)
                if best_value is not None:
                    metrics[metric_name] = best_value

            elif agg == "median":
                metrics[metric_name] = float(np.median(arr))

            elif agg == "sum":
                metrics[metric_name] = float(np.sum(arr))

            elif agg == "count":
                metrics[metric_name] = float(len(arr))

            else:
                _logger.warning(f"Unknown aggregation '{agg}' for field '{field_name}'")

        return metrics

    def _get_best_value(
        self,
        field_name: str,
        context: CollectionContext,
    ) -> float | None:
        """
        Get metadata field value from the best individual.

        Args:
            field_name: Metadata field name.
            context: Collection context.

        Returns:
            Field value from best individual, or None.
        """
        # Try to get best individual from population
        try:
            best_list = context.population.best(1)
            if not best_list:
                return None
            best = best_list[0]
        except (AttributeError, TypeError):
            return None

        if best.fitness is None:
            return None

        metadata = getattr(best.fitness, "metadata", None)
        if not metadata:
            return None

        # Handle flattened keys
        if self.flatten_nested and "." in field_name:
            # Navigate nested structure
            parts = field_name.split(".")
            current = metadata
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return self._try_numeric(field_name, current)

        if field_name in metadata:
            return self._try_numeric(field_name, metadata[field_name])

        return None
