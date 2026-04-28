"""
Ensemble Metric Collector.

Computes five population-level metrics derived purely from individual
fitness values and existing CollectionContext fields:

  - ensemble/gini_coefficient           (always present)
  - ensemble/participation_ratio        (always present)
  - ensemble/top_k_concentration        (always present)
  - ensemble/expert_turnover            (present when context.previous_elites is not None)
  - ensemble/specialization_index       (present when context.species_info is not None)

Enabled via MetricCategory.ENSEMBLE in TrackingConfig.categories.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from evolve.experiment.collectors.base import CollectionContext

_logger = logging.getLogger(__name__)


@dataclass
class EnsembleMetricCollector:
    """
    Stateless collector for ensemble diversity and concentration metrics.

    Attributes:
        top_k_percent: Percentage of top individuals used for top-k concentration
            and (when elite_size is None) expert turnover. Must be in (0.0, 100.0].
            Default: 10.0.
        elite_size: Fixed number of individuals counted as the elite set for
            Expert Turnover. When None, derived automatically as
            ceil(top_k_percent / 100 * population_size). Must be >= 1 when set.
            Default: None.
    """

    top_k_percent: float = 10.0
    elite_size: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.top_k_percent <= 100.0):
            raise ValueError(f"top_k_percent must be in (0.0, 100.0], got {self.top_k_percent}")
        if self.elite_size is not None and self.elite_size < 1:
            raise ValueError(f"elite_size must be >= 1 when set, got {self.elite_size}")

    def reset(self) -> None:
        """No-op: EnsembleMetricCollector is stateless."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_fitnesses(self, context: CollectionContext) -> npt.NDArray[np.float64] | None:
        """
        Extract scalar fitness values from the population.

        Supports two fitness access patterns:
          1. hasattr(fitness, 'values') → float(fitness.values[0])   [standard Fitness]
          2. otherwise                  → float(fitness.value)        [scalar custom]

        Individuals with None fitness are skipped with a debug log.
        An all-zero shift is applied when any value is negative.

        Returns:
            1-D numpy array of non-negative floats, or None when the array
            would be empty (all individuals have None fitness or population is empty).
        """
        raw: list[float] = []
        for ind in context.population:
            if ind.fitness is None:
                _logger.debug("Skipping individual with None fitness")
                continue
            try:
                if hasattr(ind.fitness, "values"):
                    raw.append(float(ind.fitness.values[0]))
                else:
                    raw.append(float(ind.fitness.value))  # type: ignore[attr-defined]
            except (TypeError, IndexError, AttributeError) as exc:
                _logger.debug("Could not extract fitness scalar: %s", exc)

        if not raw:
            return None

        fitnesses = np.array(raw, dtype=float)

        min_val = float(np.min(fitnesses))
        if min_val < 0.0:
            _logger.debug(
                "Negative fitness detected (min=%.4f); shifting all values by %.4f",
                min_val,
                abs(min_val),
            )
            fitnesses = fitnesses - min_val  # shift so min is 0

        return fitnesses

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect(self, context: CollectionContext) -> dict[str, float]:
        """
        Compute ensemble metrics for the current generation.

        Always-present keys (when population has >= 1 valid fitness):
          ensemble/gini_coefficient
          ensemble/participation_ratio
          ensemble/top_k_concentration

        Conditionally-present keys:
          ensemble/expert_turnover      — only when context.previous_elites is not None
          ensemble/specialization_index — only when context.species_info is not None

        Returns:
            dict of metric_key → float.  Empty dict when all individuals have
            None fitness or the population is empty.
        """
        fitnesses = self._extract_fitnesses(context)
        if fitnesses is None:
            return {}

        N = len(fitnesses)
        metrics: dict[str, float] = {}

        # ---- Gini coefficient ----------------------------------------
        total = float(np.sum(fitnesses))
        if total == 0.0:
            metrics["ensemble/gini_coefficient"] = 0.0
        else:
            sorted_f = np.sort(fitnesses)
            cumsum = np.cumsum(sorted_f)
            metrics["ensemble/gini_coefficient"] = float(
                (N + 1 - 2.0 * np.sum(cumsum) / cumsum[-1]) / N
            )

        # ---- Participation Ratio --------------------------------------
        sum_sq = float(np.sum(fitnesses**2))
        if sum_sq == 0.0:
            metrics["ensemble/participation_ratio"] = float(N)
        else:
            metrics["ensemble/participation_ratio"] = float(np.sum(fitnesses) ** 2 / sum_sq)

        # ---- Top-k Concentration -------------------------------------
        k_count = max(1, math.ceil(self.top_k_percent / 100.0 * N))
        if total == 0.0:
            metrics["ensemble/top_k_concentration"] = 0.0
        else:
            # np.partition is O(N) — avoids full sort
            top_k_vals = np.partition(fitnesses, -k_count)[-k_count:]
            metrics["ensemble/top_k_concentration"] = float(np.sum(top_k_vals) / total)

        # ---- Expert Turnover (conditional) ---------------------------
        if context.previous_elites is not None:
            elite_count = (
                self.elite_size
                if self.elite_size is not None
                else max(1, math.ceil(self.top_k_percent / 100.0 * N))
            )
            # Derive current elite by identity (sort population by fitness desc)
            population_list = list(context.population)
            sorted_inds = sorted(
                population_list,
                key=lambda ind: (
                    float(ind.fitness.values[0])
                    if ind.fitness is not None and hasattr(ind.fitness, "values")
                    else float(ind.fitness.value)  # type: ignore[attr-defined]
                    if ind.fitness is not None
                    else (float("inf") if context.minimize else float("-inf"))
                ),
                reverse=not context.minimize,
            )
            current_elite = sorted_inds[:elite_count]

            # Use UUID (.id) when available (Individual carries the same UUID
            # through with_fitness() calls), falling back to Python object
            # identity for non-framework objects (e.g., mocks in tests).
            def _key(ind: Any) -> Any:
                if hasattr(ind, "id"):
                    return ind.id
                return id(ind)

            current_keys = {_key(ind) for ind in current_elite}
            prev_keys = {_key(ind) for ind in context.previous_elites}
            if len(current_keys) == 0:
                metrics["ensemble/expert_turnover"] = 0.0
            else:
                metrics["ensemble/expert_turnover"] = float(
                    len(current_keys - prev_keys) / len(current_keys)
                )

        # ---- Specialization Index (conditional) ----------------------
        if context.species_info is not None:
            # Use UNSHIFTED fitnesses — variance is shift-invariant, but we
            # re-extract raw to keep semantics clean.
            raw_list: list[float] = []
            pop_list = list(context.population)
            for ind in pop_list:
                if ind.fitness is None:
                    continue
                try:
                    if hasattr(ind.fitness, "values"):
                        raw_list.append(float(ind.fitness.values[0]))
                    else:
                        raw_list.append(float(ind.fitness.value))  # type: ignore[attr-defined]
                except (TypeError, IndexError, AttributeError):
                    continue

            if not raw_list:
                metrics["ensemble/specialization_index"] = 0.0
            else:
                fitnesses_raw = np.array(raw_list, dtype=float)
                grand_mean = float(np.mean(fitnesses_raw))
                ss_total = float(np.sum((fitnesses_raw - grand_mean) ** 2))

                if ss_total == 0.0:
                    metrics["ensemble/specialization_index"] = 0.0
                else:
                    ss_between = 0.0
                    for _species_id, indices in context.species_info.items():
                        valid_indices = [i for i in indices if i < len(pop_list)]
                        species_fits: list[float] = []
                        for i in valid_indices:
                            ind = pop_list[i]
                            if ind.fitness is None:
                                continue
                            try:
                                if hasattr(ind.fitness, "values"):
                                    species_fits.append(float(ind.fitness.values[0]))
                                else:
                                    species_fits.append(float(ind.fitness.value))  # type: ignore[attr-defined]
                            except (TypeError, IndexError, AttributeError):
                                continue
                        n_s = len(species_fits)
                        if n_s == 0:
                            continue
                        species_mean = float(np.mean(species_fits))
                        ss_between += n_s * (species_mean - grand_mean) ** 2

                    metrics["ensemble/specialization_index"] = float(ss_between / ss_total)

        return metrics
