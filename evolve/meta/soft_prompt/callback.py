"""
ESPOCallback — Logs per-generation ESPO metrics and handles recovery.

Logs best/mean/worst fitness, diversity stats, infeasibility rate,
mutation magnitude, and best decoded text (FR-018).
Implements all-infeasible recovery (FR-012).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.core.callbacks import SimpleCallback

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

logger = logging.getLogger(__name__)


@dataclass
class ESPOCallback(SimpleCallback):
    """
    Callback for logging ESPO-specific metrics per generation.

    Computes and logs:
    - best/mean/worst fitness
    - pairwise L2 diversity stats (mean, std) over embeddings
    - infeasibility rate
    - mutation magnitude (from metrics dict if available)
    - best individual's decoded text (if decoder provided)

    Handles all-infeasible recovery (FR-012):
    - Detects when all individuals are infeasible
    - Stores previous generation for restoration
    - Signals mutation reduction via recovery_state

    Attributes:
        decoder: Optional SoftPromptDecoder for decoding best individual.
        decode_input: Input text for decoding the best genome.
        tracker: Optional MLflow tracker for logging.
    """

    decoder: SoftPromptDecoder | None = None
    decode_input: str = ""
    tracker: Any = None

    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _previous_population: Any = field(default=None, repr=False)
    _recovery_state: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def recovery_state(self) -> dict[str, Any]:
        """Current recovery state — mutation_reduction_factor and restored flag."""
        return dict(self._recovery_state)

    def on_generation_end(
        self,
        generation: int,
        population: Population[Any],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log ESPO metrics at end of each generation."""
        gen_metrics: dict[str, Any] = {"generation": generation}

        # Extract fitness values
        fitness_values: list[float] = []
        infeasible_count = 0

        for ind in population.individuals:
            if ind.fitness is not None:
                fv = float(ind.fitness.values[0])
                fitness_values.append(fv)
                if not ind.fitness.is_feasible:
                    infeasible_count += 1

        if fitness_values:
            gen_metrics["best_fitness"] = max(fitness_values)
            gen_metrics["mean_fitness"] = float(np.mean(fitness_values))
            gen_metrics["worst_fitness"] = min(fitness_values)
            gen_metrics["infeasibility_rate"] = infeasible_count / len(fitness_values)

        # Diversity: pairwise L2 distances over flat embeddings
        embeddings_list: list[np.ndarray] = []
        for ind in population.individuals:
            genome = ind.genome
            if hasattr(genome, "flat"):
                embeddings_list.append(genome.flat())

        if len(embeddings_list) >= 2:
            distances: list[float] = []
            for i in range(len(embeddings_list)):
                for j in range(i + 1, len(embeddings_list)):
                    d = float(np.linalg.norm(embeddings_list[i] - embeddings_list[j]))
                    distances.append(d)
            gen_metrics["diversity_l2_mean"] = float(np.mean(distances))
            gen_metrics["diversity_l2_std"] = float(np.std(distances))

        # Mutation magnitude from upstream metrics
        if metrics and "mutation_magnitude" in metrics:
            gen_metrics["mutation_magnitude"] = metrics["mutation_magnitude"]

        # Decode best individual
        if self.decoder is not None and self.decode_input and fitness_values:
            best_ind = max(
                population.individuals,
                key=lambda ind: float(ind.fitness.values[0]) if ind.fitness else float("-inf"),
            )
            try:
                decoded = self.decoder.decode(best_ind.genome, self.decode_input)
                gen_metrics["best_decoded_text"] = decoded
            except Exception as e:
                logger.warning(f"Failed to decode best genome at gen {generation}: {e}")

        # Log to tracker if available
        if self.tracker is not None:
            try:
                numeric_metrics = {
                    k: v for k, v in gen_metrics.items() if isinstance(v, int | float)
                }
                self.tracker.log_generation(generation, numeric_metrics)
            except Exception as e:
                logger.warning(f"Failed to log to tracker at gen {generation}: {e}")

        # All-infeasible recovery (FR-012)
        infeasibility_rate = gen_metrics.get("infeasibility_rate", 0.0)
        if fitness_values and infeasibility_rate >= 1.0:
            logger.warning(
                f"Generation {generation}: all individuals infeasible. Triggering recovery."
            )
            current_factor = self._recovery_state.get("mutation_reduction_factor", 1.0)
            self._recovery_state = {
                "all_infeasible": True,
                "restored_generation": generation,
                "mutation_reduction_factor": current_factor * 0.5,
                "previous_population": self._previous_population,
            }
            gen_metrics["recovery_triggered"] = True
            gen_metrics["mutation_reduction_factor"] = self._recovery_state[
                "mutation_reduction_factor"
            ]
        else:
            # Store current population as backup for next generation
            self._previous_population = population
            if "all_infeasible" in self._recovery_state:
                self._recovery_state.pop("all_infeasible")

        self._history.append(gen_metrics)
