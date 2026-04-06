"""
ESPOConfig — Aggregate configuration for ESPO experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evolve.representation.embedding_config import (
    EmbeddingGenomeConfig,
)


@dataclass(frozen=True)
class CoherenceDefenseConfig:
    """Configuration for the three-layer coherence defense."""

    enable_mutation_clamp: bool = True
    enable_perplexity_check: bool = True
    enable_fitness_selection: bool = True
    coherence_radius: float = 0.1
    perplexity_threshold: float = 100.0


@dataclass(frozen=True)
class ESPOConfig:
    """
    Aggregate configuration for an ESPO experiment.

    Attributes:
        genome: Embedding genome configuration.
        population_size: Number of individuals per generation.
        n_generations: Number of generations to evolve.
        mutation_rate: Per-token mutation probability.
        mutation_sigma: Gaussian noise standard deviation.
        crossover_type: Token-level crossover type.
        init_strategy: Population initialization strategy.
        coherence: Coherence defense configuration.
        device: Compute device ("cpu" or "cuda").
        seed: Random seed for reproducibility.
    """

    genome: EmbeddingGenomeConfig = field(
        default_factory=lambda: EmbeddingGenomeConfig(model_id="default")
    )
    population_size: int = 50
    n_generations: int = 50
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    crossover_type: str = "single_point"
    init_strategy: str = "noise"
    coherence: CoherenceDefenseConfig = field(default_factory=CoherenceDefenseConfig)
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {self.n_generations}")
        if not (0.0 < self.mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in (0, 1], got {self.mutation_rate}")
        if self.mutation_sigma <= 0:
            raise ValueError(f"mutation_sigma must be > 0, got {self.mutation_sigma}")
        if self.crossover_type not in ("single_point", "two_point"):
            raise ValueError(
                f"crossover_type must be 'single_point' or 'two_point', got '{self.crossover_type}'"
            )
        if self.init_strategy not in ("noise", "llm_variation"):
            raise ValueError(
                f"init_strategy must be 'noise' or 'llm_variation', got '{self.init_strategy}'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.to_dict(),
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "mutation_rate": self.mutation_rate,
            "mutation_sigma": self.mutation_sigma,
            "crossover_type": self.crossover_type,
            "init_strategy": self.init_strategy,
            "coherence": {
                "enable_mutation_clamp": self.coherence.enable_mutation_clamp,
                "enable_perplexity_check": self.coherence.enable_perplexity_check,
                "enable_fitness_selection": self.coherence.enable_fitness_selection,
                "coherence_radius": self.coherence.coherence_radius,
                "perplexity_threshold": self.coherence.perplexity_threshold,
            },
            "device": self.device,
            "seed": self.seed,
        }
