"""
TokenAwareMutator — Per-token Gaussian mutation for EmbeddingGenome.

Implements MutationOperator[EmbeddingGenome] protocol (FR-007).
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolve.representation.embedding import EmbeddingGenome


@dataclass
class TokenAwareMutator:
    """
    Per-token Gaussian mutation with optional coherence clamping.

    For each token, with probability `mutation_rate`, adds Gaussian noise
    of standard deviation `sigma`. If `coherence_radius` is set, the
    perturbation norm is clamped.

    Attributes:
        mutation_rate: Probability of mutating each token.
        sigma: Standard deviation of Gaussian noise.
        coherence_radius: Max L2 norm of perturbation per token (optional).
    """

    mutation_rate: float = 0.1
    sigma: float = 0.1
    coherence_radius: float | None = None

    def mutate(self, genome: EmbeddingGenome, rng: Random) -> EmbeddingGenome:
        """
        Apply per-token Gaussian mutation.

        Args:
            genome: Input EmbeddingGenome.
            rng: Seeded Random instance.

        Returns:
            New EmbeddingGenome with mutated tokens.
        """
        from evolve.representation.embedding import EmbeddingGenome as EG

        new_embeddings = genome.embeddings.copy()
        n_tokens, embed_dim = new_embeddings.shape

        for i in range(n_tokens):
            if rng.random() < self.mutation_rate:
                delta = np.array(
                    [rng.gauss(0, self.sigma) for _ in range(embed_dim)],
                    dtype=np.float32,
                )

                if self.coherence_radius is not None:
                    norm = float(np.linalg.norm(delta))
                    if norm > self.coherence_radius:
                        delta = delta * (self.coherence_radius / norm)

                new_embeddings[i] = new_embeddings[i] + delta

        return EG(
            embeddings=new_embeddings,
            model_id=genome.model_id,
            seed_text=genome.seed_text,
            strategy=genome.strategy,
        )
