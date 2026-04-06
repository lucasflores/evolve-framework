"""
TokenLevelCrossover — Whole-token crossover for EmbeddingGenome.

Implements CrossoverOperator[EmbeddingGenome] protocol (FR-008).
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolve.representation.embedding import EmbeddingGenome


@dataclass
class TokenLevelCrossover:
    """
    Token-level crossover swapping whole tokens between parents.

    Supports single-point and two-point crossover.

    Attributes:
        crossover_type: "single_point" or "two_point".
    """

    crossover_type: str = "single_point"

    def crossover(
        self,
        parent1: EmbeddingGenome,
        parent2: EmbeddingGenome,
        rng: Random,
    ) -> tuple[EmbeddingGenome, EmbeddingGenome]:
        """
        Perform token-level crossover.

        Args:
            parent1: First parent genome.
            parent2: Second parent genome.
            rng: Seeded Random instance.

        Returns:
            Tuple of two child EmbeddingGenome instances.
        """
        from evolve.representation.embedding import EmbeddingGenome as EG

        # Validate parent compatibility
        if parent1.n_tokens != parent2.n_tokens:
            raise ValueError(f"Token count mismatch: {parent1.n_tokens} vs {parent2.n_tokens}")
        if parent1.embed_dim != parent2.embed_dim:
            raise ValueError(f"Embed dim mismatch: {parent1.embed_dim} vs {parent2.embed_dim}")
        if parent1.model_id != parent2.model_id:
            raise ValueError(f"Model ID mismatch: '{parent1.model_id}' vs '{parent2.model_id}'")

        n_tokens = parent1.n_tokens
        p1 = parent1.embeddings.copy()
        p2 = parent2.embeddings.copy()

        if self.crossover_type == "single_point":
            if n_tokens < 2:
                # Can't crossover with 1 token, return copies
                return (
                    parent1.copy(),
                    parent2.copy(),
                )
            k = rng.randint(1, n_tokens - 1)
            c1 = np.vstack([p1[:k], p2[k:]])
            c2 = np.vstack([p2[:k], p1[k:]])

        elif self.crossover_type == "two_point":
            if n_tokens < 3:
                # Fall back to single-point
                k = rng.randint(1, max(1, n_tokens - 1))
                c1 = np.vstack([p1[:k], p2[k:]])
                c2 = np.vstack([p2[:k], p1[k:]])
            else:
                points = sorted(rng.sample(range(1, n_tokens), 2))
                k1, k2 = points
                c1 = np.vstack([p1[:k1], p2[k1:k2], p1[k2:]])
                c2 = np.vstack([p2[:k1], p1[k1:k2], p2[k2:]])
        else:
            raise ValueError(
                f"crossover_type must be 'single_point' or 'two_point', got '{self.crossover_type}'"
            )

        return (
            EG(
                embeddings=c1,
                model_id=parent1.model_id,
                seed_text=None,
                strategy=parent1.strategy,
            ),
            EG(
                embeddings=c2,
                model_id=parent1.model_id,
                seed_text=None,
                strategy=parent1.strategy,
            ),
        )
