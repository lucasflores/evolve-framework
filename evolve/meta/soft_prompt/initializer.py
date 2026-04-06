"""
PopulationInitializer — Create initial ESPO populations from seed text.

Supports noise-based and LLM-variation initialization strategies (FR-010, FR-011).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
    from evolve.representation.embedding_config import EmbeddingGenomeConfig

from evolve.representation.embedding import EmbeddingGenome

logger = logging.getLogger(__name__)


@dataclass
class PopulationInitializer:
    """
    Initialize ESPO populations from seed text.

    Attributes:
        config: Genome configuration (n_tokens, embed_dim, etc).
        decoder: SoftPromptDecoder for embedding seed text.
    """

    config: EmbeddingGenomeConfig
    decoder: SoftPromptDecoder

    def noise_init(
        self,
        population_size: int,
        seed: int = 42,
    ) -> list[EmbeddingGenome]:
        """
        Initialize population via calibrated Gaussian noise (FR-010).

        1. Embed seed text to get base embeddings.
        2. Pad or truncate to n_tokens.
        3. Add Gaussian noise scaled by coherence_radius to create variants.

        Args:
            population_size: Number of individuals to create.
            seed: Random seed for reproducibility.

        Returns:
            List of EmbeddingGenome instances.
        """
        rng = np.random.default_rng(seed)

        # Get seed embeddings
        if self.config.seed_text is not None:
            base_embeddings = self.decoder.embed_text(self.config.seed_text)
            base_embeddings = self._pad_or_truncate(base_embeddings, self.config.n_tokens)
        else:
            # No seed text: use small random initialization
            base_embeddings = (
                rng.standard_normal((self.config.n_tokens, self.config.embed_dim)).astype(
                    np.float32
                )
                * 0.02
            )

        population: list[EmbeddingGenome] = []

        for _ in range(population_size):
            noise = rng.standard_normal(base_embeddings.shape).astype(np.float32)
            noise *= self.config.coherence_radius
            perturbed = base_embeddings + noise

            genome = EmbeddingGenome(
                embeddings=perturbed,
                model_id=self.config.model_id,
                seed_text=self.config.seed_text,
                strategy=self.config.strategy,
            )
            population.append(genome)

        return population

    def _pad_or_truncate(self, embeddings: np.ndarray, target_tokens: int) -> np.ndarray:
        """Pad or truncate embeddings to target_tokens rows."""
        current = embeddings.shape[0]

        if current == target_tokens:
            return embeddings.astype(np.float32)
        elif current > target_tokens:
            return embeddings[:target_tokens].astype(np.float32)
        else:
            # Pad with mean embedding
            pad_rows = target_tokens - current
            mean_embed = embeddings.mean(axis=0, keepdims=True)
            padding = np.tile(mean_embed, (pad_rows, 1))
            return np.vstack([embeddings, padding]).astype(np.float32)

    def llm_variation_init(
        self,
        population_size: int,
        seed: int = 42,
        paraphrases: list[str] | None = None,
    ) -> list[EmbeddingGenome]:
        """
        Initialize population via LLM-discovered text variations (FR-011).

        Embeds paraphrase variants of the seed text. Falls back to
        noise_init if no paraphrases or seed text is available.

        Args:
            population_size: Number of individuals to create.
            seed: Random seed for reproducibility.
            paraphrases: Pre-computed paraphrase variants of the seed text.

        Returns:
            List of EmbeddingGenome instances.
        """
        if self.config.seed_text is None or not paraphrases:
            logger.info("No seed_text or paraphrases available, falling back to noise_init")
            return self.noise_init(population_size, seed=seed)

        rng = np.random.default_rng(seed)

        population: list[EmbeddingGenome] = []

        for i in range(population_size):
            # Cycle through paraphrases
            text = paraphrases[i % len(paraphrases)]
            embeddings = self.decoder.embed_text(text)
            embeddings = self._pad_or_truncate(embeddings, self.config.n_tokens)

            # Add small noise for uniqueness
            noise = rng.standard_normal(embeddings.shape).astype(np.float32)
            noise *= self.config.coherence_radius * 0.1
            embeddings = embeddings + noise

            genome = EmbeddingGenome(
                embeddings=embeddings,
                model_id=self.config.model_id,
                seed_text=text,
                strategy=self.config.strategy,
            )
            population.append(genome)

        return population
