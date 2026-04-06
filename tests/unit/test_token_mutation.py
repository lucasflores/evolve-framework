"""Unit tests for TokenAwareMutator (T020)."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.core.operators.token_mutation import TokenAwareMutator
from evolve.representation.embedding import EmbeddingGenome


@pytest.fixture
def genome() -> EmbeddingGenome:
    return EmbeddingGenome(
        embeddings=np.ones((4, 8), dtype=np.float32),
        model_id="test",
    )


class TestTokenAwareMutator:
    def test_basic_mutation(self, genome: EmbeddingGenome) -> None:
        mutator = TokenAwareMutator(mutation_rate=1.0, sigma=0.5)
        rng = Random(42)
        result = mutator.mutate(genome, rng)

        assert isinstance(result, EmbeddingGenome)
        assert result.n_tokens == genome.n_tokens
        assert result.embed_dim == genome.embed_dim
        assert result.model_id == genome.model_id
        # With rate=1.0, all tokens should be mutated
        assert not np.array_equal(result.embeddings, genome.embeddings)

    def test_no_mutation_rate_zero(self, genome: EmbeddingGenome) -> None:
        mutator = TokenAwareMutator(mutation_rate=0.0, sigma=0.5)
        rng = Random(42)
        result = mutator.mutate(genome, rng)
        np.testing.assert_array_equal(result.embeddings, genome.embeddings)

    def test_unmutated_tokens_identical(self) -> None:
        """Unmutated tokens should be bitwise identical to input."""
        rng = Random(42)
        genome = EmbeddingGenome(
            embeddings=np.ones((100, 8), dtype=np.float32),
            model_id="test",
        )
        mutator = TokenAwareMutator(mutation_rate=0.1, sigma=0.5)
        result = mutator.mutate(genome, rng)

        # At least some tokens should be unchanged
        unchanged = sum(
            1
            for i in range(genome.n_tokens)
            if np.array_equal(result.embeddings[i], genome.embeddings[i])
        )
        assert unchanged > 0

    def test_coherence_radius_clamping(self, genome: EmbeddingGenome) -> None:
        radius = 0.01
        mutator = TokenAwareMutator(mutation_rate=1.0, sigma=10.0, coherence_radius=radius)
        rng = Random(42)
        result = mutator.mutate(genome, rng)

        for i in range(genome.n_tokens):
            delta = result.embeddings[i] - genome.embeddings[i]
            norm = float(np.linalg.norm(delta))
            if norm > 0:  # token was mutated
                assert norm <= radius + 1e-6

    def test_deterministic_with_seed(self, genome: EmbeddingGenome) -> None:
        mutator = TokenAwareMutator(mutation_rate=0.5, sigma=0.3)
        r1 = mutator.mutate(genome, Random(42))
        r2 = mutator.mutate(genome, Random(42))
        np.testing.assert_array_equal(r1.embeddings, r2.embeddings)

    def test_preserves_metadata(self, genome: EmbeddingGenome) -> None:  # noqa: ARG002
        genome_with_text = EmbeddingGenome(
            embeddings=np.ones((4, 8), dtype=np.float32),
            model_id="test",
            seed_text="hello",
        )
        mutator = TokenAwareMutator(mutation_rate=1.0, sigma=0.1)
        result = mutator.mutate(genome_with_text, Random(1))
        assert result.seed_text == "hello"
        assert result.strategy == genome_with_text.strategy
