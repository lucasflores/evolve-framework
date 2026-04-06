"""Unit tests for TokenLevelCrossover (T021)."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.core.operators.token_crossover import TokenLevelCrossover
from evolve.representation.embedding import EmbeddingGenome


@pytest.fixture
def parent1() -> EmbeddingGenome:
    return EmbeddingGenome(
        embeddings=np.zeros((6, 8), dtype=np.float32),
        model_id="test",
    )


@pytest.fixture
def parent2() -> EmbeddingGenome:
    return EmbeddingGenome(
        embeddings=np.ones((6, 8), dtype=np.float32),
        model_id="test",
    )


class TestTokenLevelCrossover:
    def test_single_point_shapes(self, parent1: EmbeddingGenome, parent2: EmbeddingGenome) -> None:
        cx = TokenLevelCrossover(crossover_type="single_point")
        c1, c2 = cx.crossover(parent1, parent2, Random(42))
        assert c1.n_tokens == 6
        assert c1.embed_dim == 8
        assert c2.n_tokens == 6
        assert c2.embed_dim == 8

    def test_two_point_shapes(self, parent1: EmbeddingGenome, parent2: EmbeddingGenome) -> None:
        cx = TokenLevelCrossover(crossover_type="two_point")
        c1, c2 = cx.crossover(parent1, parent2, Random(42))
        assert c1.n_tokens == 6
        assert c2.n_tokens == 6

    def test_whole_token_preservation(
        self, parent1: EmbeddingGenome, parent2: EmbeddingGenome
    ) -> None:
        """Each child token must come entirely from one parent."""
        cx = TokenLevelCrossover(crossover_type="single_point")
        c1, c2 = cx.crossover(parent1, parent2, Random(42))

        for i in range(c1.n_tokens):
            from_p1 = np.array_equal(c1.embeddings[i], parent1.embeddings[i])
            from_p2 = np.array_equal(c1.embeddings[i], parent2.embeddings[i])
            assert from_p1 or from_p2, f"Child1 token {i} not from either parent"

        for i in range(c2.n_tokens):
            from_p1 = np.array_equal(c2.embeddings[i], parent1.embeddings[i])
            from_p2 = np.array_equal(c2.embeddings[i], parent2.embeddings[i])
            assert from_p1 or from_p2, f"Child2 token {i} not from either parent"

    def test_token_mismatch_raises(self) -> None:
        p1 = EmbeddingGenome(embeddings=np.zeros((4, 8), dtype=np.float32), model_id="test")
        p2 = EmbeddingGenome(embeddings=np.zeros((6, 8), dtype=np.float32), model_id="test")
        cx = TokenLevelCrossover()
        with pytest.raises(ValueError, match="Token count"):
            cx.crossover(p1, p2, Random(1))

    def test_dim_mismatch_raises(self) -> None:
        p1 = EmbeddingGenome(embeddings=np.zeros((4, 8), dtype=np.float32), model_id="test")
        p2 = EmbeddingGenome(embeddings=np.zeros((4, 16), dtype=np.float32), model_id="test")
        cx = TokenLevelCrossover()
        with pytest.raises(ValueError, match="Embed dim"):
            cx.crossover(p1, p2, Random(1))

    def test_model_id_mismatch_raises(self) -> None:
        p1 = EmbeddingGenome(embeddings=np.zeros((4, 8), dtype=np.float32), model_id="a")
        p2 = EmbeddingGenome(embeddings=np.zeros((4, 8), dtype=np.float32), model_id="b")
        cx = TokenLevelCrossover()
        with pytest.raises(ValueError, match="Model ID"):
            cx.crossover(p1, p2, Random(1))

    def test_children_seed_text_is_none(
        self, parent1: EmbeddingGenome, parent2: EmbeddingGenome
    ) -> None:
        cx = TokenLevelCrossover()
        c1, c2 = cx.crossover(parent1, parent2, Random(42))
        assert c1.seed_text is None
        assert c2.seed_text is None

    def test_deterministic(self, parent1: EmbeddingGenome, parent2: EmbeddingGenome) -> None:
        cx = TokenLevelCrossover()
        c1a, c2a = cx.crossover(parent1, parent2, Random(42))
        c1b, c2b = cx.crossover(parent1, parent2, Random(42))
        np.testing.assert_array_equal(c1a.embeddings, c1b.embeddings)
        np.testing.assert_array_equal(c2a.embeddings, c2b.embeddings)

    def test_single_token_returns_copies(self) -> None:
        """With 1 token, crossover can't swap — returns copies."""
        p1 = EmbeddingGenome(embeddings=np.zeros((1, 8), dtype=np.float32), model_id="test")
        p2 = EmbeddingGenome(embeddings=np.ones((1, 8), dtype=np.float32), model_id="test")
        cx = TokenLevelCrossover()
        c1, c2 = cx.crossover(p1, p2, Random(42))
        assert c1 == p1
        assert c2 == p2
