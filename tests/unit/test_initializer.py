"""Unit tests for PopulationInitializer (T025)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from evolve.meta.soft_prompt.initializer import PopulationInitializer
from evolve.representation.embedding import EmbeddingGenome
from evolve.representation.embedding_config import (
    DimensionalityStrategy,
    EmbeddingGenomeConfig,
)


@dataclass
class MockDecoder:
    """Mock decoder for initializer tests."""

    model_id: str = "test-model"
    _embed_dim: int = 16

    def embed_text(self, text: str) -> np.ndarray:
        """Return deterministic embeddings based on text length."""
        n_tokens = max(1, len(text.split()))
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.standard_normal((n_tokens, self._embed_dim)).astype(np.float32)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


@pytest.fixture
def config() -> EmbeddingGenomeConfig:
    return EmbeddingGenomeConfig(
        n_tokens=4,
        embed_dim=16,
        model_id="test-model",
        seed_text="answer the question",
    )


@pytest.fixture
def config_no_seed() -> EmbeddingGenomeConfig:
    return EmbeddingGenomeConfig(
        n_tokens=4,
        embed_dim=16,
        model_id="test-model",
    )


@pytest.fixture
def decoder() -> MockDecoder:
    return MockDecoder()


class TestNoiseInit:
    def test_population_size(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        init = PopulationInitializer(config=config, decoder=decoder)
        pop = init.noise_init(population_size=10, seed=42)
        assert len(pop) == 10

    def test_genome_shape(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        init = PopulationInitializer(config=config, decoder=decoder)
        pop = init.noise_init(population_size=5, seed=42)
        for g in pop:
            assert isinstance(g, EmbeddingGenome)
            assert g.n_tokens == 4
            assert g.embed_dim == 16
            assert g.model_id == "test-model"

    def test_mutual_distinctness(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        """All genomes should be pairwise distinct (L2 > 0)."""
        init = PopulationInitializer(config=config, decoder=decoder)
        pop = init.noise_init(population_size=10, seed=42)

        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                dist = float(np.linalg.norm(pop[i].embeddings - pop[j].embeddings))
                assert dist > 0, f"Genomes {i} and {j} are identical"

    def test_no_seed_text(
        self, config_no_seed: EmbeddingGenomeConfig, decoder: MockDecoder
    ) -> None:
        """Without seed text, should still produce valid population."""
        init = PopulationInitializer(config=config_no_seed, decoder=decoder)
        pop = init.noise_init(population_size=5, seed=42)
        assert len(pop) == 5
        for g in pop:
            assert g.n_tokens == 4
            assert g.embed_dim == 16

    def test_deterministic(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        init = PopulationInitializer(config=config, decoder=decoder)
        pop1 = init.noise_init(population_size=5, seed=42)
        pop2 = init.noise_init(population_size=5, seed=42)
        for g1, g2 in zip(pop1, pop2):
            np.testing.assert_array_equal(g1.embeddings, g2.embeddings)


class TestPadTruncate:
    def test_truncate_long_seed(self, decoder: MockDecoder) -> None:
        """When seed text has more tokens than n_tokens, truncate."""
        cfg = EmbeddingGenomeConfig(
            n_tokens=2,
            embed_dim=16,
            model_id="test-model",
            seed_text="this is a very long seed text with many tokens",
            strategy=DimensionalityStrategy.FULL_SPACE,
        )
        init = PopulationInitializer(config=cfg, decoder=decoder)
        pop = init.noise_init(population_size=3, seed=42)
        for g in pop:
            assert g.n_tokens == 2

    def test_pad_short_seed(self, decoder: MockDecoder) -> None:
        """When seed text has fewer tokens than n_tokens, pad."""
        cfg = EmbeddingGenomeConfig(
            n_tokens=8,
            embed_dim=16,
            model_id="test-model",
            seed_text="short",
        )
        init = PopulationInitializer(config=cfg, decoder=decoder)
        pop = init.noise_init(population_size=3, seed=42)
        for g in pop:
            assert g.n_tokens == 8


class TestLLMVariationInit:
    def test_with_paraphrases(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        init = PopulationInitializer(config=config, decoder=decoder)
        paraphrases = ["variant one two three", "variant four five six", "variant seven eight"]
        pop = init.llm_variation_init(population_size=6, seed=42, paraphrases=paraphrases)

        assert len(pop) == 6
        for g in pop:
            assert g.n_tokens == 4
            assert g.embed_dim == 16

    def test_fallback_without_paraphrases(
        self, config: EmbeddingGenomeConfig, decoder: MockDecoder
    ) -> None:
        """Without paraphrases, falls back to noise_init."""
        init = PopulationInitializer(config=config, decoder=decoder)
        pop = init.llm_variation_init(population_size=5, seed=42, paraphrases=None)
        assert len(pop) == 5

    def test_fallback_without_seed_text(
        self, config_no_seed: EmbeddingGenomeConfig, decoder: MockDecoder
    ) -> None:
        """Without seed text, falls back to noise_init."""
        init = PopulationInitializer(config=config_no_seed, decoder=decoder)
        pop = init.llm_variation_init(population_size=5, seed=42, paraphrases=["var1", "var2"])
        assert len(pop) == 5

    def test_mutual_distinctness(self, config: EmbeddingGenomeConfig, decoder: MockDecoder) -> None:
        init = PopulationInitializer(config=config, decoder=decoder)
        paraphrases = ["alpha beta gamma delta", "one two three four"]
        pop = init.llm_variation_init(population_size=5, seed=42, paraphrases=paraphrases)

        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                dist = float(np.linalg.norm(pop[i].embeddings - pop[j].embeddings))
                assert dist > 0
