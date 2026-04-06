"""Tests for text-mediated cross-model transfer (FR-014, FR-017)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from evolve.meta.soft_prompt.transfer import text_mediated_transfer
from evolve.representation.embedding import EmbeddingGenome
from evolve.representation.embedding_config import EmbeddingGenomeConfig


def _make_genome(model_id: str = "model-a") -> EmbeddingGenome:
    rng = np.random.RandomState(42)
    return EmbeddingGenome(
        embeddings=rng.randn(4, 16).astype(np.float32),
        model_id=model_id,
    )


def _make_decoder(
    model_id: str = "model-a", decode_result: str = "evolved prompt text"
) -> MagicMock:
    decoder = MagicMock()
    decoder.model_id = model_id
    decoder.decode.return_value = decode_result
    return decoder


class TestTextMediatedTransfer:
    def test_produces_non_empty_text(self) -> None:
        """Transfer produces a non-empty text string."""
        genome = _make_genome()
        decoder = _make_decoder()
        result = text_mediated_transfer(genome, decoder)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_stripped_text(self) -> None:
        """Transfer strips whitespace from decoded text."""
        genome = _make_genome()
        decoder = _make_decoder(decode_result="  some text  \n")
        result = text_mediated_transfer(genome, decoder)
        assert result == "some text"

    def test_model_mismatch_raises(self) -> None:
        """Mismatched model IDs raise ValueError."""
        genome = _make_genome("model-a")
        decoder = _make_decoder("model-b")
        with pytest.raises(ValueError, match="Model mismatch"):
            text_mediated_transfer(genome, decoder)

    def test_empty_output_raises(self) -> None:
        """Empty decode result raises ValueError."""
        genome = _make_genome()
        decoder = _make_decoder(decode_result="   ")
        with pytest.raises(ValueError, match="empty text"):
            text_mediated_transfer(genome, decoder)

    def test_passes_probe_input(self) -> None:
        """Probe input is forwarded to decoder.decode()."""
        genome = _make_genome()
        decoder = _make_decoder()
        text_mediated_transfer(genome, decoder, probe_input="Tell me about AI")
        decoder.decode.assert_called_once_with(genome, "Tell me about AI", max_tokens=None)

    def test_passes_max_tokens(self) -> None:
        """max_tokens is forwarded to decoder."""
        genome = _make_genome()
        decoder = _make_decoder()
        text_mediated_transfer(genome, decoder, max_tokens=50)
        decoder.decode.assert_called_once_with(genome, "", max_tokens=50)


class TestReseeding:
    """Test that transferred text can be used as seed for a new model (FR-017)."""

    def test_transferred_text_as_seed(self) -> None:
        """Transferred text can be used as seed_text for a different model."""
        genome = _make_genome("model-a")
        decoder = _make_decoder("model-a", decode_result="Summarize the key concepts")

        transferred = text_mediated_transfer(genome, decoder)

        # Use transferred text as seed for model-b config
        config_b = EmbeddingGenomeConfig(
            n_tokens=8,
            embed_dim=32,
            model_id="model-b",
            seed_text=transferred,
        )

        assert config_b.seed_text == "Summarize the key concepts"
        assert config_b.model_id == "model-b"

    def test_different_model_configs(self) -> None:
        """Transferred text works with any model_id and embed_dim."""
        transferred = "evolved instructions for the task"

        for model_id, embed_dim in [
            ("mistral-7b", 4096),
            ("llama-8b", 4096),
            ("gpt2", 768),
        ]:
            config = EmbeddingGenomeConfig(
                n_tokens=4,
                embed_dim=embed_dim,
                model_id=model_id,
                seed_text=transferred,
            )
            assert config.seed_text == transferred
            assert config.model_id == model_id
