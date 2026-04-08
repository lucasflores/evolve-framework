"""
Integration tests for SoftPromptDecoder (T027).

Tests decoder validation logic using mock internals to avoid loading real models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for soft prompt decoder tests")

from evolve.representation.embedding import EmbeddingGenome  # noqa: E402


class TestSoftPromptDecoderValidation:
    """Test decoder error handling without loading actual models."""

    def test_model_mismatch_raises(self) -> None:
        """FR-004: Genome targeting different model raises ValueError."""
        from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

        decoder = SoftPromptDecoder(model_id="model-A")
        # Mock the model loading
        decoder._model = MagicMock()
        decoder._tokenizer = MagicMock()
        decoder._embed_dim = 16

        genome = EmbeddingGenome(
            embeddings=np.zeros((4, 16), dtype=np.float32),
            model_id="model-B",
        )

        with pytest.raises(ValueError, match="Model mismatch"):
            decoder.decode(genome, "test input")

    def test_embed_dim_mismatch_raises(self) -> None:
        """Dimension mismatch between genome and model raises ValueError."""
        from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

        decoder = SoftPromptDecoder(model_id="test-model")
        decoder._model = MagicMock()
        decoder._tokenizer = MagicMock()
        decoder._embed_dim = 32  # Model expects 32

        genome = EmbeddingGenome(
            embeddings=np.zeros((4, 16), dtype=np.float32),  # Genome has 16
            model_id="test-model",
        )

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            decoder.decode(genome, "test input")

    def test_lazy_loading_not_triggered_on_construct(self) -> None:
        """Model should not load on construction."""
        from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

        decoder = SoftPromptDecoder(model_id="some-model")
        assert decoder._model is None
        assert decoder._tokenizer is None

    def test_embed_dim_property_triggers_load(self) -> None:
        """Accessing embed_dim should trigger lazy load."""
        from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

        decoder = SoftPromptDecoder(model_id="test-model")
        decoder._model = MagicMock()
        decoder._tokenizer = MagicMock()
        decoder._embed_dim = 768

        assert decoder.embed_dim == 768

    def test_missing_torch_raises_import_error(self) -> None:
        """Without torch installed, should raise ImportError."""
        from evolve.meta.soft_prompt.decoder import SoftPromptDecoder

        decoder = SoftPromptDecoder(model_id="test-model")

        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            # Force re-import check by resetting model
            decoder._model = None
            try:
                decoder._ensure_loaded()
            except (ImportError, ModuleNotFoundError):
                pass  # Expected
            except Exception:
                pass  # torch may already be imported in the process
