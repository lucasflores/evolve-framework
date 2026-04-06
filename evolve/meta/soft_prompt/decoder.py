"""
SoftPromptDecoder — Decode EmbeddingGenome into text via embedding-layer injection.

Requires torch and transformers (optional dependencies).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from evolve.representation.embedding import EmbeddingGenome

logger = logging.getLogger(__name__)


@dataclass
class SoftPromptDecoder:
    """
    Decodes an EmbeddingGenome into model output by injecting
    soft-prompt embeddings into a language model's embedding layer.

    Attributes:
        model_id: HuggingFace model identifier.
        device: Compute device ("cpu" or "cuda").
        max_generation_tokens: Default max tokens for generation.
    """

    model_id: str
    device: str = "cpu"
    max_generation_tokens: int = 256

    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _embed_dim: int | None = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        """Lazily load model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "SoftPromptDecoder requires torch and transformers. "
                "Install with: pip install evolve-framework[pytorch]"
            ) from err

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Cache embed_dim from the model's embedding layer
            embed_layer = self._model.get_input_embeddings()
            self._embed_dim = embed_layer.embedding_dim
        except Exception as e:
            self._model = None
            self._tokenizer = None
            raise RuntimeError(f"Failed to load model '{self.model_id}': {e}") from e

    @property
    def embed_dim(self) -> int:
        """Model embedding dimension (triggers lazy load)."""
        self._ensure_loaded()
        assert self._embed_dim is not None
        return self._embed_dim

    def decode(
        self,
        genome: EmbeddingGenome,
        task_input: str,
        max_tokens: int | None = None,
    ) -> str:
        """
        Decode genome into text by injecting embeddings into the model.

        Args:
            genome: EmbeddingGenome to decode.
            task_input: Text input to prepend soft prompt to.
            max_tokens: Override max generation tokens.

        Returns:
            Generated text string.
        """
        import torch

        self._ensure_loaded()

        # Validate model match (FR-004)
        if genome.model_id != self.model_id:
            raise ValueError(
                f"Model mismatch: genome targets '{genome.model_id}' "
                f"but decoder loaded '{self.model_id}'"
            )

        # Validate embedding dimension
        if genome.embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: genome has {genome.embed_dim}, "
                f"model has {self.embed_dim}"
            )

        effective_max = max_tokens or self.max_generation_tokens

        # Tokenize input
        input_ids = self._tokenizer.encode(task_input, return_tensors="pt").to(self.device)

        # Get input embeddings
        embed_layer = self._model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)

        # Create soft prompt embeddings tensor
        soft_prompt = torch.tensor(
            genome.embeddings, dtype=input_embeds.dtype, device=self.device
        ).unsqueeze(0)  # (1, n_tokens, embed_dim)

        # Concatenate: soft_prompt + input_embeds
        combined = torch.cat([soft_prompt, input_embeds], dim=1)

        # Generate with no gradients
        with torch.no_grad():
            outputs = self._model.generate(
                inputs_embeds=combined,
                max_new_tokens=effective_max,
                do_sample=False,
            )

        # Decode only new tokens
        generated_ids = outputs[0]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text into the model's embedding space.

        Args:
            text: Text to embed.

        Returns:
            ndarray of shape (n_tokens, embed_dim) for the tokenized text.
        """
        import torch

        self._ensure_loaded()

        input_ids = self._tokenizer.encode(text, return_tensors="pt").to(self.device)
        embed_layer = self._model.get_input_embeddings()

        with torch.no_grad():
            embeds = embed_layer(input_ids)

        # Remove batch dim, convert to numpy
        return embeds.squeeze(0).cpu().numpy()
