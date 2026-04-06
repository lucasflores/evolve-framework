"""
EmbeddingGenome — 2D continuous soft-prompt embedding representation.

Implements Genome and SerializableGenome protocols.

**FR-016**: This module MUST NOT import torch, transformers, or any ML framework.
Only numpy, dataclasses, typing, and Python stdlib are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.representation.embedding_config import DimensionalityStrategy

if TYPE_CHECKING:
    from evolve.representation.vector import VectorGenome


@dataclass(frozen=True)
class EmbeddingGenome:
    """
    2D continuous soft-prompt embedding genome.

    Stores a matrix of shape (n_tokens, embed_dim) representing
    virtual token embeddings for injection into a language model.

    Attributes:
        embeddings: 2D ndarray shape (n_tokens, embed_dim), immutable.
        model_id: Target model identifier.
        seed_text: Original seed text for initialization (optional).
        strategy: Dimensionality strategy that produced this genome.
    """

    embeddings: np.ndarray
    model_id: str
    seed_text: str | None = None
    strategy: DimensionalityStrategy = DimensionalityStrategy.MINIMAL_TOKENS

    def __post_init__(self) -> None:
        # Convert to ndarray if needed
        if not isinstance(self.embeddings, np.ndarray):
            object.__setattr__(self, "embeddings", np.asarray(self.embeddings, dtype=np.float32))

        # Ensure float32
        if self.embeddings.dtype != np.float32:
            object.__setattr__(self, "embeddings", self.embeddings.astype(np.float32))

        # Validate 2D
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {self.embeddings.ndim}D")
        if self.embeddings.shape[0] < 1 or self.embeddings.shape[1] < 1:
            raise ValueError(
                f"embeddings shape must have both dims >= 1, got {self.embeddings.shape}"
            )

        # Validate model_id
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")

        # Make immutable
        self.embeddings.flags.writeable = False

    @property
    def n_tokens(self) -> int:
        """Number of virtual tokens."""
        return self.embeddings.shape[0]

    @property
    def embed_dim(self) -> int:
        """Model embedding dimension."""
        return self.embeddings.shape[1]

    def copy(self) -> EmbeddingGenome:
        """Deep copy with copied embeddings array."""
        return EmbeddingGenome(
            embeddings=self.embeddings.copy(),
            model_id=self.model_id,
            seed_text=self.seed_text,
            strategy=self.strategy,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingGenome):
            return False
        return (
            self.model_id == other.model_id
            and self.strategy == other.strategy
            and self.seed_text == other.seed_text
            and np.array_equal(self.embeddings, other.embeddings)
        )

    def __hash__(self) -> int:
        return hash((self.model_id, self.seed_text, self.strategy, self.embeddings.tobytes()))

    # ── Flat-vector adapters ──────────────────────────────────────────

    def flat(self) -> np.ndarray:
        """Return a 1D copy of the embeddings, shape (n_tokens * embed_dim,)."""
        flat = self.embeddings.reshape(-1).copy()
        flat.flags.writeable = False
        return flat

    def to_vector_genome(self) -> VectorGenome:
        """Convert to VectorGenome for use with flat-vector operators."""
        from evolve.representation.vector import VectorGenome

        return VectorGenome(genes=self.embeddings.reshape(-1).copy())

    @classmethod
    def from_vector_genome(
        cls,
        vg: VectorGenome,
        n_tokens: int,
        model_id: str,
        seed_text: str | None = None,
        strategy: DimensionalityStrategy = DimensionalityStrategy.MINIMAL_TOKENS,
    ) -> EmbeddingGenome:
        """Reconstruct from a VectorGenome (flat representation)."""
        embed_dim = len(vg.genes) // n_tokens
        if len(vg.genes) != n_tokens * embed_dim:
            raise ValueError(
                f"VectorGenome length {len(vg.genes)} is not divisible by n_tokens={n_tokens}"
            )
        return cls(
            embeddings=vg.genes.reshape(n_tokens, embed_dim),
            model_id=model_id,
            seed_text=seed_text,
            strategy=strategy,
        )

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "embeddings": self.embeddings.tolist(),
            "model_id": self.model_id,
            "seed_text": self.seed_text,
            "strategy": self.strategy.value,
            "n_tokens": self.n_tokens,
            "embed_dim": self.embed_dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingGenome:
        """Reconstruct from dict."""
        return cls(
            embeddings=np.array(data["embeddings"], dtype=np.float32),
            model_id=data["model_id"],
            seed_text=data.get("seed_text"),
            strategy=DimensionalityStrategy(data["strategy"]),
        )
