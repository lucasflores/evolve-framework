"""
Embedding genome configuration — dimensionality strategies and config.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DimensionalityStrategy(Enum):
    """Dimensionality strategy for embedding genome evolution."""

    FULL_SPACE = "FULL_SPACE"
    COMPRESSED_SUBSPACE = "COMPRESSED_SUBSPACE"
    MINIMAL_TOKENS = "MINIMAL_TOKENS"


@dataclass(frozen=True)
class EmbeddingGenomeConfig:
    """
    Configuration for creating EmbeddingGenome instances.

    Attributes:
        n_tokens: Number of virtual tokens.
        embed_dim: Model embedding dimension.
        model_id: Target model identifier.
        strategy: Dimensionality strategy (default: MINIMAL_TOKENS).
        seed_text: Optional seed text for initialization.
        coherence_radius: L2 norm bound as fraction of mean embedding distance.
        subspace_dim: Required when strategy=COMPRESSED_SUBSPACE.
        projection_matrix: Required when strategy=COMPRESSED_SUBSPACE,
            shape (subspace_dim, embed_dim).
    """

    n_tokens: int = 8
    embed_dim: int = 768
    model_id: str = ""
    strategy: DimensionalityStrategy = DimensionalityStrategy.MINIMAL_TOKENS
    seed_text: str | None = None
    coherence_radius: float = 0.1
    subspace_dim: int | None = None
    projection_matrix: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {self.embed_dim}")
        if not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if self.coherence_radius <= 0:
            raise ValueError(f"coherence_radius must be > 0, got {self.coherence_radius}")
        if self.strategy == DimensionalityStrategy.COMPRESSED_SUBSPACE:
            if self.subspace_dim is None:
                raise ValueError("subspace_dim is required when strategy=COMPRESSED_SUBSPACE")
            if self.projection_matrix is None:
                raise ValueError("projection_matrix is required when strategy=COMPRESSED_SUBSPACE")
            if self.projection_matrix.shape != (self.subspace_dim, self.embed_dim):
                raise ValueError(
                    f"projection_matrix shape must be ({self.subspace_dim}, {self.embed_dim}), "
                    f"got {self.projection_matrix.shape}"
                )
        if self.strategy == DimensionalityStrategy.MINIMAL_TOKENS and not (4 <= self.n_tokens <= 8):
            warnings.warn(
                f"MINIMAL_TOKENS strategy typically uses 4-8 tokens, got {self.n_tokens}",
                UserWarning,
                stacklevel=2,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "n_tokens": self.n_tokens,
            "embed_dim": self.embed_dim,
            "model_id": self.model_id,
            "strategy": self.strategy.value,
            "seed_text": self.seed_text,
            "coherence_radius": self.coherence_radius,
            "subspace_dim": self.subspace_dim,
        }
        if self.projection_matrix is not None:
            result["projection_matrix"] = self.projection_matrix.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingGenomeConfig:
        """Reconstruct from dict."""
        projection = None
        if data.get("projection_matrix") is not None:
            projection = np.array(data["projection_matrix"])
        return cls(
            n_tokens=data["n_tokens"],
            embed_dim=data["embed_dim"],
            model_id=data["model_id"],
            strategy=DimensionalityStrategy(data["strategy"]),
            seed_text=data.get("seed_text"),
            coherence_radius=data.get("coherence_radius", 0.1),
            subspace_dim=data.get("subspace_dim"),
            projection_matrix=projection,
        )
