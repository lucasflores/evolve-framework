"""
VectorGenome - Fixed-length real-valued vector representation.

This is the most common genome type for continuous optimization
and neuroevolution weight encoding.

Registry name: ``"vector"``

Declarative usage::

    config = UnifiedConfig(
        genome_type="vector",
        genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
        ...
    )
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from random import Random
from typing import Any

with contextlib.suppress(ImportError):
    pass

import numpy as np


@dataclass(frozen=True)
class VectorGenome:
    """
    Fixed-length real-valued vector genome.

    Common for continuous optimization, neuroevolution weights.

    Attributes:
        genes: NumPy array of gene values, shape (n_genes,)
        bounds: Optional (lower, upper) bounds for each gene

    Example:
        >>> genome = VectorGenome(
        ...     genes=np.array([1.0, 2.0, 3.0]),
        ...     bounds=(np.zeros(3), np.ones(3) * 10)
        ... )
        >>> len(genome)
        3
        >>> genome.clip_to_bounds()  # Ensures genes within bounds
    """

    genes: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Ensure immutability and validate."""
        # Convert to numpy array if needed
        if not isinstance(self.genes, np.ndarray):
            object.__setattr__(self, "genes", np.asarray(self.genes, dtype=np.float64))

        # Ensure 1D
        if self.genes.ndim != 1:
            object.__setattr__(self, "genes", self.genes.flatten())

        # Make immutable
        self.genes.flags.writeable = False

        # Validate and freeze bounds
        if self.bounds is not None:
            lower, upper = self.bounds
            if not isinstance(lower, np.ndarray):
                lower = np.asarray(lower, dtype=np.float64)
            if not isinstance(upper, np.ndarray):
                upper = np.asarray(upper, dtype=np.float64)

            lower.flags.writeable = False
            upper.flags.writeable = False

            object.__setattr__(self, "bounds", (lower, upper))

            # Validate dimensions match
            if len(lower) != len(self.genes) or len(upper) != len(self.genes):
                raise ValueError(
                    f"Bounds dimensions {len(lower)}, {len(upper)} "
                    f"must match genes dimension {len(self.genes)}"
                )

    def copy(self) -> VectorGenome:
        """Create deep copy of genome."""
        return VectorGenome(
            genes=self.genes.copy(),
            bounds=self.bounds,
        )

    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        if not isinstance(other, VectorGenome):
            return False
        return np.array_equal(self.genes, other.genes)

    def __hash__(self) -> int:
        """Hash for set/dict membership."""
        return hash(self.genes.tobytes())

    def distance(self, other: VectorGenome) -> float:
        """Compute L2 (Euclidean) distance to another VectorGenome."""
        if not isinstance(other, VectorGenome):
            raise TypeError(
                f"Cannot compute distance between VectorGenome and {type(other).__name__}"
            )
        return float(np.linalg.norm(self.genes - other.genes))

    def __len__(self) -> int:
        """Number of genes."""
        return len(self.genes)

    def __getitem__(self, idx: int) -> float:
        """Access individual gene."""
        return float(self.genes[idx])

    def clip_to_bounds(self) -> VectorGenome:
        """
        Return genome with genes clipped to bounds.

        Returns:
            New VectorGenome with genes within bounds
        """
        if self.bounds is None:
            return self

        lower, upper = self.bounds
        clipped = np.clip(self.genes, lower, upper)
        return VectorGenome(genes=clipped, bounds=self.bounds)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "genes": self.genes.tolist(),
        }
        if self.bounds is not None:
            result["bounds"] = [
                self.bounds[0].tolist(),
                self.bounds[1].tolist(),
            ]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VectorGenome:
        """Reconstruct from dict."""
        bounds = None
        if "bounds" in data and data["bounds"] is not None:
            bounds = (
                np.array(data["bounds"][0]),
                np.array(data["bounds"][1]),
            )
        return cls(
            genes=np.array(data["genes"]),
            bounds=bounds,
        )

    @classmethod
    def random(
        cls,
        n_genes: int,
        bounds: tuple[np.ndarray, np.ndarray],
        rng: Random,
    ) -> VectorGenome:
        """
        Create random genome within bounds.

        Args:
            n_genes: Number of genes
            bounds: (lower, upper) bounds arrays
            rng: Random number generator

        Returns:
            Random VectorGenome
        """
        lower, upper = bounds
        genes = np.array([rng.uniform(float(lower[i]), float(upper[i])) for i in range(n_genes)])
        return cls(genes=genes, bounds=bounds)

    @classmethod
    def zeros(
        cls, n_genes: int, bounds: tuple[np.ndarray, np.ndarray] | None = None
    ) -> VectorGenome:
        """Create genome with all zeros."""
        return cls(genes=np.zeros(n_genes), bounds=bounds)

    @classmethod
    def ones(
        cls, n_genes: int, bounds: tuple[np.ndarray, np.ndarray] | None = None
    ) -> VectorGenome:
        """Create genome with all ones."""
        return cls(genes=np.ones(n_genes), bounds=bounds)


class VectorIdentityDecoder:
    """
    Decoder that returns genome genes as phenotype.

    For direct function optimization where genes are
    the input to the fitness function.
    """

    def decode(self, genome: VectorGenome) -> np.ndarray:
        """Return genes as phenotype."""
        return genome.genes
