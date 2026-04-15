"""
Sequence Genome - Variable-length sequence representation.

Useful for genetic programming, variable-length encodings,
and discrete optimization problems.

Registry name: ``"sequence"``

Declarative usage::

    config = UnifiedConfig(
        genome_type="sequence",
        genome_params={"length": 50, "alphabet": (0, 1)},
        ...
    )

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SequenceGenome(Generic[T]):
    """
    Variable-length sequence genome.

    Represents a sequence of genes that can grow or shrink
    during evolution. Useful for genetic programming and
    variable-length encodings.

    Attributes:
        genes: Immutable tuple of gene values
        alphabet: Optional set of valid gene values
        min_length: Minimum allowed length
        max_length: Maximum allowed length (None = unlimited)

    Example:
        >>> # Binary sequence
        >>> genome = SequenceGenome(
        ...     genes=(1, 0, 1, 1, 0),
        ...     alphabet=frozenset({0, 1}),
        ...     min_length=1,
        ...     max_length=10
        ... )
        >>>
        >>> # Symbol sequence for GP
        >>> genome = SequenceGenome(
        ...     genes=('+', 'x', '*', 'y', '2'),
        ...     alphabet=frozenset({'+', '-', '*', '/', 'x', 'y', '1', '2'})
        ... )
    """

    genes: tuple[T, ...]
    alphabet: frozenset[T] | None = None
    min_length: int = 1
    max_length: int | None = None

    def __post_init__(self) -> None:
        """Validate genome constraints."""
        if len(self.genes) < self.min_length:
            raise ValueError(f"Genome length {len(self.genes)} < min_length {self.min_length}")
        if self.max_length is not None and len(self.genes) > self.max_length:
            raise ValueError(f"Genome length {len(self.genes)} > max_length {self.max_length}")
        if self.alphabet is not None:
            for gene in self.genes:
                if gene not in self.alphabet:
                    raise ValueError(f"Gene {gene} not in alphabet {self.alphabet}")

    def copy(self) -> SequenceGenome[T]:
        """Create a copy of this genome."""
        return SequenceGenome(
            genes=self.genes,
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length,
        )

    def __eq__(self, other: object) -> bool:
        """Structural equality based on genes."""
        if not isinstance(other, SequenceGenome):
            return False
        return self.genes == other.genes

    def __hash__(self) -> int:
        """Hash for set/dict membership."""
        return hash(self.genes)

    def distance(self, other: SequenceGenome[T]) -> float:
        """Compute Levenshtein edit distance to another SequenceGenome."""
        if not isinstance(other, SequenceGenome):
            raise TypeError(
                f"Cannot compute distance between SequenceGenome and {type(other).__name__}"
            )
        # Wagner-Fischer algorithm for Levenshtein distance
        a, b = self.genes, other.genes
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return float(dp[n])

    def __len__(self) -> int:
        """Number of genes."""
        return len(self.genes)

    def __getitem__(self, idx: int | slice) -> T | tuple[T, ...]:
        """Get gene(s) by index or slice."""
        return self.genes[idx]

    def __iter__(self) -> Iterator[T]:
        """Iterate over genes."""
        return iter(self.genes)

    def with_gene(self, idx: int, value: T) -> SequenceGenome[T]:
        """
        Return copy with gene at idx replaced.

        Args:
            idx: Index to replace
            value: New gene value

        Returns:
            New genome with updated gene
        """
        if self.alphabet is not None and value not in self.alphabet:
            raise ValueError(f"Gene {value} not in alphabet")

        new_genes = list(self.genes)
        new_genes[idx] = value
        return SequenceGenome(
            genes=tuple(new_genes),
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length,
        )

    def with_insert(self, idx: int, value: T) -> SequenceGenome[T]:
        """
        Return copy with gene inserted at idx.

        Args:
            idx: Index to insert at
            value: Gene value to insert

        Returns:
            New genome with gene inserted

        Raises:
            ValueError: If would exceed max_length
        """
        if self.max_length is not None and len(self.genes) >= self.max_length:
            raise ValueError("Cannot insert: would exceed max_length")
        if self.alphabet is not None and value not in self.alphabet:
            raise ValueError(f"Gene {value} not in alphabet")

        new_genes = list(self.genes)
        new_genes.insert(idx, value)
        return SequenceGenome(
            genes=tuple(new_genes),
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length,
        )

    def with_delete(self, idx: int) -> SequenceGenome[T]:
        """
        Return copy with gene at idx removed.

        Args:
            idx: Index to remove

        Returns:
            New genome with gene removed

        Raises:
            ValueError: If would go below min_length
        """
        if len(self.genes) <= self.min_length:
            raise ValueError("Cannot delete: would go below min_length")

        new_genes = list(self.genes)
        del new_genes[idx]
        return SequenceGenome(
            genes=tuple(new_genes),
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length,
        )

    def with_append(self, value: T) -> SequenceGenome[T]:
        """
        Return copy with gene appended.

        Args:
            value: Gene value to append

        Returns:
            New genome with gene appended
        """
        return self.with_insert(len(self.genes), value)

    def with_slice(self, start: int, end: int) -> SequenceGenome[T]:
        """
        Return copy with only genes in slice.

        Args:
            start: Start index
            end: End index (exclusive)

        Returns:
            New genome with sliced genes
        """
        new_genes = self.genes[start:end]
        if len(new_genes) < self.min_length:
            raise ValueError("Slice would go below min_length")

        return SequenceGenome(
            genes=new_genes,
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "genes": list(self.genes),
            "alphabet": list(self.alphabet) if self.alphabet else None,
            "min_length": self.min_length,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceGenome:
        """Reconstruct from dict."""
        alphabet = frozenset(data["alphabet"]) if data.get("alphabet") else None
        return cls(
            genes=tuple(data["genes"]),
            alphabet=alphabet,
            min_length=data.get("min_length", 1),
            max_length=data.get("max_length"),
        )

    @classmethod
    def random(
        cls,
        length: int,
        alphabet: frozenset[T],
        rng: Random,
        min_length: int = 1,
        max_length: int | None = None,
    ) -> SequenceGenome[T]:
        """
        Create random genome from alphabet.

        Args:
            length: Number of genes
            alphabet: Set of valid gene values
            rng: Random number generator
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            Random genome
        """
        alphabet_list = list(alphabet)
        genes = tuple(rng.choice(alphabet_list) for _ in range(length))
        return cls(
            genes=genes,
            alphabet=alphabet,
            min_length=min_length,
            max_length=max_length,
        )
