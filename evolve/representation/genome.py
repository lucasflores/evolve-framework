"""
Genome protocol - Framework-neutral genetic representation.

This module defines the abstract Genome interface that all concrete
genome types must implement.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class Genome(Protocol):
    """
    Framework-neutral genetic representation.
    
    Genomes MUST:
    - Be immutable (return new instances on modification)
    - Be serializable (pickle or custom)
    - Support equality and hashing
    - NOT contain PyTorch/JAX types
    
    All evolutionary operators work with this protocol.
    Concrete implementations include:
    - VectorGenome: Fixed-length real-valued vectors
    - SequenceGenome: Variable-length sequences
    - GraphGenome: NEAT-style neural network topologies
    """

    def copy(self) -> Self:
        """
        Create deep copy of genome.
        
        Returns:
            A new genome instance with copied data
        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Structural equality.
        
        Two genomes are equal if they represent the same genetic information.
        """
        ...

    def __hash__(self) -> int:
        """
        Hash for set/dict membership.
        
        Must be consistent with __eq__: equal genomes must have equal hashes.
        """
        ...


@runtime_checkable
class SerializableGenome(Genome, Protocol):
    """
    Genome with portable serialization.
    
    Extends Genome with JSON-compatible serialization methods
    for checkpointing and experiment logging.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dict.
        
        The returned dict should contain only basic Python types
        (str, int, float, list, dict, None) for JSON compatibility.
        
        Returns:
            Dictionary representation of the genome
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Reconstruct from dict.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            Reconstructed genome instance
        """
        ...


@runtime_checkable
class MutableGenome(Protocol):
    """
    Protocol for genomes that support in-place mutation.
    
    This is an optimization for mutation operators that want
    to avoid copying. Most genomes should be immutable.
    """

    def mutate_inplace(self, **kwargs: Any) -> None:
        """
        Mutate this genome in-place.
        
        Args:
            **kwargs: Mutation parameters
        """
        ...
