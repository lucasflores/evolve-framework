"""
Phenotype and Decoder protocols.

Phenotypes are the decoded/expressed form of genomes that can be evaluated.
Decoders transform genomes into phenotypes.

NO ML FRAMEWORK IMPORTS ALLOWED (in this module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

from evolve.representation.genome import Genome

if TYPE_CHECKING:
    from collections.abc import Callable

G = TypeVar("G", bound=Genome)
P = TypeVar("P")


@runtime_checkable
class Phenotype(Protocol):
    """
    Decoded form of a genome that can be evaluated.

    Phenotypes are what the evaluator actually evaluates.
    For simple cases (like continuous optimization), the phenotype
    may just be the genome's values. For complex cases (like
    neuroevolution), the phenotype is a neural network.

    The only requirement is that phenotypes are callable,
    mapping inputs to outputs.
    """

    def __call__(self, inputs: Any) -> Any:
        """
        Apply the phenotype to inputs.

        Args:
            inputs: Input data (type depends on problem domain)

        Returns:
            Output data (type depends on problem domain)
        """
        ...


@runtime_checkable
class Decoder(Protocol[G, P]):  # type: ignore[misc]
    """
    Transforms genomes into phenotypes.

    The Decoder abstracts the genotype-phenotype mapping,
    allowing the same genome representation to be used
    with different evaluation contexts.

    Type Parameters:
        G: Genome type (input)
        P: Phenotype type (output)

    Example:
        >>> class NetworkDecoder:
        ...     def decode(self, genome: VectorGenome) -> NeuralNetwork:
        ...         weights = genome.genes.reshape(network_shape)
        ...         return NeuralNetwork(weights)
    """

    def decode(self, genome: G) -> P:
        """
        Convert genome to phenotype.

        Args:
            genome: The genetic representation

        Returns:
            The expressed phenotype
        """
        ...


class IdentityDecoder(Generic[G]):
    """
    Decoder that returns the genome unchanged.

    Used when the genome is directly evaluable (e.g., VectorGenome
    for function optimization where the genes ARE the inputs).
    """

    def decode(self, genome: G) -> G:
        """Return genome unchanged."""
        return genome


class CallableDecoder(Generic[G, P]):
    """
    Decoder wrapping a callable.

    Convenience class for creating decoders from functions.

    Example:
        >>> decoder = CallableDecoder(lambda g: g.genes)
    """

    def __init__(self, decode_fn: Callable[[G], P]) -> None:
        """
        Create decoder from function.

        Args:
            decode_fn: Function that transforms genome to phenotype
        """
        self._decode_fn = decode_fn

    def decode(self, genome: G) -> P:
        """Apply the decode function."""
        return self._decode_fn(genome)
