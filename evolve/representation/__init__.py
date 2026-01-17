"""
Representation module - Genome and phenotype abstractions.

This module contains NO ML framework imports.
Genome types use NumPy arrays and Python collections.
"""

from evolve.representation.genome import (
    Genome,
    SerializableGenome,
)
from evolve.representation.phenotype import (
    Phenotype,
    Decoder,
    IdentityDecoder,
    CallableDecoder,
)
from evolve.representation.graph import (
    NodeGene,
    ConnectionGene,
    GraphGenome,
    InnovationTracker,
)
from evolve.representation.sequence import (
    SequenceGenome,
)
from evolve.representation.network import (
    NumpyNetwork,
    RecurrentNumpyNetwork,
    NEATNetwork,
    ACTIVATIONS,
    get_activation,
)
from evolve.representation.decoder import (
    GraphToNetworkDecoder,
    GraphToMLPDecoder,
)

__all__ = [
    # Core protocols
    "Genome",
    "SerializableGenome",
    "Phenotype",
    "Decoder",
    "IdentityDecoder",
    "CallableDecoder",
    # Graph genome (NEAT)
    "NodeGene",
    "ConnectionGene",
    "GraphGenome",
    "InnovationTracker",
    # Sequence genome
    "SequenceGenome",
    # Neural networks (NumPy)
    "NumpyNetwork",
    "RecurrentNumpyNetwork",
    "NEATNetwork",
    "ACTIVATIONS",
    "get_activation",
    # Decoders
    "GraphToNetworkDecoder",
    "GraphToMLPDecoder",
]
