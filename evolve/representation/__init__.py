"""
Representation module - Genome and phenotype abstractions.

This module contains NO ML framework imports.
Genome types use NumPy arrays and Python collections.
"""

from evolve.representation.decoder import (
    GraphToMLPDecoder,
    GraphToNetworkDecoder,
)
from evolve.representation.genome import (
    Genome,
    SerializableGenome,
)
from evolve.representation.graph import (
    ConnectionGene,
    GraphGenome,
    InnovationTracker,
    NodeGene,
)
from evolve.representation.network import (
    ACTIVATIONS,
    NEATNetwork,
    NumpyNetwork,
    RecurrentNumpyNetwork,
    get_activation,
)
from evolve.representation.phenotype import (
    CallableDecoder,
    Decoder,
    IdentityDecoder,
    Phenotype,
)
from evolve.representation.scm import (
    AcyclicityMode,
    AcyclicityStrategy,
    ConflictResolution,
    SCMAlphabet,
    SCMConfig,
    SCMGenome,
    scm_distance,
    # Distance functions for ERP integration
    scm_sequence_distance,
    scm_structural_distance,
)
from evolve.representation.scm_decoder import (
    BinOp,
    Const,
    DecodedSCM,
    Expression,
    SCMDecoder,
    SCMMetadata,
    Var,
    complexity,
    evaluate,
    expr_from_dict,
    expr_to_dict,
    to_string,
    variables,
)
from evolve.representation.sequence import (
    SequenceGenome,
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
    # SCM representation
    "ConflictResolution",
    "AcyclicityMode",
    "AcyclicityStrategy",
    "SCMConfig",
    "SCMAlphabet",
    "SCMGenome",
    # SCM decoder
    "Var",
    "Const",
    "BinOp",
    "Expression",
    "complexity",
    "variables",
    "evaluate",
    "to_string",
    "expr_to_dict",
    "expr_from_dict",
    "SCMMetadata",
    "DecodedSCM",
    "SCMDecoder",
    # SCM distance functions
    "scm_sequence_distance",
    "scm_structural_distance",
    "scm_distance",
]
