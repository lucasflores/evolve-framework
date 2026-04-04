"""
Operator and Genome Registries.

Provides registries for looking up operators and genome types by name,
enabling declarative configuration of evolutionary algorithms.

Public API:
    get_operator_registry(): Get the global operator registry
    get_genome_registry(): Get the global genome registry
    reset_operator_registry(): Reset operator registry (for testing)
    reset_genome_registry(): Reset genome registry (for testing)
"""

from evolve.registry.genomes import (
    GenomeRegistry,
    get_genome_registry,
    reset_genome_registry,
)
from evolve.registry.operators import (
    OperatorRegistry,
    get_operator_registry,
    reset_operator_registry,
)

__all__ = [
    # Operator registry
    "OperatorRegistry",
    "get_operator_registry",
    "reset_operator_registry",
    # Genome registry
    "GenomeRegistry",
    "get_genome_registry",
    "reset_genome_registry",
]
