"""
Operator, Genome, Evaluator, and Callback Registries.

Provides registries for looking up operators, genome types, evaluators,
and callbacks by name, enabling declarative configuration of
evolutionary algorithms.

Public API:
    get_operator_registry(): Get the global operator registry
    get_genome_registry(): Get the global genome registry
    get_evaluator_registry(): Get the global evaluator registry
    get_callback_registry(): Get the global callback registry
    reset_operator_registry(): Reset operator registry (for testing)
    reset_genome_registry(): Reset genome registry (for testing)
    reset_evaluator_registry(): Reset evaluator registry (for testing)
    reset_callback_registry(): Reset callback registry (for testing)
"""

from evolve.registry.callbacks import (
    CallbackRegistry,
    get_callback_registry,
    reset_callback_registry,
)
from evolve.registry.evaluators import (
    EvaluatorRegistry,
    get_evaluator_registry,
    reset_evaluator_registry,
)
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
    # Evaluator registry
    "EvaluatorRegistry",
    "get_evaluator_registry",
    "reset_evaluator_registry",
    # Callback registry
    "CallbackRegistry",
    "get_callback_registry",
    "reset_callback_registry",
]
