"""
Meta-Evolution Module.

Provides infrastructure for evolving hyperparameters by running
inner evolutionary loops and optimizing configuration parameters.

Public API:
    ConfigCodec: Encode/decode configurations to/from vector genomes
    MetaEvaluator: Evaluate configurations by running inner evolution
    MetaEvolutionResult: Result of meta-evolution with best config and solution
    run_meta_evolution(): Run meta-evolution on a base configuration
"""

from evolve.meta.codec import ConfigCodec
from evolve.meta.evaluator import MetaEvaluator, run_meta_evolution
from evolve.meta.result import MetaEvolutionResult

__all__ = [
    "ConfigCodec",
    "MetaEvaluator",
    "MetaEvolutionResult",
    "run_meta_evolution",
]
