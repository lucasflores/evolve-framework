"""
Meta-Evolution Module.

Provides infrastructure for evolving hyperparameters by running
inner evolutionary loops and optimizing configuration parameters.

Public API:
    ConfigCodec: Encode/decode configurations to/from vector genomes
    ParameterDecoder: Decode a vector genome into a nested dict of parameter values
    decode_parameters(): Decode a vector against parameter specs into a nested dict
    MetaEvaluator: Evaluate configurations by running inner evolution
    MetaEvolutionResult: Result of meta-evolution with best config and solution
    run_meta_evolution(): Run meta-evolution on a base configuration
"""

from evolve.meta.codec import ConfigCodec, ParameterDecoder, decode_parameters
from evolve.meta.evaluator import MetaEvaluator, run_meta_evolution
from evolve.meta.result import MetaEvolutionResult

__all__ = [
    "ConfigCodec",
    "MetaEvaluator",
    "MetaEvolutionResult",
    "ParameterDecoder",
    "decode_parameters",
    "run_meta_evolution",
]
