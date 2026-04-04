"""
Evaluation module - Fitness computation interfaces.

This is the PRIMARY ACCELERATION BOUNDARY.
Evaluators may use GPU/JIT but must have CPU reference implementations.
"""

from evolve.evaluation.evaluator import (
    BatchEvaluator,
    DiagnosticEvaluator,
    EvaluationError,
    Evaluator,
    EvaluatorCapabilities,
    FunctionEvaluator,
)
from evolve.evaluation.scm_evaluator import (
    SCMEvaluationResult,
    SCMEvaluator,
    SCMFitnessConfig,
)
from evolve.evaluation.testing import (
    EvaluatorEquivalenceError,
    EvaluatorTester,
    assert_evaluator_equivalence,
    assert_fitness_batch_close,
    assert_fitness_close,
)

__all__ = [
    # Evaluator types
    "Evaluator",
    "EvaluatorCapabilities",
    "EvaluationError",
    "DiagnosticEvaluator",
    "FunctionEvaluator",
    "BatchEvaluator",
    # Testing utilities
    "assert_evaluator_equivalence",
    "assert_fitness_close",
    "assert_fitness_batch_close",
    "EvaluatorEquivalenceError",
    "EvaluatorTester",
    # SCM evaluator
    "SCMFitnessConfig",
    "SCMEvaluationResult",
    "SCMEvaluator",
]
