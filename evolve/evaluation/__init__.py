"""
Evaluation module - Fitness computation interfaces.

This is the PRIMARY ACCELERATION BOUNDARY.
Evaluators may use GPU/JIT but must have CPU reference implementations.
"""

from evolve.evaluation.evaluator import (
    Evaluator,
    EvaluatorCapabilities,
    EvaluationError,
    DiagnosticEvaluator,
    FunctionEvaluator,
    BatchEvaluator,
)
from evolve.evaluation.testing import (
    assert_evaluator_equivalence,
    assert_fitness_close,
    assert_fitness_batch_close,
    EvaluatorEquivalenceError,
    EvaluatorTester,
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
]
