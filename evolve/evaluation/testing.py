"""
Testing utilities for evaluators.

Provides tools for verifying evaluator correctness
and equivalence between different backends.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

from evolve.core.types import Fitness, Individual

G = TypeVar("G")


class EvaluatorEquivalenceError(AssertionError):
    """Raised when accelerated evaluator differs from reference."""

    def __init__(
        self,
        message: str,
        index: int,
        ref_values: np.ndarray,
        acc_values: np.ndarray,
        max_rel_diff: float,
    ) -> None:
        super().__init__(message)
        self.index = index
        self.ref_values = ref_values
        self.acc_values = acc_values
        self.max_rel_diff = max_rel_diff


def assert_evaluator_equivalence(
    reference: "Evaluator[G]",
    accelerated: "Evaluator[G]",
    test_individuals: Sequence[Individual[G]],
    seed: int,
    rtol: float = 1e-5,
    atol: float = 1e-10,
) -> None:
    """
    Verify accelerated evaluator matches CPU reference.
    
    Compares fitness values between a reference (CPU) evaluator
    and an accelerated (GPU/JIT) evaluator. Uses relative tolerance
    for numerical comparison.
    
    Args:
        reference: CPU reference evaluator
        accelerated: GPU/JIT accelerated evaluator
        test_individuals: Individuals to test
        seed: Random seed for reproducibility
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance for near-zero values (default: 1e-10)
        
    Raises:
        EvaluatorEquivalenceError: If results differ beyond tolerance
        
    Example:
        >>> assert_evaluator_equivalence(
        ...     cpu_evaluator, gpu_evaluator, population, seed=42
        ... )
    """
    if not test_individuals:
        return
    
    # Evaluate with both
    ref_fitness = reference.evaluate(test_individuals, seed)
    acc_fitness = accelerated.evaluate(test_individuals, seed)
    
    # Check lengths match
    if len(ref_fitness) != len(acc_fitness):
        raise EvaluatorEquivalenceError(
            f"Output length mismatch: reference={len(ref_fitness)}, "
            f"accelerated={len(acc_fitness)}",
            index=-1,
            ref_values=np.array([]),
            acc_values=np.array([]),
            max_rel_diff=float("inf"),
        )
    
    # Compare each fitness value
    for i, (ref, acc) in enumerate(zip(ref_fitness, acc_fitness)):
        ref_vals = np.asarray(ref.values)
        acc_vals = np.asarray(acc.values)
        
        # Check shapes match
        if ref_vals.shape != acc_vals.shape:
            raise EvaluatorEquivalenceError(
                f"Shape mismatch at index {i}: ref={ref_vals.shape}, "
                f"acc={acc_vals.shape}",
                index=i,
                ref_values=ref_vals,
                acc_values=acc_vals,
                max_rel_diff=float("inf"),
            )
        
        # Check for NaN
        if np.any(np.isnan(ref_vals)):
            raise EvaluatorEquivalenceError(
                f"NaN in reference fitness at index {i}",
                index=i,
                ref_values=ref_vals,
                acc_values=acc_vals,
                max_rel_diff=float("nan"),
            )
        
        if np.any(np.isnan(acc_vals)):
            raise EvaluatorEquivalenceError(
                f"NaN in accelerated fitness at index {i}",
                index=i,
                ref_values=ref_vals,
                acc_values=acc_vals,
                max_rel_diff=float("nan"),
            )
        
        # Relative tolerance check
        # Use atol as minimum denominator for near-zero values
        denom = np.maximum(np.abs(ref_vals), np.abs(acc_vals))
        denom = np.maximum(denom, atol)
        rel_diff = np.abs(ref_vals - acc_vals) / denom
        max_rel_diff = float(rel_diff.max())
        
        if max_rel_diff > rtol:
            raise EvaluatorEquivalenceError(
                f"Equivalence failed at index {i}: "
                f"max relative diff = {max_rel_diff:.2e} > rtol={rtol:.2e}",
                index=i,
                ref_values=ref_vals,
                acc_values=acc_vals,
                max_rel_diff=max_rel_diff,
            )


def assert_fitness_close(
    actual: Fitness,
    expected: Fitness,
    rtol: float = 1e-5,
    atol: float = 1e-10,
) -> None:
    """
    Assert two Fitness objects are numerically close.
    
    Args:
        actual: Actual fitness value
        expected: Expected fitness value
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Raises:
        AssertionError: If values differ beyond tolerance
    """
    actual_vals = np.asarray(actual.values)
    expected_vals = np.asarray(expected.values)
    
    np.testing.assert_allclose(
        actual_vals,
        expected_vals,
        rtol=rtol,
        atol=atol,
    )


def assert_fitness_batch_close(
    actual: Sequence[Fitness],
    expected: Sequence[Fitness],
    rtol: float = 1e-5,
    atol: float = 1e-10,
) -> None:
    """
    Assert two sequences of Fitness objects are numerically close.
    
    Args:
        actual: Actual fitness values
        expected: Expected fitness values
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Raises:
        AssertionError: If values differ beyond tolerance
    """
    assert len(actual) == len(expected), (
        f"Length mismatch: {len(actual)} vs {len(expected)}"
    )
    
    for i, (a, e) in enumerate(zip(actual, expected)):
        try:
            assert_fitness_close(a, e, rtol=rtol, atol=atol)
        except AssertionError as err:
            raise AssertionError(f"Mismatch at index {i}: {err}") from err


class EvaluatorTester:
    """
    Comprehensive evaluator testing helper.
    
    Provides utilities for testing evaluator implementations
    across different scenarios.
    
    Example:
        >>> tester = EvaluatorTester(reference_evaluator)
        >>> tester.test_determinism(population, seed=42)
        >>> tester.test_equivalence(gpu_evaluator, population)
    """

    def __init__(
        self,
        reference: "Evaluator[G]",
        rtol: float = 1e-5,
        atol: float = 1e-10,
    ) -> None:
        """
        Create evaluator tester.
        
        Args:
            reference: Reference evaluator for comparison
            rtol: Default relative tolerance
            atol: Default absolute tolerance
        """
        self._reference = reference
        self._rtol = rtol
        self._atol = atol

    def test_determinism(
        self,
        individuals: Sequence[Individual[G]],
        seed: int,
        n_trials: int = 3,
    ) -> None:
        """
        Test that evaluator produces deterministic results.
        
        Runs evaluation multiple times with same seed and
        verifies identical results.
        
        Args:
            individuals: Test individuals
            seed: Random seed
            n_trials: Number of repetitions
            
        Raises:
            AssertionError: If results differ between trials
        """
        if not individuals:
            return
        
        first_results = self._reference.evaluate(individuals, seed)
        
        for trial in range(1, n_trials):
            results = self._reference.evaluate(individuals, seed)
            assert_fitness_batch_close(
                results,
                first_results,
                rtol=0,  # Exact match required for determinism
                atol=0,
            )

    def test_equivalence(
        self,
        accelerated: "Evaluator[G]",
        individuals: Sequence[Individual[G]],
        seed: int = 42,
    ) -> None:
        """
        Test accelerated evaluator matches reference.
        
        Args:
            accelerated: Accelerated evaluator to test
            individuals: Test individuals
            seed: Random seed
            
        Raises:
            EvaluatorEquivalenceError: If results differ
        """
        assert_evaluator_equivalence(
            self._reference,
            accelerated,
            individuals,
            seed,
            rtol=self._rtol,
            atol=self._atol,
        )

    def test_batch_consistency(
        self,
        individuals: Sequence[Individual[G]],
        seed: int = 42,
    ) -> None:
        """
        Test that batch evaluation matches individual evaluation.
        
        Compares results from evaluating all at once vs one at a time.
        
        Args:
            individuals: Test individuals
            seed: Random seed
            
        Raises:
            AssertionError: If batch and individual results differ
        """
        if not individuals:
            return
        
        # Evaluate all at once
        batch_results = self._reference.evaluate(individuals, seed)
        
        # Evaluate one at a time
        individual_results = []
        for ind in individuals:
            result = self._reference.evaluate([ind], seed)
            individual_results.append(result[0])
        
        assert_fitness_batch_close(
            batch_results,
            individual_results,
            rtol=self._rtol,
            atol=self._atol,
        )


# Type hint for protocol
class Evaluator:
    """Evaluator protocol stub for type hints."""
    
    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """Evaluate individuals."""
        ...


__all__ = [
    "EvaluatorEquivalenceError",
    "assert_evaluator_equivalence",
    "assert_fitness_close",
    "assert_fitness_batch_close",
    "EvaluatorTester",
]
