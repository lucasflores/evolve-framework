"""
CPU/GPU equivalence tests.

Verifies that accelerated evaluators produce results within
tolerance of CPU reference implementations.

These tests require optional dependencies (PyTorch, JAX).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from evolve.core.types import Individual
from evolve.evaluation.evaluator import BatchEvaluator, FunctionEvaluator
from evolve.evaluation.reference.functions import ackley, rastrigin, rosenbrock, sphere
from evolve.evaluation.testing import (
    EvaluatorTester,
    assert_evaluator_equivalence,
)
from evolve.representation.vector import VectorGenome

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_population():
    """Create test population of various sizes."""
    np.random.seed(42)
    return [
        Individual(
            id=f"ind_{i}",
            genome=VectorGenome(genes=np.random.randn(10) * 2),
        )
        for i in range(100)
    ]


@pytest.fixture
def small_population():
    """Create small test population."""
    np.random.seed(42)
    return [
        Individual(
            id=f"ind_{i}",
            genome=VectorGenome(genes=np.random.randn(5)),
        )
        for i in range(10)
    ]


# ============================================================================
# CPU Reference Tests
# ============================================================================


@pytest.mark.benchmark
class TestCPUReferenceConsistency:
    """Test CPU reference implementations are consistent."""

    def test_sphere_deterministic(self, test_population):
        """Sphere function should be deterministic."""
        evaluator = FunctionEvaluator(sphere)
        tester = EvaluatorTester(evaluator)
        tester.test_determinism(test_population, seed=42)

    def test_rastrigin_deterministic(self, test_population):
        """Rastrigin function should be deterministic."""
        evaluator = FunctionEvaluator(rastrigin)
        tester = EvaluatorTester(evaluator)
        tester.test_determinism(test_population, seed=42)

    def test_rosenbrock_deterministic(self, test_population):
        """Rosenbrock function should be deterministic."""
        evaluator = FunctionEvaluator(rosenbrock)
        tester = EvaluatorTester(evaluator)
        tester.test_determinism(test_population, seed=42)

    def test_ackley_deterministic(self, test_population):
        """Ackley function should be deterministic."""
        evaluator = FunctionEvaluator(ackley)
        tester = EvaluatorTester(evaluator)
        tester.test_determinism(test_population, seed=42)

    def test_batch_vs_individual_sphere(self, small_population):
        """Batch evaluation should match individual evaluation."""

        # Create batch evaluator
        def batch_sphere(x):
            return np.sum(x**2, axis=1)

        batch_eval = BatchEvaluator(batch_sphere)
        func_eval = FunctionEvaluator(sphere)

        batch_results = batch_eval.evaluate(small_population)
        func_results = func_eval.evaluate(small_population)

        for b, f in zip(batch_results, func_results):
            np.testing.assert_allclose(b.values, f.values, rtol=1e-10)


# ============================================================================
# PyTorch Equivalence Tests
# ============================================================================


@pytest.mark.benchmark
class TestTorchEquivalence:
    """Test PyTorch evaluators match CPU reference."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_sphere_torch_equivalence(self, torch_available, test_population):
        """Torch sphere should match CPU within 1e-5."""
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            sphere_torch,
        )

        cpu_evaluator = FunctionEvaluator(sphere)
        torch_evaluator = TorchEvaluator(sphere_torch, device="cpu")

        assert_evaluator_equivalence(
            cpu_evaluator,
            torch_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_rastrigin_torch_equivalence(self, torch_available, test_population):
        """Torch rastrigin should match CPU within 1e-5."""
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            rastrigin_torch,
        )

        cpu_evaluator = FunctionEvaluator(rastrigin)
        torch_evaluator = TorchEvaluator(rastrigin_torch, device="cpu")

        assert_evaluator_equivalence(
            cpu_evaluator,
            torch_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_rosenbrock_torch_equivalence(self, torch_available, test_population):
        """Torch rosenbrock should match CPU within 1e-5."""
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            rosenbrock_torch,
        )

        cpu_evaluator = FunctionEvaluator(rosenbrock)
        torch_evaluator = TorchEvaluator(rosenbrock_torch, device="cpu")

        assert_evaluator_equivalence(
            cpu_evaluator,
            torch_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_ackley_torch_equivalence(self, torch_available, test_population):
        """Torch ackley should match CPU within 1e-5."""
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            ackley_torch,
        )

        cpu_evaluator = FunctionEvaluator(ackley)
        torch_evaluator = TorchEvaluator(ackley_torch, device="cpu")

        assert_evaluator_equivalence(
            cpu_evaluator,
            torch_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"),
        reason="CUDA not available",
    )
    def test_sphere_cuda_equivalence(self, torch_available, test_population):
        """CUDA sphere should match CPU within 1e-5."""
        import torch

        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            sphere_torch,
        )

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cpu_evaluator = FunctionEvaluator(sphere)
        cuda_evaluator = TorchEvaluator(sphere_torch, device="cuda")

        assert_evaluator_equivalence(
            cpu_evaluator,
            cuda_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )


# ============================================================================
# JAX Equivalence Tests
# ============================================================================


@pytest.mark.benchmark
class TestJaxEquivalence:
    """Test JAX evaluators match CPU reference."""

    @pytest.fixture
    def jax_available(self):
        """Check if JAX is available."""
        try:
            import jax  # noqa: F401

            return True
        except ImportError:
            pytest.skip("JAX not installed")

    def test_sphere_jax_equivalence(self, jax_available, test_population):
        """JAX sphere should match CPU within 1e-5."""
        from evolve.backends.accelerated.jax_evaluator import (
            JaxEvaluator,
            sphere_jax,
        )

        cpu_evaluator = FunctionEvaluator(sphere)
        jax_evaluator = JaxEvaluator(sphere_jax)

        assert_evaluator_equivalence(
            cpu_evaluator,
            jax_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_rastrigin_jax_equivalence(self, jax_available, test_population):
        """JAX rastrigin should match CPU within 1e-5."""
        from evolve.backends.accelerated.jax_evaluator import (
            JaxEvaluator,
            rastrigin_jax,
        )

        cpu_evaluator = FunctionEvaluator(rastrigin)
        jax_evaluator = JaxEvaluator(rastrigin_jax)

        assert_evaluator_equivalence(
            cpu_evaluator,
            jax_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_rosenbrock_jax_equivalence(self, jax_available, test_population):
        """JAX rosenbrock should match CPU within 1e-5."""
        from evolve.backends.accelerated.jax_evaluator import (
            JaxEvaluator,
            rosenbrock_jax,
        )

        cpu_evaluator = FunctionEvaluator(rosenbrock)
        jax_evaluator = JaxEvaluator(rosenbrock_jax)

        assert_evaluator_equivalence(
            cpu_evaluator,
            jax_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )

    def test_ackley_jax_equivalence(self, jax_available, test_population):
        """JAX ackley should match CPU within 1e-5."""
        from evolve.backends.accelerated.jax_evaluator import (
            JaxEvaluator,
            ackley_jax,
        )

        cpu_evaluator = FunctionEvaluator(ackley)
        jax_evaluator = JaxEvaluator(ackley_jax)

        assert_evaluator_equivalence(
            cpu_evaluator,
            jax_evaluator,
            test_population,
            seed=42,
            rtol=1e-5,
        )


# ============================================================================
# Cross-Backend Equivalence
# ============================================================================


@pytest.mark.benchmark
class TestCrossBackendEquivalence:
    """Test equivalence across different backends."""

    def test_sequential_parallel_equivalence(self, test_population):
        """Sequential and parallel backends should match."""
        from evolve.backends.parallel import ParallelBackend
        from evolve.backends.sequential import SequentialBackend

        evaluator = FunctionEvaluator(sphere)

        seq_backend = SequentialBackend()
        par_backend = ParallelBackend(n_workers=2)

        try:
            seq_results = seq_backend.map_evaluate(evaluator, test_population, seed=42)
            par_results = par_backend.map_evaluate(evaluator, test_population, seed=42)

            for s, p in zip(seq_results, par_results):
                np.testing.assert_allclose(s.values, p.values, rtol=1e-10)
        finally:
            par_backend.shutdown()
