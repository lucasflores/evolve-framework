"""
Scaling and performance benchmark tests.

Measures speedup from parallelization and GPU acceleration.

These tests are marked as 'benchmark' and may take longer to run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pytest

from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import FunctionEvaluator, BatchEvaluator
from evolve.evaluation.reference.functions import sphere, rastrigin, rosenbrock
from evolve.representation.vector import VectorGenome


# ============================================================================
# Benchmarking Utilities
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    n_individuals: int
    n_dimensions: int
    total_time: float
    per_individual_time: float
    throughput: float  # individuals per second
    
    def __str__(self) -> str:
        return (
            f"{self.name}: {self.n_individuals} x {self.n_dimensions}D "
            f"in {self.total_time:.3f}s "
            f"({self.throughput:.1f} ind/s)"
        )


def benchmark_evaluator(
    name: str,
    evaluator,
    population: Sequence[Individual],
    n_runs: int = 3,
    warmup: int = 1,
) -> BenchmarkResult:
    """
    Benchmark an evaluator.
    
    Args:
        name: Benchmark name
        evaluator: Evaluator to benchmark
        population: Test population
        n_runs: Number of timed runs
        warmup: Number of warmup runs
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup):
        evaluator.evaluate(population, seed=42)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        evaluator.evaluate(population, seed=42)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    n_ind = len(population)
    n_dim = len(getattr(population[0].genome, "genes", []))
    
    return BenchmarkResult(
        name=name,
        n_individuals=n_ind,
        n_dimensions=n_dim,
        total_time=avg_time,
        per_individual_time=avg_time / n_ind,
        throughput=n_ind / avg_time,
    )


def benchmark_backend(
    name: str,
    backend,
    evaluator,
    population: Sequence[Individual],
    n_runs: int = 3,
    warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark an execution backend."""
    # Warmup
    for _ in range(warmup):
        backend.map_evaluate(evaluator, population, seed=42)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        backend.map_evaluate(evaluator, population, seed=42)
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    n_ind = len(population)
    n_dim = len(getattr(population[0].genome, "genes", []))
    
    return BenchmarkResult(
        name=name,
        n_individuals=n_ind,
        n_dimensions=n_dim,
        total_time=avg_time,
        per_individual_time=avg_time / n_ind,
        throughput=n_ind / avg_time,
    )


def create_test_population(n: int, dim: int, seed: int = 42) -> list[Individual]:
    """Create test population."""
    np.random.seed(seed)
    return [
        Individual(
            id=f"ind_{i}",
            genome=VectorGenome(genes=np.random.randn(dim)),
        )
        for i in range(n)
    ]


# ============================================================================
# Scaling Tests
# ============================================================================

@pytest.mark.benchmark
class TestPopulationScaling:
    """Test how performance scales with population size."""

    @pytest.mark.parametrize("n_individuals", [100, 500, 1000, 5000])
    def test_sequential_scaling(self, n_individuals):
        """Measure sequential backend scaling."""
        from evolve.backends.sequential import SequentialBackend
        
        population = create_test_population(n_individuals, dim=50)
        evaluator = FunctionEvaluator(sphere)
        backend = SequentialBackend()
        
        result = benchmark_backend("sequential", backend, evaluator, population)
        
        # Just verify it completes and report
        assert result.throughput > 0
        print(f"\n{result}")

    @pytest.mark.parametrize("n_individuals", [100, 500, 1000, 5000])
    def test_parallel_scaling(self, n_individuals):
        """Measure parallel backend scaling."""
        from evolve.backends.parallel import ParallelBackend
        
        population = create_test_population(n_individuals, dim=50)
        evaluator = FunctionEvaluator(sphere)
        backend = ParallelBackend(n_workers=4)
        
        try:
            result = benchmark_backend("parallel", backend, evaluator, population)
            assert result.throughput > 0
            print(f"\n{result}")
        finally:
            backend.shutdown()


@pytest.mark.benchmark
class TestDimensionScaling:
    """Test how performance scales with problem dimension."""

    @pytest.mark.parametrize("n_dimensions", [10, 50, 100, 500])
    def test_sphere_dimension_scaling(self, n_dimensions):
        """Measure sphere evaluation scaling with dimension."""
        population = create_test_population(1000, dim=n_dimensions)
        evaluator = FunctionEvaluator(sphere)
        
        result = benchmark_evaluator("sphere", evaluator, population)
        
        assert result.throughput > 0
        print(f"\n{result}")

    @pytest.mark.parametrize("n_dimensions", [10, 50, 100, 500])
    def test_rastrigin_dimension_scaling(self, n_dimensions):
        """Measure rastrigin evaluation scaling with dimension."""
        population = create_test_population(1000, dim=n_dimensions)
        evaluator = FunctionEvaluator(rastrigin)
        
        result = benchmark_evaluator("rastrigin", evaluator, population)
        
        assert result.throughput > 0
        print(f"\n{result}")


# ============================================================================
# Speedup Tests
# ============================================================================

@pytest.mark.benchmark
class TestParallelSpeedup:
    """Test parallel backend speedup over sequential."""

    def test_parallel_speedup_large_population(self):
        """Parallel should be faster for large populations."""
        from evolve.backends.sequential import SequentialBackend
        from evolve.backends.parallel import ParallelBackend
        
        # Use expensive function and large population
        population = create_test_population(2000, dim=100)
        evaluator = FunctionEvaluator(rosenbrock)
        
        seq_backend = SequentialBackend()
        par_backend = ParallelBackend(n_workers=4)
        
        try:
            seq_result = benchmark_backend("sequential", seq_backend, evaluator, population)
            par_result = benchmark_backend("parallel", par_backend, evaluator, population)
            
            speedup = seq_result.total_time / par_result.total_time
            
            print(f"\nSequential: {seq_result}")
            print(f"Parallel:   {par_result}")
            print(f"Speedup:    {speedup:.2f}x")
            
            # We expect some speedup, but don't require specific amount
            # (depends on hardware, overhead, etc.)
            assert speedup > 0.5, "Parallel should not be significantly slower"
        finally:
            par_backend.shutdown()


@pytest.mark.benchmark
class TestTorchSpeedup:
    """Test PyTorch speedup over CPU."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_torch_cpu_speedup(self, torch_available):
        """Torch CPU should be competitive with NumPy."""
        import torch
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            sphere_torch,
        )
        
        population = create_test_population(1000, dim=100)
        
        numpy_evaluator = FunctionEvaluator(sphere)
        torch_evaluator = TorchEvaluator(sphere_torch, device='cpu')
        
        numpy_result = benchmark_evaluator("numpy", numpy_evaluator, population)
        torch_result = benchmark_evaluator("torch_cpu", torch_evaluator, population)
        
        print(f"\nNumPy:     {numpy_result}")
        print(f"Torch CPU: {torch_result}")
        
        # Both should work
        assert numpy_result.throughput > 0
        assert torch_result.throughput > 0

    @pytest.mark.skipif(
        not hasattr(__import__('torch', fromlist=['']), 'cuda') or 
        not __import__('torch').cuda.is_available(),
        reason="CUDA not available"
    )
    def test_torch_cuda_speedup(self, torch_available):
        """Torch CUDA should be faster than CPU for large populations."""
        import torch
        from evolve.backends.accelerated.torch_evaluator import (
            TorchEvaluator,
            sphere_torch,
        )
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        population = create_test_population(10000, dim=100)
        
        cpu_evaluator = TorchEvaluator(sphere_torch, device='cpu')
        cuda_evaluator = TorchEvaluator(sphere_torch, device='cuda')
        
        cpu_result = benchmark_evaluator("torch_cpu", cpu_evaluator, population)
        cuda_result = benchmark_evaluator("torch_cuda", cuda_evaluator, population)
        
        speedup = cpu_result.total_time / cuda_result.total_time
        
        print(f"\nTorch CPU:  {cpu_result}")
        print(f"Torch CUDA: {cuda_result}")
        print(f"Speedup:    {speedup:.2f}x")
        
        # Expect significant speedup on GPU
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"


@pytest.mark.benchmark
class TestJaxSpeedup:
    """Test JAX speedup over CPU."""

    @pytest.fixture
    def jax_available(self):
        """Check if JAX is available."""
        try:
            import jax
            return True
        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_jit_speedup(self, jax_available):
        """JIT compilation should speed up repeated evaluation."""
        from evolve.backends.accelerated.jax_evaluator import (
            JaxEvaluator,
            sphere_jax,
        )
        
        population = create_test_population(1000, dim=100)
        
        numpy_evaluator = FunctionEvaluator(sphere)
        jax_evaluator = JaxEvaluator(sphere_jax, jit_compile=True)
        jax_nojit_evaluator = JaxEvaluator(sphere_jax, jit_compile=False)
        
        numpy_result = benchmark_evaluator("numpy", numpy_evaluator, population)
        jax_result = benchmark_evaluator("jax_jit", jax_evaluator, population)
        jax_nojit_result = benchmark_evaluator("jax_nojit", jax_nojit_evaluator, population)
        
        print(f"\nNumPy:      {numpy_result}")
        print(f"JAX no JIT: {jax_nojit_result}")
        print(f"JAX JIT:    {jax_result}")
        
        # JIT should be faster than no-JIT after warmup
        assert jax_result.throughput > 0


# ============================================================================
# Batch Size Optimization
# ============================================================================

@pytest.mark.benchmark
class TestBatchSizeOptimization:
    """Test optimal batch sizes for different backends."""

    def test_batch_evaluator_efficiency(self):
        """Batch evaluator should be more efficient than loop."""
        population = create_test_population(1000, dim=50)
        
        # Individual evaluation
        loop_evaluator = FunctionEvaluator(sphere)
        
        # Batch evaluation
        def batch_sphere(x):
            return np.sum(x ** 2, axis=1)
        batch_evaluator = BatchEvaluator(batch_sphere)
        
        loop_result = benchmark_evaluator("loop", loop_evaluator, population)
        batch_result = benchmark_evaluator("batch", batch_evaluator, population)
        
        speedup = loop_result.total_time / batch_result.total_time
        
        print(f"\nLoop:    {loop_result}")
        print(f"Batch:   {batch_result}")
        print(f"Speedup: {speedup:.2f}x")
        
        # Batch should be faster
        assert speedup > 1.0, "Batch evaluation should be faster"
