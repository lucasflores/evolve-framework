"""
Parallel execution backend using multiprocessing.

Distributes evaluation across multiple CPU cores.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Sequence, TypeVar

from evolve.backends.base import (
    BackendCapabilities,
    BackendConfig,
    ExecutionBackend,
    Evaluator,
    derive_seed,
)
from evolve.core.types import Fitness, Individual

G = TypeVar("G")


def _get_cpu_count() -> int:
    """Get number of available CPUs, respecting container limits."""
    # Check for container CPU limits
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except (FileNotFoundError, ValueError):
        pass
    
    # Fall back to os.cpu_count
    return os.cpu_count() or 1


def _evaluate_batch(
    evaluator: Evaluator[G],
    individuals: Sequence[Individual[G]],
    seed: int | None,
) -> list[Fitness]:
    """Worker function for parallel evaluation."""
    return list(evaluator.evaluate(individuals, seed=seed))


class ParallelBackend:
    """
    Parallel execution backend using multiprocessing.
    
    Distributes evaluation across multiple CPU cores using
    a process pool. Each worker gets a derived seed for
    reproducibility.
    
    Advantages:
    - Utilizes multiple CPU cores
    - True parallelism (bypasses GIL)
    - Scales well with population size
    
    Disadvantages:
    - Serialization overhead (pickle)
    - Memory usage per worker
    - Startup cost for process pool
    
    Example:
        >>> backend = ParallelBackend(n_workers=4)
        >>> results = backend.map_evaluate(evaluator, population)
        >>> backend.shutdown()
    """

    def __init__(
        self,
        n_workers: int | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Create parallel backend.
        
        Args:
            n_workers: Number of worker processes (None = auto-detect)
            batch_size: Individuals per batch (None = auto)
            timeout: Timeout per batch in seconds (None = no timeout)
        """
        self._n_workers = n_workers or _get_cpu_count()
        self._batch_size = batch_size
        self._timeout = timeout
        self._executor: ProcessPoolExecutor | None = None
        
        self._capabilities = BackendCapabilities(
            parallel=True,
            gpu=False,
            jit=False,
            distributed=False,
            max_workers=self._n_workers,
            device_count=0,
        )

    @property
    def name(self) -> str:
        """Backend name."""
        return "parallel"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return self._capabilities

    @property
    def n_workers(self) -> int:
        """Number of worker processes."""
        return self._n_workers

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create executor pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._n_workers)
        return self._executor

    def _compute_batch_size(self, n_individuals: int) -> int:
        """Compute optimal batch size."""
        if self._batch_size is not None:
            return self._batch_size
        
        # Heuristic: aim for at least 2 batches per worker
        # but not too small (overhead dominates)
        min_batch = 10
        target_batches = self._n_workers * 2
        return max(min_batch, n_individuals // target_batches)

    def map_evaluate(
        self,
        evaluator: Evaluator[G],
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals in parallel.
        
        Splits individuals into batches and distributes
        across worker processes.
        
        Args:
            evaluator: Evaluator to use
            individuals: Individuals to evaluate
            seed: Base seed (derived seeds sent to workers)
            
        Returns:
            Fitness values in same order as individuals
        """
        if not individuals:
            return []
        
        n = len(individuals)
        
        # For small populations, sequential is faster
        if n < self._n_workers * 2:
            return evaluator.evaluate(individuals, seed=seed)
        
        batch_size = self._compute_batch_size(n)
        
        # Split into batches
        batches: list[tuple[Sequence[Individual[G]], int | None]] = []
        for i in range(0, n, batch_size):
            batch = individuals[i : i + batch_size]
            batch_seed = derive_seed(seed, i) if seed is not None else None
            batches.append((batch, batch_seed))
        
        # Submit to executor
        executor = self._get_executor()
        futures = [
            executor.submit(_evaluate_batch, evaluator, batch, batch_seed)
            for batch, batch_seed in batches
        ]
        
        # Collect results in order
        results: list[Fitness] = []
        for future in futures:
            try:
                batch_results = future.result(timeout=self._timeout)
                results.extend(batch_results)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Evaluation timed out after {self._timeout} seconds"
                )
            except Exception as e:
                raise RuntimeError(f"Parallel evaluation failed: {e}") from e
        
        return results

    def shutdown(self) -> None:
        """Shutdown worker pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.shutdown()

    def __repr__(self) -> str:
        return f"ParallelBackend(n_workers={self._n_workers})"

    def __enter__(self) -> "ParallelBackend":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.shutdown()


__all__ = ["ParallelBackend"]
