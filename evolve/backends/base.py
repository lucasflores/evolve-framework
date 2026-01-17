"""
Base execution backend protocol.

Defines the interface for all execution backends.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar

from evolve.core.types import Fitness, Individual

G = TypeVar("G")


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Describes capabilities of an execution backend.
    
    Attributes:
        parallel: Can run evaluations in parallel
        gpu: Can use GPU acceleration
        jit: Supports JIT compilation
        distributed: Can distribute across machines
        max_workers: Maximum parallel workers (None = unlimited)
        device_count: Number of available devices (GPUs)
    """

    parallel: bool = False
    gpu: bool = False
    jit: bool = False
    distributed: bool = False
    max_workers: int | None = None
    device_count: int = 0


@dataclass
class BackendConfig:
    """
    Configuration for execution backend.
    
    Attributes:
        n_workers: Number of parallel workers (None = auto)
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        seed: Random seed for reproducibility
        batch_size: Preferred batch size for evaluation
        timeout: Timeout per evaluation in seconds
    """

    n_workers: int | None = None
    device: str = "cpu"
    seed: int | None = None
    batch_size: int | None = None
    timeout: float | None = None


class ExecutionBackend(Protocol):
    """
    Protocol for execution backends.
    
    Execution backends control HOW evaluations are performed:
    - Sequential vs parallel
    - CPU vs GPU
    - Local vs distributed
    
    The backend is responsible for:
    - Managing worker processes/threads
    - Handling device placement (GPU)
    - Seed derivation for parallel execution
    - Resource cleanup
    
    Example:
        >>> backend = ParallelBackend(n_workers=4)
        >>> results = backend.map_evaluate(evaluator, population, seed=42)
    """

    @property
    def name(self) -> str:
        """Backend identifier string."""
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """Describe backend capabilities."""
        ...

    def map_evaluate(
        self,
        evaluator: "Evaluator[G]",
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals using this backend.
        
        Args:
            evaluator: Evaluator to use
            individuals: Individuals to evaluate
            seed: Base seed for reproducibility
            
        Returns:
            Fitness values in same order as individuals
        """
        ...

    def shutdown(self) -> None:
        """
        Release backend resources.
        
        Called when backend is no longer needed.
        Should be idempotent.
        """
        ...


class Evaluator(Protocol[G]):
    """Evaluator protocol (re-exported for convenience)."""

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """Evaluate individuals."""
        ...


def derive_seed(base_seed: int, index: int) -> int:
    """
    Derive a reproducible seed from base seed and index.
    
    Used for parallel execution where each worker needs
    a unique but reproducible seed.
    
    Uses a simple hash-based approach that produces
    well-distributed seeds.
    
    Args:
        base_seed: Base random seed
        index: Worker/batch index
        
    Returns:
        Derived seed for this index
        
    Example:
        >>> seeds = [derive_seed(42, i) for i in range(4)]
        >>> len(set(seeds))  # All unique
        4
    """
    # Use multiplicative hash with prime
    # This gives good distribution while being deterministic
    PRIME = 0x9E3779B1  # Golden ratio derived prime
    combined = ((base_seed * PRIME) ^ (index * PRIME)) & 0xFFFFFFFF
    return combined


__all__ = [
    "BackendCapabilities",
    "BackendConfig",
    "ExecutionBackend",
    "Evaluator",
    "derive_seed",
]
