"""
PyTorch-based accelerated evaluator.

Provides GPU-accelerated batch evaluation using PyTorch.
This module requires PyTorch to be installed.

Example:
    >>> from evolve.backends.accelerated import TorchBackend
    >>> backend = TorchBackend(device='cuda')
    >>> results = backend.map_evaluate(evaluator, population)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for TorchBackend. Install with: pip install torch")

import numpy as np

from evolve.backends.base import BackendCapabilities
from evolve.core.types import Fitness, Individual

G = TypeVar("G")


def _get_device(device: str | None) -> torch.device:
    """Get torch device, defaulting to best available."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TorchEvaluator:
    """
    GPU-accelerated evaluator using PyTorch.

    Wraps a fitness function to run efficiently on GPU.
    The fitness function should accept torch tensors
    and return fitness values.

    Example:
        >>> def sphere_torch(x: torch.Tensor) -> torch.Tensor:
        ...     return torch.sum(x ** 2, dim=1)
        >>> evaluator = TorchEvaluator(sphere_torch)
        >>> fitness = evaluator.evaluate(individuals)
    """

    def __init__(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        n_objectives: int = 1,
    ) -> None:
        """
        Create PyTorch evaluator.

        Args:
            fitness_fn: Function mapping (N, D) tensor → (N,) or (N, M) tensor
            device: Device to use ('cpu', 'cuda', 'cuda:0', 'mps', etc.)
            dtype: Data type for tensors
            n_objectives: Number of objectives
        """
        self._fitness_fn = fitness_fn
        self._device = _get_device(device)
        self._dtype = dtype
        self._n_objectives = n_objectives

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    @property
    def capabilities(self) -> BackendCapabilities:
        """Evaluator capabilities."""
        return BackendCapabilities(
            parallel=True,
            gpu=self._device.type in ("cuda", "mps"),
            jit=False,
            device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals using PyTorch.

        Converts genomes to a batched tensor, runs evaluation
        on the device, and converts back to Fitness objects.

        Args:
            individuals: Individuals to evaluate
            seed: Random seed (sets torch seed)

        Returns:
            Fitness values
        """
        if not individuals:
            return []

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Extract genes and convert to tensor
        genes_list = [np.asarray(getattr(ind.genome, "genes", ind.genome)) for ind in individuals]
        genes_np = np.vstack(genes_list)
        genes_tensor = torch.tensor(genes_np, device=self._device, dtype=self._dtype)

        # Evaluate
        with torch.no_grad():
            fitness_tensor = self._fitness_fn(genes_tensor)

        # Convert back to Fitness objects
        fitness_np = fitness_tensor.cpu().numpy()

        if fitness_np.ndim == 1:
            return [Fitness.scalar(float(v)) for v in fitness_np]
        else:
            return [Fitness(values=row) for row in fitness_np]

    def __repr__(self) -> str:
        return f"TorchEvaluator(device={self._device})"


class TorchBackend:
    """
    PyTorch execution backend.

    Uses PyTorch for GPU-accelerated batch evaluation.
    Automatically handles data transfer between CPU and GPU.

    Example:
        >>> backend = TorchBackend(device='cuda')
        >>> results = backend.map_evaluate(evaluator, population)
        >>> backend.shutdown()
    """

    def __init__(
        self,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Create PyTorch backend.

        Args:
            device: Device to use (None = auto-detect)
            dtype: Data type for tensors
        """
        self._device = _get_device(device)
        self._dtype = dtype

        self._capabilities = BackendCapabilities(
            parallel=True,
            gpu=self._device.type in ("cuda", "mps"),
            jit=False,
            device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )

    @property
    def name(self) -> str:
        """Backend name."""
        return "torch"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return self._capabilities

    @property
    def device(self) -> torch.device:
        """Current device."""
        return self._device

    def map_evaluate(
        self,
        evaluator: Any,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate using PyTorch acceleration.

        If evaluator is a TorchEvaluator, uses its GPU path.
        Otherwise, falls back to standard evaluation.

        Args:
            evaluator: Evaluator to use
            individuals: Individuals to evaluate
            seed: Random seed

        Returns:
            Fitness values
        """
        if isinstance(evaluator, TorchEvaluator):
            return evaluator.evaluate(individuals, seed=seed)

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Fall back to standard evaluation
        return evaluator.evaluate(individuals, seed=seed)

    def shutdown(self) -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"TorchBackend(device={self._device})"


# Standard torch implementations of benchmark functions
def sphere_torch(x: torch.Tensor) -> torch.Tensor:
    """Sphere function in PyTorch."""
    return torch.sum(x**2, dim=1)


def rastrigin_torch(x: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    """Rastrigin function in PyTorch."""
    n = x.shape[1]
    return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x), dim=1)


def rosenbrock_torch(x: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function in PyTorch."""
    return torch.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
        dim=1,
    )


def ackley_torch(
    x: torch.Tensor, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> torch.Tensor:
    """Ackley function in PyTorch."""
    n = x.shape[1]
    sum1 = torch.sum(x**2, dim=1)
    sum2 = torch.sum(torch.cos(c * x), dim=1)
    return -a * torch.exp(-b * torch.sqrt(sum1 / n)) - torch.exp(sum2 / n) + a + np.e


__all__ = [
    "TorchEvaluator",
    "TorchBackend",
    "sphere_torch",
    "rastrigin_torch",
    "rosenbrock_torch",
    "ackley_torch",
]
