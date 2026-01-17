"""
Accelerated backends for GPU/JIT evaluation.

This module provides optional GPU and JIT-compiled backends
for accelerated fitness evaluation. These backends require
optional dependencies (PyTorch, JAX).

Import Handling:
    Backends are only imported if their dependencies are available.
    Use has_torch() or has_jax() to check availability before importing.

Example:
    >>> from evolve.backends import has_torch
    >>> if has_torch():
    ...     from evolve.backends.accelerated import TorchBackend
    ...     backend = TorchBackend(device='cuda')
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Track which backends are available
_TORCH_AVAILABLE = False
_JAX_AVAILABLE = False

# Try importing torch backend
try:
    from evolve.backends.accelerated.torch_evaluator import TorchBackend, TorchEvaluator
    _TORCH_AVAILABLE = True
except ImportError:
    TorchBackend = None  # type: ignore
    TorchEvaluator = None  # type: ignore

# Try importing jax backend
try:
    from evolve.backends.accelerated.jax_evaluator import JaxBackend, JaxEvaluator
    _JAX_AVAILABLE = True
except ImportError:
    JaxBackend = None  # type: ignore
    JaxEvaluator = None  # type: ignore


def torch_available() -> bool:
    """Check if PyTorch backend is available."""
    return _TORCH_AVAILABLE


def jax_available() -> bool:
    """Check if JAX backend is available."""
    return _JAX_AVAILABLE


__all__ = [
    "torch_available",
    "jax_available",
]

if _TORCH_AVAILABLE:
    __all__.extend(["TorchBackend", "TorchEvaluator"])

if _JAX_AVAILABLE:
    __all__.extend(["JaxBackend", "JaxEvaluator"])
