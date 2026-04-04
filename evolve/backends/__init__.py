"""
Execution backends for the Evolve Framework.

Provides different execution strategies for evaluation:
- SequentialBackend: Default CPU sequential execution
- ParallelBackend: Multiprocessing-based parallelism
- Accelerated backends: Optional GPU/JIT support (torch, jax)

Backend Detection:
    >>> from evolve.backends import get_available_backends, get_default_backend
    >>> print(get_available_backends())  # ['sequential', 'parallel', ...]
    >>> backend = get_default_backend()

NO ML FRAMEWORK IMPORTS IN THIS FILE.
ML frameworks are only imported in accelerated/ submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolve.backends.base import ExecutionBackend

# Lazy imports to avoid loading unnecessary dependencies
_BACKEND_REGISTRY: dict[str, type[ExecutionBackend]] = {}
_BACKENDS_LOADED = False


def _load_backends() -> None:
    """Load available backends lazily."""
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return

    from evolve.backends.parallel import ParallelBackend
    from evolve.backends.sequential import SequentialBackend

    _BACKEND_REGISTRY["sequential"] = SequentialBackend
    _BACKEND_REGISTRY["parallel"] = ParallelBackend

    # Try loading accelerated backends
    try:
        from evolve.backends.accelerated import TorchBackend

        _BACKEND_REGISTRY["torch"] = TorchBackend
    except ImportError:
        pass

    try:
        from evolve.backends.accelerated import JaxBackend

        _BACKEND_REGISTRY["jax"] = JaxBackend
    except ImportError:
        pass

    _BACKENDS_LOADED = True


def get_available_backends() -> list[str]:
    """
    Get list of available backend names.

    Returns:
        List of backend names that can be instantiated

    Example:
        >>> backends = get_available_backends()
        >>> print(backends)  # ['sequential', 'parallel', 'torch', ...]
    """
    _load_backends()
    return list(_BACKEND_REGISTRY.keys())


def get_backend(name: str) -> ExecutionBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ('sequential', 'parallel', 'torch', 'jax')

    Returns:
        Backend instance

    Raises:
        ValueError: If backend is not available

    Example:
        >>> backend = get_backend('parallel')
    """
    _load_backends()
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Backend '{name}' not available. Options: {available}")
    return _BACKEND_REGISTRY[name]()


def get_default_backend() -> ExecutionBackend:
    """
    Get the default backend.

    Returns sequential backend, which is always available.

    Returns:
        Default backend instance
    """
    return get_backend("sequential")


def has_gpu_support() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if torch or jax backend is available
    """
    _load_backends()
    return "torch" in _BACKEND_REGISTRY or "jax" in _BACKEND_REGISTRY


def has_torch() -> bool:
    """Check if PyTorch backend is available."""
    _load_backends()
    return "torch" in _BACKEND_REGISTRY


def has_jax() -> bool:
    """Check if JAX backend is available."""
    _load_backends()
    return "jax" in _BACKEND_REGISTRY


__all__ = [
    "get_available_backends",
    "get_backend",
    "get_default_backend",
    "has_gpu_support",
    "has_torch",
    "has_jax",
]
