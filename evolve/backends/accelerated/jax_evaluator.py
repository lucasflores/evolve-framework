"""
JAX-based accelerated evaluator.

Provides JIT-compiled and GPU-accelerated evaluation using JAX.
This module requires JAX to be installed.

Example:
    >>> from evolve.backends.accelerated import JaxBackend
    >>> backend = JaxBackend()
    >>> results = backend.map_evaluate(evaluator, population)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError as err:
    JAX_AVAILABLE = False
    raise ImportError(
        "JAX is required for JaxBackend. Install with: pip install jax jaxlib"
    ) from err

import numpy as np

from evolve.backends.base import BackendCapabilities
from evolve.core.types import Fitness, Individual

G = TypeVar("G")


def _get_device_info() -> tuple[str, int]:
    """Get JAX device info."""
    devices = jax.devices()
    if devices:
        device_type = devices[0].platform  # 'cpu', 'gpu', 'tpu'
        device_count = len([d for d in devices if d.platform == device_type])
        return device_type, device_count
    return "cpu", 1


class JaxEvaluator:
    """
    JIT-compiled evaluator using JAX.

    Wraps a fitness function with JAX's JIT compilation
    and vmap vectorization for efficient batch evaluation.

    Example:
        >>> def sphere_jax(x: jnp.ndarray) -> float:
        ...     return jnp.sum(x ** 2)
        >>> evaluator = JaxEvaluator(sphere_jax)
        >>> fitness = evaluator.evaluate(individuals)
    """

    def __init__(
        self,
        fitness_fn: Callable[[jnp.ndarray], jnp.ndarray | float],
        jit_compile: bool = True,
        n_objectives: int = 1,
    ) -> None:
        """
        Create JAX evaluator.

        Args:
            fitness_fn: Function mapping (D,) array → scalar or (D,) → (M,)
            jit_compile: Whether to JIT-compile the function
            n_objectives: Number of objectives
        """
        self._raw_fn = fitness_fn
        self._n_objectives = n_objectives
        self._jit_compile = jit_compile

        # Vectorize and optionally JIT compile
        vmapped = vmap(fitness_fn)
        if jit_compile:
            self._fitness_fn = jit(vmapped)
        else:
            self._fitness_fn = vmapped

        # Compile once to avoid first-call latency during evolution
        self._compiled = False

    def _ensure_compiled(self, example_input: jnp.ndarray) -> None:
        """Ensure function is compiled with correct shape."""
        if not self._compiled and self._jit_compile:
            # Warm up with example input
            _ = self._fitness_fn(example_input)
            self._compiled = True

    @property
    def capabilities(self) -> BackendCapabilities:
        """Evaluator capabilities."""
        device_type, device_count = _get_device_info()
        return BackendCapabilities(
            parallel=True,
            gpu=device_type in ("gpu", "tpu"),
            jit=self._jit_compile,
            device_count=device_count,
        )

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        _seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals using JAX.

        Converts genomes to a JAX array, runs JIT-compiled
        vectorized evaluation, and converts back to Fitness objects.

        Args:
            individuals: Individuals to evaluate
            seed: Random seed (sets JAX PRNG key)

        Returns:
            Fitness values
        """
        if not individuals:
            return []

        # Extract genes and convert to JAX array
        genes_list = [np.asarray(getattr(ind.genome, "genes", ind.genome)) for ind in individuals]
        genes_np = np.vstack(genes_list)
        genes_jax = jnp.array(genes_np)

        # Ensure compiled
        self._ensure_compiled(genes_jax[:1])

        # Evaluate (deterministic - JAX functions should use passed keys)
        fitness_jax = self._fitness_fn(genes_jax)

        # Convert back to Fitness objects
        fitness_np = np.asarray(fitness_jax)

        if fitness_np.ndim == 1:
            return [Fitness.scalar(float(v)) for v in fitness_np]
        else:
            return [Fitness(values=row) for row in fitness_np]

    def __repr__(self) -> str:
        return f"JaxEvaluator(jit={self._jit_compile})"


class JaxBackend:
    """
    JAX execution backend.

    Uses JAX for JIT-compiled, GPU-accelerated batch evaluation.
    Supports CPU, GPU, and TPU execution.

    Example:
        >>> backend = JaxBackend()
        >>> results = backend.map_evaluate(evaluator, population)
    """

    def __init__(self) -> None:
        """Create JAX backend."""
        device_type, device_count = _get_device_info()

        self._capabilities = BackendCapabilities(
            parallel=True,
            gpu=device_type in ("gpu", "tpu"),
            jit=True,
            device_count=device_count,
        )
        self._device_type = device_type

    @property
    def name(self) -> str:
        """Backend name."""
        return "jax"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return self._capabilities

    @property
    def device_type(self) -> str:
        """Device type ('cpu', 'gpu', 'tpu')."""
        return self._device_type

    def map_evaluate(
        self,
        evaluator: Any,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate using JAX acceleration.

        If evaluator is a JaxEvaluator, uses its JIT path.
        Otherwise, falls back to standard evaluation.

        Args:
            evaluator: Evaluator to use
            individuals: Individuals to evaluate
            seed: Random seed

        Returns:
            Fitness values
        """
        if isinstance(evaluator, JaxEvaluator):
            return evaluator.evaluate(individuals, seed=seed)

        # Fall back to standard evaluation
        return evaluator.evaluate(individuals, seed=seed)

    def shutdown(self) -> None:
        """Clear JAX caches."""
        jax.clear_caches()

    def __repr__(self) -> str:
        return f"JaxBackend(device={self._device_type})"


# Standard JAX implementations of benchmark functions
def sphere_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Sphere function in JAX (single input)."""
    return jnp.sum(x**2)


def rastrigin_jax(x: jnp.ndarray, A: float = 10.0) -> jnp.ndarray:
    """Rastrigin function in JAX (single input)."""
    n = len(x)
    return A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))


def rosenbrock_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Rosenbrock function in JAX (single input)."""
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_jax(
    x: jnp.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * jnp.pi
) -> jnp.ndarray:
    """Ackley function in JAX (single input)."""
    n = len(x)
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(c * x))
    return -a * jnp.exp(-b * jnp.sqrt(sum1 / n)) - jnp.exp(sum2 / n) + a + jnp.e


__all__ = [
    "JaxEvaluator",
    "JaxBackend",
    "sphere_jax",
    "rastrigin_jax",
    "rosenbrock_jax",
    "ackley_jax",
]
