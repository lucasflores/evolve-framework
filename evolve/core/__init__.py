"""
Core module - Backend-agnostic evolution engine interfaces.

This module contains NO ML framework imports (PyTorch, JAX, TensorFlow).
All types use NumPy and Python stdlib only.
"""

from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.core.population import Population

__all__ = [
    "Fitness",
    "Individual",
    "IndividualMetadata",
    "Population",
]
