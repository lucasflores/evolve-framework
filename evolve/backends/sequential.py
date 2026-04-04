"""
Sequential execution backend.

The default backend - runs evaluations sequentially on CPU.
Always available, no external dependencies.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from evolve.backends.base import (
    BackendCapabilities,
    Evaluator,
)
from evolve.core.types import Fitness, Individual

G = TypeVar("G")


class SequentialBackend:
    """
    Sequential execution backend (default).

    Runs evaluations one at a time on the CPU.
    This is the simplest backend with no parallelism.

    Advantages:
    - Always available
    - Easiest to debug
    - Deterministic ordering
    - No serialization overhead

    Use this backend when:
    - Debugging evaluation issues
    - Evaluation is already fast
    - Population is small
    - Reproducibility is critical

    Example:
        >>> backend = SequentialBackend()
        >>> results = backend.map_evaluate(evaluator, population)
    """

    def __init__(self) -> None:
        """Create sequential backend."""
        self._capabilities = BackendCapabilities(
            parallel=False,
            gpu=False,
            jit=False,
            distributed=False,
            max_workers=1,
            device_count=0,
        )

    @property
    def name(self) -> str:
        """Backend name."""
        return "sequential"

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities."""
        return self._capabilities

    def map_evaluate(
        self,
        evaluator: Evaluator[G],
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals sequentially.

        Simply calls evaluator.evaluate() directly.

        Args:
            evaluator: Evaluator to use
            individuals: Individuals to evaluate
            seed: Random seed (passed to evaluator)

        Returns:
            Fitness values in same order
        """
        return evaluator.evaluate(individuals, seed=seed)

    def shutdown(self) -> None:
        """No cleanup needed for sequential backend."""
        pass

    def __repr__(self) -> str:
        return "SequentialBackend()"


__all__ = ["SequentialBackend"]
