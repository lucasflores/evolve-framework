"""
Evaluator Registry.

Provides a registry mapping evaluator type names to factory functions
for creating evaluator instances declaratively.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "EvaluatorRegistry",
    "get_evaluator_registry",
    "reset_evaluator_registry",
]


class EvaluatorRegistry:
    """
    Registry mapping evaluator type names to factory callables.

    Built-in types: benchmark, function, llm_judge, ground_truth, scm, rl, meta

    Uses lazy initialization - built-in evaluators registered on first access.

    Example:
        >>> registry = get_evaluator_registry()
        >>> evaluator = registry.get("benchmark", function_name="sphere", dimensions=10)
        >>> registry.register("custom", custom_factory)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, Callable[..., Any]] = {}
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in evaluators.

        Called automatically on first access.
        """
        if self._initialized:
            return
        self._initialized = True
        _register_builtin_evaluators(self)

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register an evaluator factory.

        Args:
            name: Evaluator name (e.g., "benchmark", "llm_judge").
            factory: Callable that returns an Evaluator instance.

        Raises:
            TypeError: If factory is not callable.
        """
        if not callable(factory):
            raise TypeError(
                f"EvaluatorRegistry: factory for '{name}' must be callable, "
                f"got {type(factory).__name__}"
            )
        self._factories[name] = factory

    def get(self, name: str, **params: Any) -> Any:
        """
        Instantiate an evaluator by name.

        Args:
            name: Registered evaluator name.
            **params: Keyword arguments passed to the factory.

        Returns:
            Evaluator instance.

        Raises:
            KeyError: If evaluator name not registered (lists available).
        """
        self._ensure_initialized()

        if name not in self._factories:
            available = self.list_evaluators()
            raise KeyError(
                f"EvaluatorRegistry: '{name}' is not registered. Available evaluators: {available}"
            )

        factory = self._factories[name]
        try:
            return factory(**params)
        except Exception as exc:
            raise type(exc)(
                f"Failed to create evaluator '{name}' with params {params}: {exc}"
            ) from exc

    def is_registered(self, name: str) -> bool:
        """
        Check if an evaluator name is registered.

        Args:
            name: Evaluator name.

        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return name in self._factories

    def list_evaluators(self) -> list[str]:
        """
        List all registered evaluator names.

        Returns:
            Sorted list of registered names.
        """
        self._ensure_initialized()
        return sorted(self._factories.keys())


def _register_builtin_evaluators(registry: EvaluatorRegistry) -> None:
    """
    Register all built-in evaluator types.

    Called during lazy initialization. ML-dependent evaluators use
    deferred imports to avoid requiring ML packages at import time.
    """
    from evolve.evaluation.evaluator import FunctionEvaluator
    from evolve.evaluation.reference.functions import BENCHMARK_FUNCTIONS

    # -----------------------------------------
    # benchmark: wraps BENCHMARK_FUNCTIONS in FunctionEvaluator
    # -----------------------------------------
    def create_benchmark_evaluator(
        function_name: str = "sphere",
        minimize: bool = True,
        **kwargs: Any,
    ) -> FunctionEvaluator:
        """Create a FunctionEvaluator from a named benchmark function."""
        if function_name not in BENCHMARK_FUNCTIONS:
            available = sorted(BENCHMARK_FUNCTIONS.keys())
            raise KeyError(f"Unknown benchmark function '{function_name}'. Available: {available}")
        fn = BENCHMARK_FUNCTIONS[function_name]
        return FunctionEvaluator(fitness_fn=fn, minimize=minimize, **kwargs)

    registry.register("benchmark", create_benchmark_evaluator)

    # -----------------------------------------
    # function: wraps an arbitrary callable via FunctionEvaluator
    # -----------------------------------------
    def create_function_evaluator(
        fitness_fn: Callable[..., float] | None = None,
        **kwargs: Any,
    ) -> FunctionEvaluator:
        """Create a FunctionEvaluator from a callable."""
        if fitness_fn is None:
            raise ValueError("fitness_fn is required for 'function' evaluator")
        return FunctionEvaluator(fitness_fn=fitness_fn, **kwargs)

    registry.register("function", create_function_evaluator)

    # -----------------------------------------
    # llm_judge: deferred import
    # -----------------------------------------
    def create_llm_judge_evaluator(**kwargs: Any) -> Any:
        """Create an LLMJudgeEvaluator (deferred import)."""
        from evolve.evaluation.llm_judge import LLMJudgeEvaluator

        return LLMJudgeEvaluator(**kwargs)

    registry.register("llm_judge", create_llm_judge_evaluator)

    # -----------------------------------------
    # ground_truth: deferred import
    # -----------------------------------------
    def create_ground_truth_evaluator(**kwargs: Any) -> Any:
        """Create a GroundTruthEvaluator (deferred import)."""
        from evolve.evaluation.benchmark import GroundTruthEvaluator

        return GroundTruthEvaluator(**kwargs)

    registry.register("ground_truth", create_ground_truth_evaluator)

    # -----------------------------------------
    # scm
    # -----------------------------------------
    def create_scm_evaluator(**kwargs: Any) -> Any:
        """Create an SCMEvaluator."""
        from evolve.evaluation.scm_evaluator import SCMEvaluator

        return SCMEvaluator(**kwargs)

    registry.register("scm", create_scm_evaluator)

    # -----------------------------------------
    # rl: deferred import
    # -----------------------------------------
    def create_rl_evaluator(**kwargs: Any) -> Any:
        """Create an RLEvaluator (deferred import)."""
        from evolve.rl.evaluator import RLEvaluator

        return RLEvaluator(**kwargs)

    registry.register("rl", create_rl_evaluator)

    # -----------------------------------------
    # meta
    # -----------------------------------------
    def create_meta_evaluator(**kwargs: Any) -> Any:
        """Create a MetaEvaluator."""
        from evolve.meta.evaluator import MetaEvaluator

        return MetaEvaluator(**kwargs)

    registry.register("meta", create_meta_evaluator)


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_evaluator_registry: EvaluatorRegistry | None = None


def get_evaluator_registry() -> EvaluatorRegistry:
    """
    Get the global evaluator registry.

    Creates and initializes on first call (lazy singleton).

    Returns:
        Global EvaluatorRegistry instance.
    """
    global _evaluator_registry
    if _evaluator_registry is None:
        _evaluator_registry = EvaluatorRegistry()
    return _evaluator_registry


def reset_evaluator_registry() -> None:
    """
    Reset global registry (for testing).

    Clears the singleton, causing re-initialization on next access.
    """
    global _evaluator_registry
    _evaluator_registry = None
