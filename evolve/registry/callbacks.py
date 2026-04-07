"""
Callback Registry.

Provides a registry mapping callback type names to factory functions
for creating callback instances declaratively.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "CallbackRegistry",
    "get_callback_registry",
    "reset_callback_registry",
]


class CallbackRegistry:
    """
    Registry mapping callback type names to factory callables.

    Built-in types: logging, checkpoint, print, history

    Uses lazy initialization - built-in callbacks registered on first access.

    Example:
        >>> registry = get_callback_registry()
        >>> callback = registry.get("print", print_every=5)
        >>> registry.register("custom", custom_factory)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, Callable[..., Any]] = {}
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in callbacks.

        Called automatically on first access.
        """
        if self._initialized:
            return
        self._initialized = True
        _register_builtin_callbacks(self)

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register a callback factory.

        Args:
            name: Callback name (e.g., "logging", "print").
            factory: Callable that returns a Callback instance.

        Raises:
            TypeError: If factory is not callable.
        """
        if not callable(factory):
            raise TypeError(
                f"CallbackRegistry: factory for '{name}' must be callable, "
                f"got {type(factory).__name__}"
            )
        self._factories[name] = factory

    def get(self, name: str, **params: Any) -> Any:
        """
        Instantiate a callback by name.

        Args:
            name: Registered callback name.
            **params: Keyword arguments passed to the factory.

        Returns:
            Callback instance.

        Raises:
            KeyError: If callback name not registered (lists available).
        """
        self._ensure_initialized()

        if name not in self._factories:
            available = self.list_callbacks()
            raise KeyError(
                f"CallbackRegistry: '{name}' is not registered. Available callbacks: {available}"
            )

        factory = self._factories[name]
        try:
            return factory(**params)
        except Exception as exc:
            raise type(exc)(
                f"Failed to create callback '{name}' with params {params}: {exc}"
            ) from exc

    def is_registered(self, name: str) -> bool:
        """
        Check if a callback name is registered.

        Args:
            name: Callback name.

        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return name in self._factories

    def list_callbacks(self) -> list[str]:
        """
        List all registered callback names.

        Returns:
            Sorted list of registered names.
        """
        self._ensure_initialized()
        return sorted(self._factories.keys())


def _register_builtin_callbacks(registry: CallbackRegistry) -> None:
    """
    Register all built-in callback types.

    Called during lazy initialization.
    """
    from evolve.core.callbacks import (
        CheckpointCallback,
        HistoryCallback,
        LoggingCallback,
        PrintCallback,
    )

    # -----------------------------------------
    # logging
    # -----------------------------------------
    def create_logging_callback(**kwargs: Any) -> LoggingCallback:
        """Create a LoggingCallback."""
        return LoggingCallback(**kwargs)

    registry.register("logging", create_logging_callback)

    # -----------------------------------------
    # checkpoint
    # -----------------------------------------
    def create_checkpoint_callback(**kwargs: Any) -> CheckpointCallback:
        """Create a CheckpointCallback."""
        return CheckpointCallback(**kwargs)

    registry.register("checkpoint", create_checkpoint_callback)

    # -----------------------------------------
    # print
    # -----------------------------------------
    def create_print_callback(**kwargs: Any) -> PrintCallback:
        """Create a PrintCallback."""
        return PrintCallback(**kwargs)

    registry.register("print", create_print_callback)

    # -----------------------------------------
    # history
    # -----------------------------------------
    def create_history_callback(**kwargs: Any) -> HistoryCallback:
        """Create a HistoryCallback."""
        return HistoryCallback(**kwargs)

    registry.register("history", create_history_callback)


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_callback_registry: CallbackRegistry | None = None


def get_callback_registry() -> CallbackRegistry:
    """
    Get the global callback registry.

    Creates and initializes on first call (lazy singleton).

    Returns:
        Global CallbackRegistry instance.
    """
    global _callback_registry
    if _callback_registry is None:
        _callback_registry = CallbackRegistry()
    return _callback_registry


def reset_callback_registry() -> None:
    """
    Reset global registry (for testing).

    Clears the singleton, causing re-initialization on next access.
    """
    global _callback_registry
    _callback_registry = None
