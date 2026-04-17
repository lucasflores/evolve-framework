"""
Decoder Registry.

Provides a registry mapping decoder type names to factory functions
for creating decoder instances declaratively.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "DecoderRegistry",
    "get_decoder_registry",
    "reset_decoder_registry",
]


class DecoderRegistry:
    """
    Registry mapping decoder type names to factory callables.

    Built-in types: identity, graph_to_network, graph_to_mlp

    Uses lazy initialization - built-in decoders registered on first access.

    Example:
        >>> registry = get_decoder_registry()
        >>> decoder = registry.get("graph_to_network")
        >>> registry.register("custom", custom_factory)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, Callable[..., Any]] = {}
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in decoders.

        Called automatically on first access.
        """
        if self._initialized:
            return
        self._initialized = True
        _register_builtin_decoders(self)

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        """
        Register a decoder factory.

        Args:
            name: Decoder name (e.g., "graph_to_network", "identity").
            factory: Callable that returns a Decoder instance.

        Raises:
            TypeError: If factory is not callable.
        """
        if not callable(factory):
            raise TypeError(
                f"DecoderRegistry: factory for '{name}' must be callable, "
                f"got {type(factory).__name__}"
            )
        self._factories[name] = factory

    def get(self, name: str, **params: Any) -> Any:
        """
        Instantiate a decoder by name.

        Args:
            name: Registered decoder name.
            **params: Keyword arguments passed to the factory.

        Returns:
            Decoder instance.

        Raises:
            KeyError: If decoder name not registered (lists available).
        """
        self._ensure_initialized()

        if name not in self._factories:
            available = self.list_decoders()
            raise KeyError(
                f"DecoderRegistry: '{name}' is not registered. Available decoders: {available}"
            )

        factory = self._factories[name]
        try:
            return factory(**params)
        except Exception as exc:
            raise type(exc)(
                f"Failed to create decoder '{name}' with params {params}: {exc}"
            ) from exc

    def is_registered(self, name: str) -> bool:
        """
        Check if a decoder name is registered.

        Args:
            name: Decoder name.

        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return name in self._factories

    def list_decoders(self) -> list[str]:
        """
        List all registered decoder names.

        Returns:
            Sorted list of registered names.
        """
        self._ensure_initialized()
        return sorted(self._factories.keys())


def _register_builtin_decoders(registry: DecoderRegistry) -> None:
    """
    Register all built-in decoder types.

    Called during lazy initialization.
    """
    from evolve.representation.phenotype import IdentityDecoder

    # -----------------------------------------
    # identity: returns genome unchanged
    # -----------------------------------------
    def create_identity_decoder(**_kwargs: Any) -> IdentityDecoder:
        """Create an IdentityDecoder."""
        return IdentityDecoder()

    registry.register("identity", create_identity_decoder)

    # -----------------------------------------
    # graph_to_network: GraphGenome → NEATNetwork
    # -----------------------------------------
    def create_graph_to_network_decoder(**kwargs: Any) -> Any:
        """Create a GraphToNetworkDecoder."""
        from evolve.representation.decoder import GraphToNetworkDecoder

        return GraphToNetworkDecoder(**kwargs)

    registry.register("graph_to_network", create_graph_to_network_decoder)

    # -----------------------------------------
    # graph_to_mlp: GraphGenome → NumpyNetwork (layer-structured)
    # -----------------------------------------
    def create_graph_to_mlp_decoder(**kwargs: Any) -> Any:
        """Create a GraphToMLPDecoder."""
        from evolve.representation.decoder import GraphToMLPDecoder

        return GraphToMLPDecoder(**kwargs)

    registry.register("graph_to_mlp", create_graph_to_mlp_decoder)


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_decoder_registry: DecoderRegistry | None = None


def get_decoder_registry() -> DecoderRegistry:
    """
    Get the global decoder registry.

    Creates and initializes on first call (lazy singleton).

    Returns:
        Global DecoderRegistry instance.
    """
    global _decoder_registry
    if _decoder_registry is None:
        _decoder_registry = DecoderRegistry()
    return _decoder_registry


def reset_decoder_registry() -> None:
    """
    Reset global registry (for testing).

    Clears the singleton, causing re-initialization on next access.
    """
    global _decoder_registry
    _decoder_registry = None
