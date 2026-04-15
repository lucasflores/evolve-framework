"""
Genome Registry.

Provides a registry mapping genome type names to factory functions
for creating genome instances.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from random import Random
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from evolve.representation.embedding import EmbeddingGenome as EmbeddingGenome

G = TypeVar("G")

# Type for genome factory functions
GenomeFactory = Callable[..., G]


class GenomeRegistry:
    """
    Registry mapping genome type names to factory functions.

    Built-in types: vector, sequence, graph, scm (FR-024)

    Uses lazy initialization - built-in genomes registered on first access.

    Example:
        >>> registry = get_genome_registry()
        >>> factory = registry.get_factory("vector")
        >>> genome = factory(dimensions=10, bounds=(-5, 5), rng=rng)
        >>> registry.register("custom", custom_factory)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, GenomeFactory] = {}
        self._default_params: dict[str, dict[str, Any]] = {}
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in genomes (FR-023).

        Called automatically on first access.
        """
        if self._initialized:
            return
        self._initialized = True
        _register_builtin_genomes(self)

    def register(
        self,
        name: str,
        factory: GenomeFactory,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a genome type (FR-026).

        Args:
            name: Unique genome type name.
            factory: Factory function that creates genome instances.
            default_params: Default parameters for the factory.
        """
        self._factories[name] = factory
        if default_params is not None:
            self._default_params[name] = default_params

    def get_factory(self, name: str) -> GenomeFactory:
        """
        Get factory function for genome type (FR-025).

        Args:
            name: Registered genome type name.

        Returns:
            Factory function.

        Raises:
            KeyError: If genome type not registered.
        """
        self._ensure_initialized()

        if name not in self._factories:
            available = self.list_types()
            raise KeyError(f"Genome type '{name}' not found. Available: {available}")

        return self._factories[name]

    def create(self, name: str, rng: Random | None = None, **params: Any) -> Any:
        """
        Create a genome instance.

        Validates params against the factory's signature before invocation.
        Factories accepting ``**kwargs`` skip strict validation.

        Args:
            name: Registered genome type name.
            rng: Random number generator.
            **params: Parameters for genome creation.

        Returns:
            New genome instance.

        Raises:
            ValueError: If unrecognized parameters are provided.
        """
        self._ensure_initialized()

        factory = self.get_factory(name)

        # Merge default params with provided params
        merged_params = dict(self._default_params.get(name, {}))
        merged_params.update(params)

        if rng is not None:
            merged_params["rng"] = rng

        # Validate params against factory signature
        _validate_factory_params(name, factory, merged_params)

        return factory(**merged_params)

    def list_types(self) -> list[str]:
        """
        List registered genome types.

        Returns:
            List of registered type names.
        """
        self._ensure_initialized()
        return list(self._factories.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if genome type is registered.

        Args:
            name: Genome type name.

        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return name in self._factories

    def get_default_params(self, name: str) -> dict[str, Any]:
        """
        Get default parameters for genome type.

        Args:
            name: Registered genome type name.

        Returns:
            Default parameters dictionary.
        """
        self._ensure_initialized()
        return dict(self._default_params.get(name, {}))


def _validate_factory_params(
    genome_type: str,
    factory: GenomeFactory,
    params: dict[str, Any],
) -> None:
    """
    Validate params against the factory's signature using inspect.signature().

    Skips validation if the factory accepts **kwargs (VAR_KEYWORD).

    Args:
        genome_type: Genome type name (for error messages).
        factory: Factory callable.
        params: Merged params dict (including injected 'rng').

    Raises:
        ValueError: If unrecognized parameter names are found.
    """
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return  # Can't introspect — skip validation

    # Check if factory accepts **kwargs → skip strict validation
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return

    accepted = set(sig.parameters.keys())
    unknown = set(params.keys()) - accepted

    if unknown:
        raise ValueError(
            f"Unrecognized genome_params for '{genome_type}': {sorted(unknown)}. "
            f"Accepted parameters: {sorted(accepted)}"
        )


def _register_builtin_genomes(registry: GenomeRegistry) -> None:
    """
    Register all built-in genome types (FR-024).

    Called during lazy initialization.
    """
    # Import genome classes (deferred to avoid circular imports)
    import numpy as np

    from evolve.representation.graph import GraphGenome
    from evolve.representation.scm import SCMGenome
    from evolve.representation.sequence import SequenceGenome
    from evolve.representation.vector import VectorGenome

    # -----------------------------------------
    # Vector genome factory
    # -----------------------------------------
    def create_vector_genome(
        dimensions: int = 10,
        bounds: tuple[float, float] = (-1.0, 1.0),
        rng: Random | None = None,
    ) -> VectorGenome:
        """Create a random VectorGenome."""
        if rng is None:
            rng = Random()

        lower, upper = bounds
        genes = [rng.uniform(lower, upper) for _ in range(dimensions)]

        # Convert bounds to numpy arrays (required by VectorGenome)
        lower_arr = np.full(dimensions, lower, dtype=np.float64)
        upper_arr = np.full(dimensions, upper, dtype=np.float64)

        return VectorGenome(genes=np.array(genes), bounds=(lower_arr, upper_arr))

    registry.register(
        "vector",
        create_vector_genome,
        default_params={"dimensions": 10, "bounds": (-1.0, 1.0)},
    )

    # -----------------------------------------
    # Sequence genome factory
    # -----------------------------------------
    def create_sequence_genome(
        length: int = 10,
        alphabet: tuple = (0, 1),
        rng: Random | None = None,
    ) -> SequenceGenome:
        """Create a random SequenceGenome."""
        if rng is None:
            rng = Random()

        genes = [rng.choice(alphabet) for _ in range(length)]
        return SequenceGenome(genes=tuple(genes))

    registry.register(
        "sequence",
        create_sequence_genome,
        default_params={"length": 10, "alphabet": (0, 1)},
    )

    # -----------------------------------------
    # Graph genome factory
    # -----------------------------------------
    def create_graph_genome(
        input_nodes: int = 2,
        output_nodes: int = 1,
        rng: Random | None = None,
    ) -> GraphGenome:
        """Create a minimal GraphGenome (inputs -> outputs directly)."""
        if rng is None:
            rng = Random()

        # Create a minimal network with input-output connections
        return cast(
            GraphGenome,
            GraphGenome.create_minimal(  # type: ignore[attr-defined]
                n_inputs=input_nodes,
                n_outputs=output_nodes,
                rng=rng,
            ),
        )

    registry.register(
        "graph",
        create_graph_genome,
        default_params={"input_nodes": 2, "output_nodes": 1},
    )

    # -----------------------------------------
    # SCM genome factory
    # -----------------------------------------
    def create_scm_genome(
        num_variables: int = 5,
        rng: Random | None = None,
    ) -> SCMGenome:
        """Create a random SCMGenome."""
        if rng is None:
            rng = Random()

        return cast(
            SCMGenome,
            SCMGenome.create_random(  # type: ignore[attr-defined]
                n_variables=num_variables,
                rng=rng,
            ),
        )

    registry.register(
        "scm",
        create_scm_genome,
        default_params={"num_variables": 5},
    )

    # -----------------------------------------
    # Embedding genome factory
    # -----------------------------------------
    def create_embedding_genome(
        n_tokens: int = 8,
        embed_dim: int = 768,
        model_id: str = "default",
        rng: Random | None = None,
    ) -> EmbeddingGenome:  # noqa: F821
        """Create a random EmbeddingGenome."""
        from evolve.representation.embedding import EmbeddingGenome

        if rng is None:
            rng = Random()

        np_rng = np.random.default_rng(rng.getrandbits(64))
        embeddings = np_rng.standard_normal((n_tokens, embed_dim)).astype(np.float32)

        return EmbeddingGenome(
            embeddings=embeddings,
            model_id=model_id,
        )

    registry.register(
        "embedding",
        create_embedding_genome,
        default_params={"n_tokens": 8, "embed_dim": 768, "model_id": "default"},
    )


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

_genome_registry: GenomeRegistry | None = None


def get_genome_registry() -> GenomeRegistry:
    """
    Get the global genome registry.

    Creates and initializes on first call (lazy singleton).

    Returns:
        Global GenomeRegistry instance.
    """
    global _genome_registry
    if _genome_registry is None:
        _genome_registry = GenomeRegistry()
    return _genome_registry


def reset_genome_registry() -> None:
    """
    Reset global registry (for testing).

    Clears the singleton, causing re-initialization on next access.
    """
    global _genome_registry
    _genome_registry = None
