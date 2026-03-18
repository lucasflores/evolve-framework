"""
Genome Registry.

Provides a registry mapping genome type names to factory functions
for creating genome instances.
"""

from __future__ import annotations

from random import Random
from typing import Any, Callable, TypeVar

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
            raise KeyError(
                f"Genome type '{name}' not found. "
                f"Available: {available}"
            )
        
        return self._factories[name]
    
    def create(self, name: str, rng: Random | None = None, **params: Any) -> Any:
        """
        Create a genome instance.
        
        Args:
            name: Registered genome type name.
            rng: Random number generator.
            **params: Parameters for genome creation.
            
        Returns:
            New genome instance.
        """
        self._ensure_initialized()
        
        factory = self.get_factory(name)
        
        # Merge default params with provided params
        merged_params = dict(self._default_params.get(name, {}))
        merged_params.update(params)
        
        if rng is not None:
            merged_params["rng"] = rng
        
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


def _register_builtin_genomes(registry: GenomeRegistry) -> None:
    """
    Register all built-in genome types (FR-024).
    
    Called during lazy initialization.
    """
    # Import genome classes (deferred to avoid circular imports)
    from evolve.representation.vector import VectorGenome
    from evolve.representation.sequence import SequenceGenome
    from evolve.representation.graph import GraphGenome
    from evolve.representation.scm import SCMGenome
    import numpy as np
    
    # -----------------------------------------
    # Vector genome factory
    # -----------------------------------------
    def create_vector_genome(
        dimensions: int = 10,
        bounds: tuple[float, float] = (-1.0, 1.0),
        rng: Random | None = None,
        **kwargs: Any,
    ) -> VectorGenome:
        """Create a random VectorGenome."""
        if rng is None:
            rng = Random()
        
        lower, upper = bounds
        genes = [rng.uniform(lower, upper) for _ in range(dimensions)]
        
        # Convert bounds to numpy arrays (required by VectorGenome)
        lower_arr = np.full(dimensions, lower, dtype=np.float64)
        upper_arr = np.full(dimensions, upper, dtype=np.float64)
        
        return VectorGenome(genes=genes, bounds=(lower_arr, upper_arr))
    
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
        **kwargs: Any,
    ) -> SequenceGenome:
        """Create a random SequenceGenome."""
        if rng is None:
            rng = Random()
        
        genes = [rng.choice(alphabet) for _ in range(length)]
        return SequenceGenome(genes=genes)
    
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
        **kwargs: Any,
    ) -> GraphGenome:
        """Create a minimal GraphGenome (inputs -> outputs directly)."""
        if rng is None:
            rng = Random()
        
        # Create a minimal network with input-output connections
        return GraphGenome.create_minimal(
            n_inputs=input_nodes,
            n_outputs=output_nodes,
            rng=rng,
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
        **kwargs: Any,
    ) -> SCMGenome:
        """Create a random SCMGenome."""
        if rng is None:
            rng = Random()
        
        return SCMGenome.create_random(
            n_variables=num_variables,
            rng=rng,
        )
    
    registry.register(
        "scm",
        create_scm_genome,
        default_params={"num_variables": 5},
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
