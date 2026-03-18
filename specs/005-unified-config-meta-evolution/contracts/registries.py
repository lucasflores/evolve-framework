"""
Registry Contracts

Defines interfaces for the operator and genome registries.
Registries provide lazy initialization and extensibility.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Any, Callable, Generic, TypeVar

G = TypeVar("G")
T = TypeVar("T")


# =============================================================================
# Base Registry Protocol
# =============================================================================


class Registry(ABC, Generic[T]):
    """
    Abstract base for registries.
    
    Provides common registration and lookup functionality.
    """
    
    @abstractmethod
    def register(self, name: str, factory: T, **metadata: Any) -> None:
        """
        Register an item.
        
        Args:
            name: Unique identifier for lookup.
            factory: Factory function or class to register.
            **metadata: Additional metadata (e.g., compatibility info).
        """
        ...
    
    @abstractmethod
    def get(self, name: str, **params: Any) -> Any:
        """
        Get and instantiate an item.
        
        Args:
            name: Registered identifier.
            **params: Parameters for instantiation.
            
        Returns:
            Instantiated item.
            
        Raises:
            KeyError: If name not registered.
        """
        ...
    
    @abstractmethod
    def list_names(self) -> list[str]:
        """
        List all registered names.
        
        Returns:
            List of registered identifiers.
        """
        ...
    
    @abstractmethod
    def is_registered(self, name: str) -> bool:
        """
        Check if name is registered.
        
        Args:
            name: Identifier to check.
            
        Returns:
            True if registered.
        """
        ...


# =============================================================================
# Operator Registry
# =============================================================================


class OperatorRegistry:
    """
    Registry mapping (category, name) to operator classes.
    
    Categories:
        - "selection": Selection operators
        - "crossover": Crossover operators
        - "mutation": Mutation operators
    
    Tracks genome compatibility metadata for validation.
    Uses lazy initialization - built-in operators registered on first access.
    
    Example:
        >>> registry = get_operator_registry()
        >>> selection = registry.get("selection", "tournament", tournament_size=5)
        >>> registry.register("mutation", "custom", CustomMutation, compatible_genomes={"vector"})
    """
    
    CATEGORIES = ("selection", "crossover", "mutation")
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self._operators: dict[tuple[str, str], type] = {}
        self._compatibility: dict[str, set[str]] = {}
        self._initialized: bool = False
    
    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in operators.
        
        Called automatically on first access.
        """
        if self._initialized:
            return
        self._initialized = True
        _register_builtin_operators(self)
    
    def register(
        self,
        category: str,
        name: str,
        cls: type,
        compatible_genomes: set[str] | None = None,
    ) -> None:
        """
        Register an operator.
        
        Args:
            category: Operator category ("selection", "crossover", "mutation").
            name: Unique name within category.
            cls: Operator class.
            compatible_genomes: Set of compatible genome types.
                Use {"*"} for all genomes. None means unspecified.
        
        Raises:
            ValueError: If category is invalid.
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}. Must be one of {self.CATEGORIES}")
        
        self._operators[(category, name)] = cls
        if compatible_genomes is not None:
            self._compatibility[name] = compatible_genomes
    
    def get(self, category: str, name: str, **params: Any) -> Any:
        """
        Instantiate an operator.
        
        Args:
            category: Operator category.
            name: Registered operator name.
            **params: Constructor parameters.
            
        Returns:
            Instantiated operator.
            
        Raises:
            KeyError: If operator not registered.
        """
        self._ensure_initialized()
        
        key = (category, name)
        if key not in self._operators:
            available = self.list_operators(category)
            raise KeyError(
                f"Operator '{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        
        cls = self._operators[key]
        return cls(**params)
    
    def is_compatible(self, operator_name: str, genome_type: str) -> bool:
        """
        Check if operator is compatible with genome type.
        
        Args:
            operator_name: Registered operator name.
            genome_type: Genome type name.
            
        Returns:
            True if compatible or compatibility unspecified.
        """
        self._ensure_initialized()
        
        if operator_name not in self._compatibility:
            # Unspecified = assumed compatible
            return True
        
        compatible = self._compatibility[operator_name]
        return "*" in compatible or genome_type in compatible
    
    def list_operators(self, category: str) -> list[str]:
        """
        List operators in category.
        
        Args:
            category: Operator category.
            
        Returns:
            List of registered operator names.
        """
        self._ensure_initialized()
        return [name for (cat, name) in self._operators if cat == category]
    
    def list_all(self) -> dict[str, list[str]]:
        """
        List all operators by category.
        
        Returns:
            Dictionary mapping category to list of names.
        """
        self._ensure_initialized()
        return {cat: self.list_operators(cat) for cat in self.CATEGORIES}
    
    def get_compatibility(self, operator_name: str) -> set[str]:
        """
        Get compatible genome types for operator.
        
        Args:
            operator_name: Registered operator name.
            
        Returns:
            Set of compatible genome types.
            {"*"} if compatible with all.
            Empty set if unspecified.
        """
        self._ensure_initialized()
        return self._compatibility.get(operator_name, set())


def _register_builtin_operators(registry: OperatorRegistry) -> None:
    """
    Register all built-in operators.
    
    Called during lazy initialization.
    """
    # Import operators (deferred to avoid circular imports)
    from evolve.core.operators.selection import (
        TournamentSelection,
        RouletteSelection,
        RankSelection,
    )
    from evolve.core.operators.crossover import (
        UniformCrossover,
        SinglePointCrossover,
        TwoPointCrossover,
    )
    from evolve.core.operators.mutation import (
        GaussianMutation,
        UniformMutation,
    )
    
    # Selection (compatible with all genomes)
    registry.register("selection", "tournament", TournamentSelection, compatible_genomes={"*"})
    registry.register("selection", "roulette", RouletteSelection, compatible_genomes={"*"})
    registry.register("selection", "rank", RankSelection, compatible_genomes={"*"})
    
    # Crossover (genome-specific)
    registry.register("crossover", "uniform", UniformCrossover, compatible_genomes={"vector", "sequence"})
    registry.register("crossover", "single_point", SinglePointCrossover, compatible_genomes={"vector", "sequence"})
    registry.register("crossover", "two_point", TwoPointCrossover, compatible_genomes={"vector", "sequence"})
    
    # Mutation (genome-specific)
    registry.register("mutation", "gaussian", GaussianMutation, compatible_genomes={"vector"})
    registry.register("mutation", "uniform", UniformMutation, compatible_genomes={"vector"})
    
    # TODO: Register SBX, Blend, Polynomial, NEATCrossover, NEATMutation
    # TODO: Register CrowdedTournamentSelection for multi-objective


# Module-level singleton
_operator_registry: OperatorRegistry | None = None


def get_operator_registry() -> OperatorRegistry:
    """
    Get the global operator registry.
    
    Creates and initializes on first call (lazy singleton).
    
    Returns:
        Global OperatorRegistry instance.
    """
    global _operator_registry
    if _operator_registry is None:
        _operator_registry = OperatorRegistry()
    return _operator_registry


def reset_operator_registry() -> None:
    """
    Reset global registry (for testing).
    
    Clears the singleton, causing re-initialization on next access.
    """
    global _operator_registry
    _operator_registry = None


# =============================================================================
# Genome Registry
# =============================================================================


# Type for genome factory functions
GenomeFactory = Callable[..., G]


class GenomeRegistry:
    """
    Registry mapping genome type names to factory functions.
    
    Built-in types: vector, sequence, graph, scm
    
    Uses lazy initialization - built-in genomes registered on first access.
    
    Example:
        >>> registry = get_genome_registry()
        >>> genome = registry.create("vector", dimensions=10, bounds=(-5, 5), rng=rng)
        >>> registry.register("custom", custom_factory)
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self._factories: dict[str, GenomeFactory] = {}
        self._default_params: dict[str, dict[str, Any]] = {}
        self._initialized: bool = False
    
    def _ensure_initialized(self) -> None:
        """
        Lazy initialization of built-in genomes.
        
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
        Register a genome type.
        
        Args:
            name: Unique type name.
            factory: Factory function that creates genomes.
            default_params: Default parameters for factory.
        """
        self._factories[name] = factory
        if default_params:
            self._default_params[name] = default_params
    
    def create(self, name: str, rng: Random, **params: Any) -> Any:
        """
        Create a genome instance.
        
        Args:
            name: Registered type name.
            rng: Random number generator.
            **params: Factory parameters (override defaults).
            
        Returns:
            New genome instance.
            
        Raises:
            KeyError: If type not registered.
        """
        self._ensure_initialized()
        
        if name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(
                f"Genome type '{name}' not registered. Available: {available}"
            )
        
        # Merge defaults with provided params
        final_params = dict(self._default_params.get(name, {}))
        final_params.update(params)
        final_params["rng"] = rng
        
        factory = self._factories[name]
        return factory(**final_params)
    
    def get_factory(self, name: str) -> GenomeFactory:
        """
        Get factory function for type.
        
        Args:
            name: Registered type name.
            
        Returns:
            Factory function.
        """
        self._ensure_initialized()
        if name not in self._factories:
            raise KeyError(f"Genome type '{name}' not registered")
        return self._factories[name]
    
    def list_types(self) -> list[str]:
        """
        List registered genome types.
        
        Returns:
            List of type names.
        """
        self._ensure_initialized()
        return list(self._factories.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if type is registered.
        
        Args:
            name: Type name to check.
            
        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return name in self._factories


def _register_builtin_genomes(registry: GenomeRegistry) -> None:
    """
    Register built-in genome types.
    
    Called during lazy initialization.
    """
    from evolve.representation.vector import VectorGenome
    from evolve.representation.sequence import SequenceGenome
    from evolve.representation.graph import GraphGenome
    from evolve.representation.scm import SCMGenome
    
    registry.register(
        "vector",
        VectorGenome.random,
        default_params={"dimensions": 10, "bounds": (-1.0, 1.0)},
    )
    registry.register(
        "sequence",
        SequenceGenome.random,
        default_params={"length": 10, "alphabet": "ACGT"},
    )
    registry.register(
        "graph",
        GraphGenome.random,
        default_params={"input_nodes": 2, "output_nodes": 1},
    )
    registry.register(
        "scm",
        SCMGenome.random,
        default_params={"num_variables": 5},
    )


# Module-level singleton
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
