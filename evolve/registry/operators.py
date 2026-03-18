"""
Operator Registry.

Provides a registry mapping operator names to implementations,
with lazy initialization and genome compatibility tracking.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable


class OperatorRegistry:
    """
    Registry mapping (category, name) to operator classes.
    
    Categories:
        - "selection": Selection operators
        - "crossover": Crossover operators
        - "mutation": Mutation operators
    
    Tracks genome compatibility metadata for validation at factory time.
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
        Lazy initialization of built-in operators (FR-015).
        
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
        Register an operator (FR-019).
        
        Args:
            category: Operator category ("selection", "crossover", "mutation").
            name: Unique name within category.
            cls: Operator class.
            compatible_genomes: Set of compatible genome types.
                Use {"*"} for all genomes. None means all compatible.
        
        Raises:
            ValueError: If category is invalid.
        """
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category: {category!r}. "
                f"Must be one of {self.CATEGORIES}"
            )
        
        self._operators[(category, name)] = cls
        if compatible_genomes is not None:
            self._compatibility[name] = compatible_genomes
        else:
            # Default: compatible with all
            self._compatibility[name] = {"*"}
    
    def get(self, category: str, name: str, **params: Any) -> Any:
        """
        Instantiate an operator (FR-020).
        
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
        Check if operator is compatible with genome type (FR-021).
        
        Args:
            operator_name: Registered operator name.
            genome_type: Genome type name.
            
        Returns:
            True if compatible or unspecified.
        """
        self._ensure_initialized()
        
        if operator_name not in self._compatibility:
            # Unspecified = assumed compatible
            return True
        
        compatible = self._compatibility[operator_name]
        return "*" in compatible or genome_type in compatible
    
    def get_compatibility(self, operator_name: str) -> set[str]:
        """
        Get compatible genome types for operator.
        
        Args:
            operator_name: Registered operator name.
            
        Returns:
            Set of compatible genome types.
            {"*"} if compatible with all.
            Empty set if not registered.
        """
        self._ensure_initialized()
        return self._compatibility.get(operator_name, set())
    
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
    
    def is_registered(self, category: str, name: str) -> bool:
        """
        Check if operator is registered.
        
        Args:
            category: Operator category.
            name: Operator name.
            
        Returns:
            True if registered.
        """
        self._ensure_initialized()
        return (category, name) in self._operators


def _register_builtin_operators(registry: OperatorRegistry) -> None:
    """
    Register all built-in operators (FR-016, FR-017, FR-018).
    
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
        BlendCrossover,
        SimulatedBinaryCrossover,
        NEATCrossover,
    )
    from evolve.core.operators.mutation import (
        GaussianMutation,
        UniformMutation,
        PolynomialMutation,
        CreepMutation,
        NEATMutation,
    )
    from evolve.multiobjective.selection import CrowdedTournamentSelection
    
    # -----------------------------------------
    # Selection operators (FR-016)
    # -----------------------------------------
    # All selection operators work with any genome type
    
    registry.register(
        "selection", "tournament",
        TournamentSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection", "roulette",
        RouletteSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection", "rank",
        RankSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection", "crowded_tournament",
        CrowdedTournamentSelection,
        compatible_genomes={"*"},
    )
    
    # -----------------------------------------
    # Crossover operators (FR-017)
    # -----------------------------------------
    
    registry.register(
        "crossover", "uniform",
        UniformCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover", "single_point",
        SinglePointCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover", "two_point",
        TwoPointCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover", "blend",
        BlendCrossover,
        compatible_genomes={"vector"},
    )
    registry.register(
        "crossover", "sbx",
        SimulatedBinaryCrossover,
        compatible_genomes={"vector"},
    )
    registry.register(
        "crossover", "neat",
        NEATCrossover,
        compatible_genomes={"graph"},
    )
    
    # -----------------------------------------
    # Mutation operators (FR-018)
    # -----------------------------------------
    
    registry.register(
        "mutation", "gaussian",
        GaussianMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation", "uniform",
        UniformMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation", "polynomial",
        PolynomialMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation", "creep",
        CreepMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation", "neat",
        NEATMutation,
        compatible_genomes={"graph"},
    )


# -----------------------------------------------------------------------------
# Module-level singleton
# -----------------------------------------------------------------------------

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
