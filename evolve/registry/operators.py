"""
Operator Registry.

Provides a registry mapping operator names to implementations,
with lazy initialization and genome compatibility tracking.

Used internally by ``create_engine(UnifiedConfig(...))`` to resolve
operator names like ``"tournament"``, ``"gaussian"``, ``"sbx"`` to
their implementing classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evolve.registry._params import accepts_keyword

if TYPE_CHECKING:
    pass


class OperatorRegistry:
    """
    Registry mapping (category, name) to operator classes.

    Categories:
        - "selection": Selection operators
        - "crossover": Crossover operators
        - "mutation": Mutation operators
        - "merge": Symbiogenetic merge operators
        - "distance": Distances between two genomes, called as ``distance(a, b)``

    Tracks genome compatibility metadata for validation at factory time.
    Uses lazy initialization - built-in operators registered on first access.

    Example:
        >>> registry = get_operator_registry()
        >>> selection = registry.get("selection", "tournament", tournament_size=5)
        >>> registry.register("mutation", "custom", CustomMutation, compatible_genomes={"vector"})
    """

    CATEGORIES = ("selection", "crossover", "mutation", "merge", "distance")

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._operators: dict[tuple[str, str], type] = {}
        # Keyed by (category, name): a name may recur across categories ("neat")
        self._compatibility: dict[tuple[str, str], set[str]] = {}
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
            raise ValueError(f"Invalid category: {category!r}. Must be one of {self.CATEGORIES}")

        self._operators[(category, name)] = cls
        if compatible_genomes is not None:
            self._compatibility[(category, name)] = compatible_genomes
        else:
            # Default: compatible with all
            self._compatibility[(category, name)] = {"*"}

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
                f"Operator '{name}' not found in category '{category}'. Available: {available}"
            )

        cls = self._operators[key]
        return cls(**params)

    def accepts_param(self, category: str, name: str, param: str) -> bool:
        """
        Check whether the operator class registered as ``name`` takes ``param``.

        Args:
            category: Operator category.
            name: Registered operator name.
            param: Constructor keyword (e.g. ``"minimize"``).

        Returns:
            True if ``param`` can be passed to the constructor by keyword
            (see ``accepts_keyword``).
        """
        self._ensure_initialized()
        cls = self._operators.get((category, name))
        return cls is not None and accepts_keyword(cls, param)

    def is_compatible(
        self, operator_name: str, genome_type: str, category: str | None = None
    ) -> bool:
        """
        Check if operator is compatible with genome type (FR-021).

        Args:
            operator_name: Registered operator name.
            genome_type: Genome type name.
            category: The operator's category. Without it, a name registered
                in several categories is compatible if any of them is.

        Returns:
            True if compatible or unspecified.
        """
        sets = self._compatible_sets(operator_name, category)
        if not sets:
            # Unspecified = assumed compatible
            return True
        return any("*" in compatible or genome_type in compatible for compatible in sets)

    def get_compatibility(self, operator_name: str, category: str | None = None) -> set[str]:
        """
        Get compatible genome types for operator.

        Args:
            operator_name: Registered operator name.
            category: The operator's category; without it, every category's
                registration of the name counts.

        Returns:
            Set of compatible genome types.
            {"*"} if compatible with all.
            Empty set if not registered.
        """
        return set().union(*self._compatible_sets(operator_name, category))

    def _compatible_sets(self, name: str, category: str | None) -> list[set[str]]:
        """Compatibility of each registration of ``name`` (in ``category``, if given)."""
        self._ensure_initialized()
        return [
            compatible
            for (cat, registered), compatible in self._compatibility.items()
            if registered == name and (category is None or cat == category)
        ]

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
    from evolve.core.operators.crossover import (
        BlendCrossover,
        NEATCrossover,
        SimulatedBinaryCrossover,
        SinglePointCrossover,
        TwoPointCrossover,
        UniformCrossover,
    )
    from evolve.core.operators.mutation import (
        ByKindMutation,
        CreepMutation,
        GaussianMutation,
        NEATMutation,
        PolynomialMutation,
        UniformMutation,
    )
    from evolve.core.operators.selection import (
        RankSelection,
        RouletteSelection,
        TournamentSelection,
    )
    from evolve.multiobjective.selection import CrowdedTournamentSelection

    # -----------------------------------------
    # Selection operators (FR-016)
    # -----------------------------------------
    # All selection operators work with any genome type

    registry.register(
        "selection",
        "tournament",
        TournamentSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection",
        "roulette",
        RouletteSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection",
        "rank",
        RankSelection,
        compatible_genomes={"*"},
    )
    registry.register(
        "selection",
        "crowded_tournament",
        CrowdedTournamentSelection,
        compatible_genomes={"*"},
    )

    # -----------------------------------------
    # Crossover operators (FR-017)
    # -----------------------------------------

    registry.register(
        "crossover",
        "uniform",
        UniformCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover",
        "single_point",
        SinglePointCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover",
        "two_point",
        TwoPointCrossover,
        compatible_genomes={"vector", "sequence"},
    )
    registry.register(
        "crossover",
        "blend",
        BlendCrossover,
        compatible_genomes={"vector"},
    )
    registry.register(
        "crossover",
        "sbx",
        SimulatedBinaryCrossover,
        compatible_genomes={"vector"},
    )
    registry.register(
        "crossover",
        "neat",
        NEATCrossover,
        compatible_genomes={"graph"},
    )

    # -----------------------------------------
    # Mutation operators (FR-018)
    # -----------------------------------------

    registry.register(
        "mutation",
        "gaussian",
        GaussianMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation",
        "uniform",
        UniformMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation",
        "by_kind",
        ByKindMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation",
        "polynomial",
        PolynomialMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation",
        "creep",
        CreepMutation,
        compatible_genomes={"vector"},
    )
    registry.register(
        "mutation",
        "neat",
        NEATMutation,
        compatible_genomes={"graph"},
    )

    # -----------------------------------------
    # Embedding (token-aware) operators
    # -----------------------------------------
    from evolve.core.operators.token_crossover import TokenLevelCrossover
    from evolve.core.operators.token_mutation import TokenAwareMutator

    registry.register(
        "mutation",
        "token_gaussian",
        TokenAwareMutator,
        compatible_genomes={"embedding"},
    )
    registry.register(
        "crossover",
        "token_single_point",
        TokenLevelCrossover,
        compatible_genomes={"embedding"},
    )
    registry.register(
        "crossover",
        "token_two_point",
        lambda: TokenLevelCrossover(crossover_type="two_point"),  # type: ignore[arg-type]
        compatible_genomes={"embedding"},
    )

    # -----------------------------------------
    # Merge operators
    # -----------------------------------------
    from evolve.core.operators.merge import (
        GraphSymbiogeneticMerge,
    )

    registry.register(
        "merge",
        "graph_symbiogenetic",
        GraphSymbiogeneticMerge,
        compatible_genomes={"graph"},
    )

    # -----------------------------------------
    # Distances between two genomes (clearing)
    # -----------------------------------------
    from evolve.diversity.speciation import GenomeDistance, NEATDistance
    from evolve.meta.codec import ParameterDistance

    registry.register(
        "distance",
        "genome",
        GenomeDistance,
        compatible_genomes={"vector", "sequence", "embedding"},
    )
    registry.register(
        "distance",
        "neat",
        NEATDistance,
        compatible_genomes={"graph"},
    )
    registry.register(
        "distance",
        "parameters",
        ParameterDistance,
        compatible_genomes={"vector"},
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
