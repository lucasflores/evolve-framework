"""
Unit tests for OperatorRegistry.

Tests registration, retrieval, compatibility checking, and built-in operators.
"""

import pytest

from evolve.registry.operators import (
    OperatorRegistry,
    get_operator_registry,
    reset_operator_registry,
)


class TestOperatorRegistryBasics:
    """Test basic registry operations."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_get_operator_registry_returns_instance(self) -> None:
        """get_operator_registry returns an OperatorRegistry."""
        registry = get_operator_registry()
        assert isinstance(registry, OperatorRegistry)

    def test_get_operator_registry_is_singleton(self) -> None:
        """get_operator_registry returns the same instance."""
        registry1 = get_operator_registry()
        registry2 = get_operator_registry()
        assert registry1 is registry2

    def test_reset_creates_new_instance(self) -> None:
        """reset_operator_registry creates a new instance."""
        registry1 = get_operator_registry()
        reset_operator_registry()
        registry2 = get_operator_registry()
        assert registry1 is not registry2


class TestBuiltInOperators:
    """Test that built-in operators are registered."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_selection_operators_registered(self) -> None:
        """All selection operators are registered."""
        registry = get_operator_registry()
        selection_ops = registry.list_operators("selection")

        assert "tournament" in selection_ops
        assert "roulette" in selection_ops
        assert "rank" in selection_ops
        assert "crowded_tournament" in selection_ops

    def test_crossover_operators_registered(self) -> None:
        """All crossover operators are registered."""
        registry = get_operator_registry()
        crossover_ops = registry.list_operators("crossover")

        assert "uniform" in crossover_ops
        assert "single_point" in crossover_ops
        assert "two_point" in crossover_ops
        assert "blend" in crossover_ops
        assert "sbx" in crossover_ops
        assert "neat" in crossover_ops

    def test_mutation_operators_registered(self) -> None:
        """All mutation operators are registered."""
        registry = get_operator_registry()
        mutation_ops = registry.list_operators("mutation")

        assert "gaussian" in mutation_ops
        assert "uniform" in mutation_ops
        assert "polynomial" in mutation_ops
        assert "creep" in mutation_ops
        assert "neat" in mutation_ops


class TestOperatorRetrieval:
    """Test retrieving operators from registry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_get_selection_operator(self) -> None:
        """Can retrieve selection operator instance."""
        registry = get_operator_registry()
        selection = registry.get("selection", "tournament", tournament_size=3)

        assert selection is not None
        assert callable(selection) or hasattr(selection, "select")

    def test_get_crossover_operator(self) -> None:
        """Can retrieve crossover operator instance."""
        registry = get_operator_registry()
        crossover = registry.get("crossover", "uniform")

        assert crossover is not None

    def test_get_mutation_operator(self) -> None:
        """Can retrieve mutation operator instance."""
        registry = get_operator_registry()
        mutation = registry.get("mutation", "gaussian", sigma=0.1)

        assert mutation is not None

    def test_get_unknown_operator_raises(self) -> None:
        """Getting unknown operator raises KeyError."""
        registry = get_operator_registry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("selection", "nonexistent_operator")

    def test_get_invalid_category_raises(self) -> None:
        """Getting from invalid category raises KeyError."""
        registry = get_operator_registry()

        with pytest.raises(KeyError):
            registry.get("invalid_category", "tournament")


class TestOperatorCompatibility:
    """Test genome compatibility checking."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_selection_compatible_with_all(self) -> None:
        """Selection operators are compatible with all genome types."""
        registry = get_operator_registry()

        assert registry.is_compatible("tournament", "vector")
        assert registry.is_compatible("tournament", "sequence")
        assert registry.is_compatible("tournament", "graph")
        assert registry.is_compatible("tournament", "scm")

    def test_gaussian_mutation_compatible_with_vector(self) -> None:
        """Gaussian mutation is compatible with vector genomes."""
        registry = get_operator_registry()
        assert registry.is_compatible("gaussian", "vector")

    def test_gaussian_mutation_incompatible_with_graph(self) -> None:
        """Gaussian mutation is not compatible with graph genomes."""
        registry = get_operator_registry()
        assert not registry.is_compatible("gaussian", "graph")

    def test_neat_mutation_compatible_with_graph(self) -> None:
        """NEAT mutation is compatible with graph genomes."""
        registry = get_operator_registry()
        assert registry.is_compatible("neat", "graph")

    def test_neat_mutation_incompatible_with_vector(self) -> None:
        """NEAT mutation is not compatible with vector genomes."""
        registry = get_operator_registry()
        assert not registry.is_compatible("neat", "vector")

    def test_get_compatibility_returns_set(self) -> None:
        """get_compatibility returns set of compatible genomes."""
        registry = get_operator_registry()
        compat = registry.get_compatibility("gaussian")

        assert isinstance(compat, set)
        assert "vector" in compat


class TestCustomOperatorRegistration:
    """Test registering custom operators."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_register_custom_mutation(self) -> None:
        """Can register custom mutation operator."""
        registry = get_operator_registry()

        class CustomMutation:
            def __init__(self, rate: float = 0.1):
                self.rate = rate

        registry.register(
            "mutation",
            "custom_mutation",
            CustomMutation,
            compatible_genomes={"vector"},
        )

        op = registry.get("mutation", "custom_mutation", rate=0.2)
        assert op.rate == 0.2

    def test_register_invalid_category_raises(self) -> None:
        """Registering to invalid category raises ValueError."""
        registry = get_operator_registry()

        class DummyOp:
            pass

        with pytest.raises(ValueError, match="Invalid category"):
            registry.register("invalid", "dummy", DummyOp)

    def test_is_registered_returns_true_for_registered(self) -> None:
        """is_registered returns True for registered operators."""
        registry = get_operator_registry()
        assert registry.is_registered("selection", "tournament")

    def test_is_registered_returns_false_for_unregistered(self) -> None:
        """is_registered returns False for unregistered operators."""
        registry = get_operator_registry()
        assert not registry.is_registered("selection", "nonexistent")


class TestListOperators:
    """Test listing operators."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_operator_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_operator_registry()

    def test_list_operators_returns_list(self) -> None:
        """list_operators returns a list."""
        registry = get_operator_registry()
        ops = registry.list_operators("selection")
        assert isinstance(ops, list)

    def test_list_all_returns_dict(self) -> None:
        """list_all returns a dict with all categories."""
        registry = get_operator_registry()
        all_ops = registry.list_all()

        assert isinstance(all_ops, dict)
        assert "selection" in all_ops
        assert "crossover" in all_ops
        assert "mutation" in all_ops
