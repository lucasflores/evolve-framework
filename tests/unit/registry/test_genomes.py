"""
Unit tests for GenomeRegistry.

Tests registration, retrieval, and built-in genome types.
"""

from random import Random

import pytest

from evolve.registry.genomes import (
    GenomeRegistry,
    get_genome_registry,
    reset_genome_registry,
)


class TestGenomeRegistryBasics:
    """Test basic registry operations."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_get_genome_registry_returns_instance(self) -> None:
        """get_genome_registry returns a GenomeRegistry."""
        registry = get_genome_registry()
        assert isinstance(registry, GenomeRegistry)

    def test_get_genome_registry_is_singleton(self) -> None:
        """get_genome_registry returns the same instance."""
        registry1 = get_genome_registry()
        registry2 = get_genome_registry()
        assert registry1 is registry2

    def test_reset_creates_new_instance(self) -> None:
        """reset_genome_registry creates a new instance."""
        registry1 = get_genome_registry()
        reset_genome_registry()
        registry2 = get_genome_registry()
        assert registry1 is not registry2


class TestBuiltInGenomes:
    """Test that built-in genome types are registered."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_vector_genome_registered(self) -> None:
        """Vector genome type is registered."""
        registry = get_genome_registry()
        assert registry.is_registered("vector")

    def test_sequence_genome_registered(self) -> None:
        """Sequence genome type is registered."""
        registry = get_genome_registry()
        assert registry.is_registered("sequence")

    def test_graph_genome_registered(self) -> None:
        """Graph genome type is registered."""
        registry = get_genome_registry()
        assert registry.is_registered("graph")

    def test_scm_genome_registered(self) -> None:
        """SCM genome type is registered."""
        registry = get_genome_registry()
        assert registry.is_registered("scm")

    def test_list_types_includes_all(self) -> None:
        """list_types includes all built-in types."""
        registry = get_genome_registry()
        types = registry.list_types()

        assert "vector" in types
        assert "sequence" in types
        assert "graph" in types
        assert "scm" in types


class TestGenomeCreation:
    """Test creating genome instances."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_create_vector_genome(self) -> None:
        """Can create vector genome."""
        registry = get_genome_registry()
        rng = Random(42)

        genome = registry.create("vector", rng=rng, dimensions=5, bounds=(-1.0, 1.0))

        assert hasattr(genome, "genes")
        assert len(genome.genes) == 5

    def test_create_sequence_genome(self) -> None:
        """Can create sequence genome."""
        registry = get_genome_registry()
        rng = Random(42)

        genome = registry.create("sequence", rng=rng, length=10, alphabet=(0, 1))

        assert hasattr(genome, "genes")
        assert len(genome.genes) == 10

    def test_create_with_defaults(self) -> None:
        """Can create genome using default parameters."""
        registry = get_genome_registry()
        rng = Random(42)

        # Should use defaults (dimensions=10)
        genome = registry.create("vector", rng=rng)

        assert hasattr(genome, "genes")

    def test_create_unknown_type_raises(self) -> None:
        """Creating unknown genome type raises KeyError."""
        registry = get_genome_registry()

        with pytest.raises(KeyError, match="not found"):
            registry.create("nonexistent_type")


class TestGenomeFactories:
    """Test getting genome factory functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_get_factory_returns_callable(self) -> None:
        """get_factory returns a callable."""
        registry = get_genome_registry()
        factory = registry.get_factory("vector")

        assert callable(factory)

    def test_factory_creates_genome(self) -> None:
        """Factory can create genome instances."""
        registry = get_genome_registry()
        factory = registry.get_factory("vector")

        genome = factory(dimensions=3, rng=Random(42))

        assert hasattr(genome, "genes")

    def test_get_factory_unknown_raises(self) -> None:
        """get_factory raises KeyError for unknown type."""
        registry = get_genome_registry()

        with pytest.raises(KeyError):
            registry.get_factory("nonexistent")


class TestCustomGenomeRegistration:
    """Test registering custom genome types."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_register_custom_genome(self) -> None:
        """Can register custom genome factory."""
        registry = get_genome_registry()

        class CustomGenome:
            def __init__(self, size: int):
                self.size = size

        def custom_factory(size: int = 5, **_kwargs) -> CustomGenome:
            return CustomGenome(size)

        registry.register("custom", custom_factory, default_params={"size": 10})

        genome = registry.create("custom")
        assert genome.size == 10

    def test_override_defaults_on_create(self) -> None:
        """Can override defaults when creating."""
        registry = get_genome_registry()

        class CustomGenome:
            def __init__(self, size: int):
                self.size = size

        def custom_factory(size: int = 5, **_kwargs) -> CustomGenome:
            return CustomGenome(size)

        registry.register("custom", custom_factory, default_params={"size": 10})

        genome = registry.create("custom", size=20)
        assert genome.size == 20


class TestDefaultParams:
    """Test default parameter handling."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_get_default_params(self) -> None:
        """Can get default params for genome type."""
        registry = get_genome_registry()
        defaults = registry.get_default_params("vector")

        assert isinstance(defaults, dict)
        assert "dimensions" in defaults or "bounds" in defaults

    def test_get_default_params_unknown(self) -> None:
        """get_default_params returns empty dict for unknown."""
        registry = get_genome_registry()
        defaults = registry.get_default_params("nonexistent")

        assert defaults == {}


class TestIsRegistered:
    """Test registration checking."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_genome_registry()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_genome_registry()

    def test_is_registered_true_for_builtin(self) -> None:
        """is_registered returns True for built-in types."""
        registry = get_genome_registry()
        assert registry.is_registered("vector")

    def test_is_registered_false_for_unknown(self) -> None:
        """is_registered returns False for unknown types."""
        registry = get_genome_registry()
        assert not registry.is_registered("nonexistent")
