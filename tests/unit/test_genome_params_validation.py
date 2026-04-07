"""Unit tests for genome_params validation via signature introspection."""

from __future__ import annotations

from random import Random

import pytest

from evolve.registry.genomes import get_genome_registry, reset_genome_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton before and after each test."""
    reset_genome_registry()
    yield
    reset_genome_registry()


class TestGenomeParamsValidation:
    """US5: Unrecognized genome_params raise clear validation error."""

    def test_reject_unrecognized_key(self):
        reg = get_genome_registry()
        rng = Random(42)
        with pytest.raises(ValueError, match="Unrecognized genome_params"):
            reg.create("vector", rng=rng, dimensons=10)  # typo: dimensons

    def test_error_lists_unknown_and_accepted(self):
        reg = get_genome_registry()
        rng = Random(42)
        with pytest.raises(ValueError, match="dimensons") as exc_info:
            reg.create("vector", rng=rng, dimensons=10)
        # Should also mention accepted params
        assert "dimensions" in str(exc_info.value) or "Accepted" in str(exc_info.value)

    def test_accept_valid_keys(self):
        reg = get_genome_registry()
        rng = Random(42)
        # Should NOT raise for valid params
        genome = reg.create("vector", rng=rng, dimensions=5, bounds=(-1.0, 1.0))
        assert genome is not None

    def test_skip_validation_for_kwargs_factories(self):
        """Factories with **kwargs skip strict validation."""
        reg = get_genome_registry()

        def flexible_factory(**kwargs):
            return {"params": kwargs}

        reg.register("flexible", flexible_factory)
        # Should NOT raise even with arbitrary params
        result = reg.create("flexible", anything_goes=True, made_up_param=42)
        assert result["params"]["anything_goes"] is True

    def test_injected_rng_excluded_from_validation(self):
        """The 'rng' param is injected, not user-provided — should not appear as 'unrecognized'."""
        reg = get_genome_registry()
        rng = Random(42)
        # 'rng' is accepted by vector factory, should not be flagged
        genome = reg.create("vector", rng=rng, dimensions=5)
        assert genome is not None
