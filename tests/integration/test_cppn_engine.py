"""Integration test: ES-HyperNEAT via UnifiedConfig + create_engine()."""

from __future__ import annotations

import pytest

from evolve.config.unified import UnifiedConfig
from evolve.registry.decoders import get_decoder_registry, reset_decoder_registry
from evolve.representation.cppn_decoder import CPPNToNetworkDecoder


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton before and after each test."""
    reset_decoder_registry()
    yield
    reset_decoder_registry()


class TestCPPNToNetworkRegistry:
    """Registry integration for cppn_to_network decoder."""

    def test_cppn_to_network_in_listing(self):
        reg = get_decoder_registry()
        assert "cppn_to_network" in reg.list_decoders()

    def test_cppn_to_network_is_registered(self):
        reg = get_decoder_registry()
        assert reg.is_registered("cppn_to_network")

    def test_cppn_to_network_resolves(self):
        reg = get_decoder_registry()
        decoder = reg.get(
            "cppn_to_network",
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
        )
        assert isinstance(decoder, CPPNToNetworkDecoder)

    def test_cppn_to_network_passes_params(self):
        reg = get_decoder_registry()
        decoder = reg.get(
            "cppn_to_network",
            input_positions=[(-1.0, -1.0), (1.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.5,
            variance_threshold=0.1,
            max_quadtree_depth=3,
            distance_input=True,
            hidden_activation="relu",
        )
        assert isinstance(decoder, CPPNToNetworkDecoder)
        assert decoder.weight_threshold == 0.5
        assert decoder.variance_threshold == 0.1
        assert decoder.max_quadtree_depth == 3
        assert decoder.distance_input is True
        assert decoder.hidden_activation == "relu"

    def test_missing_required_params_raises(self):
        reg = get_decoder_registry()
        with pytest.raises(TypeError):
            reg.get("cppn_to_network")  # missing input_positions, output_positions


class TestUnifiedConfigDecoder:
    """UnifiedConfig with decoder='cppn_to_network'."""

    def test_config_accepts_cppn_decoder(self):
        config = UnifiedConfig(
            population_size=50,
            max_generations=10,
            selection="tournament",
            crossover="neat",
            mutation="neat",
            genome_type="graph",
            decoder="cppn_to_network",
            decoder_params={
                "input_positions": [(0.0, -1.0)],
                "output_positions": [(0.0, 1.0)],
            },
        )
        assert config.decoder == "cppn_to_network"
        assert config.decoder_params["input_positions"] == [(0.0, -1.0)]
