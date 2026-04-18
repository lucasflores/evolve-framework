"""Unit tests for CPPNToNetworkDecoder (ES-HyperNEAT)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from evolve.representation.cppn_decoder import CPPNToNetworkDecoder, DecodeStats
from evolve.representation.graph import ConnectionGene, GraphGenome, NodeGene
from evolve.representation.network import NEATNetwork

# ---------------------------------------------------------------------------
# Helper: build a CPPN GraphGenome with predictable behavior
# ---------------------------------------------------------------------------


def _make_cppn_genome(
    n_inputs: int = 4,
    weight: float = 1.0,
    activation: str = "sigmoid",
) -> GraphGenome:
    """
    Create a minimal CPPN genome (no hidden nodes).

    Inputs connect directly to a single output with the given weight.
    """
    nodes = []
    for i in range(n_inputs):
        nodes.append(NodeGene(id=i, node_type="input", activation="identity"))
    output_id = n_inputs
    nodes.append(NodeGene(id=output_id, node_type="output", activation=activation))

    connections = []
    for i in range(n_inputs):
        connections.append(
            ConnectionGene(
                innovation=i,
                from_node=i,
                to_node=output_id,
                weight=weight,
                enabled=True,
            )
        )

    return GraphGenome(
        nodes=frozenset(nodes),
        connections=frozenset(connections),
        input_ids=tuple(range(n_inputs)),
        output_ids=(output_id,),
    )


def _make_uniform_cppn(n_inputs: int = 4) -> GraphGenome:
    """CPPN that produces near-constant output (low variance everywhere)."""
    # Use identity activation with very small weights → output ≈ 0
    return _make_cppn_genome(n_inputs=n_inputs, weight=0.001, activation="identity")


def _make_varying_cppn(n_inputs: int = 4) -> GraphGenome:
    """CPPN that produces spatially varying output (high variance)."""
    # sin activation with large weight → high variance across space
    return _make_cppn_genome(n_inputs=n_inputs, weight=5.0, activation="sin")


# ---------------------------------------------------------------------------
# Init validation tests (FR-015, FR-016, FR-006)
# ---------------------------------------------------------------------------


class TestCPPNToNetworkDecoderInit:
    """Initialization and parameter validation."""

    def test_valid_init(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
        )
        assert decoder.weight_threshold == 0.3
        assert decoder.variance_threshold == 0.03
        assert decoder.max_quadtree_depth == 4
        assert decoder.distance_input is False
        assert decoder.hidden_activation == "sigmoid"
        assert decoder.last_decode_stats is None

    def test_empty_input_positions_raises(self):
        with pytest.raises(ValueError, match="input_positions must not be empty"):
            CPPNToNetworkDecoder(
                input_positions=[],
                output_positions=[(0.0, 1.0)],
            )

    def test_empty_output_positions_raises(self):
        with pytest.raises(ValueError, match="output_positions must not be empty"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[],
            )

    def test_invalid_hidden_activation_raises(self):
        with pytest.raises(ValueError, match="not a registered activation"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[(0.0, 1.0)],
                hidden_activation="nonexistent",
            )

    def test_zero_weight_threshold_raises(self):
        with pytest.raises(ValueError, match="weight_threshold must be > 0"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[(0.0, 1.0)],
                weight_threshold=0.0,
            )

    def test_negative_weight_threshold_raises(self):
        with pytest.raises(ValueError, match="weight_threshold must be > 0"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[(0.0, 1.0)],
                weight_threshold=-0.1,
            )

    def test_zero_variance_threshold_raises(self):
        with pytest.raises(ValueError, match="variance_threshold must be > 0"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[(0.0, 1.0)],
                variance_threshold=0.0,
            )

    def test_zero_max_quadtree_depth_raises(self):
        with pytest.raises(ValueError, match="max_quadtree_depth must be >= 1"):
            CPPNToNetworkDecoder(
                input_positions=[(0.0, -1.0)],
                output_positions=[(0.0, 1.0)],
                max_quadtree_depth=0,
            )


# ---------------------------------------------------------------------------
# Core decode tests (US1)
# ---------------------------------------------------------------------------


class TestDecodeBasic:
    """Core decoding behavior."""

    def test_decode_returns_neat_network(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        network = decoder.decode(genome)
        assert isinstance(network, NEATNetwork)

    def test_decoded_network_is_callable(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        network = decoder.decode(genome)
        output = network(np.array([1.0]))
        assert isinstance(output, np.ndarray)
        assert output.shape == (1,)

    def test_uniform_cppn_no_hidden_neurons(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            variance_threshold=0.5,
            weight_threshold=0.1,
            max_quadtree_depth=3,
        )
        genome = _make_uniform_cppn()
        decoder.decode(genome)

        stats = decoder.last_decode_stats
        assert stats is not None
        assert stats.neurons_discovered == 0

    def test_varying_cppn_discovers_hidden_neurons(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            variance_threshold=0.001,
            weight_threshold=0.01,
            max_quadtree_depth=3,
        )
        genome = _make_varying_cppn()
        decoder.decode(genome)

        stats = decoder.last_decode_stats
        assert stats is not None
        assert stats.neurons_discovered > 0

    def test_deterministic_decoding(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()

        net1 = decoder.decode(genome)
        net2 = decoder.decode(genome)

        inp = np.array([1.0])
        out1 = net1(inp)
        out2 = net2(inp)
        np.testing.assert_array_equal(out1, out2)

    def test_all_weights_below_threshold_no_connections(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=999.0,  # impossibly high
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        decoder.decode(genome)

        stats = decoder.last_decode_stats
        assert stats is not None
        assert stats.connections_after_pruning == 0

    def test_duplicate_positions_handled(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, 0.0), (0.0, 0.0)],
            output_positions=[(1.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        network = decoder.decode(genome)
        assert isinstance(network, NEATNetwork)
        assert len(network.input_ids) == 2

    def test_minimal_cppn_no_hidden_nodes(self):
        """Minimal CPPN (no hidden nodes) still decodes."""
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        assert genome.n_hidden == 0
        network = decoder.decode(genome)
        assert isinstance(network, NEATNetwork)

    def test_multiple_inputs_and_outputs(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(-1.0, -1.0), (1.0, -1.0)],
            output_positions=[(-1.0, 1.0), (1.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        network = decoder.decode(genome)
        output = network(np.array([0.5, 0.5]))
        assert output.shape == (2,)


# ---------------------------------------------------------------------------
# Decode stats / observability (FR-014, T015b)
# ---------------------------------------------------------------------------


class TestDecodeStats:
    """DecodeStats population and observability."""

    def test_decode_stats_populated(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()
        decoder.decode(genome)

        stats = decoder.last_decode_stats
        assert stats is not None
        assert isinstance(stats, DecodeStats)
        assert stats.neurons_discovered >= 0
        assert stats.connections_before_pruning >= 0
        assert stats.neurons_pruned >= 0
        assert stats.connections_after_pruning >= 0
        assert stats.neurons_final >= 2  # at least input + output

    def test_decode_stats_frozen(self):
        stats = DecodeStats(
            neurons_discovered=5,
            connections_before_pruning=10,
            neurons_pruned=2,
            connections_after_pruning=6,
            neurons_final=5,
        )
        with pytest.raises(AttributeError):
            stats.neurons_discovered = 99  # type: ignore[misc]

    def test_decode_emits_log_message(self, caplog):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome()

        with caplog.at_level(logging.INFO, logger="evolve.representation.cppn_decoder"):
            decoder.decode(genome)

        assert any("ES-HyperNEAT decode complete" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Pruning tests (US4, T023)
# ---------------------------------------------------------------------------


class TestPruning:
    """Pruning of disconnected hidden neurons."""

    def test_prune_neuron_no_path_to_output(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        # Test pruning directly
        connections = {(0, 2): 1.0}  # input->hidden, but hidden has no path to output(1)
        pruned_conns, surviving = decoder._prune_disconnected(connections, {0}, {1}, {2})
        assert 2 not in surviving
        assert len(pruned_conns) == 0

    def test_prune_neuron_no_path_from_input(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        connections = {(2, 1): 1.0}  # hidden->output, but no input->hidden
        pruned_conns, surviving = decoder._prune_disconnected(connections, {0}, {1}, {2})
        assert 2 not in surviving

    def test_valid_path_survives(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        connections = {(0, 2): 1.0, (2, 1): 1.0}  # input->hidden->output
        pruned_conns, surviving = decoder._prune_disconnected(connections, {0}, {1}, {2})
        assert 2 in surviving
        assert len(pruned_conns) == 2

    def test_connections_involving_pruned_neurons_removed(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        # hidden 2 connects to output; hidden 3 only connects to hidden 2
        connections = {
            (0, 2): 1.0,  # input->h2
            (2, 1): 1.0,  # h2->output
            (0, 3): 1.0,  # input->h3
            (3, 4): 1.0,  # h3->h4 (dead end)
        }
        pruned_conns, surviving = decoder._prune_disconnected(connections, {0}, {1}, {2, 3, 4})
        assert 2 in surviving
        assert 3 not in surviving
        assert 4 not in surviving
        # Only connections involving surviving nodes remain
        assert (0, 3) not in pruned_conns
        assert (3, 4) not in pruned_conns


# ---------------------------------------------------------------------------
# Distance input tests (US5, T024)
# ---------------------------------------------------------------------------


class TestDistanceInput:
    """Distance input configuration."""

    def test_distance_input_true_5_inputs(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            distance_input=True,
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome(n_inputs=5)
        network = decoder.decode(genome)
        assert isinstance(network, NEATNetwork)

    def test_distance_input_false_4_inputs(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            distance_input=False,
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome(n_inputs=4)
        network = decoder.decode(genome)
        assert isinstance(network, NEATNetwork)

    def test_cppn_input_mismatch_raises(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            distance_input=True,  # expects 5 inputs
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome(n_inputs=4)  # only 4 inputs
        with pytest.raises(ValueError, match="distance_input"):
            decoder.decode(genome)

    def test_distance_input_false_mismatch_raises(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            distance_input=False,  # expects 4 inputs
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        genome = _make_cppn_genome(n_inputs=5)  # 5 inputs
        with pytest.raises(ValueError, match="distance_input"):
            decoder.decode(genome)


# ---------------------------------------------------------------------------
# Scale test (SC-005, T015c)
# ---------------------------------------------------------------------------


class TestScaleDecode:
    """Decode with high quadtree depth produces large networks."""

    def test_high_depth_decode_completes(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(-1.0, -1.0), (1.0, -1.0)],
            output_positions=[(-1.0, 1.0), (1.0, 1.0)],
            variance_threshold=0.01,
            weight_threshold=0.05,
            max_quadtree_depth=4,
        )
        genome = _make_varying_cppn()
        network = decoder.decode(genome)

        assert isinstance(network, NEATNetwork)
        # Should discover many neurons at depth 6
        stats = decoder.last_decode_stats
        assert stats is not None
        # Network should be callable
        output = network(np.array([0.5, 0.5]))
        assert isinstance(output, np.ndarray)
        assert output.shape == (2,)


# ---------------------------------------------------------------------------
# Hidden activation tests
# ---------------------------------------------------------------------------


class TestHiddenActivation:
    """Configurable hidden activation for decoded neurons."""

    def test_custom_hidden_activation(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
            hidden_activation="relu",
            weight_threshold=0.1,
            max_quadtree_depth=2,
        )
        assert decoder.hidden_activation == "relu"

    def test_default_hidden_activation_is_sigmoid(self):
        decoder = CPPNToNetworkDecoder(
            input_positions=[(0.0, -1.0)],
            output_positions=[(0.0, 1.0)],
        )
        assert decoder.hidden_activation == "sigmoid"
