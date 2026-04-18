"""
ES-HyperNEAT Decoder — CPPN-to-Network indirect encoding.

Decodes a CPPN (Compositional Pattern-Producing Network) represented
as a GraphGenome into a substrate neural network (NEATNetwork) with
automatically discovered topology.

The decoding pipeline has three phases:
1. Build an executable CPPN from the GraphGenome
2. Discover hidden neuron positions via quadtree decomposition
3. Query CPPN for connection weights, threshold, and prune

Registry name: ``"cppn_to_network"``

Declarative usage::

    config = UnifiedConfig(
        decoder="cppn_to_network",
        decoder_params={
            "input_positions": [(0.0, -1.0)],
            "output_positions": [(0.0, 1.0)],
        },
    )

References:
    Stanley, D'Ambrosio, Gauci 2009 — A Hypercube-Based Encoding
    Risi & Stanley 2012 — An Enhanced Hypercube-Based Encoding

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from evolve.representation.decoder import GraphToNetworkDecoder
from evolve.representation.graph import GraphGenome
from evolve.representation.network import NEATNetwork, get_activation, sigmoid

__all__ = ["CPPNToNetworkDecoder", "DecodeStats"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodeStats:
    """Statistics from a single ES-HyperNEAT decode operation."""

    neurons_discovered: int
    connections_before_pruning: int
    neurons_pruned: int
    connections_after_pruning: int
    neurons_final: int


@dataclass
class QuadTreeNode:
    """Internal quadtree node for spatial decomposition."""

    x: float
    y: float
    half_size: float
    depth: int
    children: list[QuadTreeNode] | None = None


class CPPNToNetworkDecoder:
    """
    ES-HyperNEAT decoder: CPPN GraphGenome -> NEATNetwork.

    Discovers hidden neuron positions via quadtree decomposition of
    CPPN output variance, queries pairwise connection weights, prunes
    disconnected neurons, and returns a callable NEATNetwork.

    Registry name: "cppn_to_network"
    """

    def __init__(
        self,
        input_positions: list[tuple[float, float]],
        output_positions: list[tuple[float, float]],
        weight_threshold: float = 0.3,
        variance_threshold: float = 0.03,
        max_quadtree_depth: int = 4,
        distance_input: bool = False,
        hidden_activation: str = "sigmoid",
    ) -> None:
        """
        Initialize the decoder.

        Args:
            input_positions: 2D coordinates for input neurons.
            output_positions: 2D coordinates for output neurons.
            weight_threshold: Minimum |CPPN output| to create a connection.
            variance_threshold: Minimum variance for quadtree subdivision.
            max_quadtree_depth: Maximum quadtree recursion depth.
            distance_input: If True, CPPN receives 5 inputs (x1,y1,x2,y2,d).
            hidden_activation: Activation function name for hidden neurons.

        Raises:
            ValueError: If input_positions or output_positions is empty.
            ValueError: If hidden_activation is not a registered activation.
            ValueError: If weight_threshold <= 0, variance_threshold <= 0,
                or max_quadtree_depth < 1.
        """
        if not input_positions:
            raise ValueError("input_positions must not be empty")
        if not output_positions:
            raise ValueError("output_positions must not be empty")
        if weight_threshold <= 0:
            raise ValueError(f"weight_threshold must be > 0, got {weight_threshold}")
        if variance_threshold <= 0:
            raise ValueError(f"variance_threshold must be > 0, got {variance_threshold}")
        if max_quadtree_depth < 1:
            raise ValueError(f"max_quadtree_depth must be >= 1, got {max_quadtree_depth}")
        try:
            self._hidden_activation_fn = get_activation(hidden_activation)
        except KeyError:
            raise ValueError(
                f"hidden_activation '{hidden_activation}' is not a registered activation function"
            ) from None

        self.input_positions = list(input_positions)
        self.output_positions = list(output_positions)
        self.weight_threshold = weight_threshold
        self.variance_threshold = variance_threshold
        self.max_quadtree_depth = max_quadtree_depth
        self.distance_input = distance_input
        self.hidden_activation = hidden_activation
        self._last_decode_stats: DecodeStats | None = None
        self._graph_decoder = GraphToNetworkDecoder()

    @property
    def last_decode_stats(self) -> DecodeStats | None:
        """Statistics from the most recent decode() call."""
        return self._last_decode_stats

    def decode(self, genome: GraphGenome) -> NEATNetwork:
        """
        Decode a CPPN GraphGenome into a callable NEATNetwork.

        Three-phase pipeline:
        1. Build executable CPPN from GraphGenome
        2. Discover hidden neuron positions via quadtree
        3. Query connections, threshold, prune, build network

        Args:
            genome: CPPN represented as a GraphGenome.

        Returns:
            Callable NEATNetwork with discovered topology.

        Raises:
            ValueError: If CPPN input count doesn't match expected.
        """
        # Phase 1: Build executable CPPN
        cppn = self._build_cppn(genome)

        # Phase 2: Discover hidden neuron positions
        hidden_positions = self._discover_hidden_neurons(cppn)

        # Assign neuron IDs
        n_inputs = len(self.input_positions)
        n_outputs = len(self.output_positions)

        all_positions: list[tuple[float, float]] = []
        all_positions.extend(self.input_positions)
        all_positions.extend(self.output_positions)
        all_positions.extend(hidden_positions)

        input_ids = tuple(range(n_inputs))
        output_ids = tuple(range(n_inputs, n_inputs + n_outputs))
        hidden_ids = list(range(n_inputs + n_outputs, len(all_positions)))

        # Phase 3: Query connections and threshold
        connections = self._query_connections(
            cppn, all_positions, input_ids, output_ids, hidden_ids
        )

        connections_before = len(connections)

        # Prune disconnected hidden neurons
        connections, surviving_hidden = self._prune_disconnected(
            connections, set(input_ids), set(output_ids), set(hidden_ids)
        )

        neurons_pruned = len(hidden_ids) - len(surviving_hidden)

        # Build the final NEATNetwork
        all_node_ids = list(input_ids) + list(output_ids) + sorted(surviving_hidden)

        node_biases: dict[int, float] = dict.fromkeys(all_node_ids, 0.0)
        node_activations: dict[int, Any] = {}
        for nid in input_ids:
            node_activations[nid] = get_activation("identity")
        for nid in output_ids:
            node_activations[nid] = sigmoid
        for nid in surviving_hidden:
            node_activations[nid] = self._hidden_activation_fn

        # Topological sort
        node_order = self._topological_sort(all_node_ids, connections)

        network = NEATNetwork(
            node_order=node_order,
            node_biases=node_biases,
            node_activations=node_activations,
            connections=connections,
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
        )

        # Build stats
        stats = DecodeStats(
            neurons_discovered=len(hidden_positions),
            connections_before_pruning=connections_before,
            neurons_pruned=neurons_pruned,
            connections_after_pruning=len(connections),
            neurons_final=len(all_node_ids),
        )
        self._last_decode_stats = stats

        logger.info(
            "ES-HyperNEAT decode complete: "
            "neurons_discovered=%d, connections_before=%d, "
            "neurons_pruned=%d, connections_after=%d, neurons_final=%d",
            stats.neurons_discovered,
            stats.connections_before_pruning,
            stats.neurons_pruned,
            stats.connections_after_pruning,
            stats.neurons_final,
        )

        return network

    def _build_cppn(self, genome: GraphGenome) -> NEATNetwork:
        """Build an executable CPPN from the GraphGenome."""
        expected_inputs = 5 if self.distance_input else 4
        if genome.n_inputs != expected_inputs:
            raise ValueError(
                f"CPPN genome has {genome.n_inputs} inputs but decoder expects "
                f"{expected_inputs} (distance_input={self.distance_input}). "
                f"Ensure the CPPN genome matches the decoder's distance_input setting."
            )
        return self._graph_decoder.decode(genome)

    def _discover_hidden_neurons(self, cppn: NEATNetwork) -> list[tuple[float, float]]:
        """
        Discover hidden neuron positions via quadtree decomposition.

        Subdivides [-1,1]x[-1,1] where CPPN output variance is high.
        """
        root = QuadTreeNode(x=0.0, y=0.0, half_size=1.0, depth=0)
        positions: list[tuple[float, float]] = []
        self._subdivide(root, cppn, positions)
        return positions

    def _subdivide(
        self,
        node: QuadTreeNode,
        cppn: NEATNetwork,
        positions: list[tuple[float, float]],
    ) -> None:
        """Recursively subdivide quadtree based on CPPN output variance."""
        # Sample CPPN at four quadrant centers
        hs = node.half_size / 2.0
        sample_points = [
            (node.x - hs, node.y - hs),
            (node.x + hs, node.y - hs),
            (node.x - hs, node.y + hs),
            (node.x + hs, node.y + hs),
        ]

        outputs = []
        for sx, sy in sample_points:
            # Query CPPN with (x1, y1, x2=0, y2=0) to get output at this point
            # For neuron discovery, we query the CPPN's response at each point
            # using (0, 0) as a reference target
            if self.distance_input:
                d = np.sqrt(sx * sx + sy * sy)
                inp = np.array([sx, sy, 0.0, 0.0, d])
            else:
                inp = np.array([sx, sy, 0.0, 0.0])
            out = cppn(inp)
            outputs.append(float(out[0]))

        variance = float(np.var(outputs))

        if variance > self.variance_threshold and node.depth < self.max_quadtree_depth:
            # Subdivide into four children
            node.children = []
            for sx, sy in sample_points:
                child = QuadTreeNode(x=sx, y=sy, half_size=hs, depth=node.depth + 1)
                node.children.append(child)
                self._subdivide(child, cppn, positions)
        elif variance > self.variance_threshold:
            # At max depth — place neurons at the quadrant centers
            for sx, sy in sample_points:
                positions.append((sx, sy))
        # If variance <= threshold, don't place neurons here

    def _query_connections(
        self,
        cppn: NEATNetwork,
        all_positions: list[tuple[float, float]],
        input_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        hidden_ids: list[int],
    ) -> dict[tuple[int, int], float]:
        """Query CPPN for all pairwise connection weights."""
        connections: dict[tuple[int, int], float] = {}
        non_input_ids = list(output_ids) + hidden_ids

        for src_id in list(input_ids) + hidden_ids:
            sx, sy = all_positions[src_id]
            for tgt_id in non_input_ids:
                if src_id == tgt_id:
                    continue
                tx, ty = all_positions[tgt_id]

                if self.distance_input:
                    d = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
                    inp = np.array([sx, sy, tx, ty, d])
                else:
                    inp = np.array([sx, sy, tx, ty])

                out = cppn(inp)
                weight = float(out[0])

                if abs(weight) > self.weight_threshold:
                    connections[(src_id, tgt_id)] = weight

        return connections

    def _prune_disconnected(
        self,
        connections: dict[tuple[int, int], float],
        input_ids: set[int],
        output_ids: set[int],
        hidden_ids: set[int],
    ) -> tuple[dict[tuple[int, int], float], set[int]]:
        """
        Prune hidden neurons with no input-to-output path.

        Uses forward BFS from inputs and backward BFS from outputs.
        Only hidden neurons in the intersection survive.
        """
        # Build adjacency lists
        forward_adj: dict[int, set[int]] = {}
        backward_adj: dict[int, set[int]] = {}
        for src, tgt in connections:
            forward_adj.setdefault(src, set()).add(tgt)
            backward_adj.setdefault(tgt, set()).add(src)

        # Forward BFS from inputs
        forward_reachable: set[int] = set()
        queue: deque[int] = deque(input_ids)
        while queue:
            node = queue.popleft()
            if node in forward_reachable:
                continue
            forward_reachable.add(node)
            for neighbor in forward_adj.get(node, set()):
                if neighbor not in forward_reachable:
                    queue.append(neighbor)

        # Backward BFS from outputs
        backward_reachable: set[int] = set()
        queue = deque(output_ids)
        while queue:
            node = queue.popleft()
            if node in backward_reachable:
                continue
            backward_reachable.add(node)
            for neighbor in backward_adj.get(node, set()):
                if neighbor not in backward_reachable:
                    queue.append(neighbor)

        # Surviving hidden = intersection of both
        surviving = hidden_ids & forward_reachable & backward_reachable

        # Filter connections to only involve surviving nodes
        valid_nodes = input_ids | output_ids | surviving
        pruned_connections = {
            (src, tgt): w
            for (src, tgt), w in connections.items()
            if src in valid_nodes and tgt in valid_nodes
        }

        return pruned_connections, surviving

    def _topological_sort(
        self,
        all_node_ids: list[int],
        connections: dict[tuple[int, int], float],
    ) -> list[int]:
        """Topologically sort nodes for feedforward evaluation."""
        node_set = set(all_node_ids)
        in_degree: dict[int, int] = dict.fromkeys(node_set, 0)
        adjacency: dict[int, list[int]] = {nid: [] for nid in node_set}

        for src, tgt in connections:
            adjacency[src].append(tgt)
            in_degree[tgt] += 1

        queue: deque[int] = deque(nid for nid in node_set if in_degree[nid] == 0)
        result: list[int] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle any remaining nodes (shouldn't happen in feedforward)
        if len(result) != len(node_set):
            remaining = [n for n in node_set if n not in set(result)]
            result.extend(remaining)

        return result
