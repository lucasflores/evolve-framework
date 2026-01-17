"""
Decoder implementations for converting genomes to phenotypes.

Provides decoders for graph genomes to neural networks and
other genome-to-phenotype transformations.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

from evolve.representation.graph import GraphGenome, NodeGene, ConnectionGene
from evolve.representation.network import (
    NEATNetwork,
    NumpyNetwork,
    get_activation,
    sigmoid,
)


class GraphToNetworkDecoder:
    """
    Decode graph genome to feedforward network.
    
    Performs topological sort to determine evaluation order
    and builds a NEATNetwork that can be executed.
    
    Example:
        >>> decoder = GraphToNetworkDecoder()
        >>> network = decoder.decode(genome)
        >>> output = network(np.array([1.0, 2.0]))
    """
    
    def decode(self, genome: GraphGenome) -> NEATNetwork:
        """
        Create network from graph structure.
        
        Performs topological sort to determine evaluation order.
        
        Args:
            genome: Graph genome to decode
            
        Returns:
            Executable NEATNetwork
        """
        # Get topological order
        node_order = self._topological_sort(genome)
        
        # Build node properties
        node_biases: dict[int, float] = {}
        node_activations: dict[int, Callable[[np.ndarray], np.ndarray]] = {}
        
        for node in genome.nodes:
            node_biases[node.id] = node.bias
            try:
                node_activations[node.id] = get_activation(node.activation)
            except KeyError:
                node_activations[node.id] = sigmoid
        
        # Build connection dict (only enabled connections)
        connections: dict[tuple[int, int], float] = {}
        for conn in genome.connections:
            if conn.enabled:
                connections[(conn.from_node, conn.to_node)] = conn.weight
        
        return NEATNetwork(
            node_order=node_order,
            node_biases=node_biases,
            node_activations=node_activations,
            connections=connections,
            input_ids=genome.input_ids,
            output_ids=genome.output_ids,
        )
    
    def _topological_sort(self, genome: GraphGenome) -> list[int]:
        """
        Topologically sort nodes for feedforward evaluation.
        
        Uses Kahn's algorithm with special handling for cycles
        (which shouldn't exist in feedforward networks).
        
        Args:
            genome: Graph genome
            
        Returns:
            List of node IDs in topological order
        """
        # Build adjacency list and in-degree count
        all_node_ids = {node.id for node in genome.nodes}
        in_degree: dict[int, int] = {node_id: 0 for node_id in all_node_ids}
        adjacency: dict[int, list[int]] = {node_id: [] for node_id in all_node_ids}
        
        for conn in genome.connections:
            if conn.enabled:
                adjacency[conn.from_node].append(conn.to_node)
                in_degree[conn.to_node] += 1
        
        # Start with nodes that have no incoming edges (inputs)
        queue = deque([
            node_id for node_id in all_node_ids 
            if in_degree[node_id] == 0
        ])
        
        result: list[int] = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we didn't process all nodes, there's a cycle
        if len(result) != len(all_node_ids):
            # Handle cycles by adding remaining nodes at the end
            remaining = [n for n in all_node_ids if n not in result]
            result.extend(remaining)
        
        return result


class GraphToMLPDecoder:
    """
    Decode graph genome to layer-structured MLP.
    
    Attempts to organize the graph into layers for more
    efficient matrix multiplication. Falls back to per-node
    evaluation if the graph doesn't fit layer structure.
    
    This is more efficient than NEATNetwork when the graph
    happens to have a regular layer structure.
    """
    
    def decode(self, genome: GraphGenome) -> NumpyNetwork | NEATNetwork:
        """
        Decode genome, using MLP structure if possible.
        
        Args:
            genome: Graph genome
            
        Returns:
            NumpyNetwork if layer structure detected, else NEATNetwork
        """
        # Try to identify layer structure
        layers = self._identify_layers(genome)
        
        if layers is None:
            # Fall back to per-node network
            return GraphToNetworkDecoder().decode(genome)
        
        # Build layer-structured network
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        activations: list[Callable[[np.ndarray], np.ndarray]] = []
        
        for i in range(len(layers) - 1):
            from_layer = layers[i]
            to_layer = layers[i + 1]
            
            # Build weight matrix
            w = np.zeros((len(from_layer), len(to_layer)))
            b = np.zeros(len(to_layer))
            
            for j, to_id in enumerate(to_layer):
                node = genome.get_node(to_id)
                if node:
                    b[j] = node.bias
                
                for k, from_id in enumerate(from_layer):
                    conn = genome.get_connection_by_nodes(from_id, to_id)
                    if conn and conn.enabled:
                        w[k, j] = conn.weight
            
            weights.append(w)
            biases.append(b)
            
            # Get activation from first node in to_layer
            if to_layer:
                node = genome.get_node(to_layer[0])
                act_name = node.activation if node else "sigmoid"
                try:
                    activations.append(get_activation(act_name))
                except KeyError:
                    activations.append(sigmoid)
        
        return NumpyNetwork(
            weights=weights,
            biases=biases,
            activations=activations,
        )
    
    def _identify_layers(
        self, genome: GraphGenome
    ) -> list[list[int]] | None:
        """
        Try to identify layer structure in genome.
        
        Returns None if genome doesn't have clean layer structure.
        """
        # Start with inputs as layer 0
        layers: list[set[int]] = [set(genome.input_ids)]
        assigned = set(genome.input_ids)
        
        # BFS to assign layers
        while True:
            next_layer: set[int] = set()
            
            for conn in genome.connections:
                if not conn.enabled:
                    continue
                if conn.from_node in assigned and conn.to_node not in assigned:
                    next_layer.add(conn.to_node)
            
            if not next_layer:
                break
            
            layers.append(next_layer)
            assigned.update(next_layer)
        
        # Check if all nodes assigned
        all_nodes = {node.id for node in genome.nodes}
        if assigned != all_nodes:
            return None
        
        # Check if outputs are in last layer
        if not set(genome.output_ids).issubset(layers[-1]):
            return None
        
        # Convert to sorted lists
        return [sorted(layer) for layer in layers]
