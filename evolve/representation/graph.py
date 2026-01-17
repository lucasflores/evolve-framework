"""
Graph Genome - NEAT-style topology-evolving genome.

Implements graph-based genomes with node and connection genes
identified by innovation numbers for crossover alignment.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class NodeGene:
    """
    Node in a graph genome.
    
    Attributes:
        id: Unique node identifier
        node_type: One of "input", "output", "hidden"
        activation: Activation function name (e.g., "sigmoid", "tanh", "relu")
        bias: Node bias value
    """
    id: int
    node_type: str  # "input" | "output" | "hidden"
    activation: str = "sigmoid"
    bias: float = 0.0
    
    def with_bias(self, bias: float) -> "NodeGene":
        """Return copy with updated bias."""
        return replace(self, bias=bias)
    
    def with_activation(self, activation: str) -> "NodeGene":
        """Return copy with updated activation."""
        return replace(self, activation=activation)


@dataclass(frozen=True)
class ConnectionGene:
    """
    Connection (edge) in a graph genome.
    
    Attributes:
        innovation: Global innovation number for crossover alignment
        from_node: Source node ID
        to_node: Target node ID
        weight: Connection weight
        enabled: Whether connection is active
    """
    innovation: int
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True
    
    def with_weight(self, weight: float) -> "ConnectionGene":
        """Return copy with updated weight."""
        return replace(self, weight=weight)
    
    def with_enabled(self, enabled: bool) -> "ConnectionGene":
        """Return copy with updated enabled status."""
        return replace(self, enabled=enabled)


@dataclass(frozen=True)
class GraphGenome:
    """
    NEAT-style graph genome for evolving neural network topologies.
    
    Nodes and connections are identified by innovation numbers
    for alignment during crossover. The genome is immutable.
    
    Attributes:
        nodes: Frozenset of node genes
        connections: Frozenset of connection genes
        input_ids: Tuple of input node IDs
        output_ids: Tuple of output node IDs
    
    Example:
        >>> # Create minimal genome with 2 inputs, 1 output
        >>> nodes = frozenset([
        ...     NodeGene(0, "input"), NodeGene(1, "input"),
        ...     NodeGene(2, "output")
        ... ])
        >>> connections = frozenset([
        ...     ConnectionGene(1, 0, 2, 0.5),
        ...     ConnectionGene(2, 1, 2, -0.3)
        ... ])
        >>> genome = GraphGenome(nodes, connections, (0, 1), (2,))
    """
    
    nodes: frozenset[NodeGene]
    connections: frozenset[ConnectionGene]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    
    def copy(self) -> "GraphGenome":
        """Create a copy of this genome."""
        return GraphGenome(
            nodes=self.nodes,
            connections=self.connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
        )
    
    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        if not isinstance(other, GraphGenome):
            return False
        return (
            self.nodes == other.nodes
            and self.connections == other.connections
            and self.input_ids == other.input_ids
            and self.output_ids == other.output_ids
        )
    
    def __hash__(self) -> int:
        """Hash for set/dict membership."""
        return hash((self.nodes, self.connections, self.input_ids, self.output_ids))
    
    @property
    def n_inputs(self) -> int:
        """Number of input nodes."""
        return len(self.input_ids)
    
    @property
    def n_outputs(self) -> int:
        """Number of output nodes."""
        return len(self.output_ids)
    
    @property
    def n_hidden(self) -> int:
        """Number of hidden nodes."""
        return len(self.nodes) - self.n_inputs - self.n_outputs
    
    @property
    def n_connections(self) -> int:
        """Total number of connections."""
        return len(self.connections)
    
    @property
    def n_enabled_connections(self) -> int:
        """Number of enabled connections."""
        return sum(1 for c in self.connections if c.enabled)
    
    def get_node(self, node_id: int) -> NodeGene | None:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_connection(self, innovation: int) -> ConnectionGene | None:
        """Get connection by innovation number."""
        for conn in self.connections:
            if conn.innovation == innovation:
                return conn
        return None
    
    def get_connection_by_nodes(
        self, from_node: int, to_node: int
    ) -> ConnectionGene | None:
        """Get connection by source and target nodes."""
        for conn in self.connections:
            if conn.from_node == from_node and conn.to_node == to_node:
                return conn
        return None
    
    def enabled_connections(self) -> frozenset[ConnectionGene]:
        """Get only enabled connections."""
        return frozenset(c for c in self.connections if c.enabled)
    
    def hidden_nodes(self) -> frozenset[NodeGene]:
        """Get only hidden nodes."""
        return frozenset(
            n for n in self.nodes 
            if n.node_type == "hidden"
        )
    
    def input_nodes(self) -> frozenset[NodeGene]:
        """Get input nodes."""
        return frozenset(
            n for n in self.nodes 
            if n.node_type == "input"
        )
    
    def output_nodes(self) -> frozenset[NodeGene]:
        """Get output nodes."""
        return frozenset(
            n for n in self.nodes 
            if n.node_type == "output"
        )
    
    def add_node(
        self,
        node: NodeGene,
        split_connection: ConnectionGene,
        new_conn1: ConnectionGene,
        new_conn2: ConnectionGene,
    ) -> "GraphGenome":
        """
        Add node by splitting a connection (NEAT add-node mutation).
        
        The split connection is disabled, and two new connections
        are added: one from the source to the new node, one from
        the new node to the target.
        
        Args:
            node: New hidden node to add
            split_connection: Connection to split
            new_conn1: Connection from source to new node
            new_conn2: Connection from new node to target
            
        Returns:
            New genome with node added
        """
        # Add new node
        new_nodes = self.nodes | {node}
        
        # Disable split connection, add new connections
        disabled_conn = split_connection.with_enabled(False)
        new_connections = (
            self.connections - {split_connection}
            | {disabled_conn, new_conn1, new_conn2}
        )
        
        return GraphGenome(
            nodes=new_nodes,
            connections=new_connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
        )
    
    def add_connection(self, connection: ConnectionGene) -> "GraphGenome":
        """
        Add a new connection to the genome.
        
        Args:
            connection: Connection to add
            
        Returns:
            New genome with connection added
        """
        return GraphGenome(
            nodes=self.nodes,
            connections=self.connections | {connection},
            input_ids=self.input_ids,
            output_ids=self.output_ids,
        )
    
    def with_connection_weight(
        self, innovation: int, weight: float
    ) -> "GraphGenome":
        """Return genome with updated connection weight."""
        old_conn = self.get_connection(innovation)
        if old_conn is None:
            return self
        
        new_conn = old_conn.with_weight(weight)
        new_connections = (self.connections - {old_conn}) | {new_conn}
        
        return GraphGenome(
            nodes=self.nodes,
            connections=new_connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
        )
    
    def with_node_bias(self, node_id: int, bias: float) -> "GraphGenome":
        """Return genome with updated node bias."""
        old_node = self.get_node(node_id)
        if old_node is None:
            return self
        
        new_node = old_node.with_bias(bias)
        new_nodes = (self.nodes - {old_node}) | {new_node}
        
        return GraphGenome(
            nodes=new_nodes,
            connections=self.connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type,
                    "activation": n.activation,
                    "bias": n.bias,
                }
                for n in sorted(self.nodes, key=lambda n: n.id)
            ],
            "connections": [
                {
                    "innovation": c.innovation,
                    "from": c.from_node,
                    "to": c.to_node,
                    "weight": c.weight,
                    "enabled": c.enabled,
                }
                for c in sorted(self.connections, key=lambda c: c.innovation)
            ],
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphGenome":
        """Reconstruct from dict."""
        nodes = frozenset(
            NodeGene(
                id=n["id"],
                node_type=n["type"],
                activation=n.get("activation", "sigmoid"),
                bias=n.get("bias", 0.0),
            )
            for n in data["nodes"]
        )
        connections = frozenset(
            ConnectionGene(
                innovation=c["innovation"],
                from_node=c["from"],
                to_node=c["to"],
                weight=c["weight"],
                enabled=c.get("enabled", True),
            )
            for c in data["connections"]
        )
        return cls(
            nodes=nodes,
            connections=connections,
            input_ids=tuple(data["input_ids"]),
            output_ids=tuple(data["output_ids"]),
        )
    
    @classmethod
    def minimal(
        cls,
        n_inputs: int,
        n_outputs: int,
        innovation_tracker: "InnovationTracker",
    ) -> "GraphGenome":
        """
        Create minimal fully-connected genome.
        
        Creates input and output nodes with connections from
        every input to every output (no hidden nodes).
        
        Args:
            n_inputs: Number of input nodes
            n_outputs: Number of output nodes
            innovation_tracker: Tracker for innovation numbers
            
        Returns:
            Minimal genome
        """
        nodes: set[NodeGene] = set()
        connections: set[ConnectionGene] = set()
        
        # Create input nodes (IDs 0 to n_inputs-1)
        input_ids = tuple(range(n_inputs))
        for i in input_ids:
            nodes.add(NodeGene(id=i, node_type="input", activation="identity"))
        
        # Create output nodes (IDs n_inputs to n_inputs+n_outputs-1)
        output_ids = tuple(range(n_inputs, n_inputs + n_outputs))
        for i in output_ids:
            nodes.add(NodeGene(id=i, node_type="output", activation="sigmoid"))
        
        # Connect all inputs to all outputs
        for in_id in input_ids:
            for out_id in output_ids:
                innov = innovation_tracker.get_innovation(in_id, out_id)
                connections.add(ConnectionGene(
                    innovation=innov,
                    from_node=in_id,
                    to_node=out_id,
                    weight=0.0,  # Will be randomized by mutation
                    enabled=True,
                ))
        
        return cls(
            nodes=frozenset(nodes),
            connections=frozenset(connections),
            input_ids=input_ids,
            output_ids=output_ids,
        )


class InnovationTracker:
    """
    Tracks global innovation numbers for NEAT crossover alignment.
    
    Innovation numbers ensure that matching genes in different
    genomes can be identified for proper crossover alignment.
    
    Attributes:
        _next_innovation: Next available innovation number
        _next_node_id: Next available node ID
        _connection_innovations: Cache of (from, to) -> innovation
    
    Example:
        >>> tracker = InnovationTracker()
        >>> innov1 = tracker.get_innovation(0, 2)  # New connection 0->2
        >>> innov2 = tracker.get_innovation(1, 2)  # New connection 1->2
        >>> innov3 = tracker.get_innovation(0, 2)  # Same as innov1 (cached)
        >>> assert innov1 == innov3
    """
    
    def __init__(
        self,
        start_innovation: int = 0,
        start_node_id: int = 0,
    ):
        """
        Initialize tracker.
        
        Args:
            start_innovation: Starting innovation number
            start_node_id: Starting node ID
        """
        self._next_innovation = start_innovation
        self._next_node_id = start_node_id
        self._connection_innovations: dict[tuple[int, int], int] = {}
    
    def get_innovation(self, from_node: int, to_node: int) -> int:
        """
        Get innovation number for a connection.
        
        Returns existing innovation if this connection was seen before,
        otherwise assigns a new innovation number.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            
        Returns:
            Innovation number for this connection
        """
        key = (from_node, to_node)
        if key not in self._connection_innovations:
            self._connection_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._connection_innovations[key]
    
    def get_new_node_id(self) -> int:
        """
        Get a new unique node ID.
        
        Returns:
            New node ID
        """
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id
    
    def reserve_node_ids(self, count: int) -> None:
        """
        Reserve node IDs (e.g., for input/output nodes).
        
        Args:
            count: Number of IDs to reserve
        """
        self._next_node_id = max(self._next_node_id, count)
    
    @property
    def current_innovation(self) -> int:
        """Current (next available) innovation number."""
        return self._next_innovation
    
    @property
    def current_node_id(self) -> int:
        """Current (next available) node ID."""
        return self._next_node_id
    
    def reset_generation(self) -> None:
        """
        Reset per-generation innovation cache.
        
        In NEAT, innovations are tracked per generation to allow
        same structural mutations in a generation to share innovations.
        Call this at the start of each generation.
        """
        self._connection_innovations.clear()
