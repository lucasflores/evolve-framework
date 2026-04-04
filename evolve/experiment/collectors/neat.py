"""
NEAT Metric Collector.

Collects metrics for neuroevolution (NEAT) experiments:
- average_node_count: Mean number of nodes across genomes
- average_connection_count: Mean number of connections
- topology_innovations: Count of new structural innovations

Implements FR-017 from the tracking specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.experiment.collectors.base import CollectionContext

if TYPE_CHECKING:
    pass


_logger = logging.getLogger(__name__)


@dataclass
class NEATMetricCollector:
    """
    Collector for NEAT neuroevolution metrics.

    Tracks network topology statistics for evolving neural networks.
    Works with graph-based genomes that have nodes and connections.

    Attributes:
        track_innovations: Whether to track innovation counts.
        node_attr: Attribute name for nodes on genome (default: "nodes").
        connection_attr: Attribute name for connections (default: "connections").

    Example:
        >>> from evolve.experiment.collectors.neat import NEATMetricCollector
        >>>
        >>> collector = NEATMetricCollector()
        >>> context = CollectionContext(generation=10, population=population)
        >>> metrics = collector.collect(context)
        >>> metrics.get("average_node_count")
        12.5
    """

    track_innovations: bool = True
    node_attr: str = "nodes"
    connection_attr: str = "connections"

    # Track seen innovations across generations
    _seen_innovations: set[Any] = field(default_factory=set)
    _new_innovations_this_gen: int = 0

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._seen_innovations = set()
        self._new_innovations_this_gen = 0

    def collect(self, context: CollectionContext) -> dict[str, Any]:
        """
        Collect NEAT metrics from context.

        Args:
            context: Collection context with population.

        Returns:
            Dictionary of NEAT topology metrics.
        """
        metrics: dict[str, Any] = {}

        node_counts = []
        connection_counts = []
        new_innovations = 0

        for ind in context.population.individuals:
            if ind.genome is None:
                continue

            # Get node count
            nodes = self._get_nodes(ind.genome)
            if nodes is not None:
                node_counts.append(len(nodes))

            # Get connection count
            connections = self._get_connections(ind.genome)
            if connections is not None:
                connection_counts.append(len(connections))

                # Track innovations
                if self.track_innovations:
                    new_innovations += self._count_new_innovations(connections)

        if node_counts:
            metrics["average_node_count"] = float(np.mean(node_counts))
            metrics["min_node_count"] = int(np.min(node_counts))
            metrics["max_node_count"] = int(np.max(node_counts))

        if connection_counts:
            metrics["average_connection_count"] = float(np.mean(connection_counts))
            metrics["min_connection_count"] = int(np.min(connection_counts))
            metrics["max_connection_count"] = int(np.max(connection_counts))

        if self.track_innovations:
            metrics["topology_innovations"] = new_innovations
            metrics["total_innovations"] = len(self._seen_innovations)

        return metrics

    def reset(self) -> None:
        """Reset internal state between runs."""
        self._seen_innovations = set()
        self._new_innovations_this_gen = 0

    def _get_nodes(self, genome: Any) -> list[Any] | None:
        """
        Get nodes from genome.

        Args:
            genome: The genome object.

        Returns:
            List of nodes, or None if not available.
        """
        # Try standard attribute
        if hasattr(genome, self.node_attr):
            nodes = getattr(genome, self.node_attr)
            if hasattr(nodes, "__len__"):
                return list(nodes) if not isinstance(nodes, list) else nodes
            return None

        # Try GraphGenome structure
        if hasattr(genome, "graph"):
            graph = genome.graph
            if hasattr(graph, "nodes"):
                return list(graph.nodes())

        # Try node_genes (NEAT-style)
        if hasattr(genome, "node_genes"):
            return list(genome.node_genes)

        return None

    def _get_connections(self, genome: Any) -> list[Any] | None:
        """
        Get connections/edges from genome.

        Args:
            genome: The genome object.

        Returns:
            List of connections, or None if not available.
        """
        # Try standard attribute
        if hasattr(genome, self.connection_attr):
            conns = getattr(genome, self.connection_attr)
            if hasattr(conns, "__len__"):
                return list(conns) if not isinstance(conns, list) else conns
            return None

        # Try 'edges' attribute
        if hasattr(genome, "edges"):
            edges = genome.edges
            if hasattr(edges, "__len__"):
                return list(edges) if not isinstance(edges, list) else edges

        # Try GraphGenome structure
        if hasattr(genome, "graph"):
            graph = genome.graph
            if hasattr(graph, "edges"):
                return list(graph.edges())

        # Try connection_genes (NEAT-style)
        if hasattr(genome, "connection_genes"):
            return list(genome.connection_genes)

        return None

    def _count_new_innovations(self, connections: list[Any]) -> int:
        """
        Count new innovations in connections.

        An innovation is a unique structural change (new connection or node).

        Args:
            connections: List of connections.

        Returns:
            Count of new innovations.
        """
        new_count = 0

        for conn in connections:
            # Try to get innovation number
            innovation_id = self._get_innovation_id(conn)
            if innovation_id is not None and innovation_id not in self._seen_innovations:
                self._seen_innovations.add(innovation_id)
                new_count += 1

        return new_count

    def _get_innovation_id(self, connection: Any) -> Any:
        """
        Extract innovation ID from a connection.

        Args:
            connection: A connection/edge object.

        Returns:
            Innovation ID or hashable identifier, or None.
        """
        # Try innovation attribute
        if hasattr(connection, "innovation"):
            return connection.innovation

        # Try innovation_number
        if hasattr(connection, "innovation_number"):
            return connection.innovation_number

        # Try id attribute
        if hasattr(connection, "id"):
            return connection.id

        # Try tuple (source, target) as identifier
        if isinstance(connection, tuple) and len(connection) >= 2:
            return connection[:2]

        # Try getting from graph edge
        if hasattr(connection, "source") and hasattr(connection, "target"):
            return (connection.source, connection.target)

        return None
