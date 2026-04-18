"""
Symbiogenetic merge operators — permanently absorb a symbiont into a host.

Provides the ``SymbiogeneticMerge`` protocol and the
``GraphSymbiogeneticMerge`` implementation for variable-topology
GraphGenome.  Fixed-length representations (vector, sequence,
embedding) are not supported because concatenation changes
dimensionality, which is incompatible with fixed-length operators.

Registry category: ``"merge"``

Declarative usage::

    config = UnifiedConfig(
        merge=MergeConfig(merge_rate=0.1),
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

G = TypeVar("G")


@runtime_checkable
class SymbiogeneticMerge(Protocol[G]):
    """
    Protocol for symbiogenetic merge operators.

    Absorbs a symbiont genome permanently into a host genome,
    producing a single merged offspring.

    Implementations must raise ``ValueError`` when ``host`` and
    ``symbiont`` are the same genome instance.
    """

    def merge(self, host: G, symbiont: G, rng: Random) -> G:
        """
        Merge symbiont into host, producing a new genome.

        Args:
            host: The host genome that absorbs the symbiont.
            symbiont: The symbiont genome absorbed into the host.
            rng: Random number generator for stochastic decisions.

        Returns:
            A new merged genome.

        Raises:
            ValueError: If host and symbiont are the same instance.
        """
        ...


@dataclass
class GraphSymbiogeneticMerge(Generic[G]):
    """
    NEAT-style graph genome merge operator.

    Absorbs the symbiont's hidden-layer topology into the host,
    remapping node IDs and adding interface connections.

    Attributes:
        interface_count: Number of interface connections to add.
        interface_ratio: Fraction of interface connections that are host→symbiont direction.
        weight_method: How to initialise interface connection weights.
        weight_mean: Mean for Gaussian weight init (``weight_method="random"``).
        weight_std: Std dev for Gaussian weight init (``weight_method="random"``).
    """

    interface_count: int = 4
    interface_ratio: float = 0.5
    weight_method: str = "mean"
    weight_mean: float = 0.0
    weight_std: float = 1.0

    def merge(self, host: G, symbiont: G, rng: Random) -> G:
        """Merge symbiont hidden topology into host graph genome."""
        if host is symbiont:
            raise ValueError("host and symbiont must be different genome instances")

        from evolve.representation.graph import (
            ConnectionGene,
            GraphGenome,
            NodeGene,
        )

        if not isinstance(host, GraphGenome) or not isinstance(symbiont, GraphGenome):
            raise TypeError("GraphSymbiogeneticMerge requires GraphGenome instances")

        # 1. Collect symbiont hidden nodes
        symbiont_hidden = [n for n in symbiont.nodes if n.node_type == "hidden"]

        # 2. Remap symbiont node IDs to avoid collisions with host
        host_node_ids = {n.id for n in host.nodes}
        max_host_id = max(host_node_ids) if host_node_ids else 0
        remap: dict[int, int] = {}

        next_id = max_host_id + 1
        for node in symbiont_hidden:
            remap[node.id] = next_id
            next_id += 1

        # Also map symbiont input/output IDs for connection remapping
        for node in symbiont.nodes:
            if node.node_type != "hidden" and node.id not in remap:
                # Map symbiont I/O to matching host I/O by position
                if node.node_type == "input":
                    idx = sorted(n.id for n in symbiont.nodes if n.node_type == "input").index(
                        node.id
                    )
                    host_inputs = sorted(n.id for n in host.nodes if n.node_type == "input")
                    if idx < len(host_inputs):
                        remap[node.id] = host_inputs[idx]
                elif node.node_type == "output":
                    idx = sorted(n.id for n in symbiont.nodes if n.node_type == "output").index(
                        node.id
                    )
                    host_outputs = sorted(n.id for n in host.nodes if n.node_type == "output")
                    if idx < len(host_outputs):
                        remap[node.id] = host_outputs[idx]

        # 3. Create remapped hidden nodes
        new_hidden_nodes: set[NodeGene] = set()
        for node in symbiont_hidden:
            new_hidden_nodes.add(
                NodeGene(
                    id=remap[node.id],
                    node_type="hidden",
                    activation=node.activation,
                    bias=node.bias,
                )
            )

        # 4. Remap symbiont internal connections (between hidden nodes or hidden↔I/O)
        new_connections: set[ConnectionGene] = set()
        max_innov = 0
        for c in host.connections:
            if c.innovation > max_innov:
                max_innov = c.innovation

        for conn in symbiont.connections:
            from_id = remap.get(conn.from_node)
            to_id = remap.get(conn.to_node)
            if from_id is None or to_id is None:
                continue
            # Only keep connections that involve at least one remapped hidden node
            from_is_hidden = conn.from_node in {n.id for n in symbiont_hidden}
            to_is_hidden = conn.to_node in {n.id for n in symbiont_hidden}
            if not from_is_hidden and not to_is_hidden:
                continue
            max_innov += 1
            new_connections.add(
                ConnectionGene(
                    innovation=max_innov,
                    from_node=from_id,
                    to_node=to_id,
                    weight=conn.weight,
                    enabled=conn.enabled,
                )
            )

        # 5. Create interface connections
        remapped_hidden_ids = [remap[n.id] for n in symbiont_hidden]
        if remapped_hidden_ids:
            host_hidden_and_io = [n.id for n in host.nodes if n.node_type != "input"]
            if not host_hidden_and_io:
                host_hidden_and_io = list(host_node_ids)

            n_host_to_symbiont = int(self.interface_count * self.interface_ratio)
            n_symbiont_to_host = self.interface_count - n_host_to_symbiont

            for _ in range(n_host_to_symbiont):
                from_id = rng.choice(host_hidden_and_io)
                to_id = rng.choice(remapped_hidden_ids)
                weight = self._init_weight(host, symbiont, rng)
                max_innov += 1
                new_connections.add(
                    ConnectionGene(
                        innovation=max_innov,
                        from_node=from_id,
                        to_node=to_id,
                        weight=weight,
                        enabled=True,
                    )
                )

            for _ in range(n_symbiont_to_host):
                from_id = rng.choice(remapped_hidden_ids)
                to_id = rng.choice(host_hidden_and_io)
                weight = self._init_weight(host, symbiont, rng)
                max_innov += 1
                new_connections.add(
                    ConnectionGene(
                        innovation=max_innov,
                        from_node=from_id,
                        to_node=to_id,
                        weight=weight,
                        enabled=True,
                    )
                )

        # 6. Assemble merged genome
        merged_nodes = host.nodes | frozenset(new_hidden_nodes)
        merged_connections = host.connections | frozenset(new_connections)

        return GraphGenome(  # type: ignore[return-value]
            nodes=merged_nodes,
            connections=merged_connections,
            input_ids=host.input_ids,
            output_ids=host.output_ids,
        )

    def _init_weight(self, host: Any, symbiont: Any, rng: Random) -> float:
        """Compute interface connection weight based on weight_method."""
        if self.weight_method == "random":
            return rng.gauss(self.weight_mean, self.weight_std)
        elif self.weight_method == "host_biased":
            from evolve.representation.graph import GraphGenome

            if isinstance(host, GraphGenome):
                host_weights = [c.weight for c in host.connections if c.enabled]
                if host_weights:
                    return sum(host_weights) / len(host_weights)
            return 0.0
        else:  # "mean"
            from evolve.representation.graph import GraphGenome

            weights: list[float] = []
            if isinstance(host, GraphGenome):
                weights.extend(c.weight for c in host.connections if c.enabled)
            if isinstance(symbiont, GraphGenome):
                weights.extend(c.weight for c in symbiont.connections if c.enabled)
            return sum(weights) / len(weights) if weights else 0.0


__all__ = [
    "SymbiogeneticMerge",
    "GraphSymbiogeneticMerge",
]
