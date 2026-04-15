"""
Mutation operators - Modify individual genomes.

Registry names (for ``UnifiedConfig(mutation=...)``)::

    "gaussian", "uniform", "polynomial"

Mutation operators MUST:
- Accept explicit RNG for determinism
- Return new genome instances (not modify in place)
- Handle bounds checking for numeric genomes
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from evolve.representation.graph import GraphGenome, InnovationTracker
    from evolve.representation.vector import VectorGenome

G = TypeVar("G")


@runtime_checkable
class MutationOperator(Protocol[G]):
    """
    Modifies an individual genome.

    Mutation operators introduce variation by
    randomly modifying genetic information.
    """

    def mutate(
        self,
        genome: G,
        rng: Random,
    ) -> G:
        """
        Create mutated copy of genome.

        Args:
            genome: Genome to mutate
            rng: Random number generator

        Returns:
            New mutated genome
        """
        ...


@dataclass
class GaussianMutation:
    """
    Gaussian mutation for vector genomes.

    Each gene is mutated by adding Gaussian noise
    with probability mutation_rate.

    Attributes:
        mutation_rate: Probability of mutating each gene (default: 0.1)
        sigma: Standard deviation of Gaussian noise (default: 0.1)
        adaptive: If True, sigma is relative to gene range
    """

    mutation_rate: float = 0.1
    sigma: float = 0.1
    adaptive: bool = False

    def mutate(
        self,
        genome: VectorGenome,
        rng: Random,
    ) -> VectorGenome:
        """Apply Gaussian mutation."""
        from evolve.representation.vector import VectorGenome

        genes = genome.genes.copy()

        for i in range(len(genes)):
            if rng.random() < self.mutation_rate:
                # Compute sigma (adaptive or fixed)
                if self.adaptive and genome.bounds is not None:
                    gene_range = genome.bounds[1][i] - genome.bounds[0][i]
                    sigma = self.sigma * gene_range
                else:
                    sigma = self.sigma

                # Add Gaussian noise
                genes[i] += rng.gauss(0, sigma)

        mutated = VectorGenome(genes=genes, bounds=genome.bounds)

        # Clip to bounds if present
        if genome.bounds is not None:
            mutated = mutated.clip_to_bounds()

        return mutated


@dataclass
class UniformMutation:
    """
    Uniform mutation for vector genomes.

    Each gene is replaced with a uniform random value
    within bounds with probability mutation_rate.

    Attributes:
        mutation_rate: Probability of mutating each gene (default: 0.1)
    """

    mutation_rate: float = 0.1

    def mutate(
        self,
        genome: VectorGenome,
        rng: Random,
    ) -> VectorGenome:
        """Apply uniform mutation."""
        from evolve.representation.vector import VectorGenome

        if genome.bounds is None:
            raise ValueError("UniformMutation requires bounds")

        genes = genome.genes.copy()
        lower, upper = genome.bounds

        for i in range(len(genes)):
            if rng.random() < self.mutation_rate:
                genes[i] = rng.uniform(lower[i], upper[i])

        return VectorGenome(genes=genes, bounds=genome.bounds)


@dataclass
class PolynomialMutation:
    """
    Polynomial mutation (commonly used in NSGA-II).

    Creates mutations with distribution similar to
    Gaussian but bounded within a range.

    Attributes:
        mutation_rate: Probability of mutating each gene (default: 1/n_genes)
        eta: Distribution index (higher = smaller mutations)
    """

    mutation_rate: float | None = None  # Default: 1/n_genes
    eta: float = 20.0

    def mutate(
        self,
        genome: VectorGenome,
        rng: Random,
    ) -> VectorGenome:
        """Apply polynomial mutation."""
        from evolve.representation.vector import VectorGenome

        genes = genome.genes.copy()
        n_genes = len(genes)

        # Default mutation rate
        mutation_rate = self.mutation_rate or (1.0 / n_genes)

        # Get bounds (use default if not provided)
        if genome.bounds is not None:
            lower, upper = genome.bounds
        else:
            lower = np.full(n_genes, -1e10)
            upper = np.full(n_genes, 1e10)

        for i in range(n_genes):
            if rng.random() < mutation_rate:
                y = genes[i]
                yl, yu = lower[i], upper[i]

                if yu - yl > 1e-10:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)

                    u = rng.random()
                    mut_pow = 1.0 / (self.eta + 1.0)

                    if u < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (self.eta + 1.0))
                        deltaq = (val**mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (self.eta + 1.0))
                        deltaq = 1.0 - (val**mut_pow)

                    y = y + deltaq * (yu - yl)
                    genes[i] = np.clip(y, yl, yu)

        return VectorGenome(genes=genes, bounds=genome.bounds)


@dataclass
class CreepMutation:
    """
    Creep mutation - small random changes.

    Similar to Gaussian but uses uniform distribution
    within a small range around current value.

    Attributes:
        mutation_rate: Probability of mutating each gene (default: 0.1)
        creep_range: Maximum change as fraction of gene range (default: 0.1)
    """

    mutation_rate: float = 0.1
    creep_range: float = 0.1

    def mutate(
        self,
        genome: VectorGenome,
        rng: Random,
    ) -> VectorGenome:
        """Apply creep mutation."""
        from evolve.representation.vector import VectorGenome

        genes = genome.genes.copy()

        for i in range(len(genes)):
            if rng.random() < self.mutation_rate:
                # Compute creep amount
                if genome.bounds is not None:
                    gene_range = genome.bounds[1][i] - genome.bounds[0][i]
                    delta = self.creep_range * gene_range
                else:
                    delta = self.creep_range

                # Apply uniform creep
                genes[i] += rng.uniform(-delta, delta)

        mutated = VectorGenome(genes=genes, bounds=genome.bounds)

        if genome.bounds is not None:
            mutated = mutated.clip_to_bounds()

        return mutated


@dataclass
class NEATMutation:
    """
    NEAT-style mutation for graph genomes.

    Supports three types of mutations:
    1. Add node: Split an existing connection with a new hidden node
    2. Add connection: Create new connection between unconnected nodes
    3. Weight mutation: Perturb or reset connection weights

    Attributes:
        add_node_prob: Probability of adding a new node (default: 0.03)
        add_connection_prob: Probability of adding a new connection (default: 0.05)
        weight_mutation_prob: Probability of mutating weights (default: 0.8)
        weight_perturb_prob: Probability of perturbing vs resetting weight (default: 0.9)
        weight_perturb_sigma: Sigma for weight perturbation (default: 0.5)
        weight_reset_range: Range for weight reset (default: (-2, 2))
        bias_mutation_prob: Probability of mutating biases (default: 0.2)
        bias_perturb_sigma: Sigma for bias perturbation (default: 0.3)
        allow_recurrent: Whether to allow recurrent connections (default: False)
    """

    add_node_prob: float = 0.03
    add_connection_prob: float = 0.05
    weight_mutation_prob: float = 0.8
    weight_perturb_prob: float = 0.9
    weight_perturb_sigma: float = 0.5
    weight_reset_range: tuple[float, float] = (-2.0, 2.0)
    bias_mutation_prob: float = 0.2
    bias_perturb_sigma: float = 0.3
    allow_recurrent: bool = False

    # Innovation tracker must be set before use
    innovation_tracker: InnovationTracker | None = None

    def mutate(
        self,
        genome: GraphGenome,
        rng: Random,
    ) -> GraphGenome:
        """
        Apply NEAT mutations to genome.

        Mutations are applied in order:
        1. Possibly add a node
        2. Possibly add a connection
        3. Possibly mutate weights
        4. Possibly mutate biases
        """

        if self.innovation_tracker is None:
            raise ValueError("innovation_tracker must be set before mutation")

        result = genome

        # 1. Add node mutation
        if rng.random() < self.add_node_prob:
            result = self._add_node_mutation(result, rng)

        # 2. Add connection mutation
        if rng.random() < self.add_connection_prob:
            result = self._add_connection_mutation(result, rng)

        # 3. Weight mutations
        if rng.random() < self.weight_mutation_prob:
            result = self._weight_mutation(result, rng)

        # 4. Bias mutations
        if rng.random() < self.bias_mutation_prob:
            result = self._bias_mutation(result, rng)

        return result

    def _add_node_mutation(
        self,
        genome: GraphGenome,
        rng: Random,
    ) -> GraphGenome:
        """Add a new node by splitting an existing connection."""
        from evolve.representation.graph import ConnectionGene, NodeGene

        # Get enabled connections
        enabled = [c for c in genome.connections if c.enabled]
        if not enabled:
            return genome

        # Select random connection to split
        conn = rng.choice(enabled)

        # Create new node
        new_node_id = self.innovation_tracker.get_new_node_id()  # type: ignore
        new_node = NodeGene(
            id=new_node_id,
            node_type="hidden",
            activation="sigmoid",
            bias=0.0,
        )

        # Create two new connections
        # Connection from source to new node (weight 1.0 for minimal impact)
        innov1 = self.innovation_tracker.get_innovation(conn.from_node, new_node_id)  # type: ignore
        conn1 = ConnectionGene(
            innovation=innov1,
            from_node=conn.from_node,
            to_node=new_node_id,
            weight=1.0,  # Standard NEAT: weight 1 going in
            enabled=True,
        )

        # Connection from new node to target (preserve original weight)
        innov2 = self.innovation_tracker.get_innovation(new_node_id, conn.to_node)  # type: ignore
        conn2 = ConnectionGene(
            innovation=innov2,
            from_node=new_node_id,
            to_node=conn.to_node,
            weight=conn.weight,  # Standard NEAT: preserve weight going out
            enabled=True,
        )

        return genome.add_node(new_node, conn, conn1, conn2)

    def _add_connection_mutation(
        self,
        genome: GraphGenome,
        rng: Random,
    ) -> GraphGenome:
        """Add a new connection between unconnected nodes."""
        from evolve.representation.graph import ConnectionGene

        # Get all possible connections
        all_nodes = list(genome.nodes)
        existing = {(c.from_node, c.to_node) for c in genome.connections}

        # Build candidate pairs
        candidates: list[tuple[int, int]] = []

        for n1 in all_nodes:
            for n2 in all_nodes:
                # Skip self-connections
                if n1.id == n2.id:
                    continue

                # Skip if connection exists
                if (n1.id, n2.id) in existing:
                    continue

                # Skip connections TO input nodes
                if n2.node_type == "input":
                    continue

                # Skip connections FROM output nodes (unless recurrent allowed)
                if n1.node_type == "output" and not self.allow_recurrent:
                    continue

                # Check for cycles if not allowing recurrent
                if not self.allow_recurrent and self._would_create_cycle(genome, n1.id, n2.id):
                    continue

                candidates.append((n1.id, n2.id))

        if not candidates:
            return genome

        # Select random candidate
        from_id, to_id = rng.choice(candidates)

        # Create new connection
        innov = self.innovation_tracker.get_innovation(from_id, to_id)  # type: ignore
        new_conn = ConnectionGene(
            innovation=innov,
            from_node=from_id,
            to_node=to_id,
            weight=rng.uniform(*self.weight_reset_range),
            enabled=True,
        )

        return genome.add_connection(new_conn)

    def _would_create_cycle(
        self,
        genome: GraphGenome,
        from_node: int,
        to_node: int,
    ) -> bool:
        """Check if adding edge from_node -> to_node would create a cycle."""
        # BFS from to_node to see if we can reach from_node
        visited: set[int] = set()
        queue = [to_node]

        # Build adjacency from enabled connections
        adj: dict[int, list[int]] = {}
        for c in genome.connections:
            if c.enabled:
                if c.from_node not in adj:
                    adj[c.from_node] = []
                adj[c.from_node].append(c.to_node)

        while queue:
            current = queue.pop(0)
            if current == from_node:
                return True  # Would create cycle
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adj.get(current, []))

        return False

    def _weight_mutation(
        self,
        genome: GraphGenome,
        rng: Random,
    ) -> GraphGenome:
        """Mutate connection weights."""
        from evolve.representation.graph import ConnectionGene, GraphGenome

        new_connections: set[ConnectionGene] = set()

        for conn in genome.connections:
            if rng.random() < self.weight_perturb_prob:
                # Perturb weight
                new_weight = conn.weight + rng.gauss(0, self.weight_perturb_sigma)
            else:
                # Reset weight
                new_weight = rng.uniform(*self.weight_reset_range)

            new_connections.add(conn.with_weight(new_weight))

        return GraphGenome(
            nodes=genome.nodes,
            connections=frozenset(new_connections),
            input_ids=genome.input_ids,
            output_ids=genome.output_ids,
        )

    def _bias_mutation(
        self,
        genome: GraphGenome,
        rng: Random,
    ) -> GraphGenome:
        """Mutate node biases."""
        from evolve.representation.graph import GraphGenome, NodeGene

        new_nodes: set[NodeGene] = set()

        for node in genome.nodes:
            # Only mutate hidden and output nodes
            if node.node_type == "input":
                new_nodes.add(node)
            else:
                new_bias = node.bias + rng.gauss(0, self.bias_perturb_sigma)
                new_nodes.add(node.with_bias(new_bias))

        return GraphGenome(
            nodes=frozenset(new_nodes),
            connections=genome.connections,
            input_ids=genome.input_ids,
            output_ids=genome.output_ids,
        )
