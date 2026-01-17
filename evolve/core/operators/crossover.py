"""
Crossover operators - Combine genetic material from parents.

Crossover operators MUST:
- Accept explicit RNG for determinism
- Return new genome instances (not modify parents)
- Handle bounds checking for numeric genomes
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

G = TypeVar("G")


@runtime_checkable
class CrossoverOperator(Protocol[G]):
    """
    Combines genetic material from two parents.
    
    Crossover operators create offspring by mixing
    genetic information from parent genomes.
    """

    def crossover(
        self,
        parent1: G,
        parent2: G,
        rng: Random,
    ) -> tuple[G, G]:
        """
        Create two offspring from two parents.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            rng: Random number generator
            
        Returns:
            Two offspring genomes
        """
        ...


@dataclass
class UniformCrossover:
    """
    Uniform crossover for vector genomes.
    
    Each gene independently comes from either parent
    with equal probability.
    
    Attributes:
        swap_prob: Probability of taking gene from parent2 (default: 0.5)
    """

    swap_prob: float = 0.5

    def crossover(
        self,
        parent1: "VectorGenome",  # type: ignore[name-defined]
        parent2: "VectorGenome",  # type: ignore[name-defined]
        rng: Random,
    ) -> tuple["VectorGenome", "VectorGenome"]:  # type: ignore[name-defined]
        """Create offspring via uniform crossover."""
        from evolve.representation.vector import VectorGenome
        
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Swap genes based on probability
        for i in range(len(genes1)):
            if rng.random() < self.swap_prob:
                genes1[i], genes2[i] = genes2[i], genes1[i]
        
        child1 = VectorGenome(genes=genes1, bounds=parent1.bounds)
        child2 = VectorGenome(genes=genes2, bounds=parent2.bounds)
        
        return child1, child2


@dataclass
class SinglePointCrossover:
    """
    Single-point crossover for vector genomes.
    
    Select a random crossover point; offspring get genes
    from one parent before the point and the other after.
    """

    def crossover(
        self,
        parent1: "VectorGenome",  # type: ignore[name-defined]
        parent2: "VectorGenome",  # type: ignore[name-defined]
        rng: Random,
    ) -> tuple["VectorGenome", "VectorGenome"]:  # type: ignore[name-defined]
        """Create offspring via single-point crossover."""
        from evolve.representation.vector import VectorGenome
        
        n_genes = len(parent1.genes)
        point = rng.randint(1, n_genes - 1) if n_genes > 1 else 1
        
        genes1 = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        genes2 = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        child1 = VectorGenome(genes=genes1, bounds=parent1.bounds)
        child2 = VectorGenome(genes=genes2, bounds=parent2.bounds)
        
        return child1, child2


@dataclass
class TwoPointCrossover:
    """
    Two-point crossover for vector genomes.
    
    Select two crossover points; swap the segment between them.
    """

    def crossover(
        self,
        parent1: "VectorGenome",  # type: ignore[name-defined]
        parent2: "VectorGenome",  # type: ignore[name-defined]
        rng: Random,
    ) -> tuple["VectorGenome", "VectorGenome"]:  # type: ignore[name-defined]
        """Create offspring via two-point crossover."""
        from evolve.representation.vector import VectorGenome
        
        n_genes = len(parent1.genes)
        
        if n_genes < 3:
            # Fall back to single-point for short genomes
            return SinglePointCrossover().crossover(parent1, parent2, rng)
        
        # Select two distinct points
        points = sorted(rng.sample(range(1, n_genes), 2))
        p1, p2 = points[0], points[1]
        
        genes1 = np.concatenate([
            parent1.genes[:p1],
            parent2.genes[p1:p2],
            parent1.genes[p2:],
        ])
        genes2 = np.concatenate([
            parent2.genes[:p1],
            parent1.genes[p1:p2],
            parent2.genes[p2:],
        ])
        
        child1 = VectorGenome(genes=genes1, bounds=parent1.bounds)
        child2 = VectorGenome(genes=genes2, bounds=parent2.bounds)
        
        return child1, child2


@dataclass
class BlendCrossover:
    """
    BLX-α crossover for continuous optimization.
    
    Creates offspring by interpolating/extrapolating
    between parent gene values.
    
    Attributes:
        alpha: Extension factor (default: 0.5)
               0.0 = interpolation only
               0.5 = standard BLX-α (can extend 50% beyond parents)
    """

    alpha: float = 0.5

    def crossover(
        self,
        parent1: "VectorGenome",  # type: ignore[name-defined]
        parent2: "VectorGenome",  # type: ignore[name-defined]
        rng: Random,
    ) -> tuple["VectorGenome", "VectorGenome"]:  # type: ignore[name-defined]
        """Create offspring via blend crossover."""
        from evolve.representation.vector import VectorGenome
        
        genes1_list = []
        genes2_list = []
        
        for g1, g2 in zip(parent1.genes, parent2.genes):
            d = abs(g2 - g1)
            low = min(g1, g2) - self.alpha * d
            high = max(g1, g2) + self.alpha * d
            
            genes1_list.append(rng.uniform(low, high))
            genes2_list.append(rng.uniform(low, high))
        
        genes1 = np.array(genes1_list)
        genes2 = np.array(genes2_list)
        
        # Clip to bounds if present
        child1 = VectorGenome(genes=genes1, bounds=parent1.bounds)
        child2 = VectorGenome(genes=genes2, bounds=parent2.bounds)
        
        if parent1.bounds is not None:
            child1 = child1.clip_to_bounds()
            child2 = child2.clip_to_bounds()
        
        return child1, child2


@dataclass
class SimulatedBinaryCrossover:
    """
    SBX crossover commonly used in NSGA-II.
    
    Creates offspring that have similar distribution
    to parent distances as single-point crossover
    on binary strings.
    
    Attributes:
        eta: Distribution index (higher = children closer to parents)
    """

    eta: float = 15.0

    def crossover(
        self,
        parent1: "VectorGenome",  # type: ignore[name-defined]
        parent2: "VectorGenome",  # type: ignore[name-defined]
        rng: Random,
    ) -> tuple["VectorGenome", "VectorGenome"]:  # type: ignore[name-defined]
        """Create offspring via SBX crossover."""
        from evolve.representation.vector import VectorGenome
        
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        for i in range(len(genes1)):
            if rng.random() <= 0.5:
                if abs(genes1[i] - genes2[i]) > 1e-14:
                    y1 = min(genes1[i], genes2[i])
                    y2 = max(genes1[i], genes2[i])
                    
                    u = rng.random()
                    beta = (
                        (2.0 * u) ** (1.0 / (self.eta + 1))
                        if u <= 0.5
                        else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1))
                    )
                    
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    genes1[i] = c1
                    genes2[i] = c2
        
        child1 = VectorGenome(genes=genes1, bounds=parent1.bounds)
        child2 = VectorGenome(genes=genes2, bounds=parent2.bounds)
        
        if parent1.bounds is not None:
            child1 = child1.clip_to_bounds()
            child2 = child2.clip_to_bounds()
        
        return child1, child2


@dataclass
class NEATCrossover:
    """
    NEAT-style crossover for graph genomes.
    
    Aligns genes by innovation number and combines them:
    - Matching genes: randomly choose from either parent
    - Disjoint/Excess genes: inherit from fitter parent
    
    In NEAT, the fitter parent's topology is preserved for
    disjoint and excess genes.
    
    Attributes:
        match_prob: Probability of taking matching gene from parent1 (default: 0.5)
        disabled_gene_inherit_prob: Probability of inheriting disabled gene
            as disabled (default: 0.75)
    """

    match_prob: float = 0.5
    disabled_gene_inherit_prob: float = 0.75

    def crossover(
        self,
        parent1: "GraphGenome",  # type: ignore[name-defined]
        parent2: "GraphGenome",  # type: ignore[name-defined]
        rng: Random,
        parent1_fitter: bool = True,
    ) -> tuple["GraphGenome", "GraphGenome"]:  # type: ignore[name-defined]
        """
        Create offspring via NEAT crossover.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            rng: Random number generator
            parent1_fitter: If True, parent1 is considered fitter
            
        Returns:
            Two offspring genomes (second is symmetric crossover)
        """
        from evolve.representation.graph import GraphGenome, NodeGene, ConnectionGene
        
        # Build connection maps by innovation number
        p1_conns = {c.innovation: c for c in parent1.connections}
        p2_conns = {c.innovation: c for c in parent2.connections}
        
        all_innovations = set(p1_conns.keys()) | set(p2_conns.keys())
        
        # Child 1: parent1_fitter determines which parent donates disjoint/excess
        child1_connections: set[ConnectionGene] = set()
        child2_connections: set[ConnectionGene] = set()
        
        for innov in all_innovations:
            c1 = p1_conns.get(innov)
            c2 = p2_conns.get(innov)
            
            if c1 is not None and c2 is not None:
                # Matching gene - randomly choose
                chosen_for_child1 = c1 if rng.random() < self.match_prob else c2
                chosen_for_child2 = c2 if rng.random() < self.match_prob else c1
                
                # Handle disabled genes
                if not c1.enabled or not c2.enabled:
                    if rng.random() < self.disabled_gene_inherit_prob:
                        chosen_for_child1 = chosen_for_child1.with_enabled(False)
                    if rng.random() < self.disabled_gene_inherit_prob:
                        chosen_for_child2 = chosen_for_child2.with_enabled(False)
                
                child1_connections.add(chosen_for_child1)
                child2_connections.add(chosen_for_child2)
                
            elif c1 is not None:
                # Disjoint/excess from parent1
                if parent1_fitter:
                    child1_connections.add(c1)
                else:
                    child2_connections.add(c1)
                    
            else:  # c2 is not None
                # Disjoint/excess from parent2
                if parent1_fitter:
                    child2_connections.add(c2)
                else:
                    child1_connections.add(c2)
        
        # Collect all required nodes for each child
        def get_required_nodes(connections: set[ConnectionGene]) -> set[int]:
            nodes: set[int] = set()
            for c in connections:
                nodes.add(c.from_node)
                nodes.add(c.to_node)
            return nodes
        
        child1_node_ids = get_required_nodes(child1_connections)
        child2_node_ids = get_required_nodes(child2_connections)
        
        # Always include input and output nodes
        child1_node_ids |= set(parent1.input_ids) | set(parent1.output_ids)
        child2_node_ids |= set(parent1.input_ids) | set(parent1.output_ids)
        
        # Build node maps
        p1_nodes = {n.id: n for n in parent1.nodes}
        p2_nodes = {n.id: n for n in parent2.nodes}
        
        child1_nodes: set[NodeGene] = set()
        child2_nodes: set[NodeGene] = set()
        
        for node_id in child1_node_ids:
            # Prefer fitter parent's node, fall back to other
            if parent1_fitter:
                node = p1_nodes.get(node_id) or p2_nodes.get(node_id)
            else:
                node = p2_nodes.get(node_id) or p1_nodes.get(node_id)
            if node:
                child1_nodes.add(node)
        
        for node_id in child2_node_ids:
            # Opposite preference for child2
            if parent1_fitter:
                node = p2_nodes.get(node_id) or p1_nodes.get(node_id)
            else:
                node = p1_nodes.get(node_id) or p2_nodes.get(node_id)
            if node:
                child2_nodes.add(node)
        
        child1 = GraphGenome(
            nodes=frozenset(child1_nodes),
            connections=frozenset(child1_connections),
            input_ids=parent1.input_ids,
            output_ids=parent1.output_ids,
        )
        
        child2 = GraphGenome(
            nodes=frozenset(child2_nodes),
            connections=frozenset(child2_connections),
            input_ids=parent1.input_ids,
            output_ids=parent1.output_ids,
        )
        
        return child1, child2
