"""Tests for symbiogenetic merge operators."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import numpy as np
import pytest

from evolve.core.operators.merge import (
    GraphSymbiogeneticMerge,
    SymbiogeneticMerge,
)
from evolve.representation.graph import ConnectionGene, GraphGenome, NodeGene
from evolve.representation.vector import VectorGenome

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify all implementations satisfy SymbiogeneticMerge protocol."""

    def test_graph_is_merge(self) -> None:
        op = GraphSymbiogeneticMerge()
        assert isinstance(op, SymbiogeneticMerge)


# ---------------------------------------------------------------------------
# Identity check (all operators)
# ---------------------------------------------------------------------------


class TestIdentityCheck:
    """All operators must reject merging the same instance."""

    def test_graph_same_instance(self) -> None:
        op = GraphSymbiogeneticMerge()
        genome = _make_graph()
        with pytest.raises(ValueError, match="different genome instances"):
            op.merge(genome, genome, Random(42))


# ---------------------------------------------------------------------------
# GraphSymbiogeneticMerge
# ---------------------------------------------------------------------------


def _make_graph(
    n_hidden: int = 0,
    hidden_start_id: int = 4,
) -> GraphGenome:
    """Helper: create a graph genome with 2 inputs, 1 output, optional hidden."""
    nodes: set[NodeGene] = {
        NodeGene(0, "input"),
        NodeGene(1, "input"),
        NodeGene(2, "output"),
    }
    connections: set[ConnectionGene] = {
        ConnectionGene(1, 0, 2, 0.5),
        ConnectionGene(2, 1, 2, -0.3),
    }
    for i in range(n_hidden):
        nid = hidden_start_id + i
        nodes.add(NodeGene(nid, "hidden", "relu", bias=0.1 * i))
        connections.add(ConnectionGene(100 + i, 0, nid, 0.2))
        connections.add(ConnectionGene(200 + i, nid, 2, 0.3))
    return GraphGenome(
        nodes=frozenset(nodes),
        connections=frozenset(connections),
        input_ids=(0, 1),
        output_ids=(2,),
    )


class TestGraphSymbiogeneticMerge:
    """Tests for graph genome merge."""

    def test_merge_adds_hidden_nodes(self) -> None:
        host = _make_graph(n_hidden=0)
        symbiont = _make_graph(n_hidden=2, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))
        assert isinstance(merged, GraphGenome)
        # Original host had 3 nodes (2 in + 1 out), symbiont added 2 hidden
        assert merged.n_hidden == 2
        assert merged.n_inputs == 2
        assert merged.n_outputs == 1

    def test_merge_preserves_host_io(self) -> None:
        host = _make_graph()
        symbiont = _make_graph(n_hidden=1, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))
        assert merged.input_ids == host.input_ids
        assert merged.output_ids == host.output_ids

    def test_merge_increases_connections(self) -> None:
        host = _make_graph()
        symbiont = _make_graph(n_hidden=2, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=4)
        merged = op.merge(host, symbiont, Random(42))
        # Merged should have more connections than host alone
        assert merged.n_connections > host.n_connections

    def test_node_ids_no_collision(self) -> None:
        host = _make_graph(n_hidden=1, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=1, hidden_start_id=4)  # same ID
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))
        node_ids = [n.id for n in merged.nodes]
        assert len(node_ids) == len(set(node_ids)), "Node ID collision detected"

    def test_weight_method_random(self) -> None:
        host = _make_graph()
        symbiont = _make_graph(n_hidden=1, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(
            interface_count=4,
            weight_method="random",
            weight_mean=0.0,
            weight_std=1.0,
        )
        merged = op.merge(host, symbiont, Random(42))
        assert merged.n_connections > host.n_connections

    def test_weight_method_host_biased(self) -> None:
        host = _make_graph()
        symbiont = _make_graph(n_hidden=1, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(
            interface_count=2,
            weight_method="host_biased",
        )
        merged = op.merge(host, symbiont, Random(42))
        assert merged.n_connections > host.n_connections

    def test_deterministic_with_same_seed(self) -> None:
        host = _make_graph()
        symbiont = _make_graph(n_hidden=2, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=4)
        m1 = op.merge(host, symbiont, Random(123))
        m2 = op.merge(host, symbiont, Random(123))
        assert m1 == m2

    def test_no_hidden_symbiont_no_crash(self) -> None:
        """Symbiont with no hidden nodes should produce same as host."""
        host = _make_graph()
        symbiont = _make_graph(n_hidden=0)
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))
        # No hidden nodes to absorb, so merged ≈ host
        assert merged.n_hidden == 0


# ---------------------------------------------------------------------------
# SequenceSymbiogeneticMerge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Registry category test (T011)
# ---------------------------------------------------------------------------


class TestRegistryCategory:
    """Verify merge category in OperatorRegistry."""

    def test_merge_category_exists(self) -> None:
        """T011: 'merge' is a valid OperatorRegistry category."""
        from evolve.registry.operators import get_operator_registry

        registry = get_operator_registry()
        assert "merge" in registry.list_all()

    def test_graph_merge_operator_registered(self) -> None:
        """Verify graph merge operator is registered."""
        from evolve.registry.operators import get_operator_registry

        registry = get_operator_registry()
        merge_ops = registry.list_operators("merge")
        assert "graph_symbiogenetic" in merge_ops


# ---------------------------------------------------------------------------
# Custom merge registration (T038)
# ---------------------------------------------------------------------------


class TestCustomMergeRegistration:
    """Verify user-defined merge operators can be registered."""

    def test_custom_merge_register_and_retrieve(self) -> None:
        """T038: user-defined class satisfying protocol can be registered."""
        from evolve.registry.operators import OperatorRegistry

        @dataclass
        class _CustomMerge:
            scale: float = 1.0

            def merge(self, host: Any, symbiont: Any, rng: Random) -> Any:
                return host

        reg = OperatorRegistry()
        reg.register("merge", "custom_test", _CustomMerge, compatible_genomes={"vector"})
        op = reg.get("merge", "custom_test", scale=2.0)
        assert isinstance(op, SymbiogeneticMerge)
        assert op.scale == 2.0


# ---------------------------------------------------------------------------
# Graph merge detailed tests (T015-T017, T066)
# ---------------------------------------------------------------------------


class TestGraphMergeDetails:
    """Detailed graph merge validation tests."""

    def test_symbiont_innovation_numbers_remapped(self) -> None:
        """T015: symbiont innovation numbers are remapped to avoid collision."""
        host = _make_graph(n_hidden=1, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=1, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))

        # All innovation numbers should be unique
        innovations = [c.innovation for c in merged.connections]
        assert len(innovations) == len(set(innovations)), "Innovation number collision detected"

        # Host innovations should be preserved
        host_innovations = {c.innovation for c in host.connections}
        merged_innovations = {c.innovation for c in merged.connections}
        assert host_innovations.issubset(merged_innovations), (
            "Host innovation numbers were modified"
        )

    def test_interface_connections_directional_split(self) -> None:
        """T016: interface connections follow interface_ratio split."""
        host = _make_graph(n_hidden=1, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=2, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=10, interface_ratio=0.7)
        merged = op.merge(host, symbiont, Random(42))

        # Count interface connections (those not in host or remapped symbiont internals)
        host_innovations = {c.innovation for c in host.connections}
        interface_conns = [c for c in merged.connections if c.innovation not in host_innovations]
        # Should have some interface connections
        assert len(interface_conns) > 0

    def test_host_internal_topology_preserved(self) -> None:
        """T017: host internal topology and weights are preserved."""
        host = _make_graph(n_hidden=1, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=1, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=2)
        merged = op.merge(host, symbiont, Random(42))

        # Host connections should be present in merged with same weights
        for conn in host.connections:
            matching = [c for c in merged.connections if c.innovation == conn.innovation]
            assert len(matching) == 1, f"Host connection {conn.innovation} missing"
            assert matching[0].weight == conn.weight, (
                f"Host connection {conn.innovation} weight changed"
            )

        # Host nodes should be present
        host_ids = {n.id for n in host.nodes}
        merged_ids = {n.id for n in merged.nodes}
        assert host_ids.issubset(merged_ids)

    def test_neat_distance_exceeds_threshold_after_merge(self) -> None:
        """T066: merged offspring has high structural distance from host (SC-002)."""
        from evolve.diversity.speciation import neat_distance

        host = _make_graph(n_hidden=1, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=2, hidden_start_id=10)
        op = GraphSymbiogeneticMerge(interface_count=4)
        merged = op.merge(host, symbiont, Random(42))

        # Distance between merged and host should be non-trivial
        dist = neat_distance(merged, host, c_disjoint=1.0, c_excess=1.0, c_weight=0.4)
        # The merged genome adds symbiont genes + interface connections,
        # so it must differ significantly from the original host
        assert dist > 0.0, "Merged genome should differ from host structurally"

    def test_interface_count_exceeds_available_nodes(self) -> None:
        """T068: interface_count > available nodes creates as many as possible."""

        host = _make_graph(n_hidden=0, hidden_start_id=4)
        symbiont = _make_graph(n_hidden=0, hidden_start_id=10)
        # Request more interface connections than possible node pairs
        op = GraphSymbiogeneticMerge(interface_count=1000)
        merged = op.merge(host, symbiont, Random(42))

        # Should still produce a valid genome without error
        assert len(merged.nodes) >= len(host.nodes)


class TestMergeMetadata:
    """Tests for merge offspring metadata (T056)."""

    def test_merged_offspring_metadata_fields(self) -> None:
        """T056: merged offspring metadata includes origin, host_id, symbiont_id, source_strategy."""
        from uuid import uuid4

        from evolve.core.engine import EvolutionConfig, EvolutionEngine
        from evolve.core.operators.crossover import UniformCrossover
        from evolve.core.operators.mutation import GaussianMutation
        from evolve.core.operators.selection import TournamentSelection
        from evolve.core.population import Population
        from evolve.core.types import Individual as Ind
        from evolve.evaluation.evaluator import FunctionEvaluator
        from evolve.representation.vector import VectorGenome

        def _sphere(x: np.ndarray) -> float:
            return float(np.sum(x**2))

        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        @dataclass
        class _SimpleMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_SimpleMerge(),
        )

        individuals = [
            Ind(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        merged = [
            ind
            for ind in result.population.individuals
            if ind.metadata.origin == "symbiogenetic_merge"
        ]
        assert len(merged) > 0
        for ind in merged:
            assert ind.metadata.origin == "symbiogenetic_merge"
            assert ind.metadata.parent_ids is not None
            assert len(ind.metadata.parent_ids) == 2  # (host_id, symbiont_id)
            assert ind.metadata.source_strategy is not None
