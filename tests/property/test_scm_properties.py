"""Property-based tests for SCM representation (T088, T100, T101).

Uses hypothesis to test:
- Serialization round-trips (T088)
- Decoding determinism (T100)
- Graph validity (T101)

Focused on core invariants that the implementation must maintain.
"""

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from evolve.representation.scm import (
    SCMConfig,
    SCMGenome,
)
from evolve.representation.scm_decoder import SCMDecoder

# -- Strategies for generating SCM configs --


@st.composite
def scm_config_strategy(draw):
    """Generate valid SCMConfig instances."""
    num_vars = draw(st.integers(min_value=2, max_value=6))
    var_names = tuple(f"V{i}" for i in range(num_vars))

    return SCMConfig(observed_variables=var_names)


@st.composite
def scm_genome_strategy(draw):
    """Generate random SCMGenome instances."""
    config = draw(scm_config_strategy())
    length = draw(st.integers(min_value=20, max_value=100))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    genome = SCMGenome.random(config, length=length, rng=Random(seed))
    return genome, config


class TestSerializationRoundTrip:
    """T088: Property tests for serialization round-trip."""

    @given(scm_genome_strategy())
    @settings(max_examples=50, deadline=None)
    def test_genome_to_dict_roundtrip(self, genome_and_config):
        """Genome -> dict -> Genome preserves all data."""
        genome, config = genome_and_config

        # Serialize to dict
        data = genome.to_dict()

        # Verify dict contains expected keys
        assert "genes" in data
        assert "config" in data

        # Deserialize
        restored = SCMGenome.from_dict(data)

        # Verify equality
        assert len(restored.genes) == len(genome.genes)
        assert list(restored.genes) == list(genome.genes)
        assert restored.config.observed_variables == genome.config.observed_variables


class TestGenomeBasics:
    """Basic invariants for SCMGenome."""

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_genome_has_genes(self, genome_and_config):
        """Genome always has some genes."""
        genome, config = genome_and_config
        assert len(genome.genes) > 0

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_genome_config_preserved(self, genome_and_config):
        """Genome stores its config."""
        genome, config = genome_and_config
        assert genome.config.observed_variables == config.observed_variables

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_random_deterministic_with_seed(self, genome_and_config):
        """Same seed produces same genome."""
        genome, config = genome_and_config

        # Create two genomes with same seed
        g1 = SCMGenome.random(config, length=50, rng=Random(12345))
        g2 = SCMGenome.random(config, length=50, rng=Random(12345))

        assert list(g1.genes) == list(g2.genes)


class TestDecodingDeterminism:
    """T100: Property tests for decoding determinism."""

    @given(scm_genome_strategy())
    @settings(max_examples=50, deadline=None)
    def test_same_genome_same_decoded_scm(self, genome_and_config):
        """Same genome always decodes to same DecodedSCM."""
        genome, config = genome_and_config
        decoder = SCMDecoder(config)

        # Decode the same genome twice
        scm1 = decoder.decode(genome)
        scm2 = decoder.decode(genome)

        # Equations should be identical
        assert set(scm1.equations.keys()) == set(scm2.equations.keys())

        # Graph structure should be identical
        assert set(scm1.graph.nodes()) == set(scm2.graph.nodes())
        assert set(scm1.graph.edges()) == set(scm2.graph.edges())

        # Metadata should be identical
        assert scm1.metadata.conflict_count == scm2.metadata.conflict_count
        assert scm1.metadata.junk_gene_indices == scm2.metadata.junk_gene_indices
        assert scm1.metadata.is_cyclic == scm2.metadata.is_cyclic

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_decoding_with_fresh_decoder(self, genome_and_config):
        """Fresh decoder produces same result as reused decoder."""
        genome, config = genome_and_config

        # Decode with two separate decoder instances
        decoder1 = SCMDecoder(config)
        decoder2 = SCMDecoder(config)

        scm1 = decoder1.decode(genome)
        scm2 = decoder2.decode(genome)

        assert set(scm1.equations.keys()) == set(scm2.equations.keys())
        assert set(scm1.graph.edges()) == set(scm2.graph.edges())


class TestGraphValidity:
    """T101: Property tests for graph validity (edges match equation dependencies)."""

    @given(scm_genome_strategy())
    @settings(max_examples=50, deadline=None)
    def test_edges_match_equation_dependencies(self, genome_and_config):
        """Graph edges exactly match variables in equations."""
        genome, config = genome_and_config
        decoder = SCMDecoder(config)
        scm = decoder.decode(genome)

        # Collect all dependencies from equations
        from evolve.representation.scm_decoder import variables

        expected_edges = set()
        for target, expr in scm.equations.items():
            deps = variables(expr)
            for dep in deps:
                expected_edges.add((dep, target))

        # Get actual edges from graph
        actual_edges = set(scm.graph.edges())

        # Edges should match exactly
        assert expected_edges == actual_edges, (
            f"Edge mismatch: expected {expected_edges}, got {actual_edges}"
        )

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_equation_targets_are_graph_nodes(self, genome_and_config):
        """All equation targets are nodes in the graph."""
        genome, config = genome_and_config
        decoder = SCMDecoder(config)
        scm = decoder.decode(genome)

        nodes = set(scm.graph.nodes())

        for target in scm.equations.keys():
            assert target in nodes, f"Equation target {target} not in graph nodes"

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_self_loops_detected_as_cycles(self, genome_and_config):
        """Self-loops (A -> A) are detected as cycles."""
        genome, config = genome_and_config
        decoder = SCMDecoder(config)
        scm = decoder.decode(genome)

        has_self_loop = any(source == target for source, target in scm.graph.edges())

        # If there's a self-loop, it should be marked as cyclic
        if has_self_loop:
            assert scm.metadata.is_cyclic, "Self-loop not detected as cycle"

    @given(scm_genome_strategy())
    @settings(max_examples=30, deadline=None)
    def test_cycle_detection_matches_graph(self, genome_and_config):
        """metadata.is_cyclic matches actual graph cycle detection."""
        import networkx as nx

        genome, config = genome_and_config
        decoder = SCMDecoder(config)
        scm = decoder.decode(genome)

        # Verify using networkx cycle detection
        try:
            cycles = list(nx.simple_cycles(scm.graph))
            has_cycles = len(cycles) > 0
        except nx.NetworkXError:
            has_cycles = False

        assert scm.metadata.is_cyclic == has_cycles, (
            f"Cycle mismatch: metadata says {scm.metadata.is_cyclic}, "
            f"but networkx found cycles={has_cycles}"
        )
