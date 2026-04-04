"""
Unit tests for SCM Genome module.

Tests cover:
- SCMConfig validation
- SCMAlphabet generation
- SCMGenome creation, copying, equality, hashing
- ERC Gaussian sampling and mutation
- Serialization round-trip
"""

from __future__ import annotations

import math
from random import Random

import pytest

from evolve.representation.scm import (
    AcyclicityMode,
    ConflictResolution,
    SCMAlphabet,
    SCMConfig,
    SCMGenome,
)
from evolve.representation.sequence import SequenceGenome

# === Fixtures ===


@pytest.fixture
def basic_config() -> SCMConfig:
    """Simple 3-variable configuration."""
    return SCMConfig(observed_variables=("A", "B", "C"))


@pytest.fixture
def config_with_latents() -> SCMConfig:
    """Configuration with latent variables."""
    return SCMConfig(
        observed_variables=("X", "Y", "Z"),
        max_latent_variables=2,
    )


@pytest.fixture
def rng() -> Random:
    """Seeded random number generator."""
    return Random(42)


# === SCMConfig Tests ===


class TestSCMConfig:
    """Tests for SCMConfig dataclass."""

    def test_basic_creation(self):
        """Test basic config creation with defaults."""
        config = SCMConfig(observed_variables=("A", "B", "C"))

        assert config.observed_variables == ("A", "B", "C")
        assert config.max_latent_variables == 3
        assert config.conflict_resolution == ConflictResolution.FIRST_WINS
        assert config.acyclicity_mode == AcyclicityMode.REJECT
        assert config.erc_count == 5

    def test_empty_observed_raises(self):
        """Test that empty observed_variables raises ValueError."""
        with pytest.raises(ValueError, match="observed_variables must be non-empty"):
            SCMConfig(observed_variables=())

    def test_negative_latent_raises(self):
        """Test that negative max_latent_variables raises ValueError."""
        with pytest.raises(ValueError, match="max_latent_variables must be >= 0"):
            SCMConfig(observed_variables=("A",), max_latent_variables=-1)

    def test_zero_erc_sigma_raises(self):
        """Test that zero ERC sigma raises ValueError."""
        with pytest.raises(ValueError, match="ERC sigma values must be positive"):
            SCMConfig(observed_variables=("A",), erc_sigma_init=0.0)

    def test_custom_penalties(self):
        """Test custom penalty configuration."""
        config = SCMConfig(
            observed_variables=("A",),
            cycle_penalty_per_cycle=2.0,
            conflict_penalty=0.5,
        )

        assert config.cycle_penalty_per_cycle == 2.0
        assert config.conflict_penalty == 0.5

    def test_frozen_dataclass(self, basic_config):
        """Test that config is immutable."""
        with pytest.raises(AttributeError):
            basic_config.max_latent_variables = 5


# === SCMAlphabet Tests ===


class TestSCMAlphabet:
    """Tests for SCMAlphabet generation."""

    def test_from_config_basic(self, basic_config):
        """Test alphabet generation from basic config."""
        alphabet = SCMAlphabet.from_config(basic_config)

        # Check variable refs
        assert "A" in alphabet.variable_refs
        assert "B" in alphabet.variable_refs
        assert "C" in alphabet.variable_refs
        assert "H1" in alphabet.variable_refs
        assert "H2" in alphabet.variable_refs
        assert "H3" in alphabet.variable_refs

        # Check STORE genes
        assert "STORE_A" in alphabet.store_genes
        assert "STORE_B" in alphabet.store_genes
        assert "STORE_H1" in alphabet.store_genes

        # Check operators
        assert "+" in alphabet.operators
        assert "-" in alphabet.operators
        assert "*" in alphabet.operators
        assert "/" in alphabet.operators

        # Check constants
        assert 0.0 in alphabet.constants
        assert 1.0 in alphabet.constants
        assert math.pi in alphabet.constants

        # Check ERC slots
        assert "ERC_0" in alphabet.erc_slots
        assert len(alphabet.erc_slots) == basic_config.erc_count

    def test_no_latents(self):
        """Test alphabet with no latent variables."""
        config = SCMConfig(observed_variables=("X",), max_latent_variables=0)
        alphabet = SCMAlphabet.from_config(config)

        assert "X" in alphabet.variable_refs
        assert "H1" not in alphabet.variable_refs
        assert "STORE_X" in alphabet.store_genes

    def test_all_variables_property(self, basic_config):
        """Test all_variables property."""
        alphabet = SCMAlphabet.from_config(basic_config)
        all_vars = alphabet.all_variables

        assert all_vars == alphabet.variable_refs


# === SCMGenome Tests ===


class TestSCMGenomeCreation:
    """Tests for SCMGenome creation."""

    def test_random_creation(self, basic_config, rng):
        """Test random genome creation."""
        genome = SCMGenome.random(basic_config, length=20, rng=rng)

        assert len(genome.genes) == 20
        assert genome.config == basic_config

    def test_random_erc_sampling(self, basic_config):
        """Test that ERC values are sampled from Gaussian."""
        Random(123)
        genomes = [SCMGenome.random(basic_config, length=50, rng=Random(i)) for i in range(100)]

        # Collect all ERC values
        all_erc_values = []
        for genome in genomes:
            all_erc_values.extend(val for _, val in genome.erc_values)

        if all_erc_values:
            # Check distribution is roughly centered at 0
            mean = sum(all_erc_values) / len(all_erc_values)
            assert abs(mean) < 1.0  # Should be near 0

    def test_deterministic_with_seed(self, basic_config):
        """Test that same seed produces same genome."""
        genome1 = SCMGenome.random(basic_config, length=20, rng=Random(42))
        genome2 = SCMGenome.random(basic_config, length=20, rng=Random(42))

        assert genome1.genes == genome2.genes
        assert genome1.erc_values == genome2.erc_values


class TestSCMGenomeCopy:
    """Tests for SCMGenome.copy()."""

    def test_copy_independence(self, basic_config, rng):
        """Test that copy creates independent genome."""
        original = SCMGenome.random(basic_config, length=20, rng=rng)
        copied = original.copy()

        assert original == copied
        assert original is not copied
        assert original.inner is not copied.inner

    def test_copy_preserves_values(self, basic_config, rng):
        """Test that copy preserves all values."""
        original = SCMGenome.random(basic_config, length=20, rng=rng)
        copied = original.copy()

        assert copied.genes == original.genes
        assert copied.config == original.config
        assert copied.erc_values == original.erc_values


class TestSCMGenomeEquality:
    """Tests for SCMGenome equality and hashing."""

    def test_equal_genomes(self, basic_config, _rng):
        """Test equality of identical genomes."""
        g1 = SCMGenome.random(basic_config, length=10, rng=Random(42))
        g2 = SCMGenome.random(basic_config, length=10, rng=Random(42))

        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_different_genes(self, basic_config):
        """Test inequality with different genes."""
        g1 = SCMGenome.random(basic_config, length=10, rng=Random(1))
        g2 = SCMGenome.random(basic_config, length=10, rng=Random(2))

        assert g1 != g2

    def test_hashable(self, basic_config, _rng):
        """Test that genomes can be used in sets."""
        g1 = SCMGenome.random(basic_config, length=10, rng=Random(42))
        g2 = SCMGenome.random(basic_config, length=10, rng=Random(42))
        g3 = SCMGenome.random(basic_config, length=10, rng=Random(99))

        genome_set = {g1, g2, g3}
        assert len(genome_set) == 2  # g1 and g2 are equal


class TestSCMGenomeERC:
    """Tests for ERC handling."""

    def test_get_erc_value(self, basic_config, rng):
        """Test getting ERC value by slot."""
        genome = SCMGenome.random(basic_config, length=50, rng=rng)

        if genome.erc_values:
            slot, expected = genome.erc_values[0]
            assert genome.get_erc_value(slot) == expected

    def test_get_missing_erc_raises(self, basic_config, rng):
        """Test that missing ERC slot raises KeyError."""
        genome = SCMGenome.random(basic_config, length=50, rng=rng)

        with pytest.raises(KeyError):
            genome.get_erc_value(99999)

    def test_erc_mutation(self, basic_config, rng):
        """Test ERC perturbation mutation."""
        genome = SCMGenome.random(basic_config, length=50, rng=rng)

        if genome.erc_values:
            mutated = genome.mutate_erc(Random(99))

            # Values should be different
            assert mutated.erc_values != genome.erc_values
            # Structure preserved
            assert len(mutated.erc_values) == len(genome.erc_values)

    def test_erc_mutation_single_slot(self, basic_config, rng):
        """Test mutating single ERC slot."""
        genome = SCMGenome.random(basic_config, length=50, rng=rng)

        if len(genome.erc_values) >= 2:
            slot_to_mutate = genome.erc_values[0][0]
            mutated = genome.mutate_erc(Random(99), slot=slot_to_mutate)

            # First slot should change
            assert mutated.erc_values[0][1] != genome.erc_values[0][1]
            # Other slots should remain same
            assert mutated.erc_values[1] == genome.erc_values[1]


class TestSCMGenomeSerialization:
    """Tests for SCMGenome serialization."""

    def test_to_dict(self, basic_config, rng):
        """Test serialization to dict."""
        genome = SCMGenome.random(basic_config, length=20, rng=rng)
        data = genome.to_dict()

        assert data["type"] == "SCMGenome"
        assert data["version"] == "1.0"
        assert len(data["genes"]) == 20
        assert "config" in data

    def test_from_dict(self, basic_config, rng):
        """Test deserialization from dict."""
        genome = SCMGenome.random(basic_config, length=20, rng=rng)
        data = genome.to_dict()
        restored = SCMGenome.from_dict(data)

        assert restored == genome

    def test_round_trip(self, basic_config, rng):
        """Test full serialization round-trip."""
        original = SCMGenome.random(basic_config, length=20, rng=rng)
        restored = SCMGenome.from_dict(original.to_dict())

        assert original == restored
        assert original.genes == restored.genes
        assert original.erc_values == restored.erc_values

    def test_invalid_type_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Expected type 'SCMGenome'"):
            SCMGenome.from_dict({"type": "WrongType"})

    def test_invalid_version_raises(self, basic_config, rng):
        """Test that unsupported version raises ValueError."""
        genome = SCMGenome.random(basic_config, length=10, rng=rng)
        data = genome.to_dict()
        data["version"] = "99.0"

        with pytest.raises(ValueError, match="Unsupported version"):
            SCMGenome.from_dict(data)


class TestSCMGenomeAlphabetCompatibility:
    """Tests for alphabet compatibility with SequenceGenome."""

    def test_genes_from_alphabet(self, basic_config, rng):
        """Test that all genes come from alphabet."""
        alphabet = SCMAlphabet.from_config(basic_config)
        genome = SCMGenome.random(basic_config, length=100, rng=rng)

        valid_genes = alphabet.all_gene_symbols
        for gene in genome.genes:
            # Gene should be in alphabet or a float (ERC value reference)
            if isinstance(gene, str):
                assert gene in valid_genes or gene.startswith("ERC_"), f"Invalid gene: {gene}"


# === SCM Distance and Matchability Tests (T092-T093) ===


class TestSCMSequenceDistance:
    """Tests for SCM sequence-only distance computation (T092)."""

    def test_identical_genomes_have_zero_distance(self, basic_config, rng):
        """Test that identical genomes have zero sequence distance."""
        from evolve.representation.scm import scm_sequence_distance

        genome = SCMGenome.random(basic_config, length=20, rng=rng)
        distance = scm_sequence_distance(genome, genome)
        assert distance == 0.0

    def test_different_genomes_have_positive_distance(self, basic_config, _rng):
        """Test that different genomes have positive sequence distance."""
        from evolve.representation.scm import scm_sequence_distance

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))

        distance = scm_sequence_distance(genome_a, genome_b)
        assert distance > 0.0

    def test_distance_is_symmetric(self, basic_config, _rng):
        """Test that distance(a, b) == distance(b, a)."""
        from evolve.representation.scm import scm_sequence_distance

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))

        dist_ab = scm_sequence_distance(genome_a, genome_b)
        dist_ba = scm_sequence_distance(genome_b, genome_a)

        assert dist_ab == dist_ba

    def test_distance_is_normalized(self, basic_config, _rng):
        """Test that distance is normalized to [0, 1]."""
        from evolve.representation.scm import scm_sequence_distance

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))

        distance = scm_sequence_distance(genome_a, genome_b)
        assert 0.0 <= distance <= 1.0


class TestSCMStructuralDistance:
    """Tests for SCM structural distance computation (T093)."""

    def test_identical_structures_have_zero_distance(self, basic_config, rng):
        """Test that identical decoded structures have zero structural distance."""
        from evolve.representation.scm import scm_structural_distance
        from evolve.representation.scm_decoder import SCMDecoder

        genome = SCMGenome.random(basic_config, length=20, rng=rng)
        decoder = SCMDecoder(basic_config)

        distance = scm_structural_distance(genome, genome, decoder)
        assert distance == 0.0

    def test_different_structures_have_positive_distance(self, basic_config):
        """Test that different structures have positive distance."""
        from evolve.representation.scm import scm_structural_distance
        from evolve.representation.scm_decoder import SCMDecoder

        # Create genomes with different structures
        # genome_a: A = B (one edge B -> A)
        genes_a = ("B", "STORE_A")
        genome_a = SCMGenome(
            inner=SequenceGenome(genes=genes_a),
            config=basic_config,
            erc_values=(),
        )

        # genome_b: B = C (one edge C -> B)
        genes_b = ("C", "STORE_B")
        genome_b = SCMGenome(
            inner=SequenceGenome(genes=genes_b),
            config=basic_config,
            erc_values=(),
        )

        decoder = SCMDecoder(basic_config)
        distance = scm_structural_distance(genome_a, genome_b, decoder)
        assert distance > 0.0

    def test_structural_distance_reflects_graph_differences(self, basic_config):
        """Test that structural distance increases with more edge differences."""
        from evolve.representation.scm import scm_structural_distance
        from evolve.representation.scm_decoder import SCMDecoder

        # genome_a: A = B + C (two edges)
        genes_a = ("B", "C", "+", "STORE_A")
        genome_a = SCMGenome(
            inner=SequenceGenome(genes=genes_a),
            config=basic_config,
            erc_values=(),
        )

        # genome_b: A = B (one edge)
        genes_b = ("B", "STORE_A")
        genome_b = SCMGenome(
            inner=SequenceGenome(genes=genes_b),
            config=basic_config,
            erc_values=(),
        )

        # genome_c: empty (no edges)
        genes_c = ("1", "2", "+")  # Just junk - no STORE
        genome_c = SCMGenome(
            inner=SequenceGenome(genes=genes_c),
            config=basic_config,
            erc_values=(),
        )

        decoder = SCMDecoder(basic_config)

        dist_ab = scm_structural_distance(genome_a, genome_b, decoder)
        dist_ac = scm_structural_distance(genome_a, genome_c, decoder)

        # Genome C is more different from A than B is
        assert dist_ac > dist_ab


class TestSCMCombinedDistance:
    """Tests for combined SCM distance with structural_weight (T093)."""

    def test_weight_zero_uses_sequence_only(self, basic_config, _rng):
        """Test that structural_weight=0 gives sequence-only distance."""
        from evolve.representation.scm import scm_distance, scm_sequence_distance
        from evolve.representation.scm_decoder import SCMDecoder

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))
        decoder = SCMDecoder(basic_config)

        combined = scm_distance(genome_a, genome_b, decoder, structural_weight=0.0)
        sequence_only = scm_sequence_distance(genome_a, genome_b)

        assert combined == pytest.approx(sequence_only, abs=1e-10)

    def test_weight_one_uses_structural_only(self, basic_config, _rng):
        """Test that structural_weight=1.0 gives structural-only distance."""
        from evolve.representation.scm import scm_distance, scm_structural_distance
        from evolve.representation.scm_decoder import SCMDecoder

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))
        decoder = SCMDecoder(basic_config)

        combined = scm_distance(genome_a, genome_b, decoder, structural_weight=1.0)
        structural_only = scm_structural_distance(genome_a, genome_b, decoder)

        assert combined == pytest.approx(structural_only, abs=1e-10)

    def test_weight_half_averages_both(self, basic_config, _rng):
        """Test that structural_weight=0.5 averages sequence and structural."""
        from evolve.representation.scm import (
            scm_distance,
            scm_sequence_distance,
            scm_structural_distance,
        )
        from evolve.representation.scm_decoder import SCMDecoder

        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))
        decoder = SCMDecoder(basic_config)

        combined = scm_distance(genome_a, genome_b, decoder, structural_weight=0.5)
        seq_dist = scm_sequence_distance(genome_a, genome_b)
        struct_dist = scm_structural_distance(genome_a, genome_b, decoder)

        expected = 0.5 * seq_dist + 0.5 * struct_dist
        assert combined == pytest.approx(expected, abs=1e-10)

    def test_similar_sequences_different_structures_affects_matchability(self, basic_config):
        """
        Test acceptance scenario: similar sequences but different decoded graphs
        reduces matchability when structural_weight > 0.
        """
        from evolve.representation.scm import scm_distance
        from evolve.representation.scm_decoder import SCMDecoder

        # Create two genomes with same genes but ERC values that may decode differently
        # This is harder to construct, so we use manually crafted genes

        # Both have same sequence but decode to different structures
        # This requires the same genes to produce different graph structures
        # Actually, same genes WILL produce same structure since decoding is deterministic
        # So let's test with slight sequence difference but major structural difference

        # genome_a: A = B (edge B -> A)
        genes_a = ("B", "STORE_A", "1", "2")  # Padding
        genome_a = SCMGenome(
            inner=SequenceGenome(genes=genes_a),
            config=basic_config,
            erc_values=(),
        )

        # genome_b: C = A (edge A -> C) - different structure, similar length
        genes_b = ("A", "STORE_C", "1", "2")  # Similar sequence
        genome_b = SCMGenome(
            inner=SequenceGenome(genes=genes_b),
            config=basic_config,
            erc_values=(),
        )

        decoder = SCMDecoder(basic_config)

        # With structural_weight=0, only sequence matters
        dist_seq_only = scm_distance(genome_a, genome_b, decoder, structural_weight=0.0)

        # With structural_weight=0.5, structure contributes
        dist_with_struct = scm_distance(genome_a, genome_b, decoder, structural_weight=0.5)

        # The distances should differ since structure is different
        # In this case, structure difference should increase overall distance
        assert dist_with_struct != dist_seq_only
