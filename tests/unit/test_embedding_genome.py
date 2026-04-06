"""Unit tests for EmbeddingGenome — protocol compliance, immutability, serialization."""

from __future__ import annotations

import ast

import numpy as np
import pytest

from evolve.representation.embedding import EmbeddingGenome
from evolve.representation.embedding_config import (
    DimensionalityStrategy,
    EmbeddingGenomeConfig,
)
from evolve.representation.genome import Genome, SerializableGenome
from evolve.representation.vector import VectorGenome

# ── Protocol Compliance (T009) ──────────────────────────────────────


class TestEmbeddingGenomeProtocol:
    def test_implements_genome_protocol(self, sample_embedding_genome: EmbeddingGenome) -> None:
        assert isinstance(sample_embedding_genome, Genome)

    def test_implements_serializable_protocol(
        self, sample_embedding_genome: EmbeddingGenome
    ) -> None:
        assert isinstance(sample_embedding_genome, SerializableGenome)

    def test_copy_equals_original(self, sample_embedding_genome: EmbeddingGenome) -> None:
        copied = sample_embedding_genome.copy()
        assert copied == sample_embedding_genome
        assert copied is not sample_embedding_genome

    def test_copy_is_deep(self, sample_embedding_genome: EmbeddingGenome) -> None:
        copied = sample_embedding_genome.copy()
        assert not np.shares_memory(copied.embeddings, sample_embedding_genome.embeddings)

    def test_hash_consistent_with_eq(self, sample_embedding_genome: EmbeddingGenome) -> None:
        copied = sample_embedding_genome.copy()
        assert hash(copied) == hash(sample_embedding_genome)

    def test_different_genomes_not_equal(self, np_rng: np.random.Generator) -> None:
        g1 = EmbeddingGenome(
            embeddings=np_rng.standard_normal((4, 16)).astype(np.float32),
            model_id="model-a",
        )
        g2 = EmbeddingGenome(
            embeddings=np_rng.standard_normal((4, 16)).astype(np.float32),
            model_id="model-a",
        )
        assert g1 != g2

    def test_different_model_id_not_equal(self, sample_embedding_genome: EmbeddingGenome) -> None:
        other = EmbeddingGenome(
            embeddings=sample_embedding_genome.embeddings.copy(),
            model_id="other-model",
        )
        assert sample_embedding_genome != other


class TestEmbeddingGenomeImmutability:
    def test_embeddings_not_writeable(self, sample_embedding_genome: EmbeddingGenome) -> None:
        assert not sample_embedding_genome.embeddings.flags.writeable

    def test_frozen_dataclass(self, sample_embedding_genome: EmbeddingGenome) -> None:
        with pytest.raises(AttributeError):
            sample_embedding_genome.model_id = "new"  # type: ignore[misc]


class TestEmbeddingGenomeProperties:
    def test_n_tokens(self, sample_embedding_genome: EmbeddingGenome) -> None:
        assert sample_embedding_genome.n_tokens == 4

    def test_embed_dim(self, sample_embedding_genome: EmbeddingGenome) -> None:
        assert sample_embedding_genome.embed_dim == 16

    def test_shape_from_embeddings(self) -> None:
        g = EmbeddingGenome(
            embeddings=np.zeros((3, 128), dtype=np.float32),
            model_id="m",
        )
        assert g.n_tokens == 3
        assert g.embed_dim == 128


class TestEmbeddingGenomeValidation:
    def test_1d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            EmbeddingGenome(embeddings=np.zeros(10), model_id="m")

    def test_3d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            EmbeddingGenome(embeddings=np.zeros((2, 3, 4)), model_id="m")

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            EmbeddingGenome(embeddings=np.zeros((2, 4)), model_id="")

    def test_zero_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="both dims >= 1"):
            EmbeddingGenome(embeddings=np.zeros((0, 4)), model_id="m")

    def test_zero_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="both dims >= 1"):
            EmbeddingGenome(embeddings=np.zeros((4, 0)), model_id="m")


class TestEmbeddingGenomeSerialization:
    def test_roundtrip(self, sample_embedding_genome: EmbeddingGenome) -> None:
        d = sample_embedding_genome.to_dict()
        restored = EmbeddingGenome.from_dict(d)
        assert restored == sample_embedding_genome

    def test_dict_structure(self, sample_embedding_genome: EmbeddingGenome) -> None:
        d = sample_embedding_genome.to_dict()
        assert "embeddings" in d
        assert "model_id" in d
        assert "strategy" in d
        assert "n_tokens" in d
        assert "embed_dim" in d
        assert d["n_tokens"] == 4
        assert d["embed_dim"] == 16

    def test_from_dict_with_seed_text(self) -> None:
        g = EmbeddingGenome(
            embeddings=np.ones((2, 4), dtype=np.float32),
            model_id="test",
            seed_text="hello",
        )
        d = g.to_dict()
        restored = EmbeddingGenome.from_dict(d)
        assert restored.seed_text == "hello"


# ── Flat-Vector Adapter (T009 + contract) ───────────────────────────


class TestFlatVectorAdapter:
    def test_flat_shape(self, sample_embedding_genome: EmbeddingGenome) -> None:
        flat = sample_embedding_genome.flat()
        assert flat.shape == (4 * 16,)

    def test_flat_immutable(self, sample_embedding_genome: EmbeddingGenome) -> None:
        flat = sample_embedding_genome.flat()
        assert not flat.flags.writeable

    def test_to_vector_genome_type(self, sample_embedding_genome: EmbeddingGenome) -> None:
        vg = sample_embedding_genome.to_vector_genome()
        assert isinstance(vg, VectorGenome)
        assert len(vg.genes) == 4 * 16

    def test_roundtrip_via_vector_genome(self, sample_embedding_genome: EmbeddingGenome) -> None:
        vg = sample_embedding_genome.to_vector_genome()
        restored = EmbeddingGenome.from_vector_genome(
            vg,
            n_tokens=sample_embedding_genome.n_tokens,
            model_id=sample_embedding_genome.model_id,
            seed_text=sample_embedding_genome.seed_text,
            strategy=sample_embedding_genome.strategy,
        )
        assert np.allclose(restored.embeddings, sample_embedding_genome.embeddings, atol=1e-6)

    def test_from_vector_genome_bad_length(self) -> None:
        vg = VectorGenome(genes=np.zeros(10))
        with pytest.raises(ValueError, match="not divisible"):
            EmbeddingGenome.from_vector_genome(vg, n_tokens=3, model_id="m")


# ── DimensionalityStrategy & Config (T010) ─────────────────────────


class TestDimensionalityStrategy:
    def test_enum_values(self) -> None:
        assert DimensionalityStrategy.FULL_SPACE.value == "FULL_SPACE"
        assert DimensionalityStrategy.COMPRESSED_SUBSPACE.value == "COMPRESSED_SUBSPACE"
        assert DimensionalityStrategy.MINIMAL_TOKENS.value == "MINIMAL_TOKENS"


class TestEmbeddingGenomeConfig:
    def test_defaults(self) -> None:
        cfg = EmbeddingGenomeConfig(embed_dim=768, model_id="m")
        assert cfg.n_tokens == 8
        assert cfg.strategy == DimensionalityStrategy.MINIMAL_TOKENS
        assert cfg.coherence_radius == 0.1

    def test_bad_n_tokens(self) -> None:
        with pytest.raises(ValueError, match="n_tokens"):
            EmbeddingGenomeConfig(n_tokens=0, embed_dim=768, model_id="m")

    def test_bad_embed_dim(self) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            EmbeddingGenomeConfig(n_tokens=4, embed_dim=0, model_id="m")

    def test_empty_model_id(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            EmbeddingGenomeConfig(embed_dim=768, model_id="")

    def test_compressed_requires_subspace_dim(self) -> None:
        with pytest.raises(ValueError, match="subspace_dim"):
            EmbeddingGenomeConfig(
                embed_dim=768,
                model_id="m",
                strategy=DimensionalityStrategy.COMPRESSED_SUBSPACE,
            )

    def test_compressed_requires_projection(self) -> None:
        with pytest.raises(ValueError, match="projection_matrix"):
            EmbeddingGenomeConfig(
                embed_dim=768,
                model_id="m",
                strategy=DimensionalityStrategy.COMPRESSED_SUBSPACE,
                subspace_dim=32,
            )

    def test_compressed_valid(self) -> None:
        proj = np.random.randn(32, 768)
        cfg = EmbeddingGenomeConfig(
            embed_dim=768,
            model_id="m",
            strategy=DimensionalityStrategy.COMPRESSED_SUBSPACE,
            subspace_dim=32,
            projection_matrix=proj,
        )
        assert cfg.subspace_dim == 32

    def test_minimal_tokens_warning(self) -> None:
        with pytest.warns(UserWarning, match="4-8 tokens"):
            EmbeddingGenomeConfig(n_tokens=20, embed_dim=768, model_id="m")

    def test_roundtrip(self) -> None:
        cfg = EmbeddingGenomeConfig(
            n_tokens=6,
            embed_dim=512,
            model_id="test",
            seed_text="hello",
        )
        d = cfg.to_dict()
        restored = EmbeddingGenomeConfig.from_dict(d)
        assert restored.n_tokens == cfg.n_tokens
        assert restored.embed_dim == cfg.embed_dim
        assert restored.model_id == cfg.model_id
        assert restored.seed_text == cfg.seed_text


# ── Import Boundary (T011 / SC-009) ────────────────────────────────


class TestImportBoundary:
    """Verify embedding.py imports no ML frameworks (FR-016)."""

    FORBIDDEN_MODULES = {"torch", "transformers", "tensorflow", "jax"}

    def test_no_ml_imports(self) -> None:
        import evolve.representation.embedding as mod

        source_path = mod.__file__
        assert source_path is not None

        with open(source_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in self.FORBIDDEN_MODULES, (
                        f"embedding.py imports forbidden module: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                top = node.module.split(".")[0]
                assert top not in self.FORBIDDEN_MODULES, (
                    f"embedding.py imports from forbidden module: {node.module}"
                )
