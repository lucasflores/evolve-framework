"""
Pytest configuration and shared fixtures.

This file provides common test fixtures and configuration
for the entire test suite.
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.utils.random import create_rng

# ============================================================================
# Random/Determinism Fixtures
# ============================================================================


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def rng(seed: int) -> Random:
    """Seeded random number generator."""
    return create_rng(seed)


@pytest.fixture
def np_rng(seed: int) -> np.random.Generator:
    """Seeded NumPy random generator."""
    return np.random.default_rng(seed)


# ============================================================================
# Genome Fixtures
# ============================================================================


@pytest.fixture
def vector_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Standard bounds for 10-dimensional vector optimization."""
    lower = np.full(10, -5.0)
    upper = np.full(10, 5.0)
    return (lower, upper)


@pytest.fixture
def random_genes(
    np_rng: np.random.Generator, vector_bounds: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Random genes within bounds."""
    lower, upper = vector_bounds
    return np_rng.uniform(lower, upper)


# ============================================================================
# ESPO Fixtures
# ============================================================================


@pytest.fixture
def embedding_genome_config():
    """Standard EmbeddingGenomeConfig for tests."""
    from evolve.representation.embedding_config import EmbeddingGenomeConfig

    return EmbeddingGenomeConfig(
        n_tokens=4,
        embed_dim=16,
        model_id="test-model",
    )


@pytest.fixture
def sample_embedding_genome(np_rng: np.random.Generator):
    """A small EmbeddingGenome for tests (4 tokens x 16 dim)."""
    from evolve.representation.embedding import EmbeddingGenome

    embeddings = np_rng.standard_normal((4, 16)).astype(np.float32)
    return EmbeddingGenome(
        embeddings=embeddings,
        model_id="test-model",
        seed_text="test prompt",
    )


@pytest.fixture
def sample_task_spec():
    """A minimal TaskSpec for tests."""
    from evolve.evaluation.task_spec import TaskSpec

    return TaskSpec(
        task_type="qa",
        inputs=(
            {"input": "What is 2+2?"},
            {"input": "What is the capital of France?"},
        ),
        ground_truth=("4", "Paris"),
    )


# ============================================================================
# Population Fixtures
# ============================================================================


@pytest.fixture
def population_size() -> int:
    """Standard population size for tests."""
    return 50


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "property: marks property-based tests")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


# ============================================================================
# Hypothesis Settings
# ============================================================================

from hypothesis import Verbosity, settings  # noqa: E402

# Register slower profile for CI
settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=50)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
