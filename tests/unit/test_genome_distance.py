"""
Tests for genome distance methods.

Covers T021 (L2 for VectorGenome, edit distance for SequenceGenome, type mismatch).
"""

from __future__ import annotations

import numpy as np
import pytest

from evolve.representation.sequence import SequenceGenome
from evolve.representation.vector import VectorGenome


class TestVectorGenomeDistance:
    """T019/T021: VectorGenome.distance() with L2 norm."""

    def test_identical_genomes_zero_distance(self):
        g = VectorGenome(genes=np.array([1.0, 2.0, 3.0]))
        assert g.distance(g.copy()) == 0.0

    def test_known_l2_distance(self):
        g1 = VectorGenome(genes=np.array([0.0, 0.0]))
        g2 = VectorGenome(genes=np.array([3.0, 4.0]))
        assert g1.distance(g2) == pytest.approx(5.0)

    def test_distance_is_symmetric(self):
        g1 = VectorGenome(genes=np.array([1.0, 2.0, 3.0]))
        g2 = VectorGenome(genes=np.array([4.0, 5.0, 6.0]))
        assert g1.distance(g2) == pytest.approx(g2.distance(g1))

    def test_type_mismatch_raises(self):
        g = VectorGenome(genes=np.array([1.0]))
        with pytest.raises(TypeError):
            g.distance(SequenceGenome(genes=(1, 2)))  # type: ignore[arg-type]


class TestSequenceGenomeDistance:
    """T020/T021: SequenceGenome.distance() with edit distance."""

    def test_identical_sequences_zero_distance(self):
        g = SequenceGenome(genes=(1, 2, 3))
        assert g.distance(g.copy()) == 0.0

    def test_single_substitution(self):
        g1 = SequenceGenome(genes=("a", "b", "c"))
        g2 = SequenceGenome(genes=("a", "x", "c"))
        assert g1.distance(g2) == 1.0

    def test_insertion(self):
        g1 = SequenceGenome(genes=("a", "b"))
        g2 = SequenceGenome(genes=("a", "b", "c"), max_length=5)
        assert g1.distance(g2) == 1.0

    def test_completely_different(self):
        g1 = SequenceGenome(genes=("a", "b", "c"))
        g2 = SequenceGenome(genes=("x", "y", "z"))
        assert g1.distance(g2) == 3.0

    def test_empty_vs_nonempty(self):
        g1 = SequenceGenome(genes=(), min_length=0)
        g2 = SequenceGenome(genes=("a", "b", "c"), min_length=0)
        assert g1.distance(g2) == 3.0

    def test_distance_is_symmetric(self):
        g1 = SequenceGenome(genes=(1, 2, 3))
        g2 = SequenceGenome(genes=(1, 3, 2))
        assert g1.distance(g2) == g2.distance(g1)

    def test_type_mismatch_raises(self):
        g = SequenceGenome(genes=(1, 2))
        with pytest.raises(TypeError):
            g.distance(VectorGenome(genes=np.array([1.0])))  # type: ignore[arg-type]
