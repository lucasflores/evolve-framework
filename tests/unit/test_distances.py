"""Distances by name: the "distance" category of the operator registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evolve.config.meta import ParameterSpec
from evolve.diversity.speciation import GenomeDistance, NEATDistance, neat_distance
from evolve.meta.codec import ParameterDecoder, ParameterDistance
from evolve.registry.operators import get_operator_registry, reset_operator_registry
from evolve.representation.graph import ConnectionGene, GraphGenome, NodeGene
from evolve.representation.vector import VectorGenome

ENC = ParameterSpec(path="enc", param_type="categorical", choices=("a", "b", "c", "d"))
LR = ParameterSpec(path="lr", bounds=(1e-4, 1e-1), log_scale=True)
EPOCHS = ParameterSpec(path="epochs", param_type="integer", bounds=(1, 5))
POOL = ParameterSpec(path="pool", param_type="subset", choices=("x", "y", "z", "w"))
FLAG = ParameterSpec(path="flag", param_type="categorical", choices=(False, True))
GATED = ParameterSpec(path="t", bounds=(0.0, 1.0), parent="flag", active_values=(True,))


def _v(*genes: float) -> VectorGenome:
    return VectorGenome(genes=np.array(genes, dtype=float))


@pytest.fixture(autouse=True)
def _reset():
    reset_operator_registry()
    yield
    reset_operator_registry()


class TestParameterDistance:
    """Gower distance over decoded values: one share per gene, averaged."""

    SPECS = (ENC, LR, EPOCHS, POOL)

    def _d(self, a: VectorGenome, b: VectorGenome, specs=None) -> float:
        return ParameterDistance(decoder=ParameterDecoder(specs or self.SPECS))(a, b)

    def test_genomes_decoding_alike_are_at_zero(self) -> None:
        # Different positions, same bins: enc "a", same lr, epochs 3, pool {x}
        a = _v(0.05, 0.5, 0.5, 0.9, 0.1, 0.2, 0.3)
        b = _v(0.2, 0.5, 0.55, 0.6, 0.4, 0.0, 0.49)
        assert self._d(a, b) == 0.0

    def test_each_kind_contributes_its_share(self) -> None:
        a = _v(0.1, 0.0, 0.0, 0.9, 0.9, 0.1, 0.1)  # a, lr 1e-4, epochs 1, {x, y}
        b = _v(0.4, 1.0, 0.25, 0.9, 0.1, 0.9, 0.1)  # b, lr 1e-1, epochs 2, {x, z}
        # categorical 1; log-scale lr across the whole range 1; epochs 1 of 4
        # steps 0.25; pool Jaccard 1 - 1/3
        expected = (1 + 1 + 0.25 + (1 - 1 / 3)) / 4
        assert self._d(a, b) == pytest.approx(expected)
        assert self._d(b, a) == pytest.approx(expected)

    def test_log_scale_compares_logs(self) -> None:
        a = _v(0.1, 0.0, 0.5, 0.9, 0.1, 0.1, 0.1)  # lr 1e-4
        b = _v(0.1, 1 / 3, 0.5, 0.9, 0.1, 0.1, 0.1)  # lr 1e-3: a third of the way in logs
        assert self._d(a, b) == pytest.approx((1 / 3) / 4)

    def test_two_empty_subsets_are_equal(self) -> None:
        empty = _v(0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1)
        assert self._d(empty, empty) == 0.0

    def test_a_gene_active_in_one_only_counts_one(self) -> None:
        specs = (FLAG, GATED)
        off = _v(0.25, 0.3)  # flag False, t inactive
        on = _v(0.75, 0.3)  # flag True, t 0.3
        # flag differs (1) and t applies to one only (1), over the two genes
        assert self._d(off, on, specs) == 1.0
        # neither has t: only flag counts
        assert self._d(off, _v(0.25, 0.9), specs) == 0.0

    def test_decodes_each_genome_once_and_bounds_what_it_keeps(self) -> None:
        distance = ParameterDistance(decoder=ParameterDecoder(self.SPECS))
        a, b = _v(0.1, 0.0, 0.0, 0.9, 0.9, 0.1, 0.1), _v(0.4, 1.0, 0.25, 0.9, 0.1, 0.9, 0.1)
        for _ in range(5):
            distance(a, b)
        info = distance._values.cache_info()
        assert (info.misses, info.hits) == (2, 8)
        assert info.maxsize == 10_000

    def test_takes_a_decoder_exposing_one(self) -> None:
        class Wrapper:
            parameter_decoder = ParameterDecoder((ENC,))

        assert ParameterDistance(decoder=Wrapper())(_v(0.1), _v(0.9)) == 1.0

    def test_refuses_without_a_parameter_decoder(self) -> None:
        with pytest.raises(ValueError, match="needs a decoder built on ParameterDecoder"):
            ParameterDistance(decoder=None)


class TestGenomeAndNEATDistances:
    def test_genome_distance_is_the_genomes_own(self) -> None:
        a, b = _v(0.0, 0.0), _v(3.0, 4.0)
        assert GenomeDistance()(a, b) == 5.0 == a.distance(b)

    def test_neat_distance_with_its_coefficients(self) -> None:
        nodes = frozenset({NodeGene(id=0, node_type="input"), NodeGene(id=1, node_type="output")})
        a = GraphGenome(
            nodes=nodes,
            connections=frozenset(
                {ConnectionGene(innovation=0, from_node=0, to_node=1, weight=1.0, enabled=True)}
            ),
            input_ids=(0,),
            output_ids=(1,),
        )
        b = GraphGenome(
            nodes=nodes,
            connections=frozenset(
                {ConnectionGene(innovation=0, from_node=0, to_node=1, weight=2.0, enabled=True)}
            ),
            input_ids=(0,),
            output_ids=(1,),
        )
        assert NEATDistance(c_weight=0.5)(a, b) == neat_distance(a, b, c_weight=0.5) == 0.5


class TestRegistry:
    def test_distances_are_registered_by_name(self) -> None:
        registry = get_operator_registry()
        assert sorted(registry.list_operators("distance")) == ["genome", "neat", "parameters"]
        assert "distance" in registry.list_all()

    def test_compatibility(self) -> None:
        registry = get_operator_registry()
        assert registry.is_compatible("genome", "vector")
        assert registry.is_compatible("genome", "sequence")
        assert not registry.is_compatible("genome", "graph")
        assert registry.is_compatible("neat", "graph")
        assert registry.is_compatible("parameters", "vector")
        assert not registry.is_compatible("parameters", "graph")

    def test_a_name_in_two_categories_keeps_each_compatibility(self) -> None:
        registry = get_operator_registry()
        registry.register("distance", "gaussian", GenomeDistance)  # compatible with all
        assert not registry.is_compatible("gaussian", "graph", "mutation")
        assert registry.get_compatibility("gaussian", "mutation") == {"vector"}
        assert registry.is_compatible("gaussian", "graph", "distance")

    def test_parameters_takes_the_decoder(self) -> None:
        registry = get_operator_registry()
        assert registry.accepts_param("distance", "parameters", "decoder")
        assert not registry.accepts_param("distance", "genome", "decoder")
        distance = registry.get("distance", "parameters", decoder=ParameterDecoder((ENC,)))
        assert distance(_v(0.1), _v(0.3)) == 1.0

    def test_neat_takes_its_coefficients(self) -> None:
        distance = get_operator_registry().get("distance", "neat", c_weight=0.1)
        assert math.isclose(distance.c_weight, 0.1)
