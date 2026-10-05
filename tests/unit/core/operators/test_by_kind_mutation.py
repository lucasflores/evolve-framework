"""Tests for ByKindMutation, the mutation that moves each gene by its kind."""

from __future__ import annotations

import math
from collections import Counter
from random import Random
from typing import Any

import numpy as np
import pytest

from evolve.config.meta import ParameterSpec
from evolve.config.unified import UnifiedConfig
from evolve.core.operators.mutation import ByKindMutation
from evolve.factory.engine import create_engine, create_initial_population
from evolve.meta.codec import ParameterDecoder
from evolve.representation.vector import VectorGenome

CONT = ParameterSpec(path="lr", bounds=(0.0, 1.0))
INT = ParameterSpec(path="epochs", param_type="integer", bounds=(1, 5))
WIDE = ParameterSpec(path="dim", param_type="integer", bounds=(16, 512))
CAT = ParameterSpec(path="enc", param_type="categorical", choices=("a", "b", "c"))
FLAG = ParameterSpec(path="flag", param_type="categorical", choices=(False, True))
GATED = ParameterSpec(
    path="centred",
    param_type="categorical",
    choices=(False, True),
    parent="flag",
    active_values=(True,),
)
LENGTH = ParameterSpec(
    path="length",
    param_type="categorical",
    parent="enc",
    choices_by_parent={"a": (128,), "b": (256, 512, 1024)},
)
POOL = ParameterSpec(path="pool", param_type="subset", choices=("x", "y", "z"))
ONE = ParameterSpec(path="one", param_type="categorical", choices=("only",))
FIXED = ParameterSpec(path="fixed", bounds=(2.0, 2.0))


def _op(specs: tuple[ParameterSpec, ...], **params: Any) -> ByKindMutation:
    return ByKindMutation(decoder=ParameterDecoder(specs), **params)


def _genome(*positions: float) -> VectorGenome:
    n = len(positions)
    return VectorGenome(genes=np.array(positions, dtype=float), bounds=(np.zeros(n), np.ones(n)))


def _within(observed: int, n: int, p: float) -> bool:
    """Observed count within 4.5 standard deviations of a binomial(n, p)."""
    return abs(observed - n * p) <= 4.5 * math.sqrt(n * p * (1 - p)) + 1


class TestRates:
    """Every active gene changes at exactly its kind's rate."""

    SPECS = (CONT, INT, CAT, FLAG, POOL)

    @pytest.mark.parametrize(
        "positions",
        [
            # mid-bin: as ibis places a known candidate
            (0.5, 0.5, 0.5, 0.25, 0.75, 0.25, 0.25),
            # beside a bin boundary, or at a bound
            (0.99, 0.0, 0.01, 0.49, 0.51, 0.49, 0.0),
        ],
    )
    def test_each_gene_changes_at_its_rate_wherever_it_sits(
        self, positions: tuple[float, ...]
    ) -> None:
        decoder = ParameterDecoder(self.SPECS)
        op = ByKindMutation(decoder=decoder, mutation_rate=0.3, discrete_rate=0.15)
        parent = _genome(*positions)
        before = decoder.values(parent.genes.tolist())
        rng, n = Random(11), 20_000
        changed: Counter[str] = Counter()
        for _ in range(n):
            after = decoder.values(op.mutate(parent, rng).genes.tolist())
            changed.update(path for path in before if after[path] != before[path])
        expected = {"lr": 0.3, "epochs": 0.15, "enc": 0.15, "flag": 0.15, "pool": 0.15}
        for path, p in expected.items():
            assert _within(changed[path], n, p), (path, changed[path] / n)

    def test_discrete_rate_defaults_to_mutation_rate(self) -> None:
        op = _op(self.SPECS, mutation_rate=0.4)
        assert op.discrete_rate == 0.4


class TestInteger:
    """An integer moves at least one step, stays in range, and steps scale with it."""

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_always_moves_and_stays_in_range(self, value: int) -> None:
        decoder = ParameterDecoder((INT,))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(value)
        parent = _genome((value - 1) / 4)
        for _ in range(300):
            got = decoder.values(op.mutate(parent, rng).genes.tolist())["epochs"]
            assert got != value
            assert 1 <= got <= 5

    def test_a_narrow_range_moves_one_step(self) -> None:
        decoder = ParameterDecoder((INT,))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(0)
        steps = [
            abs(decoder.values(op.mutate(_genome(0.5), rng).genes.tolist())["epochs"] - 3)
            for _ in range(2000)
        ]
        assert sum(s == 1 for s in steps) / len(steps) > 0.95

    def test_a_wide_range_takes_proportionate_steps(self) -> None:
        decoder = ParameterDecoder((WIDE,))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0, sigma=0.1)
        rng = Random(0)
        steps = [
            abs(decoder.values(op.mutate(_genome(0.5), rng).genes.tolist())["dim"] - 264)
            for _ in range(2000)
        ]
        # E|N(0, 0.1 * 496)| is about 39.6
        assert 30 < sum(steps) / len(steps) < 50


class TestCategorical:
    """A categorical moves to a different option among those open to it."""

    def test_lands_on_each_other_option_alike(self) -> None:
        decoder = ParameterDecoder((CAT,))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(0)
        got = Counter(
            decoder.values(op.mutate(_genome(0.5), rng).genes.tolist())["enc"] for _ in range(4000)
        )
        assert got["b"] == 0
        assert _within(got["a"], 4000, 0.5)

    def test_a_dependent_chooses_among_its_parents_options(self) -> None:
        decoder = ParameterDecoder((CAT, LENGTH))
        op = ByKindMutation(decoder=decoder, discrete_rate=0.5)
        rng = Random(0)
        parent = _genome(0.5, 0.5)  # enc "b", length 512
        moved = Counter()
        for _ in range(4000):
            after = decoder.values(op.mutate(parent, rng).genes.tolist())
            if after["enc"] == "b" and after["length"] != 512:
                moved[after["length"]] += 1
        assert set(moved) == {256, 1024}


class TestSubset:
    """A subset flips one choice its parent allows."""

    def test_exactly_one_choice_flips(self) -> None:
        decoder = ParameterDecoder((POOL,))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(0)
        flipped = Counter()
        for _ in range(3000):
            after = set(
                decoder.values(op.mutate(_genome(0.75, 0.25, 0.75), rng).genes.tolist())["pool"]
            )
            diff = after ^ {"x", "z"}
            assert len(diff) == 1
            flipped.update(diff)
        assert set(flipped) == {"x", "y", "z"}

    def test_a_choice_its_parent_disallows_never_flips(self) -> None:
        only_a = ParameterSpec(path="enc", param_type="categorical", choices=("a",))
        by_parent = ParameterSpec(
            path="pool",
            param_type="subset",
            choices=("x", "y", "z"),
            parent="enc",
            choices_by_parent={"a": ("x", "y")},
        )
        op = _op((only_a, by_parent), discrete_rate=1.0)
        rng = Random(0)
        flipped = Counter()
        for _ in range(500):
            child = op.mutate(_genome(0.5, 0.75, 0.25, 0.6), rng)
            assert child.genes[3] == 0.6
            flipped.update(i for i in (1, 2) if child.genes[i] != [0.75, 0.25][i - 1])
        assert set(flipped) == {1, 2}


class TestActivity:
    """Inactive genes stay put; a gene whose parent just changed is re-drawn
    when its old position means nothing."""

    def test_an_inactive_gene_keeps_its_position(self) -> None:
        threshold = ParameterSpec(path="t", bounds=(0.0, 1.0), parent="flag", active_values=(True,))
        op = _op((FLAG, threshold, CONT), mutation_rate=1.0, discrete_rate=0.0)
        rng = Random(0)
        for _ in range(200):
            child = op.mutate(_genome(0.25, 0.3, 0.5), rng)
            assert child.genes[1] == 0.3
            assert child.genes[2] != 0.5

    def test_a_newly_active_gene_is_redrawn(self) -> None:
        decoder = ParameterDecoder((FLAG, GATED))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(0)
        # centred sits at 0.5, the boundary: left alone it would decode True
        got = Counter(
            decoder.values(op.mutate(_genome(0.25, 0.5), rng).genes.tolist())["centred"]
            for _ in range(4000)
        )
        assert _within(got[False], 4000, 0.5)

    def test_options_that_follow_a_changed_parent_are_redrawn(self) -> None:
        decoder = ParameterDecoder((CAT, LENGTH))
        op = ByKindMutation(decoder=decoder, discrete_rate=1.0)
        rng = Random(0)
        got = Counter()
        for _ in range(6000):
            after = decoder.values(op.mutate(_genome(0.1, 0.5), rng).genes.tolist())
            if after["enc"] == "b":
                got[after["length"]] += 1
        # from "a", a length at 0.5 would decode to 512 under "b" every time
        total = sum(got.values())
        for length in (256, 512, 1024):
            assert _within(got[length], total, 1 / 3), got

    @pytest.mark.parametrize("relative", [False, True])
    def test_a_subset_keeps_its_choices_when_its_parent_changes(self, relative: bool) -> None:
        # Each position says whether one choice is in, under any parent value,
        # so only the one flip the subset's own mutation makes may differ.
        if relative:
            parent = ParameterSpec(path="p", param_type="subset", choices=("x", "y", "z"))
            child = ParameterSpec(
                path="s", param_type="subset", choices=("x", "y", "z"), parent="p"
            )
            start = (0.75, 0.75, 0.75)
        else:
            parent = ParameterSpec(path="p", param_type="categorical", choices=("a", "b"))
            child = ParameterSpec(
                path="s",
                param_type="subset",
                choices=("x", "y", "z"),
                parent="p",
                choices_by_parent={"a": ("x", "y"), "b": ("x", "y", "z")},
            )
            start = (0.25,)
        op = _op((parent, child), discrete_rate=1.0)
        rng = Random(0)
        positions = (*start, 0.75, 0.25, 0.75)
        for _ in range(500):
            genes = op.mutate(_genome(*positions), rng).genes.tolist()
            subset = genes[len(start) :]
            assert sum(a != b for a, b in zip(subset, positions[len(start) :])) <= 1

    def test_a_gene_still_active_under_its_parents_new_value_keeps_its_meaning(self) -> None:
        parent = ParameterSpec(path="p", param_type="categorical", choices=("x", "y", "z"))
        child = ParameterSpec(path="c", bounds=(0.0, 1.0), parent="p", active_values=("x", "y"))
        decoder = ParameterDecoder((parent, child))
        op = ByKindMutation(decoder=decoder, mutation_rate=0.0, discrete_rate=1.0)
        rng = Random(0)
        for _ in range(500):
            after = decoder.values(op.mutate(_genome(1 / 6, 0.3), rng).genes.tolist())
            if after["p"] == "y":
                assert after["c"] == pytest.approx(0.3)


class TestDeterminismAndShape:
    def test_same_seed_same_child(self) -> None:
        op = _op((CONT, INT, CAT, FLAG, GATED, LENGTH, POOL), mutation_rate=0.5, discrete_rate=0.5)
        parent = _genome(0.2, 0.5, 0.4, 0.25, 0.5, 0.5, 0.75, 0.25, 0.25)
        first = op.mutate(parent, Random(42)).genes
        assert np.array_equal(op.mutate(parent, Random(42)).genes, first)

    def test_parent_untouched_and_bounds_kept(self) -> None:
        op = _op((CONT, POOL), mutation_rate=1.0, discrete_rate=1.0)
        parent = _genome(0.5, 0.75, 0.25, 0.25)
        child = op.mutate(parent, Random(0))
        assert parent.genes.tolist() == [0.5, 0.75, 0.25, 0.25]
        assert child.bounds is not None and parent.bounds is not None
        assert all(np.array_equal(c, p) for c, p in zip(child.bounds, parent.bounds))

    def test_a_continuous_nudge_is_clipped_to_the_bounds(self) -> None:
        op = _op((CONT,), mutation_rate=1.0, sigma=5.0)
        rng = Random(0)
        for _ in range(200):
            assert 0.0 <= op.mutate(_genome(0.5), rng).genes[0] <= 1.0

    def test_genes_with_one_value_draw_nothing(self) -> None:
        op = _op((ONE, FIXED), mutation_rate=1.0, discrete_rate=1.0)
        rng = Random(5)
        state = rng.getstate()
        assert op.mutate(_genome(0.3, 0.6), rng).genes.tolist() == [0.3, 0.6]
        assert rng.getstate() == state


class TestRefusals:
    def test_needs_a_decoder(self) -> None:
        with pytest.raises(ValueError, match="needs a decoder built on ParameterDecoder"):
            ByKindMutation()

    def test_needs_a_parameter_decoder(self) -> None:
        with pytest.raises(ValueError, match="needs a decoder built on ParameterDecoder"):
            ByKindMutation(decoder=object())

    def test_takes_a_decoder_exposing_one(self) -> None:
        class Wrapper:
            parameter_decoder = ParameterDecoder((CAT,))

        op = ByKindMutation(decoder=Wrapper(), discrete_rate=1.0)
        assert op.mutate(_genome(0.5), Random(0)).genes[0] != 0.5

    @pytest.mark.parametrize(
        "params",
        [{"mutation_rate": 1.5}, {"discrete_rate": -0.1}, {"sigma": -1.0}],
    )
    def test_refuses_bad_settings(self, params: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            _op((CAT,), **params)


class TestFromConfig:
    """`mutation="by_kind"` with a declared parameters decoder."""

    SPECS = (CAT, LENGTH, INT, POOL)

    def _config(self, **fields: Any) -> UnifiedConfig:
        if "decoder" in fields:
            fields["decoder_params"] = {"params": [s.to_dict() for s in self.SPECS]}
        return UnifiedConfig(
            population_size=6,
            max_generations=3,
            selection="tournament",
            crossover="sbx",
            mutation="by_kind",
            mutation_params={"discrete_rate": 0.3},
            genome_type="vector",
            genome_params={
                "dimensions": ParameterDecoder(self.SPECS).dimensions,
                "bounds": (0.0, 1.0),
            },
            seed=3,
            **fields,
        )

    def test_runs_with_the_declared_decoder(self) -> None:
        config = self._config(decoder="parameters")
        engine = create_engine(config, evaluator=lambda spec: float(spec["epochs"]))
        assert isinstance(engine.mutation, ByKindMutation)
        assert engine.mutation.discrete_rate == 0.3
        assert engine.mutation.decoder is engine.evaluator._decoder
        engine.run(create_initial_population(config))

    def test_refused_without_a_decoder(self) -> None:
        with pytest.raises(ValueError, match="needs a decoder built on ParameterDecoder"):
            create_engine(self._config(), evaluator=lambda _genome: 0.0)
