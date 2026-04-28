"""
Unit tests for EnsembleMetricCollector.

Tests Gini coefficient, Participation Ratio, Top-k Concentration,
Expert Turnover, and edge cases for the ensemble observability collector.

Specialization Index (US3) tests are in TestSpecializationIndex below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from evolve.core.types import Fitness
from evolve.experiment.collectors.base import CollectionContext
from evolve.experiment.collectors.ensemble import EnsembleMetricCollector

# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockIndividual:
    """Mock individual: supports both .fitness.values[0] and None fitness."""

    fitness: Fitness | None = None


@dataclass
class MockScalarFitnessOnly:
    """Mock fitness object that has .value but NOT .values — tests alternate path."""

    value: float


@dataclass
class MockIndividualScalar:
    """Mock individual using the scalar fitness path (fitness.value)."""

    fitness: MockScalarFitnessOnly | None = None


class MockPopulation:
    """Minimal mock population: iterable, sized, and indexable."""

    def __init__(self, individuals: list[Any]) -> None:
        self._individuals = individuals

    def __iter__(self):  # type: ignore[override]
        return iter(self._individuals)

    def __len__(self) -> int:
        return len(self._individuals)

    def __getitem__(self, idx: int) -> Any:
        return self._individuals[idx]


def _ind(f: float | None) -> MockIndividual:
    """Build a MockIndividual with the given fitness value (or None)."""
    if f is None:
        return MockIndividual(fitness=None)
    return MockIndividual(fitness=Fitness.scalar(f))


def make_context(
    fitnesses: list[float | None],
    *,
    previous_elites: list[Any] | None = None,
    species_info: dict[int, list[int]] | None = None,
    generation: int = 0,
) -> CollectionContext:
    """Build a CollectionContext from a flat list of fitness values."""
    individuals = [_ind(f) for f in fitnesses]
    population = MockPopulation(individuals)
    return CollectionContext(
        generation=generation,
        population=population,  # type: ignore[arg-type]
        previous_elites=previous_elites,
        species_info=species_info,
    )


# ---------------------------------------------------------------------------
# Phase 2 Foundational: MetricCategory.ENSEMBLE must exist
# ---------------------------------------------------------------------------


class TestMetricCategoryEnsemble:
    """MetricCategory.ENSEMBLE must be importable and equal 'ensemble'."""

    def test_ensemble_category_value(self) -> None:
        from evolve.config.tracking import MetricCategory

        assert MetricCategory.ENSEMBLE.value == "ensemble"


# ---------------------------------------------------------------------------
# TestGiniCoefficient
# ---------------------------------------------------------------------------


class TestGiniCoefficient:
    """Gini coefficient correctness across all edge cases."""

    def test_uniform_population_is_zero(self) -> None:
        """Uniform fitness → perfect equality → Gini = 0."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 1.0, 1.0, 1.0])
        result = collector.collect(ctx)
        assert result["ensemble/gini_coefficient"] == pytest.approx(0.0, abs=1e-9)

    def test_monopoly_is_n_minus_1_over_n(self) -> None:
        """One individual holds all fitness → Gini = (N-1)/N."""
        collector = EnsembleMetricCollector()
        N = 5
        # One non-zero, rest zero
        ctx = make_context([10.0] + [0.0] * (N - 1))
        result = collector.collect(ctx)
        expected = (N - 1) / N
        assert result["ensemble/gini_coefficient"] == pytest.approx(expected, rel=1e-6)

    def test_zero_total_fitness_returns_zero(self) -> None:
        """All-zero fitness → degenerate case → 0.0."""
        collector = EnsembleMetricCollector()
        ctx = make_context([0.0, 0.0, 0.0])
        result = collector.collect(ctx)
        assert result["ensemble/gini_coefficient"] == pytest.approx(0.0, abs=1e-9)

    def test_ground_truth_three_element(self) -> None:
        """[1, 2, 3] → G = 2/9 ≈ 0.2222."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 2.0, 3.0])
        result = collector.collect(ctx)
        assert result["ensemble/gini_coefficient"] == pytest.approx(2 / 9, rel=1e-6)

    def test_value_in_unit_interval(self) -> None:
        """Gini must always be in [0, 1]."""
        collector = EnsembleMetricCollector()
        ctx = make_context([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        result = collector.collect(ctx)
        g = result["ensemble/gini_coefficient"]
        assert 0.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# TestParticipationRatio
# ---------------------------------------------------------------------------


class TestParticipationRatio:
    """Participation Ratio correctness and range."""

    def test_uniform_returns_float_n(self) -> None:
        """Uniform fitness → all contribute equally → PR = float(N)."""
        collector = EnsembleMetricCollector()
        N = 4
        ctx = make_context([1.0] * N)
        result = collector.collect(ctx)
        assert result["ensemble/participation_ratio"] == pytest.approx(float(N), rel=1e-6)

    def test_monopoly_returns_one(self) -> None:
        """One dominant individual → PR = 1.0."""
        collector = EnsembleMetricCollector()
        ctx = make_context([100.0] + [0.0] * 9)
        result = collector.collect(ctx)
        assert result["ensemble/participation_ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_zero_total_returns_float_n(self) -> None:
        """Zero total fitness (0/0) → degenerate → float(N)."""
        collector = EnsembleMetricCollector()
        N = 6
        ctx = make_context([0.0] * N)
        result = collector.collect(ctx)
        assert result["ensemble/participation_ratio"] == pytest.approx(float(N), abs=1e-9)

    def test_value_in_range_one_to_n(self) -> None:
        """PR must be in [1, N]."""
        collector = EnsembleMetricCollector()
        N = 5
        ctx = make_context([1.0, 2.0, 3.0, 4.0, 5.0])
        result = collector.collect(ctx)
        pr = result["ensemble/participation_ratio"]
        assert 1.0 <= pr <= float(N)


# ---------------------------------------------------------------------------
# TestTopKConcentration
# ---------------------------------------------------------------------------


class TestTopKConcentration:
    """Top-k concentration correctness and range."""

    def test_monopoly_returns_one(self) -> None:
        """Top individual holds everything → concentration = 1.0."""
        collector = EnsembleMetricCollector(top_k_percent=10.0)
        ctx = make_context([100.0] + [0.0] * 9)
        result = collector.collect(ctx)
        assert result["ensemble/top_k_concentration"] == pytest.approx(1.0, abs=1e-9)

    def test_uniform_returns_top_k_fraction(self) -> None:
        """Uniform fitness: top-k fraction should equal k/100 of total."""
        # 10 individuals, all fitness=1.0, top_k_percent=10 → 1 individual → 1/10 = 0.1
        collector = EnsembleMetricCollector(top_k_percent=10.0)
        ctx = make_context([1.0] * 10)
        result = collector.collect(ctx)
        assert result["ensemble/top_k_concentration"] == pytest.approx(0.1, rel=1e-6)

    def test_zero_total_returns_zero(self) -> None:
        """Zero total fitness (all 0s) → degenerate → 0.0."""
        collector = EnsembleMetricCollector()
        ctx = make_context([0.0, 0.0, 0.0, 0.0])
        result = collector.collect(ctx)
        assert result["ensemble/top_k_concentration"] == pytest.approx(0.0, abs=1e-9)

    def test_top_k_count_clamped_to_at_least_one(self) -> None:
        """Very small k% that rounds to 0 → clamped to 1 individual."""
        # 5 individuals, top_k_percent=1 → ceil(1/100 * 5) = 1
        collector = EnsembleMetricCollector(top_k_percent=1.0)
        ctx = make_context([1.0, 2.0, 3.0, 4.0, 5.0])
        result = collector.collect(ctx)
        # top 1 of 5 = 5.0 / 15.0
        assert result["ensemble/top_k_concentration"] == pytest.approx(5.0 / 15.0, rel=1e-6)

    def test_value_in_unit_interval(self) -> None:
        """Top-k concentration must be in [0, 1]."""
        collector = EnsembleMetricCollector(top_k_percent=20.0)
        ctx = make_context([1.0, 3.0, 5.0, 2.0, 4.0])
        result = collector.collect(ctx)
        c = result["ensemble/top_k_concentration"]
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# TestExpertTurnover
# ---------------------------------------------------------------------------


class TestExpertTurnover:
    """Expert Turnover correctness.

    For mock objects without an ``.id`` attribute the collector falls back to
    Python object identity (``id()``).  For real ``Individual`` objects it uses
    the stable UUID ``.id`` so that an elite carried over through
    ``with_fitness()`` is correctly recognised as the same individual.
    """

    def _make_individuals(self, fitnesses: list[float]) -> list[MockIndividual]:
        return [MockIndividual(fitness=Fitness.scalar(f)) for f in fitnesses]

    def test_fully_stable_elite_returns_zero(self) -> None:
        """Same objects in both elite sets → turnover = 0.0."""
        collector = EnsembleMetricCollector(top_k_percent=50.0)
        # 4 individuals; top-50% = top 2
        inds = self._make_individuals([5.0, 4.0, 2.0, 1.0])
        population = MockPopulation(inds)
        # Elite is top-2 = inds[0], inds[1]
        ctx = CollectionContext(
            generation=1,
            population=population,  # type: ignore[arg-type]
            previous_elites=[inds[0], inds[1]],
        )
        result = collector.collect(ctx)
        assert result["ensemble/expert_turnover"] == pytest.approx(0.0, abs=1e-9)

    def test_fully_replaced_elite_returns_one(self) -> None:
        """Completely different objects in elite set → turnover = 1.0."""
        collector = EnsembleMetricCollector(top_k_percent=50.0)
        inds = self._make_individuals([5.0, 4.0, 2.0, 1.0])
        prev_inds = self._make_individuals([3.0, 2.5])  # different objects
        population = MockPopulation(inds)
        ctx = CollectionContext(
            generation=1,
            population=population,  # type: ignore[arg-type]
            previous_elites=prev_inds,  # type: ignore[arg-type]
        )
        result = collector.collect(ctx)
        assert result["ensemble/expert_turnover"] == pytest.approx(1.0, abs=1e-9)

    def test_empty_previous_elites_returns_one(self) -> None:
        """Empty previous elite list → all current are new → turnover = 1.0."""
        collector = EnsembleMetricCollector(top_k_percent=50.0)
        inds = self._make_individuals([5.0, 4.0, 2.0, 1.0])
        population = MockPopulation(inds)
        ctx = CollectionContext(
            generation=1,
            population=population,  # type: ignore[arg-type]
            previous_elites=[],
        )
        result = collector.collect(ctx)
        assert result["ensemble/expert_turnover"] == pytest.approx(1.0, abs=1e-9)

    def test_previous_elites_none_key_absent(self) -> None:
        """previous_elites=None → expert_turnover key must be absent from result."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 2.0, 3.0], previous_elites=None)
        result = collector.collect(ctx)
        assert "ensemble/expert_turnover" not in result

    def test_partial_turnover(self) -> None:
        """One of two elite slots is new → turnover = 0.5."""
        collector = EnsembleMetricCollector(top_k_percent=50.0)
        inds = self._make_individuals([5.0, 4.0, 2.0, 1.0])
        population = MockPopulation(inds)
        # Previous elite: top ind is same object, second is different
        prev_ind_different = MockIndividual(fitness=Fitness.scalar(3.5))
        ctx = CollectionContext(
            generation=1,
            population=population,  # type: ignore[arg-type]
            previous_elites=[inds[0], prev_ind_different],  # type: ignore[arg-type]
        )
        result = collector.collect(ctx)
        # inds[0] is stable (1 of 2 is same), inds[1] is new → 1/2 turnover
        assert result["ensemble/expert_turnover"] == pytest.approx(0.5, abs=1e-9)

    def test_value_in_unit_interval(self) -> None:
        """Expert turnover must be in [0, 1]."""
        collector = EnsembleMetricCollector(top_k_percent=20.0)
        inds = [MockIndividual(fitness=Fitness.scalar(float(i))) for i in range(1, 11)]
        population = MockPopulation(inds)
        prev = inds[:3]
        ctx = CollectionContext(
            generation=5,
            population=population,  # type: ignore[arg-type]
            previous_elites=prev,  # type: ignore[arg-type]
        )
        result = collector.collect(ctx)
        t = result["ensemble/expert_turnover"]
        assert 0.0 <= t <= 1.0

    def test_uuid_stable_elite_returns_zero(self) -> None:
        """Individual carried through with_fitness() keeps same UUID → turnover = 0.0 (maximize)."""
        from uuid import uuid4

        from evolve.core.types import Individual
        from evolve.representation.vector import VectorGenome

        collector = EnsembleMetricCollector(top_k_percent=50.0)

        uid_a = uuid4()
        uid_b = uuid4()
        genome_a = VectorGenome(genes=np.array([5.0]))
        genome_b = VectorGenome(genes=np.array([4.0]))

        # Simulate gen-T elites evaluated and stored (maximize: high fitness = best)
        elite_a_prev = Individual(id=uid_a, genome=genome_a, fitness=Fitness.scalar(5.0))
        elite_b_prev = Individual(id=uid_b, genome=genome_b, fitness=Fitness.scalar(4.0))

        # Same individuals re-evaluated in gen T+1 (with_fitness creates new objects)
        elite_a_curr = elite_a_prev.with_fitness(Fitness.scalar(5.1))
        elite_b_curr = elite_b_prev.with_fitness(Fitness.scalar(4.1))

        # Population also contains two worse individuals
        worse_c = Individual(
            genome=VectorGenome(genes=np.array([2.0])), fitness=Fitness.scalar(2.0)
        )
        worse_d = Individual(
            genome=VectorGenome(genes=np.array([1.0])), fitness=Fitness.scalar(1.0)
        )

        class _Pop:
            def __iter__(self):  # type: ignore[override]
                return iter([elite_a_curr, elite_b_curr, worse_c, worse_d])

            def __len__(self) -> int:
                return 4

        ctx = CollectionContext(
            generation=1,
            population=_Pop(),  # type: ignore[arg-type]
            previous_elites=[elite_a_prev, elite_b_prev],
        )
        result = collector.collect(ctx)
        # Both current elites share UUIDs with previous elites → 0 turnover
        assert result["ensemble/expert_turnover"] == pytest.approx(0.0, abs=1e-9)

    def test_uuid_stable_elite_minimize_returns_zero(self) -> None:
        """UUID-based comparison works correctly for minimization problems (low fitness = best)."""
        from uuid import uuid4

        from evolve.core.types import Individual
        from evolve.representation.vector import VectorGenome

        collector = EnsembleMetricCollector(top_k_percent=50.0)

        uid_a = uuid4()
        uid_b = uuid4()

        # Best individuals have LOW fitness (minimization)
        elite_a_prev = Individual(
            id=uid_a, genome=VectorGenome(genes=np.array([0.1])), fitness=Fitness.scalar(0.1)
        )
        elite_b_prev = Individual(
            id=uid_b, genome=VectorGenome(genes=np.array([0.2])), fitness=Fitness.scalar(0.2)
        )

        # Same elites survive to next generation with slightly updated fitness
        elite_a_curr = elite_a_prev.with_fitness(Fitness.scalar(0.05))
        elite_b_curr = elite_b_prev.with_fitness(Fitness.scalar(0.15))

        # Two worse individuals (high fitness = bad for minimization)
        worse_c = Individual(
            genome=VectorGenome(genes=np.array([2.0])), fitness=Fitness.scalar(2.0)
        )
        worse_d = Individual(
            genome=VectorGenome(genes=np.array([3.0])), fitness=Fitness.scalar(3.0)
        )

        class _Pop:
            def __iter__(self):  # type: ignore[override]
                return iter([elite_a_curr, elite_b_curr, worse_c, worse_d])

            def __len__(self) -> int:
                return 4

        ctx = CollectionContext(
            generation=1,
            population=_Pop(),  # type: ignore[arg-type]
            minimize=True,
            previous_elites=[elite_a_prev, elite_b_prev],
        )
        result = collector.collect(ctx)
        # Minimize=True → collector picks lowest fitness as elites; UUIDs match → 0 turnover
        assert result["ensemble/expert_turnover"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty population, None fitness, negative fitness, single individual."""

    def test_empty_population_returns_empty_dict(self) -> None:
        """Empty population → collector returns {}."""
        collector = EnsembleMetricCollector()
        ctx = make_context([])
        result = collector.collect(ctx)
        assert result == {}

    def test_all_none_fitness_returns_empty_dict(self) -> None:
        """All individuals with None fitness → treated as empty → {}."""
        collector = EnsembleMetricCollector()
        ctx = make_context([None, None, None])
        result = collector.collect(ctx)
        assert result == {}

    def test_negative_fitness_values_are_shifted(self) -> None:
        """Negative fitnesses must be shifted so metrics are computed correctly."""
        collector = EnsembleMetricCollector()
        # [-1, 0, 1] → shifted to [0, 1, 2] — same as [0, 1, 2] case
        ctx_neg = make_context([-1.0, 0.0, 1.0])
        ctx_pos = make_context([0.0, 1.0, 2.0])
        result_neg = collector.collect(ctx_neg)
        result_pos = collector.collect(ctx_pos)
        assert result_neg["ensemble/gini_coefficient"] == pytest.approx(
            result_pos["ensemble/gini_coefficient"], rel=1e-6
        )
        assert result_neg["ensemble/participation_ratio"] == pytest.approx(
            result_pos["ensemble/participation_ratio"], rel=1e-6
        )
        assert result_neg["ensemble/top_k_concentration"] == pytest.approx(
            result_pos["ensemble/top_k_concentration"], rel=1e-6
        )

    def test_single_individual_uniform_gini(self) -> None:
        """Single individual → Gini = 0, PR = 1.0, top_k = 1.0."""
        collector = EnsembleMetricCollector()
        ctx = make_context([5.0])
        result = collector.collect(ctx)
        assert result["ensemble/gini_coefficient"] == pytest.approx(0.0, abs=1e-9)
        assert result["ensemble/participation_ratio"] == pytest.approx(1.0, rel=1e-6)
        assert result["ensemble/top_k_concentration"] == pytest.approx(1.0, abs=1e-9)

    def test_mixed_none_and_valid_fitness(self) -> None:
        """None fitness individuals are skipped; valid ones still computed."""
        collector = EnsembleMetricCollector()
        ctx = make_context([None, 2.0, None, 2.0])
        result = collector.collect(ctx)
        # Two valid individuals with same fitness → Gini = 0
        assert result["ensemble/gini_coefficient"] == pytest.approx(0.0, abs=1e-9)

    def test_scalar_fitness_path(self) -> None:
        """Fitness objects with .value (no .values) are read via scalar path."""
        collector = EnsembleMetricCollector()
        inds = [
            MockIndividualScalar(fitness=MockScalarFitnessOnly(value=1.0)),
            MockIndividualScalar(fitness=MockScalarFitnessOnly(value=2.0)),
            MockIndividualScalar(fitness=MockScalarFitnessOnly(value=3.0)),
        ]
        population = MockPopulation(inds)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
        )
        result = collector.collect(ctx)
        assert result["ensemble/gini_coefficient"] == pytest.approx(2 / 9, rel=1e-6)

    def test_validation_rejects_invalid_top_k_percent(self) -> None:
        """top_k_percent outside (0, 100] must raise ValueError."""
        with pytest.raises(ValueError):
            EnsembleMetricCollector(top_k_percent=0.0)
        with pytest.raises(ValueError):
            EnsembleMetricCollector(top_k_percent=-5.0)
        with pytest.raises(ValueError):
            EnsembleMetricCollector(top_k_percent=101.0)

    def test_validation_rejects_invalid_elite_size(self) -> None:
        """elite_size < 1 must raise ValueError."""
        with pytest.raises(ValueError):
            EnsembleMetricCollector(elite_size=0)
        with pytest.raises(ValueError):
            EnsembleMetricCollector(elite_size=-1)

    def test_always_present_keys(self) -> None:
        """Gini, PR, and top_k_concentration are always present for valid population."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 2.0, 3.0])
        result = collector.collect(ctx)
        assert "ensemble/gini_coefficient" in result
        assert "ensemble/participation_ratio" in result
        assert "ensemble/top_k_concentration" in result

    def test_reset_is_no_op(self) -> None:
        """reset() must not raise and collector must remain functional."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 2.0])
        collector.reset()
        result = collector.collect(ctx)
        assert "ensemble/gini_coefficient" in result


# ---------------------------------------------------------------------------
# TestSpecializationIndex (T016 - US3)
# ---------------------------------------------------------------------------


class TestSpecializationIndex:
    """Specialization Index (eta-squared) correctness and guard."""

    def test_species_info_none_key_absent(self) -> None:
        """species_info=None → specialization_index absent from result."""
        collector = EnsembleMetricCollector()
        ctx = make_context([1.0, 2.0, 3.0], species_info=None)
        result = collector.collect(ctx)
        assert "ensemble/specialization_index" not in result

    def test_maximally_divergent_species_approaches_one(self) -> None:
        """Two species with fully divergent fitness → eta² close to 1.0."""
        collector = EnsembleMetricCollector()
        # Species 0: [10, 10], Species 1: [0, 0] — all variance is between-species
        individuals = [_ind(f) for f in [10.0, 10.0, 0.0, 0.0]]
        population = MockPopulation(individuals)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
            species_info={0: [0, 1], 1: [2, 3]},
        )
        result = collector.collect(ctx)
        assert result["ensemble/specialization_index"] == pytest.approx(1.0, rel=1e-6)

    def test_identical_species_distributions_approaches_zero(self) -> None:
        """Two species with same fitness distribution → eta² close to 0.0."""
        collector = EnsembleMetricCollector()
        # Both species have identical distributions: [1, 3] each
        individuals = [_ind(f) for f in [1.0, 3.0, 1.0, 3.0]]
        population = MockPopulation(individuals)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
            species_info={0: [0, 1], 1: [2, 3]},
        )
        result = collector.collect(ctx)
        assert result["ensemble/specialization_index"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_total_variance_returns_zero(self) -> None:
        """All-identical fitness (SS_total=0) → degenerate → 0.0."""
        collector = EnsembleMetricCollector()
        individuals = [_ind(5.0) for _ in range(4)]
        population = MockPopulation(individuals)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
            species_info={0: [0, 1], 1: [2, 3]},
        )
        result = collector.collect(ctx)
        assert result["ensemble/specialization_index"] == pytest.approx(0.0, abs=1e-9)

    def test_single_individual_with_species_info_returns_zero(self) -> None:
        """Single individual → SS_total = 0 → degenerate → 0.0."""
        collector = EnsembleMetricCollector()
        individuals = [_ind(7.0)]
        population = MockPopulation(individuals)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
            species_info={0: [0]},
        )
        result = collector.collect(ctx)
        assert result["ensemble/specialization_index"] == pytest.approx(0.0, abs=1e-9)

    def test_value_in_unit_interval(self) -> None:
        """Specialization index must be in [0, 1]."""
        collector = EnsembleMetricCollector()
        individuals = [_ind(f) for f in [1.0, 2.0, 5.0, 6.0, 3.0, 4.0]]
        population = MockPopulation(individuals)
        ctx = CollectionContext(
            generation=0,
            population=population,  # type: ignore[arg-type]
            species_info={0: [0, 1, 2], 1: [3, 4, 5]},
        )
        result = collector.collect(ctx)
        si = result["ensemble/specialization_index"]
        assert 0.0 <= si <= 1.0
