"""
Property-based tests for Pareto dominance and multi-objective operations.

Tests verify mathematical properties:
- Dominance antisymmetry
- Dominance transitivity
- Non-dominated sorting completeness
- Crowding distance non-negativity
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from evolve.multiobjective import (
    MultiObjectiveFitness,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    hypervolume_2d,
    pareto_front,
)

# Strategies for generating test data
fitness_objectives = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=5),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)

two_d_objectives = arrays(
    dtype=np.float64,
    shape=(2,),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)


class TestDominanceProperties:
    """Property-based tests for Pareto dominance."""

    @given(two_d_objectives)
    @settings(max_examples=100)
    def test_dominance_reflexivity_strict(self, objectives):
        """Strict dominance is NOT reflexive: a does not strictly dominate a."""
        fitness = MultiObjectiveFitness(objectives)

        # Strict dominance requires at least one strictly better
        assert not dominates(fitness, fitness, strict=True)

    @given(two_d_objectives)
    @settings(max_examples=100)
    def test_dominance_reflexivity_weak(self, objectives):
        """Weak dominance IS reflexive: a weakly dominates a."""
        fitness = MultiObjectiveFitness(objectives)

        # Weak dominance allows equality
        assert dominates(fitness, fitness, strict=False)

    @given(two_d_objectives, two_d_objectives)
    @settings(max_examples=200)
    def test_dominance_antisymmetry(self, obj_a, obj_b):
        """If a strictly dominates b, then b cannot dominate a."""
        fitness_a = MultiObjectiveFitness(obj_a)
        fitness_b = MultiObjectiveFitness(obj_b)

        if dominates(fitness_a, fitness_b, strict=True):
            assert not dominates(fitness_b, fitness_a, strict=True)

    @given(two_d_objectives, two_d_objectives, two_d_objectives)
    @settings(max_examples=200)
    def test_dominance_transitivity(self, obj_a, obj_b, obj_c):
        """If a dominates b and b dominates c, then a dominates c."""
        fitness_a = MultiObjectiveFitness(obj_a)
        fitness_b = MultiObjectiveFitness(obj_b)
        fitness_c = MultiObjectiveFitness(obj_c)

        if dominates(fitness_a, fitness_b) and dominates(fitness_b, fitness_c):
            assert dominates(fitness_a, fitness_c)


class TestNonDominatedSortingProperties:
    """Property-based tests for non-dominated sorting."""

    @given(st.lists(two_d_objectives, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_sorting_covers_all_individuals(self, objectives_list):
        """Every individual must be assigned to exactly one front."""
        fitnesses = [MultiObjectiveFitness(obj) for obj in objectives_list]
        fronts = fast_non_dominated_sort(fitnesses)

        # Flatten all fronts
        all_indices = []
        for front in fronts:
            all_indices.extend(front)

        # Should contain all indices exactly once
        assert sorted(all_indices) == list(range(len(fitnesses)))

    @given(st.lists(two_d_objectives, min_size=2, max_size=20))
    @settings(max_examples=100)
    def test_front_zero_is_pareto_front(self, objectives_list):
        """Front 0 from sorting should match pareto_front result."""
        fitnesses = [MultiObjectiveFitness(obj) for obj in objectives_list]

        fronts = fast_non_dominated_sort(fitnesses)
        pf = pareto_front(fitnesses)

        assert set(fronts[0]) == set(pf)

    @given(st.lists(two_d_objectives, min_size=3, max_size=15))
    @settings(max_examples=100)
    def test_later_fronts_dominated_by_earlier(self, objectives_list):
        """Solutions in front k+1 must be dominated by at least one in front k."""
        fitnesses = [MultiObjectiveFitness(obj) for obj in objectives_list]
        fronts = fast_non_dominated_sort(fitnesses)

        for k in range(len(fronts) - 1):
            front_k = fronts[k]
            front_k1 = fronts[k + 1]

            for idx_later in front_k1:
                # Must be dominated by at least one in earlier front
                is_dominated = any(
                    dominates(fitnesses[idx_earlier], fitnesses[idx_later])
                    for idx_earlier in front_k
                )
                assert is_dominated, (
                    f"Index {idx_later} in front {k + 1} not dominated by front {k}"
                )


class TestCrowdingDistanceProperties:
    """Property-based tests for crowding distance."""

    @given(st.lists(two_d_objectives, min_size=3, max_size=15))
    @settings(max_examples=100)
    def test_crowding_distance_non_negative(self, objectives_list):
        """Crowding distance must be non-negative."""
        fitnesses = [MultiObjectiveFitness(obj) for obj in objectives_list]
        indices = list(range(len(fitnesses)))

        distances = crowding_distance(fitnesses, indices)

        for idx, dist in distances.items():
            assert dist >= 0 or dist == float("inf")

    @given(st.lists(two_d_objectives, min_size=3, max_size=15))
    @settings(max_examples=100)
    def test_crowding_boundary_infinite(self, objectives_list):
        """Boundary solutions should have infinite crowding distance."""
        assume(len(objectives_list) >= 3)

        fitnesses = [MultiObjectiveFitness(obj) for obj in objectives_list]
        indices = list(range(len(fitnesses)))

        distances = crowding_distance(fitnesses, indices)

        # At least 2 solutions should have infinite distance (boundaries)
        inf_count = sum(1 for d in distances.values() if d == float("inf"))
        assert inf_count >= 2


class TestHypervolumeProperties:
    """Property-based tests for hypervolume indicator."""

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_hypervolume_non_negative(self, n_points):
        """Hypervolume must be non-negative."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0.1, 10, size=(n_points, 2))
        reference = np.array([0.0, 0.0])

        hv = hypervolume_2d(points, reference)

        assert hv >= 0

    @given(
        arrays(
            dtype=np.float64,
            shape=(2,),
            elements=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_hypervolume_single_point(self, point):
        """Hypervolume of single point is product of coordinates minus reference."""
        reference = np.array([0.0, 0.0])
        points = point.reshape(1, 2)

        hv = hypervolume_2d(points, reference)
        expected = point[0] * point[1]

        assert hv == pytest.approx(expected, rel=1e-10)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=100)
    def test_hypervolume_subset_not_greater(self, n_points):
        """Hypervolume of subset cannot exceed hypervolume of full set."""
        rng = np.random.default_rng(123)
        points = rng.uniform(0.1, 10, size=(n_points, 2))
        reference = np.array([0.0, 0.0])

        full_hv = hypervolume_2d(points, reference)

        # Remove one point and check
        subset = points[:-1]
        subset_hv = hypervolume_2d(subset, reference)

        assert subset_hv <= full_hv + 1e-10  # Small tolerance for float comparison


class TestMultiObjectiveFitness:
    """Property-based tests for MultiObjectiveFitness dataclass."""

    @given(fitness_objectives)
    @settings(max_examples=100)
    def test_fitness_immutable(self, objectives):
        """Fitness objectives should be immutable after creation."""
        fitness = MultiObjectiveFitness(objectives.copy())

        # Attempting to modify should raise
        with pytest.raises((ValueError, TypeError)):
            fitness.objectives[0] = 999.0

    @given(fitness_objectives)
    @settings(max_examples=100)
    def test_fitness_equality(self, objectives):
        """Two fitnesses with same objectives should be equal."""
        f1 = MultiObjectiveFitness(objectives.copy())
        f2 = MultiObjectiveFitness(objectives.copy())

        assert f1 == f2
        assert hash(f1) == hash(f2)

    @given(fitness_objectives)
    @settings(max_examples=100)
    def test_n_objectives_correct(self, objectives):
        """n_objectives should match array length."""
        fitness = MultiObjectiveFitness(objectives)

        assert fitness.n_objectives == len(objectives)

    @given(
        fitness_objectives,
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=3),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_feasibility_detection(self, objectives, constraints):
        """Feasibility should match constraint satisfaction."""
        fitness = MultiObjectiveFitness(objectives=objectives, constraint_violations=constraints)

        expected_feasible = np.all(constraints <= 0)
        assert fitness.is_feasible == expected_feasible
