"""pick_distinct(): distinct feasible candidates by front, then crowding distance."""

from __future__ import annotations

import numpy as np
import pytest

from evolve.core.types import Fitness, Individual
from evolve.multiobjective.selection import NSGA2Selector, pick_distinct

MINIMIZE_BOTH = NSGA2Selector(directions=("minimize", "minimize"))


def _ind(label: str, objectives: tuple[float, float], constraints=None) -> Individual:
    fitness = Fitness(
        values=np.array(objectives, dtype=float),
        constraints=None if constraints is None else np.array(constraints, dtype=float),
    )
    return Individual(genome=label).with_fitness(fitness)


def _labels(picked: list[Individual]) -> list[str]:
    return [ind.genome for ind in picked]


def test_distinct_feasible_candidates_by_front_then_crowding() -> None:
    individuals = [
        _ind("x", (-1, -1), (1.0,)),  # dominates every point, infeasible
        _ind("c", (2, 2)),  # crowded less than "b"
        _ind("b", (1, 3)),
        _ind("a", (0, 4)),
        _ind("a", (0, 4)),  # another copy of one candidate
        _ind("d", (4, 0)),
        _ind("e", (3, 3)),  # the second front
    ]
    picked = pick_distinct(individuals, MINIMIZE_BOTH, 4, exclude={"b"})
    assert _labels(picked) == ["a", "d", "c", "e"]
    assert picked[0].fitness.values.tolist() == [0.0, 4.0]
    # Fewer distinct feasible candidates than the count: all of them, no more
    assert len(pick_distinct(individuals, MINIMIZE_BOTH, 6, exclude={"b"})) == 4


def test_copies_are_one_point_when_crowding_is_measured() -> None:
    # One front. Alone, b sits in a wider gap than c; as two copies each
    # copy's gap to the other is nil, and c would be picked over b.
    front = [
        _ind("a", (0, 10)),
        _ind("b", (5, 5)),
        _ind("b", (5, 5)),
        _ind("c", (6, 4)),
        _ind("d", (10, 0)),
    ]
    picked = _labels(pick_distinct(front, MINIMIZE_BOTH, 3))
    assert set(picked[:2]) == {"a", "d"}
    assert picked[2] == "b"


def test_identity_decides_what_counts_as_one_candidate() -> None:
    # Labels "a1" and "a2" are one candidate under an identity of the first letter
    individuals = [_ind("a1", (0, 2)), _ind("a2", (0, 2)), _ind("b1", (2, 0))]
    picked = pick_distinct(individuals, MINIMIZE_BOTH, 3, identity=lambda g: g[0])
    assert _labels(picked) == ["a1", "b1"]


def test_nothing_feasible_picks_nothing() -> None:
    assert pick_distinct([_ind("x", (0, 0), (0.5,))], MINIMIZE_BOTH, 2) == []


@pytest.mark.parametrize("count", [0, -2])
def test_no_places_picks_nothing(count: int) -> None:
    assert pick_distinct([_ind("a", (0, 1)), _ind("b", (1, 0))], MINIMIZE_BOTH, count) == []


def test_a_candidate_with_an_infeasible_copy_is_infeasible() -> None:
    # Noisy copies of "a": one evaluation broke a constraint, so "a" isn't picked
    individuals = [_ind("a", (0, 2), (0.5,)), _ind("a", (0, 2)), _ind("b", (2, 0))]
    assert _labels(pick_distinct(individuals, MINIMIZE_BOTH, 3)) == ["b"]
