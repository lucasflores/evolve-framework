"""Tests for clear(): grouping near-copies best first and holding back the surplus."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from evolve.diversity.niching import clear


def _gap(a: float, b: float) -> float:
    return abs(a - b)


def _clear(
    points: list[float],
    closeness: float,
    copies: int | None,
    order: list[int] | None = None,
    distance: Callable[[float, float], float] = _gap,
) -> tuple[list[int], list[int], list[int]]:
    return clear(points, order or list(range(len(points))), distance, closeness, copies)


class TestGroups:
    def test_zero_closeness_groups_exact_copies_only(self) -> None:
        winners, held, sizes = _clear([1.0, 1.0, 1.0 + 1e-9, 2.0, 1.0], 0.0, 1)
        assert winners == [0, 2, 3]
        assert held == [1, 4]
        assert sizes == [3, 1, 1]

    def test_groups_form_around_the_best_candidate_not_by_chaining(self) -> None:
        # 0.0 leads; 0.4 joins it; 0.8 is 0.4 from 0.4 but 0.8 from the leader
        winners, held, sizes = _clear([0.0, 0.4, 0.8], 0.5, 1)
        assert winners == [0, 2]
        assert held == [1]
        assert sizes == [2, 1]

    def test_the_first_copies_in_order_win_each_group(self) -> None:
        # Best first: 3, 1, 0, 2, 4; all one group
        winners, held, sizes = _clear([5.0] * 5, 0.0, 2, order=[3, 1, 0, 2, 4])
        assert winners == [3, 1]
        assert held == [0, 2, 4]
        assert sizes == [5]

    def test_without_a_cap_nothing_is_held_back(self) -> None:
        winners, held, sizes = _clear([1.0, 1.0, 2.0, 1.0], 0.0, None)
        assert winners == [0, 1, 3, 2]
        assert held == []
        assert sizes == [3, 1]

    def test_lists_follow_the_given_order(self) -> None:
        winners, held, _ = _clear([1.0, 2.0, 1.0, 2.0], 0.0, 1, order=[3, 2, 1, 0])
        assert winners == [3, 2]
        assert held == [1, 0]

    def test_nothing_to_clear(self) -> None:
        assert _clear([], 0.0, 1) == ([], [], [])

    @pytest.mark.parametrize("copies", [0, -1])
    def test_a_cap_below_one_is_refused(self, copies: int) -> None:
        with pytest.raises(ValueError, match="copies"):
            _clear([1.0], 0.0, copies)

    def test_a_negative_closeness_is_refused(self) -> None:
        with pytest.raises(ValueError, match="closeness"):
            _clear([1.0], -0.1, 1)

    def test_the_order_must_cover_every_candidate_once(self) -> None:
        with pytest.raises(ValueError, match="order"):
            clear([1.0, 2.0], [0, 0], _gap, 0.0, 1)
