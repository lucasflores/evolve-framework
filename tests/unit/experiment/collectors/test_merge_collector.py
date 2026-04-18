"""Tests for MergeMetricCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evolve.experiment.collectors.merge import MergeMetricCollector


class TestMergeMetricCollector:
    """Tests for merge metric collection."""

    def test_initial_state(self) -> None:
        c = MergeMetricCollector()
        ctx = MagicMock()
        metrics = c.collect(ctx)
        assert metrics["merge/count"] == 0
        assert metrics["merge/mean_genome_complexity"] == 0.0
        assert metrics["merge/complexity_delta"] == 0.0

    def test_record_single_merge(self) -> None:
        c = MergeMetricCollector()
        c.record_merge(host_complexity=5, symbiont_complexity=3, merged_complexity=7)
        ctx = MagicMock()
        metrics = c.collect(ctx)
        assert metrics["merge/count"] == 1
        assert metrics["merge/mean_genome_complexity"] == 7.0
        assert metrics["merge/complexity_delta"] == 2.0  # 7 - 5

    def test_record_multiple_merges(self) -> None:
        c = MergeMetricCollector()
        c.record_merge(host_complexity=5, symbiont_complexity=3, merged_complexity=7)
        c.record_merge(host_complexity=10, symbiont_complexity=4, merged_complexity=12)
        ctx = MagicMock()
        metrics = c.collect(ctx)
        assert metrics["merge/count"] == 2
        assert metrics["merge/mean_genome_complexity"] == pytest.approx(9.5)
        assert metrics["merge/complexity_delta"] == pytest.approx(2.0)  # mean of [2, 2]

    def test_reset(self) -> None:
        c = MergeMetricCollector()
        c.record_merge(host_complexity=5, symbiont_complexity=3, merged_complexity=7)
        c.reset()
        ctx = MagicMock()
        metrics = c.collect(ctx)
        assert metrics["merge/count"] == 0

    def test_reset_generation(self) -> None:
        c = MergeMetricCollector()
        c.record_merge(host_complexity=5, symbiont_complexity=3, merged_complexity=7)
        c.reset_generation()
        ctx = MagicMock()
        metrics = c.collect(ctx)
        assert metrics["merge/count"] == 0
