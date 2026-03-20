"""
Unit tests for timing utilities.

Tests TimingContext, GenerationTimer, and timing measurement accuracy.
"""

from __future__ import annotations

import time
import pytest

from evolve.utils.timing import (
    GenerationTimer,
    TimingResult,
    timing_context,
)


class TestTimingContext:
    """Tests for timing_context context manager."""
    
    def test_basic_timing(self) -> None:
        """timing_context should measure elapsed time."""
        with timing_context("test_phase") as results:
            time.sleep(0.01)  # 10ms
        
        assert len(results) == 1
        result = results[0]
        assert result.name == "test_phase"
        assert result.elapsed_ms >= 10.0
        assert result.elapsed_ms < 100.0  # Reasonable upper bound
    
    def test_cpu_time_captured(self) -> None:
        """timing_context should capture CPU time."""
        with timing_context("cpu_test") as results:
            # CPU-bound work
            _ = sum(i * i for i in range(10000))
        
        result = results[0]
        assert result.cpu_time_ms is not None
        assert result.cpu_time_ms >= 0
    
    def test_exception_still_captures_time(self) -> None:
        """timing_context should capture time even if exception raised."""
        results: list[TimingResult] = []
        
        with pytest.raises(ValueError):
            with timing_context("error_phase") as results:
                time.sleep(0.005)
                raise ValueError("test error")
        
        assert len(results) == 1
        assert results[0].elapsed_ms >= 5.0
    
    def test_nested_timing_contexts(self) -> None:
        """Nested timing contexts should work independently."""
        with timing_context("outer") as outer_results:
            time.sleep(0.005)
            
            with timing_context("inner") as inner_results:
                time.sleep(0.005)
        
        assert outer_results[0].elapsed_ms > inner_results[0].elapsed_ms


class TestGenerationTimer:
    """Tests for GenerationTimer class."""
    
    def test_basic_phase_timing(self) -> None:
        """GenerationTimer should time individual phases."""
        timer = GenerationTimer()
        
        timer.start("evaluation")
        time.sleep(0.01)
        elapsed = timer.stop("evaluation")
        
        assert elapsed >= 10.0
    
    def test_multiple_phases(self) -> None:
        """GenerationTimer should track multiple phases."""
        timer = GenerationTimer()
        
        timer.start("evaluation")
        time.sleep(0.005)
        timer.stop("evaluation")
        
        timer.start("selection")
        time.sleep(0.005)
        timer.stop("selection")
        
        metrics = timer.get_metrics()
        
        assert "evaluation_time_ms" in metrics
        assert "selection_time_ms" in metrics
        assert metrics["evaluation_time_ms"] >= 5.0
        assert metrics["selection_time_ms"] >= 5.0
    
    def test_generation_total_timing(self) -> None:
        """GenerationTimer should track total generation time."""
        timer = GenerationTimer()
        
        timer.start_generation()
        time.sleep(0.01)
        total = timer.end_generation()
        
        assert total >= 10.0
        
        metrics = timer.get_metrics()
        assert "generation_time_ms" in metrics
        assert metrics["generation_time_ms"] >= 10.0
    
    def test_get_metrics_without_breakdown(self) -> None:
        """get_metrics(breakdown=False) should return only total."""
        timer = GenerationTimer()
        
        timer.start_generation()
        
        timer.start("evaluation")
        time.sleep(0.005)
        timer.stop("evaluation")
        
        timer.end_generation()
        
        # With breakdown
        full_metrics = timer.get_metrics(breakdown=True)
        assert "evaluation_time_ms" in full_metrics
        
        # Without breakdown
        brief_metrics = timer.get_metrics(breakdown=False)
        assert "evaluation_time_ms" not in brief_metrics
        assert "generation_time_ms" in brief_metrics
    
    def test_stop_unknown_phase_raises(self) -> None:
        """Stopping unknown phase should raise KeyError."""
        timer = GenerationTimer()
        
        with pytest.raises(KeyError, match="unknown"):
            timer.stop("unknown")
    
    def test_end_generation_without_start_raises(self) -> None:
        """end_generation without start should raise RuntimeError."""
        timer = GenerationTimer()
        
        with pytest.raises(RuntimeError):
            timer.end_generation()
    
    def test_reset_clears_state(self) -> None:
        """reset() should clear all timing data."""
        timer = GenerationTimer()
        
        timer.start_generation()
        timer.start("evaluation")
        time.sleep(0.005)
        timer.stop("evaluation")
        timer.end_generation()
        
        # Verify data exists
        metrics = timer.get_metrics()
        assert len(metrics) > 0
        
        # Reset
        timer.reset()
        
        # Verify cleared
        empty_metrics = timer.get_metrics()
        assert len(empty_metrics) == 0
    
    def test_cpu_time_captured_for_phases(self) -> None:
        """GenerationTimer should capture CPU time for phases."""
        timer = GenerationTimer()
        
        timer.start("cpu_work")
        # CPU-bound work
        _ = [i * i for i in range(50000)]
        timer.stop("cpu_work")
        
        metrics = timer.get_metrics()
        assert "cpu_work_time_ms" in metrics
        assert "cpu_work_cpu_time_ms" in metrics
        assert metrics["cpu_work_cpu_time_ms"] >= 0


class TestTimingOverhead:
    """Tests to verify timing overhead is minimal."""
    
    def test_timing_context_overhead(self) -> None:
        """timing_context overhead should be negligible."""
        # Measure baseline
        start = time.perf_counter()
        for _ in range(1000):
            pass
        baseline = time.perf_counter() - start
        
        # Measure with timing
        start = time.perf_counter()
        for _ in range(1000):
            with timing_context("noop") as results:
                pass
        with_timing = time.perf_counter() - start
        
        # Overhead should be small (allow up to 200ms for 1000 iterations)
        # This is ~0.2ms per timing context, which is acceptable
        overhead = with_timing - baseline
        assert overhead < 0.2  # 200ms total for 1000 iterations
    
    def test_generation_timer_overhead(self) -> None:
        """GenerationTimer overhead should be negligible."""
        timer = GenerationTimer()
        
        # Measure timing operations themselves
        start = time.perf_counter()
        for _ in range(1000):
            timer.start("test")
            timer.stop("test")
            timer.reset()
        elapsed = time.perf_counter() - start
        
        # Should complete quickly (less than 50ms for 1000 iterations)
        assert elapsed < 0.05


class TestTimingResult:
    """Tests for TimingResult dataclass."""
    
    def test_timing_result_fields(self) -> None:
        """TimingResult should have expected fields."""
        result = TimingResult(
            name="test",
            elapsed_ms=10.5,
            cpu_time_ms=8.0,
        )
        
        assert result.name == "test"
        assert result.elapsed_ms == 10.5
        assert result.cpu_time_ms == 8.0
    
    def test_timing_result_cpu_time_optional(self) -> None:
        """cpu_time_ms should be optional."""
        result = TimingResult(name="test", elapsed_ms=10.5)
        
        assert result.cpu_time_ms is None
