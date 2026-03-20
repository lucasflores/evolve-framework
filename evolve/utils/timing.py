"""
Timing utilities for metric collection.

Provides context managers and utilities for measuring execution time
of evolution phases with minimal overhead.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimingResult:
    """
    Result from a timing measurement.
    
    Attributes:
        name: Name of the timed phase.
        elapsed_ms: Elapsed wall-clock time in milliseconds.
        cpu_time_ms: CPU time in milliseconds (if available).
    """
    
    name: str
    elapsed_ms: float
    cpu_time_ms: float | None = None


@contextmanager
def timing_context(name: str) -> Iterator[list[TimingResult]]:
    """
    Context manager for timing a code block.
    
    Usage:
        >>> with timing_context("evaluation") as results:
        ...     # expensive operation
        ...     evaluate_population(population)
        >>> print(results[0].elapsed_ms)
        150.5
    
    Args:
        name: Name for this timing measurement.
        
    Yields:
        List that will contain a single TimingResult after exit.
    """
    results: list[TimingResult] = []
    
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    
    try:
        yield results
    finally:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        
        elapsed_ms = (end_wall - start_wall) * 1000.0
        cpu_time_ms = (end_cpu - start_cpu) * 1000.0
        
        results.append(TimingResult(
            name=name,
            elapsed_ms=elapsed_ms,
            cpu_time_ms=cpu_time_ms,
        ))


@dataclass
class GenerationTimer:
    """
    Timer for measuring phase durations within a generation.
    
    Provides fine-grained timing breakdown of evolution phases:
    - evaluation: Fitness evaluation time
    - selection: Parent selection time
    - crossover: Crossover operation time
    - mutation: Mutation operation time
    - total: Total generation time
    
    Example:
        >>> timer = GenerationTimer()
        >>> timer.start("evaluation")
        >>> evaluate_population(population)
        >>> timer.stop("evaluation")
        >>> 
        >>> timer.start("selection")
        >>> parents = select(population)
        >>> timer.stop("selection")
        >>> 
        >>> metrics = timer.get_metrics()
        >>> # {"evaluation_time_ms": 150.0, "selection_time_ms": 10.0, ...}
    
    Note:
        For accurate total generation time, wrap the entire step in
        a timing_context or use start_generation()/end_generation().
    """
    
    _phase_starts: dict[str, float] = field(default_factory=dict)
    _phase_times: dict[str, float] = field(default_factory=dict)
    _phase_cpu_times: dict[str, float] = field(default_factory=dict)
    _generation_start: float | None = field(default=None)
    _generation_cpu_start: float | None = field(default=None)
    
    def start(self, phase: str) -> None:
        """
        Start timing a phase.
        
        Args:
            phase: Phase name (e.g., "evaluation", "selection").
        """
        self._phase_starts[phase] = time.perf_counter()
        self._phase_starts[f"{phase}_cpu"] = time.process_time()
    
    def stop(self, phase: str) -> float:
        """
        Stop timing a phase.
        
        Args:
            phase: Phase name.
            
        Returns:
            Elapsed time in milliseconds.
            
        Raises:
            KeyError: If phase was not started.
        """
        if phase not in self._phase_starts:
            raise KeyError(f"Phase '{phase}' was never started")
        
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        
        elapsed_ms = (end_wall - self._phase_starts[phase]) * 1000.0
        cpu_ms = (end_cpu - self._phase_starts[f"{phase}_cpu"]) * 1000.0
        
        self._phase_times[phase] = elapsed_ms
        self._phase_cpu_times[phase] = cpu_ms
        
        return elapsed_ms
    
    def start_generation(self) -> None:
        """Start timing the entire generation."""
        self._generation_start = time.perf_counter()
        self._generation_cpu_start = time.process_time()
    
    def end_generation(self) -> float:
        """
        End timing the entire generation.
        
        Returns:
            Total generation time in milliseconds.
        """
        if self._generation_start is None:
            raise RuntimeError("Generation timing was not started")
        
        end_wall = time.perf_counter()
        elapsed_ms = (end_wall - self._generation_start) * 1000.0
        self._phase_times["total"] = elapsed_ms
        
        if self._generation_cpu_start is not None:
            end_cpu = time.process_time()
            cpu_ms = (end_cpu - self._generation_cpu_start) * 1000.0
            self._phase_cpu_times["total"] = cpu_ms
        
        return elapsed_ms
    
    def get_metrics(self, breakdown: bool = True) -> dict[str, float]:
        """
        Get timing metrics dictionary.
        
        Args:
            breakdown: If True, include per-phase breakdowns.
                       If False, include only total generation time.
        
        Returns:
            Dictionary with timing metrics in format "{phase}_time_ms".
        """
        metrics: dict[str, float] = {}
        
        # Always include total if available
        if "total" in self._phase_times:
            metrics["generation_time_ms"] = self._phase_times["total"]
            if "total" in self._phase_cpu_times:
                metrics["generation_cpu_time_ms"] = self._phase_cpu_times["total"]
        
        if not breakdown:
            return metrics
        
        # Add phase breakdowns
        for phase, elapsed_ms in self._phase_times.items():
            if phase == "total":
                continue
            metrics[f"{phase}_time_ms"] = elapsed_ms
            
            if phase in self._phase_cpu_times:
                metrics[f"{phase}_cpu_time_ms"] = self._phase_cpu_times[phase]
        
        return metrics
    
    def reset(self) -> None:
        """Reset all timing data for next generation."""
        self._phase_starts.clear()
        self._phase_times.clear()
        self._phase_cpu_times.clear()
        self._generation_start = None
        self._generation_cpu_start = None


__all__ = [
    "GenerationTimer",
    "TimingResult",
    "timing_context",
]
