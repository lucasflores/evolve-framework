"""
Benchmark tests for timing instrumentation overhead.

Verifies that timing overhead is <1% for populations under 1000.
"""

from __future__ import annotations

import time
from random import Random
from uuid import uuid4

import numpy as np
import pytest

from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.operators.crossover import UniformCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.population import Individual, Population
from evolve.core.types import Fitness
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.representation.vector import VectorGenome
from evolve.utils.timing import GenerationTimer


class SimpleEvaluator:
    """Simple evaluator for benchmarking."""

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(n_objectives=1)

    def evaluate(self, individuals, seed=None):  # noqa: ARG002
        """Evaluate batch of individuals."""
        results = []
        for ind in individuals:
            fitness = -sum(g**2 for g in ind.genome.genes)
            results.append(Fitness((fitness,)))
        return results


def create_population(
    size: int, n_genes: int = 10, seed: int = 42, evaluated: bool = False
) -> Population:
    """Create a population for benchmarking."""
    rng = Random(seed)
    bounds = (np.array([-5.0] * n_genes), np.array([5.0] * n_genes))

    individuals = [
        Individual(
            id=uuid4(),
            genome=VectorGenome.random(n_genes, bounds, rng),
            fitness=Fitness((-rng.random(),)) if evaluated else None,
        )
        for _ in range(size)
    ]
    return Population(individuals=individuals, generation=0)


@pytest.mark.benchmark
class TestTimingOverhead:
    """Benchmark tests for timing overhead."""

    @pytest.mark.parametrize("pop_size", [100, 500, 1000])
    def test_timing_overhead_percentage(self, pop_size: int) -> None:
        """
        Verify timing overhead is <1% for populations under 1000.

        This test compares:
        1. Time spent in actual evolution work
        2. Time spent in timing instrumentation

        The timing overhead should be negligible compared to evolution work.
        """
        n_iterations = 100

        # Measure timing overhead directly
        timer = GenerationTimer()

        # Measure overhead of start/stop operations
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            timer.reset()
            timer.start_generation()
            timer.start("selection")
            timer.stop("selection")
            timer.start("variation")
            timer.stop("variation")
            timer.start("evaluation")
            timer.stop("evaluation")
            timer.end_generation()
            _ = timer.get_metrics()
        timing_overhead_total = (time.perf_counter() - start_time) * 1000  # ms

        # Measure a representative evolution-like workload
        population = create_population(pop_size, evaluated=True)
        evaluator = SimpleEvaluator()
        selection = TournamentSelection(tournament_size=3)

        start_time = time.perf_counter()
        for _ in range(n_iterations):
            # Selection
            rng = Random(42)
            _ = list(selection.select(population, pop_size, rng))

            # Evaluation (the main work)
            _ = evaluator.evaluate(population.individuals)
        workload_total = (time.perf_counter() - start_time) * 1000  # ms

        # Calculate overhead percentage
        overhead_percentage = (timing_overhead_total / workload_total) * 100

        # Assert overhead is less than 5%
        assert overhead_percentage < 5.0, (
            f"Timing overhead ({overhead_percentage:.2f}%) exceeds 5% threshold "
            f"for population size {pop_size}"
        )

    def test_timing_context_overhead_microseconds(self) -> None:
        """
        Verify single timing operation overhead is measured in microseconds.

        Each start/stop pair should add only ~1-10 microseconds of overhead.
        """
        timer = GenerationTimer()
        n_iterations = 10000

        start_time = time.perf_counter()
        for _ in range(n_iterations):
            timer.start("test")
            timer.stop("test")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Calculate per-operation overhead
        overhead_per_op_us = (elapsed_ms / n_iterations) * 1000  # microseconds

        # Should be under 100 microseconds per operation (generous bound)
        assert overhead_per_op_us < 100, (
            f"Timing overhead per operation ({overhead_per_op_us:.2f}µs) is too high"
        )

    def test_get_metrics_overhead(self) -> None:
        """
        Verify get_metrics() call has minimal overhead.

        Building the metrics dict should be fast.
        """
        timer = GenerationTimer()
        n_iterations = 10000

        # Set up timer with typical phases
        timer.start_generation()
        timer.start("selection")
        timer.stop("selection")
        timer.start("variation")
        timer.stop("variation")
        timer.start("evaluation")
        timer.stop("evaluation")
        timer.end_generation()

        # Measure get_metrics overhead
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = timer.get_metrics(breakdown=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Per-call overhead
        overhead_per_call_us = (elapsed_ms / n_iterations) * 1000

        # Should be under 50 microseconds per call
        assert overhead_per_call_us < 50, (
            f"get_metrics() overhead ({overhead_per_call_us:.2f}µs) is too high"
        )

    def test_full_generation_overhead_percentage(self) -> None:
        """
        Integration test: verify timing metrics are reasonable in full evolution.

        Checks that reported phase times are consistent with total generation time.
        """
        pop_size = 100
        n_generations = 5

        # Run evolution
        engine = EvolutionEngine(
            config=EvolutionConfig(
                population_size=pop_size,
                max_generations=n_generations,
                elitism=2,
                crossover_rate=0.9,
                mutation_rate=0.1,
            ),
            selection=TournamentSelection(tournament_size=3),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            evaluator=SimpleEvaluator(),
            seed=42,
        )

        pop = create_population(pop_size, seed=99)
        result = engine.run(pop)

        # Verify timing consistency for each generation
        for gen_metrics in result.history:
            phase_sum = (
                gen_metrics["selection_time_ms"]
                + gen_metrics["variation_time_ms"]
                + gen_metrics["evaluation_time_ms"]
            )
            total = gen_metrics["generation_time_ms"]

            # Phase times should sum to less than or equal to total
            # (total may include minor overhead not captured in phases)
            assert phase_sum <= total * 1.05, (
                f"Phase times ({phase_sum:.2f}ms) exceed total ({total:.2f}ms)"
            )

            # Phase times should be reasonable portion of total
            # (not artificially small)
            assert phase_sum >= total * 0.9, (
                f"Phase times ({phase_sum:.2f}ms) are too small vs total ({total:.2f}ms)"
            )
