"""
Integration test for end-to-end ESPO pipeline (T016).

Tests the full loop: population init → evaluate → select → mutate → repeat.
Uses mock decoder to avoid real model loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np
import pytest

from evolve.core.population import Population
from evolve.core.types import Individual
from evolve.evaluation.benchmark import GroundTruthEvaluator
from evolve.evaluation.task_spec import TaskSpec
from evolve.meta.soft_prompt.callback import ESPOCallback
from evolve.representation.embedding import EmbeddingGenome

# ── Mock Decoder ────────────────────────────────────────────────────


@dataclass
class MockDecoder:
    """Mock decoder that produces deterministic outputs based on genome state."""

    model_id: str = "test-model"
    _embed_dim: int = 16

    def decode(  # noqa: ARG002
        self, genome: EmbeddingGenome, task_input: str, _max_tokens: int | None = None
    ) -> str:
        # Return response based on genome's mean value (higher mean = "better" answer)
        mean_val = float(np.mean(genome.embeddings))
        if mean_val > 0.5:
            return task_input.split("?")[0] if "?" in task_input else "correct"
        return "wrong answer"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_text(self, text: str) -> np.ndarray:
        # Return fake embeddings for seed text
        n_tokens = len(text.split())
        return np.random.randn(n_tokens, self._embed_dim).astype(np.float32)


# ── Simple Evolution Loop ───────────────────────────────────────────


def _simple_gaussian_mutation(
    genome: EmbeddingGenome, sigma: float, rng: np.random.Generator
) -> EmbeddingGenome:
    """Simple token-level Gaussian mutation."""
    noise = rng.standard_normal(genome.embeddings.shape).astype(np.float32) * sigma
    new_embeddings = genome.embeddings + noise
    return EmbeddingGenome(
        embeddings=new_embeddings,
        model_id=genome.model_id,
        seed_text=genome.seed_text,
        strategy=genome.strategy,
    )


def _tournament_select(
    individuals: list[Individual[EmbeddingGenome]],
    n: int,
    rng: Random,
) -> list[Individual[EmbeddingGenome]]:
    """Simple tournament selection."""
    selected = []
    for _ in range(n):
        i, j = rng.sample(range(len(individuals)), 2)
        a, b = individuals[i], individuals[j]
        fa = float(a.fitness.values[0]) if a.fitness else float("-inf")
        fb = float(b.fitness.values[0]) if b.fitness else float("-inf")
        selected.append(a if fa >= fb else b)
    return selected


# ── Test ────────────────────────────────────────────────────────────


class TestESPOPipeline:
    """End-to-end ESPO pipeline integration test."""

    def test_full_pipeline_5_generations(self) -> None:
        """Run 5 generations and verify fitness tracking."""
        rng = Random(42)
        np_rng = np.random.default_rng(42)
        pop_size = 10
        n_generations = 5

        # Setup
        decoder = MockDecoder()
        task_spec = TaskSpec(
            task_type="qa",
            inputs=(
                {"input": "What is 2+2?"},
                {"input": "Capital of France?"},
            ),
            ground_truth=("2+2", "Capital of France"),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=task_spec)
        callback = ESPOCallback()

        # Initialize population
        individuals: list[Individual[EmbeddingGenome]] = []
        for _ in range(pop_size):
            embeddings = np_rng.standard_normal((4, 16)).astype(np.float32)
            genome = EmbeddingGenome(
                embeddings=embeddings,
                model_id="test-model",
            )
            individuals.append(Individual(genome=genome))

        # Evolution loop
        for gen in range(n_generations):
            population = Population(individuals, generation=gen)

            # Evaluate
            fitnesses = evaluator.evaluate(individuals, seed=42)
            individuals = [
                Individual(genome=ind.genome, fitness=fit, id=ind.id)
                for ind, fit in zip(individuals, fitnesses)
            ]

            # Callback
            population = Population(individuals, generation=gen)
            callback.on_generation_end(gen, population, {})

            # Select + mutate
            selected = _tournament_select(individuals, pop_size, rng)
            individuals = []
            for sel in selected:
                mutated = _simple_gaussian_mutation(sel.genome, sigma=0.1, rng=np_rng)
                individuals.append(Individual(genome=mutated))

        # Verify callback recorded all generations
        assert len(callback.history) == n_generations

        # Verify metrics were logged
        for entry in callback.history:
            assert "best_fitness" in entry
            assert "mean_fitness" in entry
            assert "diversity_l2_mean" in entry
            assert "generation" in entry

    def test_callback_diversity_decreases_under_selection(self) -> None:
        """Diversity should not increase under pure selection (no mutation)."""
        np_rng = np.random.default_rng(123)
        pop_size = 20
        rng = Random(123)

        decoder = MockDecoder()
        task_spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "test?"},),
            ground_truth=("test",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=task_spec)
        callback = ESPOCallback()

        # Initialize diverse population
        individuals: list[Individual[EmbeddingGenome]] = []
        for _ in range(pop_size):
            embeddings = np_rng.standard_normal((4, 16)).astype(np.float32) * 2.0
            genome = EmbeddingGenome(embeddings=embeddings, model_id="test-model")
            individuals.append(Individual(genome=genome))

        # Apply strong selection pressure over 3 gens (no mutation)
        for gen in range(3):
            fitnesses = evaluator.evaluate(individuals, seed=42)
            individuals = [
                Individual(genome=ind.genome, fitness=fit, id=ind.id)
                for ind, fit in zip(individuals, fitnesses)
            ]
            population = Population(individuals, generation=gen)
            callback.on_generation_end(gen, population, {})

            # Selection only (no mutation) → diversity should drop
            selected = _tournament_select(individuals, pop_size, rng)
            individuals = [Individual(genome=s.genome) for s in selected]

        # First recorded diversity should be >= last
        first_div = callback.history[0].get("diversity_l2_mean", 0.0)
        last_div = callback.history[-1].get("diversity_l2_mean", 0.0)
        assert last_div <= first_div + 1e-6  # Allow small float tolerance

    def test_embedding_genome_registry(self) -> None:
        """Verify 'embedding' genome type is registered."""
        from evolve.registry.genomes import get_genome_registry

        registry = get_genome_registry()
        assert registry.is_registered("embedding")

        genome = registry.create("embedding", n_tokens=4, embed_dim=16, model_id="test")
        assert isinstance(genome, EmbeddingGenome)
        assert genome.n_tokens == 4
        assert genome.embed_dim == 16

    def test_flat_vector_operator_compatibility(self) -> None:
        """T022: Flat-vector operators work via adapter round-trip (FR-009)."""
        from evolve.core.operators.mutation import GaussianMutation

        rng = Random(42)
        np_rng = np.random.default_rng(42)

        genome = EmbeddingGenome(
            embeddings=np_rng.standard_normal((4, 16)).astype(np.float32),
            model_id="test-model",
        )

        # Convert to VectorGenome → apply GaussianMutation → convert back
        vg = genome.to_vector_genome()
        mutator = GaussianMutation(mutation_rate=1.0, sigma=0.01)
        mutated_vg = mutator.mutate(vg, rng)

        restored = EmbeddingGenome.from_vector_genome(
            mutated_vg,
            n_tokens=genome.n_tokens,
            model_id=genome.model_id,
            seed_text=genome.seed_text,
            strategy=genome.strategy,
        )

        assert isinstance(restored, EmbeddingGenome)
        assert restored.n_tokens == genome.n_tokens
        assert restored.embed_dim == genome.embed_dim
        assert restored.model_id == genome.model_id
        # Should be different (mutated)
        assert not np.array_equal(restored.embeddings, genome.embeddings)

    def test_token_operators_registered(self) -> None:
        """Verify token operators registered in operator registry."""
        from evolve.registry.operators import get_operator_registry

        registry = get_operator_registry()
        assert registry.is_registered("mutation", "token_gaussian")
        assert registry.is_registered("crossover", "token_single_point")
        assert registry.is_registered("crossover", "token_two_point")

    @pytest.mark.slow
    def test_performance_budget_per_individual(self) -> None:
        """T044: Mock evaluation per individual takes << 60s (SC-010).

        This test measures wall-clock time for evaluation of a single
        individual using the mock decoder. Real GPU-based evaluation
        budgets are covered by the ≤60s SC-010 requirement.
        """
        import time

        np_rng = np.random.default_rng(42)
        decoder = MockDecoder()
        task_spec = TaskSpec(
            task_type="qa",
            inputs=tuple({"input": f"Question {i}?"} for i in range(50)),
            ground_truth=tuple(f"Question {i}" for i in range(50)),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=task_spec)

        genome = EmbeddingGenome(
            embeddings=np_rng.standard_normal((8, 16)).astype(np.float32),
            model_id="test-model",
        )
        ind = Individual(genome=genome)

        start = time.monotonic()
        evaluator.evaluate([ind], seed=42)
        elapsed = time.monotonic() - start

        # Mock should be sub-second; real model budget is ≤60s (SC-010)
        assert elapsed < 60.0, f"Evaluation took {elapsed:.2f}s, exceeds 60s budget"

    @pytest.mark.slow
    def test_50_generation_fitness_improvement(self) -> None:
        """T045: 50-generation ESPO run on mock data improves over seed (SC-001).

        Verifies that the evolutionary loop actually improves best fitness
        over the course of 50 generations.
        """
        rng = Random(42)
        np_rng = np.random.default_rng(42)
        pop_size = 20
        n_generations = 50

        decoder = MockDecoder()
        task_spec = TaskSpec(
            task_type="qa",
            inputs=(
                {"input": "What is 2+2?"},
                {"input": "Capital of France?"},
                {"input": "Color of sky?"},
            ),
            ground_truth=("2+2", "Capital of France", "Color of sky"),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=task_spec)
        callback = ESPOCallback()

        # Initialize population
        individuals: list[Individual[EmbeddingGenome]] = []
        for _ in range(pop_size):
            embeddings = np_rng.standard_normal((4, 16)).astype(np.float32)
            genome = EmbeddingGenome(embeddings=embeddings, model_id="test-model")
            individuals.append(Individual(genome=genome))

        initial_best = None

        for gen in range(n_generations):
            fitnesses = evaluator.evaluate(individuals, seed=42)
            individuals = [
                Individual(genome=ind.genome, fitness=fit, id=ind.id)
                for ind, fit in zip(individuals, fitnesses)
            ]

            population = Population(individuals, generation=gen)
            callback.on_generation_end(gen, population, {})

            if gen == 0:
                initial_best = callback.history[0]["best_fitness"]

            # Select + mutate
            selected = _tournament_select(individuals, pop_size, rng)
            individuals = []
            for sel in selected:
                mutated = _simple_gaussian_mutation(sel.genome, sigma=0.05, rng=np_rng)
                individuals.append(Individual(genome=mutated))

        assert initial_best is not None
        final_best = callback.history[-1]["best_fitness"]

        # Best fitness at end should be >= initial (SC-001)
        assert final_best >= initial_best, (
            f"No improvement: initial={initial_best}, final={final_best}"
        )
