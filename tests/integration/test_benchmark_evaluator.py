"""
Integration tests for GroundTruthEvaluator (T043).

Uses a mock decoder to test deterministic scoring without loading real models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from evolve.core.types import Individual
from evolve.evaluation.benchmark import GroundTruthEvaluator, _accuracy, _exact_match, _token_f1
from evolve.evaluation.task_spec import TaskSpec
from evolve.representation.embedding import EmbeddingGenome

# ── Mock Decoder ────────────────────────────────────────────────────


@dataclass
class MockDecoder:
    """Decoder that returns predetermined responses."""

    model_id: str = "test-model"
    device: str = "cpu"
    responses: dict[str, str] | None = None
    _embed_dim: int = 16

    def decode(  # noqa: ARG002
        self, _genome: EmbeddingGenome, task_input: str, _max_tokens: int | None = None
    ) -> str:
        if self.responses and task_input in self.responses:
            return self.responses[task_input]
        return "default answer"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# ── Metric Functions ────────────────────────────────────────────────


class TestMetricFunctions:
    def test_accuracy_match(self) -> None:
        assert _accuracy("The answer is Paris", "Paris") == 1.0

    def test_accuracy_no_match(self) -> None:
        assert _accuracy("I don't know", "Paris") == 0.0

    def test_exact_match_case_insensitive(self) -> None:
        assert _exact_match("Paris", "paris") == 1.0
        assert _exact_match(" Paris ", "paris") == 1.0

    def test_exact_match_miss(self) -> None:
        assert _exact_match("London", "Paris") == 0.0

    def test_f1_perfect(self) -> None:
        assert _token_f1("the quick brown fox", "the quick brown fox") == 1.0

    def test_f1_partial(self) -> None:
        score = _token_f1("the quick brown fox", "the brown cat")
        assert 0.0 < score < 1.0

    def test_f1_no_overlap(self) -> None:
        assert _token_f1("hello world", "goodbye moon") == 0.0

    def test_f1_empty(self) -> None:
        assert _token_f1("", "") == 1.0
        assert _token_f1("hello", "") == 0.0
        assert _token_f1("", "hello") == 0.0


# ── GroundTruthEvaluator ────────────────────────────────────────────


def _make_genome(np_rng: np.random.Generator, model_id: str = "test-model") -> EmbeddingGenome:
    embeddings = np_rng.standard_normal((4, 16)).astype(np.float32)
    return EmbeddingGenome(embeddings=embeddings, model_id=model_id)


def _make_individual(genome: EmbeddingGenome) -> Individual[EmbeddingGenome]:
    return Individual(genome=genome)


class TestGroundTruthEvaluator:
    def test_requires_ground_truth(self) -> None:
        decoder = MockDecoder()
        spec = TaskSpec(task_type="qa", inputs=({"input": "q1"},))
        with pytest.raises(ValueError, match="ground_truth"):
            GroundTruthEvaluator(decoder=decoder, task_spec=spec)

    def test_all_correct(self, np_rng: np.random.Generator) -> None:
        decoder = MockDecoder(responses={"What is 2+2?": "4", "Capital of France?": "Paris"})
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "What is 2+2?"}, {"input": "Capital of France?"}),
            ground_truth=("4", "Paris"),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        genome = _make_genome(np_rng)
        results = evaluator.evaluate([_make_individual(genome)])

        assert len(results) == 1
        assert results[0].values[0] == 1.0  # 100% accuracy

    def test_all_wrong(self, np_rng: np.random.Generator) -> None:
        decoder = MockDecoder(responses={})  # "default answer" for everything
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "q1"}, {"input": "q2"}),
            ground_truth=("Paris", "42"),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        genome = _make_genome(np_rng)
        results = evaluator.evaluate([_make_individual(genome)])

        assert results[0].values[0] == 0.0

    def test_multiple_metrics(self, np_rng: np.random.Generator) -> None:
        decoder = MockDecoder(responses={"q1": "Paris"})
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "q1"},),
            ground_truth=("Paris",),
            metrics=("accuracy", "f1", "exact_match"),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        genome = _make_genome(np_rng)
        results = evaluator.evaluate([_make_individual(genome)])

        assert results[0].values.shape == (3,)  # 3 metrics
        assert results[0].values[0] == 1.0  # accuracy
        assert results[0].values[1] == 1.0  # f1
        assert results[0].values[2] == 1.0  # exact_match

    def test_deterministic_same_seed(self, np_rng: np.random.Generator) -> None:
        """SC-006: Same inputs + same seed → same outputs."""
        decoder = MockDecoder(responses={"q1": "answer1", "q2": "answer2"})
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "q1"}, {"input": "q2"}),
            ground_truth=("answer1", "answer2"),
            metrics=("accuracy",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        genome = _make_genome(np_rng)
        ind = _make_individual(genome)

        r1 = evaluator.evaluate([ind], seed=42)
        r2 = evaluator.evaluate([ind], seed=42)

        np.testing.assert_array_equal(r1[0].values, r2[0].values)

    def test_capabilities(self) -> None:
        decoder = MockDecoder()
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "q"},),
            ground_truth=("a",),
            metrics=("accuracy", "f1"),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        caps = evaluator.capabilities

        assert caps.batchable is True
        assert caps.stochastic is False
        assert caps.n_objectives == 2
        assert caps.n_constraints == 0

    def test_multiple_individuals(self, np_rng: np.random.Generator) -> None:
        decoder = MockDecoder(responses={"q": "correct"})
        spec = TaskSpec(
            task_type="qa",
            inputs=({"input": "q"},),
            ground_truth=("correct",),
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)

        inds = [_make_individual(_make_genome(np_rng)) for _ in range(5)]
        results = evaluator.evaluate(inds)

        assert len(results) == 5
        for r in results:
            assert r.values[0] == 1.0

    def test_sample_size(self, np_rng: np.random.Generator) -> None:
        """When sample_size is set with a seed, only a subset of inputs is used."""
        call_count = 0

        class CountingDecoder(MockDecoder):
            def decode(self, genome, task_input, max_tokens=None):  # noqa: ARG002
                nonlocal call_count
                call_count += 1
                return "answer"

        decoder = CountingDecoder()
        spec = TaskSpec(
            task_type="qa",
            inputs=tuple({"input": f"q{i}"} for i in range(10)),
            ground_truth=tuple(f"a{i}" for i in range(10)),
            sample_size=3,
        )
        evaluator = GroundTruthEvaluator(decoder=decoder, task_spec=spec)
        genome = _make_genome(np_rng)
        evaluator.evaluate([_make_individual(genome)], seed=42)

        assert call_count == 3  # Only 3 inputs evaluated
