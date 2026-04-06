"""Tests for LLMJudgeEvaluator — multi-objective scoring with rubric."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evolve.core.types import Individual
from evolve.evaluation.llm_judge import EvaluationError, LLMJudgeEvaluator
from evolve.evaluation.task_spec import RubricCriterion, TaskSpec
from evolve.representation.embedding import EmbeddingGenome

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_genome(seed: int = 42) -> EmbeddingGenome:
    rng = np.random.RandomState(seed)
    return EmbeddingGenome(
        embeddings=rng.randn(4, 16).astype(np.float32),
        model_id="test-model",
    )


def _make_rubric() -> tuple[RubricCriterion, ...]:
    return (
        RubricCriterion(
            name="relevance",
            description="How relevant is the response?",
            scale_min=0.0,
            scale_max=1.0,
        ),
        RubricCriterion(
            name="coherence",
            description="How coherent is the response?",
            scale_min=0.0,
            scale_max=1.0,
        ),
    )


def _make_task_spec(rubric: tuple[RubricCriterion, ...] | None = None) -> TaskSpec:
    return TaskSpec(
        task_type="generation",
        inputs=(
            {"input": "What is AI?"},
            {"input": "Explain ML."},
        ),
        rubric=rubric or _make_rubric(),
        metrics=("accuracy",),
    )


class MockDecoder:
    """Mock decoder that returns predictable text."""

    def __init__(self, model_id: str = "test-model") -> None:
        self.model_id = model_id

    def decode(  # noqa: ARG002
        self, _genome: EmbeddingGenome, task_input: str, _max_tokens: int | None = None
    ) -> str:
        return f"Response to: {task_input}"


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_requires_rubric(self) -> None:
        """LLMJudgeEvaluator requires a TaskSpec with rubric."""
        ts = TaskSpec(
            task_type="qa",
            inputs=({"input": "x", "expected_output": "y"},),
            ground_truth=("y",),
        )
        with pytest.raises(ValueError, match="rubric"):
            LLMJudgeEvaluator(
                decoder=MockDecoder(),
                task_spec=ts,
                judge_model_id="judge-model",
            )


# ===================================================================
# Capabilities
# ===================================================================


class TestCapabilities:
    def test_n_objectives_matches_rubric(self) -> None:
        """n_objectives equals number of rubric criteria."""
        ts = _make_task_spec()
        evaluator = LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
        )
        assert evaluator.capabilities.n_objectives == 2

    def test_batchable_and_deterministic(self) -> None:
        ts = _make_task_spec()
        evaluator = LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
        )
        assert evaluator.capabilities.batchable is True
        assert evaluator.capabilities.stochastic is False


# ===================================================================
# Evaluation with mock judge
# ===================================================================


class TestEvaluate:
    def _make_evaluator(self, judge_response: str) -> LLMJudgeEvaluator:
        """Create evaluator with a fixed judge response."""
        ts = _make_task_spec()
        return LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
            _judge_fn=lambda prompt: judge_response,  # noqa: ARG005
        )

    def test_fitness_shape_matches_rubric(self) -> None:
        """Fitness has one value per rubric criterion."""
        response = json.dumps({"relevance": 0.8, "coherence": 0.9})
        evaluator = self._make_evaluator(response)

        genome = _make_genome()
        ind = Individual(genome=genome)
        results = evaluator.evaluate([ind])

        assert len(results) == 1
        assert results[0].values.shape == (2,)

    def test_scores_are_correct(self) -> None:
        """Scores match judge response."""
        response = json.dumps({"relevance": 0.7, "coherence": 0.5})
        evaluator = self._make_evaluator(response)

        ind = Individual(genome=_make_genome())
        results = evaluator.evaluate([ind])

        # Averaged over 2 inputs (same response each time)
        np.testing.assert_allclose(results[0].values, [0.7, 0.5], atol=1e-5)

    def test_multiple_individuals(self) -> None:
        """Returns one Fitness per individual."""
        response = json.dumps({"relevance": 0.6, "coherence": 0.4})
        evaluator = self._make_evaluator(response)

        inds = [Individual(genome=_make_genome(i)) for i in range(3)]
        results = evaluator.evaluate(inds)
        assert len(results) == 3

    def test_score_clamping(self) -> None:
        """Scores outside [scale_min, scale_max] are clamped."""
        response = json.dumps({"relevance": 1.5, "coherence": -0.3})
        evaluator = self._make_evaluator(response)

        ind = Individual(genome=_make_genome())
        results = evaluator.evaluate([ind])

        assert float(results[0].values[0]) == pytest.approx(1.0)
        assert float(results[0].values[1]) == pytest.approx(0.0)

    def test_malformed_json_falls_back_to_minimum(self) -> None:
        """Malformed judge response gives minimum scores."""
        evaluator = self._make_evaluator("this is not json at all")

        ind = Individual(genome=_make_genome())
        results = evaluator.evaluate([ind])

        # Both criteria should get scale_min (0.0)
        np.testing.assert_array_equal(results[0].values, [0.0, 0.0])

    def test_json_in_code_block(self) -> None:
        """Judge response wrapped in markdown code block is parsed."""
        response = '```json\n{"relevance": 0.9, "coherence": 0.8}\n```'
        evaluator = self._make_evaluator(response)

        ind = Individual(genome=_make_genome())
        results = evaluator.evaluate([ind])
        np.testing.assert_allclose(results[0].values, [0.9, 0.8], atol=1e-5)

    def test_judge_error_raises_evaluation_error(self) -> None:
        """EvaluationError propagates from judge call."""

        def failing_judge(prompt: str) -> str:  # noqa: ARG001
            raise EvaluationError("Judge unavailable")

        ts = _make_task_spec()
        evaluator = LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
            _judge_fn=failing_judge,
        )

        ind = Individual(genome=_make_genome())
        with pytest.raises(EvaluationError, match="Judge unavailable"):
            evaluator.evaluate([ind])


# ===================================================================
# Multi-objective / NSGA-II compatibility (SC-007)
# ===================================================================


class TestMultiObjective:
    def test_fitness_is_multi_dimensional(self) -> None:
        """Multi-criterion rubric produces multi-dimensional Fitness."""
        rubric = (
            RubricCriterion(name="a", description="A"),
            RubricCriterion(name="b", description="B"),
            RubricCriterion(name="c", description="C"),
        )
        ts = _make_task_spec(rubric=rubric)
        response = json.dumps({"a": 0.5, "b": 0.7, "c": 0.3})
        evaluator = LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
            _judge_fn=lambda p: response,  # noqa: ARG005
        )

        ind = Individual(genome=_make_genome())
        results = evaluator.evaluate([ind])
        assert results[0].n_objectives == 3

    def test_dominance_comparison(self) -> None:
        """Multi-objective fitness supports Pareto dominance."""
        rubric = _make_rubric()
        ts = _make_task_spec(rubric=rubric)

        # One returns high scores, another low
        responses = iter(
            [
                json.dumps({"relevance": 0.9, "coherence": 0.9}),
                json.dumps({"relevance": 0.9, "coherence": 0.9}),
                json.dumps({"relevance": 0.1, "coherence": 0.1}),
                json.dumps({"relevance": 0.1, "coherence": 0.1}),
            ]
        )
        evaluator = LLMJudgeEvaluator(
            decoder=MockDecoder(),
            task_spec=ts,
            judge_model_id="judge-model",
            _judge_fn=lambda p: next(responses),  # noqa: ARG005
        )

        ind1 = Individual(genome=_make_genome(1))
        ind2 = Individual(genome=_make_genome(2))
        results = evaluator.evaluate([ind1, ind2])

        # Higher scores dominate lower (minimize=False means higher is better)
        assert results[0].dominates(results[1], minimize=False)
        assert not results[1].dominates(results[0], minimize=False)
