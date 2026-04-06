"""Unit tests for TaskSpec and RubricCriterion dataclasses."""

from __future__ import annotations

import pytest

from evolve.evaluation.task_spec import RubricCriterion, TaskSpec

# ── RubricCriterion ─────────────────────────────────────────────────────


class TestRubricCriterion:
    def test_valid_criterion(self) -> None:
        c = RubricCriterion(name="clarity", description="How clear is the response?")
        assert c.name == "clarity"
        assert c.scale_min == 0.0
        assert c.scale_max == 1.0

    def test_custom_scale(self) -> None:
        c = RubricCriterion(
            name="relevance",
            description="Is the answer relevant?",
            scale_min=1.0,
            scale_max=5.0,
        )
        assert c.scale_min == 1.0
        assert c.scale_max == 5.0

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            RubricCriterion(name="", description="desc")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description must be non-empty"):
            RubricCriterion(name="x", description="")

    def test_scale_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RubricCriterion(name="x", description="d", scale_min=5.0, scale_max=1.0)

    def test_roundtrip_serialization(self) -> None:
        c = RubricCriterion(
            name="depth", description="Depth of reasoning", scale_min=0, scale_max=10
        )
        d = c.to_dict()
        restored = RubricCriterion.from_dict(d)
        assert restored == c


# ── TaskSpec ────────────────────────────────────────────────────────────


def _make_inputs(n: int = 3) -> tuple[dict[str, str], ...]:
    return tuple({"input": f"question {i}"} for i in range(n))


class TestTaskSpec:
    def test_valid_qa(self) -> None:
        spec = TaskSpec(task_type="qa", inputs=_make_inputs())
        assert spec.task_type == "qa"
        assert len(spec.inputs) == 3
        assert spec.metrics == ("accuracy",)
        assert spec.max_generation_tokens == 256

    def test_all_task_types(self) -> None:
        for tt in ("qa", "generation", "classification", "instruction_following"):
            spec = TaskSpec(task_type=tt, inputs=_make_inputs(1))
            assert spec.task_type == tt

    def test_invalid_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid task_type"):
            TaskSpec(task_type="foo", inputs=_make_inputs())

    def test_empty_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="inputs must be non-empty"):
            TaskSpec(inputs=())

    def test_input_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain an 'input' key"):
            TaskSpec(inputs=({"text": "hello"},))

    def test_ground_truth_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="ground_truth length"):
            TaskSpec(inputs=_make_inputs(2), ground_truth=("a",))

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            TaskSpec(inputs=_make_inputs(), metrics=("bleu",))

    def test_max_generation_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_generation_tokens"):
            TaskSpec(inputs=_make_inputs(), max_generation_tokens=0)

    def test_sample_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_size"):
            TaskSpec(inputs=_make_inputs(), sample_size=0)

    def test_roundtrip_serialization(self) -> None:
        spec = TaskSpec(
            task_type="generation",
            inputs=_make_inputs(2),
            ground_truth=("a", "b"),
            rubric=(
                RubricCriterion(name="c1", description="d1"),
                RubricCriterion(name="c2", description="d2", scale_min=1, scale_max=5),
            ),
            metrics=("accuracy", "f1"),
            max_generation_tokens=128,
            sample_size=10,
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.task_type == spec.task_type
        assert restored.inputs == spec.inputs
        assert restored.ground_truth == spec.ground_truth
        assert restored.metrics == spec.metrics
        assert restored.max_generation_tokens == spec.max_generation_tokens
        assert restored.sample_size == spec.sample_size
        assert len(restored.rubric) == 2
        assert restored.rubric[0].name == "c1"
