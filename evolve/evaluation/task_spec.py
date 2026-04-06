"""
Task specification for ESPO evaluation.

Defines what is being optimized — task type, inputs, metrics, rubric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_TASK_TYPES = {"qa", "generation", "classification", "instruction_following"}
VALID_METRICS = {"accuracy", "f1", "exact_match", "pass_at_k"}


@dataclass(frozen=True)
class RubricCriterion:
    """
    A single scoring criterion for LLM-as-judge evaluation.

    Attributes:
        name: Criterion name (e.g., "clarity", "relevance").
        description: What the judge should evaluate.
        scale_min: Minimum score.
        scale_max: Maximum score.
    """

    name: str
    description: str
    scale_min: float = 0.0
    scale_max: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Criterion name must be non-empty")
        if not self.description:
            raise ValueError("Criterion description must be non-empty")
        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"scale_min ({self.scale_min}) must be less than scale_max ({self.scale_max})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricCriterion:
        return cls(
            name=data["name"],
            description=data["description"],
            scale_min=data.get("scale_min", 0.0),
            scale_max=data.get("scale_max", 1.0),
        )


@dataclass(frozen=True)
class TaskSpec:
    """
    Configuration defining what an ESPO experiment optimizes.

    Attributes:
        task_type: One of "qa", "generation", "classification", "instruction_following".
        inputs: List of task inputs, each with at least an "input" key.
        ground_truth: Ground-truth answers (required for benchmark evaluation).
        rubric: Scoring criteria (required for LLM-as-judge evaluation).
        metrics: Metrics to compute for benchmark evaluation.
        max_generation_tokens: Maximum tokens for model generation.
        sample_size: Subsample inputs per evaluation (None = use all).
    """

    task_type: str = "qa"
    inputs: tuple[dict[str, str], ...] = ()
    ground_truth: tuple[str, ...] | None = None
    rubric: tuple[RubricCriterion, ...] | None = None
    metrics: tuple[str, ...] = ("accuracy",)
    max_generation_tokens: int = 256
    sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. Must be one of {VALID_TASK_TYPES}"
            )
        if not self.inputs:
            raise ValueError("inputs must be non-empty")
        for i, inp in enumerate(self.inputs):
            if "input" not in inp:
                raise ValueError(f"Input at index {i} must contain an 'input' key")
        if self.ground_truth is not None and len(self.ground_truth) != len(self.inputs):
            raise ValueError(
                f"ground_truth length ({len(self.ground_truth)}) must match "
                f"inputs length ({len(self.inputs)})"
            )
        for m in self.metrics:
            if m not in VALID_METRICS:
                raise ValueError(f"Invalid metric '{m}'. Must be one of {VALID_METRICS}")
        if self.max_generation_tokens < 1:
            raise ValueError(
                f"max_generation_tokens must be >= 1, got {self.max_generation_tokens}"
            )
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1 or None, got {self.sample_size}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_type": self.task_type,
            "inputs": list(self.inputs),
            "ground_truth": list(self.ground_truth) if self.ground_truth else None,
            "metrics": list(self.metrics),
            "max_generation_tokens": self.max_generation_tokens,
            "sample_size": self.sample_size,
        }
        if self.rubric is not None:
            result["rubric"] = [c.to_dict() for c in self.rubric]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        rubric = None
        if data.get("rubric") is not None:
            rubric = tuple(RubricCriterion.from_dict(c) for c in data["rubric"])
        gt = tuple(data["ground_truth"]) if data.get("ground_truth") else None
        return cls(
            task_type=data["task_type"],
            inputs=tuple(data["inputs"]),
            ground_truth=gt,
            rubric=rubric,
            metrics=tuple(data.get("metrics", ["accuracy"])),
            max_generation_tokens=data.get("max_generation_tokens", 256),
            sample_size=data.get("sample_size"),
        )
