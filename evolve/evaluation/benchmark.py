"""
GroundTruthEvaluator — Evaluate embedding genomes against ground-truth answers.

Computes accuracy, F1, exact match, and pass@k metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities

if TYPE_CHECKING:
    from evolve.evaluation.task_spec import TaskSpec
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
    from evolve.representation.embedding import EmbeddingGenome

logger = logging.getLogger(__name__)


def _token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    """Binary exact match (case-insensitive, stripped)."""
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def _accuracy(prediction: str, reference: str) -> float:
    """Accuracy as normalized containment check."""
    return 1.0 if reference.strip().lower() in prediction.strip().lower() else 0.0


METRIC_FNS = {
    "accuracy": _accuracy,
    "f1": _token_f1,
    "exact_match": _exact_match,
}


@dataclass
class GroundTruthEvaluator:
    """
    Evaluate EmbeddingGenome individuals against a benchmark task.

    Uses SoftPromptDecoder to generate text, then scores against
    ground truth using configured metrics (FR-005).

    Attributes:
        decoder: SoftPromptDecoder for text generation.
        task_spec: TaskSpec defining inputs, ground truth, metrics.
    """

    decoder: SoftPromptDecoder
    task_spec: TaskSpec

    def __post_init__(self) -> None:
        if self.task_spec.ground_truth is None:
            raise ValueError("GroundTruthEvaluator requires task_spec with ground_truth")

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            stateful=False,
            n_objectives=len(self.task_spec.metrics),
            n_constraints=0,
            supports_diagnostics=True,
            supports_gpu=True,
            supports_jit=False,
        )

    def evaluate(
        self,
        individuals: Sequence[Individual[EmbeddingGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals by decoding and scoring against ground truth.

        Deterministic: same inputs + same seed → same outputs (SC-006).

        Args:
            individuals: Individuals to evaluate.
            seed: Random seed (unused — evaluation is deterministic).

        Returns:
            One Fitness per individual, in same order.
        """
        assert self.task_spec.ground_truth is not None

        # Determine inputs to use (subsample if configured)
        inputs = list(self.task_spec.inputs)
        ground_truth = list(self.task_spec.ground_truth)

        if self.task_spec.sample_size is not None and seed is not None:
            rng = np.random.default_rng(seed)
            n = min(self.task_spec.sample_size, len(inputs))
            indices = rng.choice(len(inputs), size=n, replace=False)
            inputs = [inputs[i] for i in indices]
            ground_truth = [ground_truth[i] for i in indices]

        results: list[Fitness] = []

        for individual in individuals:
            genome = individual.genome
            metric_scores = self._evaluate_single(genome, inputs, ground_truth)
            values = np.array(
                [metric_scores[m] for m in self.task_spec.metrics],
                dtype=np.float64,
            )
            results.append(
                Fitness(
                    values=values,
                    metadata={"per_metric": metric_scores},
                )
            )

        return results

    def _evaluate_single(
        self,
        genome: EmbeddingGenome,
        inputs: list[dict[str, str]],
        ground_truth: list[str],
    ) -> dict[str, float]:
        """Score a single genome across all inputs and metrics."""
        metric_scores: dict[str, list[float]] = {m: [] for m in self.task_spec.metrics}

        for inp, gt in zip(inputs, ground_truth):
            prediction = self.decoder.decode(genome, inp["input"])

            for metric_name in self.task_spec.metrics:
                if metric_name == "pass_at_k":
                    # pass@k uses exact match as base
                    score = _exact_match(prediction, gt)
                else:
                    fn = METRIC_FNS[metric_name]
                    score = fn(prediction, gt)
                metric_scores[metric_name].append(score)

        # Average across inputs
        return {m: float(np.mean(scores)) for m, scores in metric_scores.items()}
