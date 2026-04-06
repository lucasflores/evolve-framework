"""
LLMJudgeEvaluator — Multi-objective evaluation using LLM-as-judge.

Returns per-criterion fitness scores from a rubric, supporting
multi-objective selection (NSGA-II) (FR-006, SC-007).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any

import numpy as np

from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities

if TYPE_CHECKING:
    from evolve.evaluation.task_spec import RubricCriterion, TaskSpec
    from evolve.meta.soft_prompt.decoder import SoftPromptDecoder
    from evolve.representation.embedding import EmbeddingGenome

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails due to judge unavailability."""


@dataclass
class LLMJudgeEvaluator:
    """
    Evaluate EmbeddingGenome individuals using an LLM judge.

    The judge evaluates decoded outputs against a rubric with
    per-criterion scores, producing multi-dimensional Fitness
    suitable for NSGA-II selection.

    Attributes:
        decoder: SoftPromptDecoder for text generation.
        task_spec: TaskSpec with rubric criteria.
        judge_model_id: Model identifier for the judge LLM.
        temperature: Judge temperature (0.0 for determinism).
        _judge_fn: Optional callable for judging (for testing/injection).
    """

    decoder: SoftPromptDecoder
    task_spec: TaskSpec
    judge_model_id: str
    temperature: float = 0.0
    _judge_fn: Any = None  # Callable[[str], str] | None

    def __post_init__(self) -> None:
        if not self.task_spec.rubric:
            raise ValueError(
                "LLMJudgeEvaluator requires a TaskSpec with at least one rubric criterion"
            )

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            stateful=False,
            n_objectives=len(self.task_spec.rubric),
            n_constraints=0,
            supports_diagnostics=True,
            supports_gpu=True,
            supports_jit=False,
        )

    def _format_judge_prompt(self, model_output: str, rubric: tuple[RubricCriterion, ...]) -> str:
        """Format the judge prompt with rubric criteria and model output."""
        criteria_lines = []
        for i, c in enumerate(rubric, 1):
            criteria_lines.append(
                f"{i}. {c.name}: {c.description} (scale: {c.scale_min}–{c.scale_max})"
            )
        criteria_text = "\n".join(criteria_lines)

        return (
            "You are evaluating the quality of an AI response. "
            "Score on each criterion.\n\n"
            f"Criteria:\n{criteria_text}\n\n"
            f"Response to evaluate:\n{model_output}\n\n"
            'Provide scores as JSON: {"criterion_name": score, ...}'
        )

    def _parse_scores(self, response: str, rubric: tuple[RubricCriterion, ...]) -> np.ndarray:
        """Parse per-criterion scores from JSON response.

        Clamps scores to [scale_min, scale_max]. On parse failure,
        returns minimum scores.
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response.strip()
            if "```" in text:
                # Extract content between first pair of backticks
                start = text.index("```") + 3
                if text[start:].startswith("json"):
                    start += 4
                end = text.index("```", start)
                text = text[start:end].strip()

            scores_dict = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return np.array([c.scale_min for c in rubric], dtype=np.float32)

        scores = []
        for c in rubric:
            raw = scores_dict.get(c.name, c.scale_min)
            try:
                score = float(raw)
            except (TypeError, ValueError):
                score = c.scale_min
            # Clamp to valid range
            score = max(c.scale_min, min(c.scale_max, score))
            scores.append(score)

        return np.array(scores, dtype=np.float32)

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model. Uses _judge_fn if provided, else loads model."""
        if self._judge_fn is not None:
            return self._judge_fn(prompt)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise EvaluationError(
                "LLMJudgeEvaluator requires torch and transformers. "
                "Install with: pip install evolve-framework[llm]"
            ) from err

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.judge_model_id)
            model = AutoModelForCausalLM.from_pretrained(self.judge_model_id)
            model.eval()

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            raise EvaluationError(f"Judge model '{self.judge_model_id}' failed: {e}") from e

    def evaluate(
        self,
        individuals: Sequence[Individual[EmbeddingGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate individuals using LLM-as-judge.

        For each individual:
        1. Decode genome with each task input
        2. Send judge prompt with rubric
        3. Parse per-criterion scores
        4. Average across inputs

        Args:
            individuals: Individuals to evaluate.
            seed: Random seed (used for input subsampling if sample_size set).

        Returns:
            One Fitness per individual, shape (n_criteria,).
        """
        rubric = self.task_spec.rubric
        inputs = self.task_spec.inputs

        # Subsample inputs if sample_size configured
        if self.task_spec.sample_size is not None and seed is not None:
            rng = Random(seed)
            sample_size = min(self.task_spec.sample_size, len(inputs))
            indices = rng.sample(range(len(inputs)), sample_size)
            inputs = tuple(inputs[i] for i in sorted(indices))

        results: list[Fitness] = []

        for ind in individuals:
            all_scores: list[np.ndarray] = []

            for task_input in inputs:
                input_text = task_input.get("input", "")

                # Decode genome to text
                try:
                    output = self.decoder.decode(ind.genome, input_text)
                except Exception as e:
                    logger.warning(f"Decode failed for individual: {e}")
                    all_scores.append(np.array([c.scale_min for c in rubric], dtype=np.float32))
                    continue

                # Format and send to judge
                prompt = self._format_judge_prompt(output, rubric)

                # Try judge, retry once on malformed response
                scores = None
                for attempt in range(2):
                    try:
                        response = self._call_judge(prompt)
                        scores = self._parse_scores(response, rubric)
                        # Check if we got non-minimum scores (valid parse)
                        min_scores = np.array([c.scale_min for c in rubric], dtype=np.float32)
                        if not np.array_equal(scores, min_scores) or attempt == 1:
                            break
                    except EvaluationError:
                        raise
                    except Exception as e:
                        logger.warning(f"Judge attempt {attempt + 1} failed: {e}")
                        if attempt == 1:
                            scores = np.array([c.scale_min for c in rubric], dtype=np.float32)

                if scores is None:
                    scores = np.array([c.scale_min for c in rubric], dtype=np.float32)
                all_scores.append(scores)

            # Average scores across inputs
            if all_scores:
                mean_scores = np.mean(all_scores, axis=0).astype(np.float32)
            else:
                mean_scores = np.array([c.scale_min for c in rubric], dtype=np.float32)

            results.append(Fitness(values=mean_scores))

        return results
