"""FunctionEvaluator return-value conversion."""

from __future__ import annotations

import numpy as np

from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome


def _individuals() -> list[Individual[VectorGenome]]:
    return [Individual(genome=VectorGenome(genes=np.array([1.0, 2.0])))]


def test_fitness_return_passes_through_with_constraints() -> None:
    """A fitness_fn can return a Fitness, e.g. to report constraint values."""
    returned = Fitness(values=np.array([3.0]), constraints=np.array([0.5]), metadata={"k": 1})
    evaluator = FunctionEvaluator(lambda _genes: returned, n_constraints=1)

    (fitness,) = evaluator.evaluate(_individuals())

    assert fitness is returned
    assert not fitness.is_feasible


def test_float_and_array_returns_unchanged() -> None:
    (scalar,) = FunctionEvaluator(lambda genes: float(genes.sum())).evaluate(_individuals())
    (vector,) = FunctionEvaluator(lambda genes: genes * 2).evaluate(_individuals())

    assert scalar.values.tolist() == [3.0]
    assert scalar.constraints is None
    assert vector.values.tolist() == [2.0, 4.0]
