"""
Fixed-seed runs pinned to what main produced before clearing existed.

The values below were recorded on main at b3243f6. A change that alters a
random draw, an operator or survival in a run that doesn't ask for anything
new fails here. Genes and objectives are compared to ten decimal places, so
a platform's last-bit differences in floating point do not count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from evolve.config import ObjectiveSpec, UnifiedConfig
from evolve.core.types import Fitness
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.factory import create_engine, create_initial_population


class TwoObjectives:
    """Objective ``spread`` = sum((g - 0.3)^2), minimized; ``lead`` =
    g[0] * (1 - g[1]), maximized: they pull genes 0 and 1 apart."""

    capabilities = EvaluatorCapabilities(n_objectives=2)

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        out = []
        for ind in individuals:
            g = np.asarray(ind.genome.genes)
            out.append(Fitness(values=np.array([((g - 0.3) ** 2).sum(), g[0] * (1 - g[1])])))
        return out


def multiobjective_config() -> UnifiedConfig:
    return UnifiedConfig(
        population_size=12,
        max_generations=6,
        selection="crowded_tournament",
        crossover="sbx",
        mutation="gaussian",
        genome_type="vector",
        genome_params={"dimensions": 4, "bounds": (0.0, 1.0)},
        seed=13,
    ).with_multiobjective(
        objectives=(
            ObjectiveSpec(name="spread", direction="minimize"),
            ObjectiveSpec(name="lead", direction="maximize"),
        )
    )


def single_objective_config() -> UnifiedConfig:
    return UnifiedConfig(
        population_size=10,
        max_generations=6,
        elitism=2,
        selection="tournament",
        crossover="sbx",
        mutation="gaussian",
        genome_type="vector",
        genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        seed=21,
    )


def run(kind: str, config: UnifiedConfig | None = None) -> list[list[float]]:
    """The final population as rows of genes then objective values, rounded."""
    if kind == "multiobjective":
        config = config or multiobjective_config()
        engine = create_engine(config, evaluator=TwoObjectives())
    else:
        config = config or single_objective_config()
        engine = create_engine(
            config, evaluator=lambda genes: float((np.asarray(genes) ** 2).sum())
        )
    result = engine.run(create_initial_population(config))
    return [
        [round(float(v), 10) for v in (*ind.genome.genes, *ind.fitness.values)]
        for ind in result.population.individuals
    ]


GOLDEN: dict[str, Any] = {
    "multiobjective": [
        [0.3828882978, 0.2134626712, 0.2898508789, 0.2139245736, 0.0218711629, 0.301155939],
        [0.753096199, 0.0, 0.3939870827, 0.2139075305, 0.3115416505, 0.753096199],
        [0.3828882978, 0.2134626712, 0.2898508789, 0.2139245736, 0.0218711629, 0.301155939],
        [0.6507137507, 0.0876269339, 0.2059121531, 0.2585806219, 0.1786705419, 0.5936936998],
        [0.4864300356, 0.1310099046, 0.3988628943, 0.2139075305, 0.0804995957, 0.422702883],
        [0.7370805457, 0.0, 0.3423817862, 0.2139075305, 0.2902475326, 0.7370805457],
        [0.7340236022, 0.1302130228, 0.3988628943, 0.2139075305, 0.2343898901, 0.6384441702],
        [0.4435159348, 0.1533747682, 0.2017389126, 0.1960405716, 0.0625585862, 0.3754917811],
        [0.7370805457, 0.1286874803, 0.3988628943, 0.2118862974, 0.2379252793, 0.6422275076],
        [0.3653772764, 0.1310099046, 0.3988628943, 0.2139075305, 0.0500176258, 0.3175092342],
        [0.7370805457, 0.1286874803, 0.3988628943, 0.2118862974, 0.2379252793, 0.6422275076],
        [0.3655454736, 0.1335817095, 0.2881913354, 0.2139085296, 0.0395424424, 0.3167152843],
    ],
    "single_objective": [
        [-0.0407639074, 0.032118679, 0.0056804269, 0.0027255729],
        [-0.0420841366, 0.032118679, 0.0056804269, 0.0028349514],
        [-0.0428597856, -0.0988663417, 0.0053381025, 0.0116400101],
        [-0.0418308467, 0.0358061532, 0.0160009286, 0.0032879301],
        [-0.0061717904, 0.032118679, 0.0056804269, 0.0011019678],
        [-0.0407639074, 0.032118679, 0.0056804269, 0.0027255729],
        [-0.0420841366, 0.032118679, 0.0056804269, 0.0028349514],
        [-0.0428601283, -0.1656070839, 0.0056804269, 0.0292949641],
        [-0.0417990559, 0.0359048943, 0.0061367907, 0.0030739827],
        [-0.0428601283, 0.032118679, 0.0162165317, 0.003131576],
    ],
    "multiobjective_hash": "9894b1850c9d8e6e",
}


@pytest.mark.parametrize("kind", ["multiobjective", "single_objective"])
def test_a_fixed_seed_run_matches_main(kind: str) -> None:
    assert run(kind) == GOLDEN[kind]


def test_the_config_hash_matches_main() -> None:
    assert multiobjective_config().compute_hash() == GOLDEN["multiobjective_hash"]
