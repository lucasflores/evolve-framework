"""A run's lifecycle: the evaluator's start hook, and failures reaching callbacks."""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
import pytest

from evolve.config import UnifiedConfig
from evolve.config.tracking import TrackingConfig
from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.operators.crossover import SimulatedBinaryCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.diversity.islands.engine import IslandConfig, IslandEvolutionEngine
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.factory import create_engine, create_initial_population
from evolve.representation.vector import VectorGenome


class Refused(Exception):
    """An evaluator's refusal, with fields a caller acts on."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class Recording:
    """An evaluator recording its lifecycle into a shared event list."""

    capabilities = EvaluatorCapabilities(n_objectives=1)

    def __init__(self, events: list[str], refuse: bool = False, fail_on: int = 0) -> None:
        self.events, self.refuse, self.fail_on, self.calls = events, refuse, fail_on, 0

    def on_run_start(self) -> None:
        self.events.append("evaluator.on_run_start")
        if self.refuse:
            raise Refused("the search can't open", "trainer uncommitted")

    def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
        self.calls += 1
        self.events.append("evaluate")
        if self.calls == self.fail_on:
            raise RuntimeError("evaluation broke")
        return [Fitness(values=np.array([float(np.sum(i.genome.genes))])) for i in individuals]


class Watching:
    """A callback recording the run-level hooks."""

    def __init__(self, events: list[str], fail_in_on_error: bool = False) -> None:
        self.events, self.fail_in_on_error, self.errors = events, fail_in_on_error, []

    def on_run_start(self, _config: Any) -> None:
        self.events.append("callback.on_run_start")

    def on_run_end(self, _population: Any, _reason: str) -> None:
        self.events.append("callback.on_run_end")

    def on_error(self, error: BaseException) -> None:
        self.errors.append(error)
        if self.fail_in_on_error:
            raise ValueError("on_error broke")


def _engine(evaluator: Any, generations: int = 2) -> EvolutionEngine:
    return EvolutionEngine(
        config=EvolutionConfig(population_size=4, max_generations=generations),
        evaluator=evaluator,
        selection=TournamentSelection(),
        crossover=SimulatedBinaryCrossover(),
        mutation=GaussianMutation(),
        seed=1,
    )


def _start() -> Population[VectorGenome]:
    rng = Random(0)
    return Population(
        [
            Individual(genome=VectorGenome(genes=np.array([rng.random(), rng.random()])))
            for _ in range(4)
        ]
    )


class TestEvaluatorStartHook:
    def test_runs_once_after_the_callbacks_and_before_the_first_evaluation(self) -> None:
        events: list[str] = []
        _engine(Recording(events)).run(_start(), callbacks=[Watching(events)])
        assert events[:3] == ["callback.on_run_start", "evaluator.on_run_start", "evaluate"]
        assert events.count("evaluator.on_run_start") == 1
        assert events[-1] == "callback.on_run_end"

    def test_an_evaluator_without_it_runs_as_before(self) -> None:
        class Plain:
            capabilities = EvaluatorCapabilities(n_objectives=1)

            def evaluate(self, individuals: Any, seed: int | None = None) -> list[Fitness]:
                return [Fitness(values=np.array([0.0])) for _ in individuals]

        assert _engine(Plain()).run(_start()).generations == 2


class TestFailures:
    def test_a_refused_start_reaches_the_caller_unchanged_and_the_callbacks(self) -> None:
        events: list[str] = []
        watching = Watching(events)
        with pytest.raises(Refused) as raised:
            _engine(Recording(events, refuse=True)).run(_start(), callbacks=[watching])
        assert raised.value.detail == "trainer uncommitted"
        assert watching.errors == [raised.value]
        assert "evaluate" not in events
        assert "callback.on_run_end" not in events

    def test_a_failure_mid_run_reaches_the_callbacks(self) -> None:
        events: list[str] = []
        watching = Watching(events)
        with pytest.raises(RuntimeError, match="evaluation broke") as raised:
            _engine(Recording(events, fail_on=2)).run(_start(), callbacks=[watching])
        assert watching.errors == [raised.value]

    def test_a_callback_failing_in_on_error_cannot_hide_the_runs_error(self) -> None:
        events: list[str] = []
        with pytest.warns(RuntimeWarning, match="on_error broke"), pytest.raises(Refused):
            _engine(Recording(events, refuse=True)).run(
                _start(), callbacks=[Watching(events, fail_in_on_error=True)]
            )

    def test_a_refused_start_closes_the_tracked_run_as_failed(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mlflow = pytest.importorskip("mlflow")
        monkeypatch.chdir(tmp_path)  # an sqlite store writes artifacts under ./mlruns
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        config = UnifiedConfig(
            population_size=4,
            max_generations=2,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 2, "bounds": (0.0, 1.0)},
            seed=1,
            tracking=TrackingConfig(backend="mlflow", experiment_name="refused", tracking_uri=uri),
        )
        engine = create_engine(config, evaluator=Recording([], refuse=True))
        with pytest.raises(Refused):
            engine.run(create_initial_population(config))
        assert mlflow.active_run() is None
        mlflow.set_tracking_uri(uri)
        runs = mlflow.search_runs(experiment_names=["refused"])
        assert list(runs["status"]) == ["FAILED"]


class TestIslandEngine:
    def _engine(self, evaluator: Any) -> IslandEvolutionEngine:
        return IslandEvolutionEngine(
            config=IslandConfig(n_islands=2, population_per_island=3, max_generations=2),
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=SimulatedBinaryCrossover(),
            mutation=GaussianMutation(),
            seed=1,
        )

    @staticmethod
    def _genome(rng: Random) -> VectorGenome:
        return VectorGenome(genes=np.array([rng.random(), rng.random()]))

    def test_the_start_hook_runs_before_the_first_evaluation(self) -> None:
        events: list[str] = []
        self._engine(Recording(events)).run(self._genome)
        assert events[:2] == ["evaluator.on_run_start", "evaluate"]
        assert events.count("evaluator.on_run_start") == 1

    def test_a_failure_reaches_the_callbacks(self) -> None:
        events: list[str] = []
        watching = Watching(events)
        with pytest.raises(Refused) as raised:
            self._engine(Recording(events, refuse=True)).run(self._genome, callbacks=[watching])
        assert watching.errors == [raised.value]
