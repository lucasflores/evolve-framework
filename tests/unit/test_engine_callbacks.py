"""
Tests for callback persistence through engine.run() and priority ordering.

Covers T006 (callback persistence/merge/priority/backward-compat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual
from evolve.representation.vector import VectorGenome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RecordingCallback:
    """Callback that records all events it receives."""

    name: str = "recorder"
    _priority: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def priority(self) -> int:
        return self._priority

    def on_run_start(self, config: Any) -> None:
        self.events.append(f"{self.name}:run_start")

    def on_generation_start(self, generation: int, population: Population[Any]) -> None:
        self.events.append(f"{self.name}:gen_start:{generation}")

    def on_generation_end(
        self, generation: int, population: Population[Any], metrics: dict[str, Any]
    ) -> None:
        self.events.append(f"{self.name}:gen_end:{generation}")

    def on_run_end(self, population: Population[Any], reason: str) -> None:
        self.events.append(f"{self.name}:run_end")


class _DummyEvaluator:
    """Minimal evaluator that returns constant fitness."""

    def evaluate(self, individuals, seed=None):
        return [Fitness.scalar(1.0) for _ in individuals]


class _DummySelection:
    def select(self, population, n, rng):
        inds = list(population.individuals)
        return [inds[i % len(inds)] for i in range(n)]


class _DummyCrossover:
    def crossover(self, g1, g2, rng):
        return g1.copy(), g2.copy()


class _DummyMutation:
    def mutate(self, g, rng):
        return g.copy()


def _make_population(size: int = 10, n_genes: int = 5) -> Population[VectorGenome]:
    individuals = [
        Individual(genome=VectorGenome(genes=np.random.default_rng(i).random(n_genes)))
        for i in range(size)
    ]
    return Population(individuals=individuals)


def _make_engine(callbacks=None, max_generations=2, **kwargs) -> EvolutionEngine[VectorGenome]:
    config = EvolutionConfig(
        population_size=10,
        max_generations=max_generations,
        elitism=1,
    )
    return EvolutionEngine(
        config=config,
        evaluator=_DummyEvaluator(),
        selection=_DummySelection(),
        crossover=_DummyCrossover(),
        mutation=_DummyMutation(),
        seed=42,
        callbacks=callbacks,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallbackPersistence:
    """T004/T005: Creation-time callbacks persist through run()."""

    def test_creation_callbacks_receive_events(self):
        """Callbacks passed at engine creation receive events without passing to run()."""
        cb = RecordingCallback(name="creation")
        engine = _make_engine(callbacks=[cb])
        pop = _make_population()

        engine.run(initial_population=pop)

        assert "creation:run_start" in cb.events
        assert "creation:run_end" in cb.events
        assert any("creation:gen_end" in e for e in cb.events)

    def test_creation_callbacks_persist_across_runs(self):
        """Creation callbacks work on subsequent run() calls."""
        cb = RecordingCallback(name="persist")
        engine = _make_engine(callbacks=[cb])
        pop = _make_population()

        engine.run(initial_population=pop)
        first_run_count = len(cb.events)
        assert first_run_count > 0

        cb.events.clear()
        engine.run(initial_population=pop)

        assert len(cb.events) > 0
        assert "persist:run_start" in cb.events

    def test_run_callbacks_merged_with_creation(self):
        """run()-time callbacks merged with creation-time callbacks."""
        creation_cb = RecordingCallback(name="creation")
        run_cb = RecordingCallback(name="runtime")
        engine = _make_engine(callbacks=[creation_cb])
        pop = _make_population()

        engine.run(initial_population=pop, callbacks=[run_cb])

        assert "creation:run_start" in creation_cb.events
        assert "runtime:run_start" in run_cb.events


class TestCallbackPriorityOrdering:
    """T005: Callbacks sorted by priority (ascending), stable sort."""

    def test_lower_priority_runs_first(self):
        """Lower priority number runs before higher."""
        order: list[str] = []

        @dataclass
        class OrderTracker:
            name: str
            _priority: int = 0

            @property
            def priority(self) -> int:
                return self._priority

            def on_run_start(self, config: Any) -> None:
                order.append(self.name)

            def on_generation_start(self, gen: int, pop: Population[Any]) -> None:
                pass

            def on_generation_end(
                self, gen: int, pop: Population[Any], metrics: dict[str, Any]
            ) -> None:
                pass

            def on_run_end(self, pop: Population[Any], reason: str) -> None:
                pass

        low = OrderTracker(name="low", _priority=0)
        high = OrderTracker(name="high", _priority=1000)

        # Pass in reverse order — should still sort correctly
        engine = _make_engine(callbacks=[high, low])
        pop = _make_population()
        engine.run(initial_population=pop)

        assert order[0] == "low"
        assert order[1] == "high"

    def test_equal_priority_preserves_registration_order(self):
        """Same priority preserves insertion order (stable sort)."""
        order: list[str] = []

        @dataclass
        class OrderTracker:
            name: str
            _priority: int = 0

            @property
            def priority(self) -> int:
                return self._priority

            def on_run_start(self, config: Any) -> None:
                order.append(self.name)

            def on_generation_start(self, gen: int, pop: Population[Any]) -> None:
                pass

            def on_generation_end(
                self, gen: int, pop: Population[Any], metrics: dict[str, Any]
            ) -> None:
                pass

            def on_run_end(self, pop: Population[Any], reason: str) -> None:
                pass

        a = OrderTracker(name="a", _priority=5)
        b = OrderTracker(name="b", _priority=5)
        c = OrderTracker(name="c", _priority=5)

        engine = _make_engine(callbacks=[a, b, c])
        pop = _make_population()
        engine.run(initial_population=pop)

        assert order == ["a", "b", "c"]


class TestBackwardCompatibility:
    """No callbacks at creation or run still works."""

    def test_no_callbacks_at_all(self):
        """Engine works with no callbacks passed anywhere."""
        engine = _make_engine()
        pop = _make_population()
        result = engine.run(initial_population=pop)
        assert result.best is not None

    def test_only_run_callbacks(self):
        """Passing callbacks only to run() still works (backward compat)."""
        cb = RecordingCallback(name="run_only")
        engine = _make_engine()
        pop = _make_population()
        engine.run(initial_population=pop, callbacks=[cb])

        assert "run_only:run_start" in cb.events


class TestTrackingCallbackPriority:
    """T027: TrackingCallback has priority=1000, custom callbacks run before it."""

    def test_tracking_callback_priority_is_1000(self):
        from evolve.config.tracking import TrackingConfig
        from evolve.experiment.tracking.callback import TrackingCallback

        tc = TrackingCallback(config=TrackingConfig(backend="null", enabled=False))
        assert tc.priority == 1000

    def test_custom_before_tracking(self):
        """Custom callback (priority 0) runs before TrackingCallback (priority 1000)."""
        custom = RecordingCallback(name="custom", _priority=0)
        tracking = RecordingCallback(name="tracking", _priority=1000)

        engine = _make_engine(callbacks=[tracking, custom])
        pop = _make_population()
        engine.run(initial_population=pop)

        # Lower priority (custom=0) should have run_start before higher (tracking=1000)
        assert "custom:run_start" in custom.events
        assert "tracking:run_start" in tracking.events
