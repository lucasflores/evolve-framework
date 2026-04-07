"""Integration tests for declarative engine creation."""

from __future__ import annotations

import pytest

from evolve.config.unified import UnifiedConfig
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.factory.engine import create_engine
from evolve.registry.callbacks import reset_callback_registry
from evolve.registry.evaluators import reset_evaluator_registry


@pytest.fixture(autouse=True)
def _reset_registries():
    """Reset all singletons before and after each test."""
    reset_evaluator_registry()
    reset_callback_registry()
    yield
    reset_evaluator_registry()
    reset_callback_registry()


def _sphere(x):
    """Simple sphere function for testing."""
    return sum(xi**2 for xi in x)


class TestDeclarativeEvaluatorResolution:
    """US1: Declare evaluator in config, create_engine resolves it."""

    def test_benchmark_evaluator_resolves_and_runs(self):
        config = UnifiedConfig(
            name="test_declarative",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )
        engine = create_engine(config)
        assert engine is not None

    def test_explicit_evaluator_overrides_config(self):
        config = UnifiedConfig(
            name="test_override",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )
        explicit_eval = FunctionEvaluator(fitness_fn=_sphere)
        engine = create_engine(config, evaluator=explicit_eval)
        assert engine is not None

    def test_missing_evaluator_raises_valueerror(self):
        config = UnifiedConfig(
            name="test_no_eval",
            population_size=10,
            max_generations=2,
            genome_type="vector",
        )
        with pytest.raises(ValueError, match="No evaluator provided"):
            create_engine(config)

    def test_missing_evaluator_error_lists_available(self):
        config = UnifiedConfig(
            name="test_no_eval",
            population_size=10,
            max_generations=2,
            genome_type="vector",
        )
        with pytest.raises(ValueError, match="Available evaluators"):
            create_engine(config)

    def test_callable_evaluator_still_works(self):
        config = UnifiedConfig(
            name="test_callable",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
        )
        engine = create_engine(config, evaluator=_sphere)
        assert engine is not None

    def test_runtime_overrides_merge_with_evaluator_params(self):
        config = UnifiedConfig(
            name="test_overrides",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )
        # runtime_overrides override evaluator_params
        engine = create_engine(config, runtime_overrides={"function_name": "rastrigin"})
        assert engine is not None


class TestCombinedFlow:
    """Full combined flow: config with evaluator + custom_callbacks + new fields."""

    def test_full_declarative_config_creates_engine(self):
        config = UnifiedConfig(
            name="full_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            custom_callbacks=(
                {"name": "history"},
                {"name": "print", "params": {"print_every": 1}},
            ),
        )
        engine = create_engine(config)
        assert engine is not None
        # Should have callbacks: CallbackConfig-derived (0 by default) + custom (2) + explicit (0)
        assert len(engine._callbacks) >= 2


class TestCallbackWiring:
    """US3: Declare custom callbacks in config, factory resolves them."""

    def test_custom_callbacks_resolved_from_registry(self):
        config = UnifiedConfig(
            name="cb_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            custom_callbacks=({"name": "history"},),
        )
        engine = create_engine(config)
        from evolve.core.callbacks import HistoryCallback

        history_cbs = [cb for cb in engine._callbacks if isinstance(cb, HistoryCallback)]
        assert len(history_cbs) == 1

    def test_execution_order_config_custom_explicit(self):
        """Callback order: Config-derived → Custom → Explicit."""
        from evolve.config.callbacks import CallbackConfig
        from evolve.core.callbacks import (
            HistoryCallback,
            LoggingCallback,
            PrintCallback,
        )

        config = UnifiedConfig(
            name="order_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            callbacks=CallbackConfig(enable_logging=True, enable_checkpointing=False),
            custom_callbacks=({"name": "history"},),
        )
        explicit_cb = PrintCallback(print_every=1)
        engine = create_engine(config, callbacks=[explicit_cb])

        cbs = engine._callbacks
        # Order: LoggingCallback (from Config) → HistoryCallback (custom) → PrintCallback (explicit)
        logging_idx = next(i for i, cb in enumerate(cbs) if isinstance(cb, LoggingCallback))
        history_idx = next(i for i, cb in enumerate(cbs) if isinstance(cb, HistoryCallback))
        print_idx = next(i for i, cb in enumerate(cbs) if isinstance(cb, PrintCallback))
        assert logging_idx < history_idx < print_idx

    def test_empty_custom_callbacks_is_noop(self):
        config = UnifiedConfig(
            name="noop_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            custom_callbacks=(),
        )
        engine = create_engine(config)
        # No CallbackConfig and no custom_callbacks → 0 callbacks
        assert len(engine._callbacks) == 0

    def test_unregistered_callback_name_raises(self):
        config = UnifiedConfig(
            name="bad_cb_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            custom_callbacks=({"name": "nonexistent_callback"},),
        )
        with pytest.raises(KeyError, match="not registered"):
            create_engine(config)

    def test_duplicate_callback_name_both_resolve(self):
        config = UnifiedConfig(
            name="dup_test",
            population_size=10,
            max_generations=2,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-5.12, 5.12)},
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            custom_callbacks=(
                {"name": "history"},
                {"name": "history"},
            ),
        )
        engine = create_engine(config)
        from evolve.core.callbacks import HistoryCallback

        history_cbs = [cb for cb in engine._callbacks if isinstance(cb, HistoryCallback)]
        assert len(history_cbs) == 2
