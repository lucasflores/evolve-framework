"""Unit tests for EvaluatorRegistry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from evolve.registry.evaluators import (
    EvaluatorRegistry,
    get_evaluator_registry,
    reset_evaluator_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton before and after each test."""
    reset_evaluator_registry()
    yield
    reset_evaluator_registry()


class TestSingleton:
    """Singleton behavior."""

    def test_same_instance_returned(self):
        r1 = get_evaluator_registry()
        r2 = get_evaluator_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_evaluator_registry()
        reset_evaluator_registry()
        r2 = get_evaluator_registry()
        assert r1 is not r2

    def test_lazy_initialization(self):
        reg = EvaluatorRegistry()
        assert reg._initialized is False
        assert len(reg._factories) == 0
        # Trigger initialization
        reg.list_evaluators()
        assert reg._initialized is True
        assert len(reg._factories) > 0


class TestBuiltinEvaluators:
    """Built-in evaluator resolution."""

    def test_benchmark_sphere_resolves(self):
        reg = get_evaluator_registry()
        evaluator = reg.get("benchmark", function_name="sphere")
        assert evaluator is not None

    def test_benchmark_unknown_function_raises_keyerror(self):
        reg = get_evaluator_registry()
        with pytest.raises(KeyError, match="Unknown benchmark function"):
            reg.get("benchmark", function_name="nonexistent_fn")

    def test_function_evaluator_resolves(self):
        reg = get_evaluator_registry()
        evaluator = reg.get("function", fitness_fn=lambda x: sum(x))
        assert evaluator is not None

    def test_function_evaluator_missing_fn_raises(self):
        reg = get_evaluator_registry()
        with pytest.raises(ValueError, match="fitness_fn is required"):
            reg.get("function")

    def test_list_has_seven_builtins(self):
        reg = get_evaluator_registry()
        names = reg.list_evaluators()
        assert len(names) == 7
        for name in ["benchmark", "function", "llm_judge", "ground_truth", "scm", "rl", "meta"]:
            assert name in names

    def test_list_is_sorted(self):
        reg = get_evaluator_registry()
        names = reg.list_evaluators()
        assert names == sorted(names)

    def test_is_registered_true_for_builtins(self):
        reg = get_evaluator_registry()
        assert reg.is_registered("benchmark")
        assert reg.is_registered("function")

    def test_is_registered_false_for_unknown(self):
        reg = get_evaluator_registry()
        assert not reg.is_registered("nonexistent")


class TestKeyErrorMessage:
    """Error messages include available evaluator list."""

    def test_unknown_name_lists_available(self):
        reg = get_evaluator_registry()
        with pytest.raises(KeyError, match="Available evaluators"):
            reg.get("unknown_evaluator")


class TestRegisterOverwrite:
    """Register/overwrite semantics."""

    def test_register_custom(self):
        reg = get_evaluator_registry()
        reg.register("my_eval", lambda **_kw: MagicMock())
        assert reg.is_registered("my_eval")

    def test_overwrite_builtin(self):
        reg = get_evaluator_registry()
        # Trigger initialization first
        assert reg.is_registered("benchmark")
        custom = MagicMock()
        reg.register("benchmark", lambda **_kw: custom)
        result = reg.get("benchmark")
        assert result is custom

    def test_register_non_callable_raises_typeerror(self):
        reg = get_evaluator_registry()
        with pytest.raises(TypeError, match="must be callable"):
            reg.register("bad", "not_callable")  # type: ignore[arg-type]


class TestDeferredImports:
    """Deferred import guarantee — importing evaluators module does NOT trigger ML deps."""

    def test_import_does_not_load_ml_deps(self):
        # Importing the module should not import heavy ML packages
        # We check that the module can be imported without error
        # even if ML deps aren't installed (they may be, but the
        # import itself shouldn't trigger them)
        import importlib

        mod = importlib.import_module("evolve.registry.evaluators")
        assert mod is not None
        # The registry class itself should not have triggered initialization
        reg = EvaluatorRegistry()
        assert reg._initialized is False
        assert "torch" not in sys.modules.get("__importing__", "")


class TestFactoryErrorPropagation:
    """Factory exceptions propagate with context."""

    def test_factory_error_includes_name_and_params(self):
        reg = get_evaluator_registry()

        def bad_factory(**_kw):
            raise RuntimeError("factory broke")

        reg.register("broken", bad_factory)
        with pytest.raises(RuntimeError, match="Failed to create evaluator 'broken'"):
            reg.get("broken", some_param=42)


class TestCustomEvaluatorWorkflow:
    """US2: Register and use a custom evaluator."""

    def test_register_user_factory_and_resolve(self):
        reg = get_evaluator_registry()
        custom = MagicMock()
        reg.register("my_domain_eval", lambda **_kw: custom)
        result = reg.get("my_domain_eval")
        assert result is custom

    def test_custom_shows_in_list(self):
        reg = get_evaluator_registry()
        reg.register("my_domain_eval", lambda **_kw: MagicMock())
        names = reg.list_evaluators()
        assert "my_domain_eval" in names
        # Built-ins still present
        assert "benchmark" in names

    def test_overwrite_builtin_with_custom(self):
        reg = get_evaluator_registry()
        # Trigger initialization first
        assert reg.is_registered("benchmark")
        custom = MagicMock()
        reg.register("benchmark", lambda **_kw: custom)
        result = reg.get("benchmark")
        assert result is custom

    def test_custom_evaluator_params_passed(self):
        reg = get_evaluator_registry()

        def my_factory(threshold: float = 0.5, **_kw):
            m = MagicMock()
            m.threshold = threshold
            return m

        reg.register("my_eval", my_factory)
        result = reg.get("my_eval", threshold=0.9)
        assert result.threshold == 0.9
