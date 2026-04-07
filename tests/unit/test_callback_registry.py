"""Unit tests for CallbackRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evolve.registry.callbacks import (
    CallbackRegistry,
    get_callback_registry,
    reset_callback_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton before and after each test."""
    reset_callback_registry()
    yield
    reset_callback_registry()


class TestSingleton:
    """Singleton behavior."""

    def test_same_instance_returned(self):
        r1 = get_callback_registry()
        r2 = get_callback_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_callback_registry()
        reset_callback_registry()
        r2 = get_callback_registry()
        assert r1 is not r2

    def test_lazy_initialization(self):
        reg = CallbackRegistry()
        assert reg._initialized is False
        reg.list_callbacks()
        assert reg._initialized is True


class TestBuiltinCallbacks:
    """4 built-in callbacks resolve."""

    def test_four_builtins_registered(self):
        reg = get_callback_registry()
        names = reg.list_callbacks()
        assert len(names) == 4
        for name in ["logging", "checkpoint", "print", "history"]:
            assert name in names

    def test_print_callback_resolves(self):
        reg = get_callback_registry()
        cb = reg.get("print", print_every=5)
        assert cb is not None
        assert cb.print_every == 5

    def test_history_callback_resolves(self):
        reg = get_callback_registry()
        cb = reg.get("history")
        assert cb is not None

    def test_logging_callback_resolves(self):
        reg = get_callback_registry()
        cb = reg.get("logging", log_level="DEBUG")
        assert cb is not None

    def test_checkpoint_callback_resolves(self, tmp_path):
        reg = get_callback_registry()
        cb = reg.get("checkpoint", checkpoint_dir=str(tmp_path), checkpoint_frequency=5)
        assert cb is not None


class TestKeyErrorMessage:
    """KeyError for unknown names."""

    def test_unknown_name_raises_keyerror(self):
        reg = get_callback_registry()
        with pytest.raises(KeyError, match="Available callbacks"):
            reg.get("nonexistent")


class TestRegisterOverwrite:
    """Register/overwrite semantics."""

    def test_register_custom(self):
        reg = get_callback_registry()
        reg.register("my_cb", lambda **_kw: MagicMock())
        assert reg.is_registered("my_cb")

    def test_overwrite_builtin(self):
        reg = get_callback_registry()
        assert reg.is_registered("print")
        custom = MagicMock()
        reg.register("print", lambda **_kw: custom)
        assert reg.get("print") is custom

    def test_register_non_callable_raises_typeerror(self):
        reg = get_callback_registry()
        with pytest.raises(TypeError, match="must be callable"):
            reg.register("bad", 42)  # type: ignore[arg-type]
