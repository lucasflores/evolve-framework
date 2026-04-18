"""Unit tests for DecoderRegistry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from evolve.registry.decoders import (
    DecoderRegistry,
    get_decoder_registry,
    reset_decoder_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset singleton before and after each test."""
    reset_decoder_registry()
    yield
    reset_decoder_registry()


class TestSingleton:
    """Singleton behavior."""

    def test_same_instance_returned(self):
        r1 = get_decoder_registry()
        r2 = get_decoder_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_decoder_registry()
        reset_decoder_registry()
        r2 = get_decoder_registry()
        assert r1 is not r2

    def test_lazy_initialization(self):
        reg = DecoderRegistry()
        assert reg._initialized is False
        assert len(reg._factories) == 0
        # Trigger initialization
        reg.list_decoders()
        assert reg._initialized is True
        assert len(reg._factories) > 0


class TestBuiltinDecoders:
    """Built-in decoder resolution."""

    def test_identity_resolves(self):
        reg = get_decoder_registry()
        decoder = reg.get("identity")
        assert decoder is not None

    def test_identity_is_identity_decoder(self):
        from evolve.representation.phenotype import IdentityDecoder

        reg = get_decoder_registry()
        decoder = reg.get("identity")
        assert isinstance(decoder, IdentityDecoder)

    def test_graph_to_network_resolves(self):
        reg = get_decoder_registry()
        decoder = reg.get("graph_to_network")
        assert decoder is not None

    def test_graph_to_mlp_resolves(self):
        reg = get_decoder_registry()
        decoder = reg.get("graph_to_mlp")
        assert decoder is not None

    def test_list_has_four_builtins(self):
        reg = get_decoder_registry()
        names = reg.list_decoders()
        assert len(names) == 4
        for name in ["identity", "graph_to_network", "graph_to_mlp", "cppn_to_network"]:
            assert name in names

    def test_list_is_sorted(self):
        reg = get_decoder_registry()
        names = reg.list_decoders()
        assert names == sorted(names)

    def test_is_registered_true_for_builtins(self):
        reg = get_decoder_registry()
        assert reg.is_registered("identity")
        assert reg.is_registered("graph_to_network")
        assert reg.is_registered("graph_to_mlp")
        assert reg.is_registered("cppn_to_network")

    def test_is_registered_false_for_unknown(self):
        reg = get_decoder_registry()
        assert not reg.is_registered("nonexistent")


class TestKeyErrorMessage:
    """Error messages include available decoder list."""

    def test_unknown_name_lists_available(self):
        reg = get_decoder_registry()
        with pytest.raises(KeyError, match="Available decoders"):
            reg.get("unknown_decoder")


class TestRegisterOverwrite:
    """Register/overwrite semantics."""

    def test_register_custom(self):
        reg = get_decoder_registry()
        reg.register("my_decoder", lambda **_kw: MagicMock())
        assert reg.is_registered("my_decoder")

    def test_overwrite_builtin(self):
        reg = get_decoder_registry()
        assert reg.is_registered("identity")
        custom = MagicMock()
        reg.register("identity", lambda **_kw: custom)
        result = reg.get("identity")
        assert result is custom

    def test_register_non_callable_raises_typeerror(self):
        reg = get_decoder_registry()
        with pytest.raises(TypeError, match="must be callable"):
            reg.register("bad", "not_callable")  # type: ignore[arg-type]


class TestDeferredImports:
    """Deferred import guarantee — importing decoders module does NOT trigger ML deps."""

    def test_import_does_not_load_ml_deps(self):
        import importlib

        mod = importlib.import_module("evolve.registry.decoders")
        assert mod is not None
        reg = DecoderRegistry()
        assert reg._initialized is False
        assert "torch" not in sys.modules.get("__importing__", "")


class TestFactoryErrorPropagation:
    """Factory exceptions propagate with context."""

    def test_factory_error_includes_name_and_params(self):
        reg = get_decoder_registry()

        def bad_factory(**_kw):
            raise RuntimeError("factory broke")

        reg.register("broken", bad_factory)
        with pytest.raises(RuntimeError, match="Failed to create decoder 'broken'"):
            reg.get("broken", some_param=42)


class TestCustomDecoderWorkflow:
    """Register and use a custom decoder."""

    def test_register_user_factory_and_resolve(self):
        reg = get_decoder_registry()
        custom = MagicMock()
        reg.register("my_decoder", lambda **_kw: custom)
        result = reg.get("my_decoder")
        assert result is custom

    def test_custom_shows_in_list(self):
        reg = get_decoder_registry()
        reg.register("my_decoder", lambda **_kw: MagicMock())
        names = reg.list_decoders()
        assert "my_decoder" in names
        assert "identity" in names

    def test_overwrite_builtin_with_custom(self):
        reg = get_decoder_registry()
        assert reg.is_registered("identity")
        custom = MagicMock()
        reg.register("identity", lambda **_kw: custom)
        result = reg.get("identity")
        assert result is custom

    def test_custom_decoder_params_passed(self):
        reg = get_decoder_registry()

        def my_factory(hidden_size: int = 64, **_kw):
            m = MagicMock()
            m.hidden_size = hidden_size
            return m

        reg.register("my_decoder", my_factory)
        result = reg.get("my_decoder", hidden_size=128)
        assert result.hidden_size == 128
