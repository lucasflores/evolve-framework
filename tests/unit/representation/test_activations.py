"""Unit tests for CPPN activation functions (sin, abs)."""

from __future__ import annotations

import numpy as np
import pytest

from evolve.representation.network import (
    ACTIVATIONS,
    abs_activation,
    get_activation,
    sin_activation,
)


class TestSinActivation:
    """Tests for sin activation function."""

    def test_registered_in_activations(self):
        assert "sin" in ACTIVATIONS

    def test_get_activation_returns_callable(self):
        fn = get_activation("sin")
        assert callable(fn)

    def test_sin_values(self):
        x = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        result = sin_activation(x)
        expected = np.sin(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sin_zero(self):
        result = sin_activation(np.array([0.0]))
        assert result[0] == pytest.approx(0.0)


class TestAbsActivation:
    """Tests for abs activation function."""

    def test_registered_in_activations(self):
        assert "abs" in ACTIVATIONS

    def test_get_activation_returns_callable(self):
        fn = get_activation("abs")
        assert callable(fn)

    def test_abs_values(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = abs_activation(x)
        expected = np.abs(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_abs_negative(self):
        result = abs_activation(np.array([-5.0]))
        assert result[0] == pytest.approx(5.0)


class TestExistingCPPNActivations:
    """Verify gaussian and linear already exist (T021)."""

    def test_gaussian_registered(self):
        assert "gaussian" in ACTIVATIONS
        fn = get_activation("gaussian")
        assert callable(fn)

    def test_linear_registered(self):
        assert "linear" in ACTIVATIONS
        fn = get_activation("linear")
        assert callable(fn)

    def test_gaussian_values(self):
        fn = get_activation("gaussian")
        x = np.array([0.0, 1.0, -1.0])
        result = fn(x)
        expected = np.exp(-x * x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_linear_is_identity(self):
        fn = get_activation("linear")
        x = np.array([1.0, 2.0, -3.0])
        result = fn(x)
        np.testing.assert_allclose(result, x, atol=1e-10)
