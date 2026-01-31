"""
Unit tests for sandbox module.

Tests cover:
- StepCounter counting and limit enforcement
- StepLimitExceeded exception
- sandboxed_execute context manager
- safe_execute wrapper function
"""

import pytest

from evolve.reproduction.sandbox import (
    StepCounter,
    StepLimitExceeded,
    sandboxed_execute,
    safe_execute,
)


class TestStepCounter:
    """Tests for StepCounter class."""

    def test_initial_state(self) -> None:
        counter = StepCounter(limit=100)
        assert counter.count == 0
        assert counter.limit == 100
        assert counter.remaining == 100

    def test_default_limit(self) -> None:
        counter = StepCounter()
        assert counter.limit == 1000

    def test_step_increments(self) -> None:
        counter = StepCounter(limit=100)
        counter.step()
        assert counter.count == 1
        counter.step(5)
        assert counter.count == 6

    def test_remaining_decreases(self) -> None:
        counter = StepCounter(limit=100)
        counter.step(30)
        assert counter.remaining == 70

    def test_reset(self) -> None:
        counter = StepCounter(limit=100)
        counter.step(50)
        counter.reset()
        assert counter.count == 0
        assert counter.remaining == 100

    def test_raises_when_exceeded(self) -> None:
        counter = StepCounter(limit=10)
        counter.step(10)  # At limit, should be OK
        with pytest.raises(StepLimitExceeded) as exc_info:
            counter.step()  # Exceeds limit
        assert exc_info.value.count == 11
        assert exc_info.value.limit == 10

    def test_raises_immediately_if_large_step(self) -> None:
        counter = StepCounter(limit=10)
        with pytest.raises(StepLimitExceeded):
            counter.step(20)

    def test_context_manager_resets(self) -> None:
        counter = StepCounter(limit=100)
        counter.step(50)
        with counter:
            assert counter.count == 0
        # After context, count should still be 0 (no steps in context)

    def test_remaining_floors_at_zero(self) -> None:
        counter = StepCounter(limit=10)
        counter.count = 20  # Manually exceed
        assert counter.remaining == 0


class TestStepLimitExceeded:
    """Tests for StepLimitExceeded exception."""

    def test_exception_attributes(self) -> None:
        exc = StepLimitExceeded(count=150, limit=100)
        assert exc.count == 150
        assert exc.limit == 100

    def test_exception_message(self) -> None:
        exc = StepLimitExceeded(count=150, limit=100)
        assert "150" in str(exc)
        assert "100" in str(exc)


class TestSandboxedExecute:
    """Tests for sandboxed_execute context manager."""

    def test_provides_counter(self) -> None:
        with sandboxed_execute(limit=50) as counter:
            assert isinstance(counter, StepCounter)
            assert counter.limit == 50
            assert counter.count == 0

    def test_counter_usable_in_context(self) -> None:
        with sandboxed_execute(limit=100) as counter:
            counter.step(10)
            assert counter.count == 10

    def test_default_limit(self) -> None:
        with sandboxed_execute() as counter:
            assert counter.limit == 1000


class TestSafeExecute:
    """Tests for safe_execute wrapper function."""

    def test_successful_execution(self) -> None:
        def simple_func(x: int, counter: StepCounter) -> int:
            counter.step()
            return x * 2

        result, success = safe_execute(simple_func, default=-1, x=5)
        assert result == 10
        assert success is True

    def test_returns_default_on_step_limit(self) -> None:
        def expensive_func(counter: StepCounter) -> str:
            for _ in range(1000):
                counter.step()
            return "done"

        result, success = safe_execute(expensive_func, default="failed", step_limit=10)
        assert result == "failed"
        assert success is False

    def test_returns_default_on_exception(self) -> None:
        def failing_func(counter: StepCounter) -> str:
            counter.step()
            raise ValueError("oops")

        result, success = safe_execute(failing_func, default="error", step_limit=100)
        assert result == "error"
        assert success is False

    def test_respects_step_limit_parameter(self) -> None:
        def counting_func(counter: StepCounter) -> int:
            for i in range(100):
                counter.step()
            return 42

        # Should fail with small limit
        result, success = safe_execute(counting_func, default=0, step_limit=50)
        assert result == 0
        assert success is False

        # Should succeed with large limit
        result, success = safe_execute(counting_func, default=0, step_limit=200)
        assert result == 42
        assert success is True


class TestStepLimitEnforcement:
    """Integration tests for step limit enforcement in matchability."""

    def test_adversarial_evaluation_terminates(self) -> None:
        """
        Verify that a malicious evaluation that would loop forever
        is terminated by the step counter.
        """
        def infinite_loop(counter: StepCounter) -> bool:
            while True:
                counter.step()
            return True  # Never reached

        result, success = safe_execute(
            infinite_loop,
            default=False,
            step_limit=100,
        )
        assert result is False
        assert success is False

    def test_complex_evaluation_within_limit(self) -> None:
        """
        Verify that a complex but bounded evaluation completes.
        """
        def complex_eval(counter: StepCounter) -> float:
            total = 0.0
            for i in range(50):
                counter.step()
                total += i * 0.01
            return total

        result, success = safe_execute(
            complex_eval,
            default=-1.0,
            step_limit=100,
        )
        assert success is True
        assert result == pytest.approx(12.25)  # sum(0..49) * 0.01
