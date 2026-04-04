"""
Sandboxed execution for Evolvable Reproduction Protocols.

This module provides resource limiting and safe execution of protocol
evaluation to ensure system stability even with adversarial protocols.

Key Components:
- StepCounter: Tracks execution steps and enforces limits
- StepLimitExceeded: Exception raised when limit exceeded
- sandboxed_execute: Context manager for safe protocol execution
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class StepLimitExceeded(Exception):
    """
    Raised when protocol execution exceeds step limit.

    This exception is caught by safe evaluation wrappers to ensure
    the evolutionary system continues operating even when individual
    protocols misbehave.

    Attributes:
        count: Number of steps taken
        limit: Maximum allowed steps
    """

    def __init__(self, count: int, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(f"Step limit exceeded: {count} > {limit}")


@dataclass
class StepCounter:
    """
    Counts execution steps and enforces limits.

    Used to sandbox protocol evaluation and prevent infinite loops
    or excessive computation. Each protocol evaluation should receive
    a fresh StepCounter or call reset() before use.

    Thread-safe within a single protocol evaluation (no shared state).

    Attributes:
        limit: Maximum steps allowed (default: 1000)
        count: Current step count

    Example:
        >>> counter = StepCounter(limit=100)
        >>> for i in range(50):
        ...     counter.step()  # OK
        >>> counter.count
        50
        >>> for i in range(60):
        ...     counter.step()  # Raises StepLimitExceeded at step 101
    """

    limit: int = 1000
    count: int = 0

    def step(self, n: int = 1) -> None:
        """
        Increment counter and check limit.

        Args:
            n: Number of steps to add (default: 1)

        Raises:
            StepLimitExceeded: If count exceeds limit after increment
        """
        self.count += n
        if self.count > self.limit:
            raise StepLimitExceeded(self.count, self.limit)

    def reset(self) -> None:
        """Reset counter to zero for reuse."""
        self.count = 0

    @property
    def remaining(self) -> int:
        """Number of steps remaining before limit."""
        return max(0, self.limit - self.count)

    def __enter__(self) -> StepCounter:
        """Context manager entry - reset counter."""
        self.reset()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        pass


@contextmanager
def sandboxed_execute(
    limit: int = 1000,
) -> Generator[StepCounter, None, None]:
    """
    Context manager for sandboxed protocol execution.

    Creates a fresh StepCounter and provides it for protocol evaluation.
    The counter is automatically reset on entry.

    Args:
        limit: Maximum steps allowed

    Yields:
        StepCounter for the evaluation

    Example:
        >>> with sandboxed_execute(limit=100) as counter:
        ...     # Protocol evaluation here
        ...     counter.step()
        ...     result = evaluate_something()
    """
    counter = StepCounter(limit=limit)
    yield counter


def safe_execute(
    func: Callable[..., T],
    default: T,
    *args: Any,
    step_limit: int = 1000,
    **kwargs: Any,
) -> tuple[T, bool]:
    """
    Execute a function with step limiting and exception handling.

    If the function exceeds the step limit or raises any exception,
    the default value is returned instead.

    Args:
        func: Function to execute
        default: Value to return on failure
        *args: Positional arguments for func
        step_limit: Maximum steps allowed
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (result, success) where:
        - result: Function return value or default on failure
        - success: True if function completed normally

    Example:
        >>> def risky_eval(context, counter):
        ...     for _ in range(10000):
        ...         counter.step()  # Will exceed limit
        ...     return True
        >>> result, success = safe_execute(risky_eval, False, context, step_limit=100)
        >>> success
        False
        >>> result
        False
    """
    counter = StepCounter(limit=step_limit)
    try:
        # Inject counter into kwargs if func expects it
        if "counter" in kwargs or (
            hasattr(func, "__code__") and "counter" in func.__code__.co_varnames
        ):
            kwargs["counter"] = counter
            result = func(*args, **kwargs)
        else:
            # Try calling with counter as positional arg
            result = func(*args, counter, **kwargs)
        return result, True
    except StepLimitExceeded:
        return default, False
    except Exception:
        # Catch all exceptions to ensure stability
        return default, False
