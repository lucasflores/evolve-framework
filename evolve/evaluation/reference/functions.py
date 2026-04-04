"""
Reference benchmark functions for testing optimization.

These are standard test functions for evolutionary algorithms.
All functions are implemented in pure NumPy for CPU reference.

Functions follow the convention:
- Input: numpy array of shape (n,) for single evaluation
         or shape (batch, n) for batch evaluation
- Output: scalar fitness value (lower is better for minimization)
"""

from __future__ import annotations

import numpy as np


def sphere(x: np.ndarray) -> float | np.ndarray:
    """
    Sphere function (De Jong's F1).

    f(x) = sum(x_i^2)

    Properties:
    - Unimodal, convex, separable
    - Global minimum: f(0, ..., 0) = 0
    - Search domain: usually [-5.12, 5.12]^n

    Args:
        x: Input vector(s), shape (n,) or (batch, n)

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        return float(np.sum(x**2))
    else:
        return np.sum(x**2, axis=1)


def rastrigin(x: np.ndarray, A: float = 10.0) -> float | np.ndarray:
    """
    Rastrigin function.

    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))

    Properties:
    - Highly multimodal, non-convex, separable
    - Global minimum: f(0, ..., 0) = 0
    - Search domain: usually [-5.12, 5.12]^n
    - Many local minima (~10^n for n dimensions)

    Args:
        x: Input vector(s), shape (n,) or (batch, n)
        A: Amplitude parameter (default: 10.0)

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        n = len(x)
        return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    else:
        n = x.shape[1]
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def rosenbrock(x: np.ndarray) -> float | np.ndarray:
    """
    Rosenbrock function (banana function).

    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Properties:
    - Unimodal (for n≤3), non-convex, non-separable
    - Global minimum: f(1, ..., 1) = 0
    - Search domain: usually [-5, 10]^n
    - Has a narrow, parabolic valley

    Args:
        x: Input vector(s), shape (n,) or (batch, n)

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    else:
        return np.sum(
            100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
            axis=1,
        )


def ackley(
    x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> float | np.ndarray:
    """
    Ackley function.

    Properties:
    - Multimodal, non-convex, non-separable
    - Global minimum: f(0, ..., 0) = 0
    - Search domain: usually [-32.768, 32.768]^n
    - Has many local minima but one global minimum

    Args:
        x: Input vector(s), shape (n,) or (batch, n)
        a, b, c: Function parameters

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)
    else:
        n = x.shape[1]
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(np.cos(c * x), axis=1)
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e


def griewank(x: np.ndarray) -> float | np.ndarray:
    """
    Griewank function.

    Properties:
    - Multimodal, non-convex, non-separable
    - Global minimum: f(0, ..., 0) = 0
    - Search domain: usually [-600, 600]^n

    Args:
        x: Input vector(s), shape (n,) or (batch, n)

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        n = len(x)
        i = np.arange(1, n + 1)
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(i)))
        return float(sum_term - prod_term + 1)
    else:
        n = x.shape[1]
        i = np.arange(1, n + 1)
        sum_term = np.sum(x**2, axis=1) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(i)), axis=1)
        return sum_term - prod_term + 1


def schwefel(x: np.ndarray) -> float | np.ndarray:
    """
    Schwefel function.

    Properties:
    - Multimodal, non-convex, separable
    - Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    - Search domain: usually [-500, 500]^n
    - Has deceptive structure (best local far from global)

    Args:
        x: Input vector(s), shape (n,) or (batch, n)

    Returns:
        Fitness value(s)
    """
    if x.ndim == 1:
        n = len(x)
        return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))
    else:
        n = x.shape[1]
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


# Multi-objective test functions (ZDT suite)


def zdt1(x: np.ndarray) -> np.ndarray:
    """
    ZDT1 benchmark function.

    Properties:
    - Two objectives, convex Pareto front
    - Pareto optimal front: f2 = 1 - sqrt(f1)
    - Search domain: [0, 1]^n

    Args:
        x: Input vector, shape (n,) where n >= 2

    Returns:
        Array of shape (2,) with [f1, f2]
    """
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """
    ZDT2 benchmark function.

    Properties:
    - Two objectives, non-convex Pareto front
    - Pareto optimal front: f2 = 1 - f1^2
    - Search domain: [0, 1]^n

    Args:
        x: Input vector, shape (n,) where n >= 2

    Returns:
        Array of shape (2,) with [f1, f2]
    """
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])


def zdt3(x: np.ndarray) -> np.ndarray:
    """
    ZDT3 benchmark function.

    Properties:
    - Two objectives, disconnected Pareto front
    - Search domain: [0, 1]^n

    Args:
        x: Input vector, shape (n,) where n >= 2

    Returns:
        Array of shape (2,) with [f1, f2]
    """
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
    return np.array([f1, f2])


# Function registry for easy access
BENCHMARK_FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "griewank": griewank,
    "schwefel": schwefel,
    "zdt1": zdt1,
    "zdt2": zdt2,
    "zdt3": zdt3,
}


def get_function(name: str):
    """
    Get benchmark function by name.

    Args:
        name: Function name (case-insensitive)

    Returns:
        Benchmark function

    Raises:
        KeyError: If function not found
    """
    return BENCHMARK_FUNCTIONS[name.lower()]


def get_bounds(name: str, n_dims: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get standard bounds for a benchmark function.

    Args:
        name: Function name
        n_dims: Number of dimensions

    Returns:
        (lower_bounds, upper_bounds) arrays
    """
    bounds_map = {
        "sphere": (-5.12, 5.12),
        "rastrigin": (-5.12, 5.12),
        "rosenbrock": (-5.0, 10.0),
        "ackley": (-32.768, 32.768),
        "griewank": (-600.0, 600.0),
        "schwefel": (-500.0, 500.0),
        "zdt1": (0.0, 1.0),
        "zdt2": (0.0, 1.0),
        "zdt3": (0.0, 1.0),
    }

    low, high = bounds_map[name.lower()]
    return np.full(n_dims, low), np.full(n_dims, high)
