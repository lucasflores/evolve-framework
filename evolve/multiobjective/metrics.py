"""
Multi-Objective Optimization Metrics.

Implements hypervolume indicator and other performance metrics.
"""

from __future__ import annotations

import numpy as np


def hypervolume_2d(
    points: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Calculate hypervolume for 2D Pareto front.

    Hypervolume is the volume of objective space dominated by
    the Pareto front and bounded by a reference point.

    Uses efficient O(n log n) algorithm for bi-objective problems.

    IMPORTANT: Assumes MAXIMIZATION. Points should be better than
    reference point (higher values for maximization).

    Args:
        points: Pareto front points, shape (n, 2)
        reference: Reference point (nadir), shape (2,)
                   Should be worse than all front points.

    Returns:
        Hypervolume value

    Example:
        >>> front = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
        >>> ref = np.array([0.0, 0.0])
        >>> hv = hypervolume_2d(front, ref)
        >>> hv  # Area dominated by front
        7.0
    """
    if len(points) == 0:
        return 0.0

    points = np.atleast_2d(points)
    reference = np.atleast_1d(reference)

    if points.shape[1] != 2:
        raise ValueError(f"Expected 2D points, got shape {points.shape}")
    if reference.shape != (2,):
        raise ValueError(f"Expected 2D reference, got shape {reference.shape}")

    # Filter points that are dominated by reference (shouldn't happen but be safe)
    valid_mask = np.all(points > reference, axis=1)
    points = points[valid_mask]

    if len(points) == 0:
        return 0.0

    # Sort by first objective (descending for sweepline from right to left)
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]

    hv = 0.0
    prev_y = reference[1]  # Start from reference y

    for point in sorted_points:
        if point[1] > prev_y:
            # Add rectangle from current x to reference x, height from prev_y to point y
            width = point[0] - reference[0]
            height = point[1] - prev_y
            hv += width * height
            prev_y = point[1]

    return float(hv)


def hypervolume_contribution(
    points: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Calculate each point's exclusive hypervolume contribution.

    The exclusive contribution of a point is the hypervolume that
    would be lost if that point were removed.

    Useful for hypervolume-based selection (SMS-EMOA).

    Args:
        points: Pareto front points, shape (n, 2)
        reference: Reference point

    Returns:
        Array of contributions, shape (n,)

    Example:
        >>> front = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
        >>> ref = np.array([0.0, 0.0])
        >>> contrib = hypervolume_contribution(front, ref)
    """
    if len(points) == 0:
        return np.array([])

    points = np.atleast_2d(points)
    n = len(points)
    contributions = np.zeros(n)

    total_hv = hypervolume_2d(points, reference)

    for i in range(n):
        # HV without point i
        remaining = np.delete(points, i, axis=0)
        if len(remaining) > 0:
            hv_without = hypervolume_2d(remaining, reference)
        else:
            hv_without = 0.0
        contributions[i] = total_hv - hv_without

    return contributions


def generational_distance(
    front: np.ndarray,
    reference_front: np.ndarray,
    p: float = 2.0,
) -> float:
    """
    Calculate Generational Distance (GD) from a front to a reference front.

    GD measures the average distance from each point in the front
    to the nearest point in the reference front.

    Lower is better: 0 means all points are on the reference front.

    Args:
        front: Obtained Pareto front, shape (n, m)
        reference_front: True/reference Pareto front, shape (k, m)
        p: Power parameter (default 2 for Euclidean)

    Returns:
        GD value
    """
    if len(front) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")

    distances = []
    for point in front:
        # Distance to nearest reference point
        d = np.min(np.linalg.norm(reference_front - point, axis=1))
        distances.append(d**p)

    return (np.mean(distances)) ** (1.0 / p)


def inverted_generational_distance(
    front: np.ndarray,
    reference_front: np.ndarray,
    p: float = 2.0,
) -> float:
    """
    Calculate Inverted Generational Distance (IGD).

    IGD measures the average distance from each point in the
    reference front to the nearest point in the obtained front.

    Lower is better. Measures both convergence and diversity.

    Args:
        front: Obtained Pareto front, shape (n, m)
        reference_front: True/reference Pareto front, shape (k, m)
        p: Power parameter (default 2 for Euclidean)

    Returns:
        IGD value
    """
    if len(front) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")

    distances = []
    for ref_point in reference_front:
        # Distance to nearest obtained point
        d = np.min(np.linalg.norm(front - ref_point, axis=1))
        distances.append(d**p)

    return (np.mean(distances)) ** (1.0 / p)


def spread(
    front: np.ndarray,
    reference_front: np.ndarray | None = None,
) -> float:
    """
    Calculate spread (diversity) metric for a 2D front.

    Measures how evenly distributed the points are along the front.

    Lower is better: 0 means perfectly uniform distribution.

    Args:
        front: Obtained Pareto front, shape (n, 2)
        reference_front: Optional reference for extreme points

    Returns:
        Spread value in [0, 1]
    """
    if len(front) < 2:
        return 1.0  # No spread with < 2 points

    front = np.atleast_2d(front)

    # Sort by first objective
    sorted_indices = np.argsort(front[:, 0])
    sorted_front = front[sorted_indices]

    # Calculate consecutive distances
    distances = []
    for i in range(len(sorted_front) - 1):
        d = np.linalg.norm(sorted_front[i + 1] - sorted_front[i])
        distances.append(d)

    if len(distances) == 0:
        return 1.0

    distances = np.array(distances)
    mean_d = np.mean(distances)

    if mean_d == 0:
        return 0.0  # All points at same location

    # Calculate spread
    spread_val = np.sum(np.abs(distances - mean_d)) / (len(distances) * mean_d)

    return float(min(spread_val, 1.0))


def coverage(
    front_a: np.ndarray,
    front_b: np.ndarray,
) -> float:
    """
    Calculate coverage metric C(A, B).

    C(A, B) = fraction of points in B that are dominated by at least one point in A.

    Higher C(A, B) means A is better than B.
    Note: C(A, B) + C(B, A) != 1 in general.

    Args:
        front_a: First Pareto front
        front_b: Second Pareto front

    Returns:
        Coverage value in [0, 1]
    """
    if len(front_b) == 0:
        return 1.0
    if len(front_a) == 0:
        return 0.0

    dominated_count = 0
    for b_point in front_b:
        for a_point in front_a:
            # Check if a dominates b (all objectives >= and at least one >)
            if np.all(a_point >= b_point) and np.any(a_point > b_point):
                dominated_count += 1
                break

    return dominated_count / len(front_b)
