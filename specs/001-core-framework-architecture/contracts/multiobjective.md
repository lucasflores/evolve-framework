# Multi-Objective Optimization Interfaces Contract

**Module**: `evolve.multiobjective`  
**Purpose**: Define Pareto dominance, ranking, crowding, and selection for multi-objective optimization

---

## Fitness Multi-Objective Extension

```python
from dataclasses import dataclass
import numpy as np
from typing import Sequence


@dataclass(frozen=True)
class MultiObjectiveFitness:
    """
    Fitness for multi-objective optimization.
    
    All objectives follow MAXIMIZATION convention.
    To minimize: negate or transform.
    """
    objectives: np.ndarray  # Shape: (n_objectives,)
    constraint_violations: np.ndarray | None = None  # Shape: (n_constraints,)
    
    def __post_init__(self):
        self.objectives.flags.writeable = False
        if self.constraint_violations is not None:
            self.constraint_violations.flags.writeable = False
    
    @property
    def n_objectives(self) -> int:
        return len(self.objectives)
    
    @property
    def is_feasible(self) -> bool:
        """Check if all constraints satisfied."""
        if self.constraint_violations is None:
            return True
        return np.all(self.constraint_violations <= 0)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiObjectiveFitness):
            return False
        return np.array_equal(self.objectives, other.objectives)
    
    def __hash__(self) -> int:
        return hash(self.objectives.tobytes())
```

---

## Pareto Dominance

```python
def dominates(
    a: MultiObjectiveFitness,
    b: MultiObjectiveFitness,
    *,
    strict: bool = True
) -> bool:
    """
    Check if fitness 'a' Pareto-dominates fitness 'b'.
    
    Constrained domination (feasibility-first):
    1. Feasible solutions dominate infeasible
    2. Among infeasible, less violation dominates
    3. Among feasible, Pareto dominance applies
    
    Args:
        a: First fitness
        b: Second fitness
        strict: If True, at least one objective must be strictly better
        
    Returns:
        True if a dominates b
    """
    # Constraint handling
    if a.is_feasible and not b.is_feasible:
        return True
    if not a.is_feasible and b.is_feasible:
        return False
    if not a.is_feasible and not b.is_feasible:
        # Compare constraint violations
        return np.sum(np.maximum(a.constraint_violations, 0)) < \
               np.sum(np.maximum(b.constraint_violations, 0))
    
    # Both feasible: Pareto dominance
    # a dominates b if: a >= b for all objectives AND a > b for at least one
    at_least_equal = np.all(a.objectives >= b.objectives)
    if not at_least_equal:
        return False
    
    if strict:
        strictly_better = np.any(a.objectives > b.objectives)
        return strictly_better
    return True


def pareto_front(
    fitnesses: Sequence[MultiObjectiveFitness]
) -> list[int]:
    """
    Get indices of non-dominated solutions (Pareto front).
    
    Args:
        fitnesses: List of multi-objective fitnesses
        
    Returns:
        Indices of solutions on the Pareto front
    """
    n = len(fitnesses)
    is_dominated = [False] * n
    
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(fitnesses[j], fitnesses[i]):
                is_dominated[i] = True
                break
    
    return [i for i in range(n) if not is_dominated[i]]
```

---

## NSGA-II Non-Dominated Sorting

```python
@dataclass
class RankedIndividual:
    """Individual with ranking information."""
    index: int
    rank: int  # 0 = first front (Pareto-optimal)
    crowding_distance: float


def fast_non_dominated_sort(
    fitnesses: Sequence[MultiObjectiveFitness]
) -> list[list[int]]:
    """
    NSGA-II fast non-dominated sorting.
    
    Assigns all individuals to ranked fronts.
    Front 0 is the Pareto front, front 1 is dominated only by front 0, etc.
    
    Complexity: O(M * N^2) where M = objectives, N = population size
    
    Args:
        fitnesses: Multi-objective fitnesses
        
    Returns:
        List of fronts, each containing indices of individuals
    """
    n = len(fitnesses)
    
    # domination_count[i] = number of solutions dominating i
    domination_count = [0] * n
    # dominated_set[i] = set of solutions dominated by i
    dominated_set: list[set[int]] = [set() for _ in range(n)]
    
    fronts: list[list[int]] = [[]]
    
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(fitnesses[p], fitnesses[q]):
                dominated_set[p].add(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                domination_count[p] += 1
        
        if domination_count[p] == 0:
            fronts[0].append(p)
    
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for p in fronts[current_front]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current_front += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts
```

---

## Crowding Distance

```python
def crowding_distance(
    fitnesses: Sequence[MultiObjectiveFitness],
    front_indices: list[int]
) -> dict[int, float]:
    """
    Calculate crowding distance for individuals in a front.
    
    Crowding distance measures how close an individual is to its
    neighbors in objective space. Higher = more isolated = more diverse.
    
    Boundary solutions get infinite distance to ensure preservation.
    
    Args:
        fitnesses: All fitnesses
        front_indices: Indices of individuals in this front
        
    Returns:
        Mapping from index to crowding distance
    """
    if len(front_indices) <= 2:
        return {i: float('inf') for i in front_indices}
    
    n_objectives = fitnesses[front_indices[0]].n_objectives
    distances = {i: 0.0 for i in front_indices}
    
    for m in range(n_objectives):
        # Sort by objective m
        sorted_indices = sorted(
            front_indices,
            key=lambda i: fitnesses[i].objectives[m]
        )
        
        # Boundary points get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Objective range
        obj_min = fitnesses[sorted_indices[0]].objectives[m]
        obj_max = fitnesses[sorted_indices[-1]].objectives[m]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        # Interior points
        for k in range(1, len(sorted_indices) - 1):
            prev_obj = fitnesses[sorted_indices[k - 1]].objectives[m]
            next_obj = fitnesses[sorted_indices[k + 1]].objectives[m]
            distances[sorted_indices[k]] += (next_obj - prev_obj) / obj_range
    
    return distances
```

---

## NSGA-II Selection Protocol

```python
from typing import Protocol, TypeVar
from evolve.core import Individual

G = TypeVar('G')


class MultiObjectiveSelector(Protocol[G]):
    """Selection operator for multi-objective optimization."""
    
    def select(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        rng: 'Random'
    ) -> list[Individual[G]]:
        """
        Select individuals based on rank and crowding.
        
        Args:
            population: Population with multi-objective fitnesses
            n_select: Number to select
            rng: Random number generator
            
        Returns:
            Selected individuals
        """
        ...


class NSGA2Selector:
    """
    NSGA-II selection using rank and crowding distance.
    
    Selection preference:
    1. Lower rank (closer to Pareto front)
    2. Higher crowding distance (more diverse)
    """
    
    def select(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        rng: 'Random'
    ) -> list[Individual[G]]:
        # Extract multi-objective fitnesses
        fitnesses = [
            ind.fitness for ind in population
            if isinstance(ind.fitness, MultiObjectiveFitness)
        ]
        
        # Non-dominated sorting
        fronts = fast_non_dominated_sort(fitnesses)
        
        # Build ranked list with crowding
        selected: list[Individual[G]] = []
        
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                # Add entire front
                selected.extend(population[i] for i in front)
            else:
                # Need to select subset using crowding distance
                distances = crowding_distance(fitnesses, front)
                # Sort by crowding (descending) and take remaining
                sorted_front = sorted(
                    front,
                    key=lambda i: distances[i],
                    reverse=True
                )
                remaining = n_select - len(selected)
                selected.extend(
                    population[i] for i in sorted_front[:remaining]
                )
                break
        
        return selected
```

---

## Tournament Selection (Multi-Objective)

```python
class CrowdedTournamentSelection:
    """
    Binary tournament using crowded comparison operator.
    
    Winner is the individual with:
    1. Better rank, OR
    2. Same rank but higher crowding distance
    """
    
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size
    
    def select(
        self,
        population: Sequence[Individual[G]],
        n_select: int,
        ranks: dict[int, int],
        crowding: dict[int, float],
        rng: 'Random'
    ) -> list[Individual[G]]:
        selected = []
        
        for _ in range(n_select):
            # Random tournament
            candidates = rng.sample(range(len(population)), self.tournament_size)
            
            # Crowded comparison
            best = candidates[0]
            for c in candidates[1:]:
                if self._crowded_compare(best, c, ranks, crowding) < 0:
                    best = c
            
            selected.append(population[best])
        
        return selected
    
    def _crowded_compare(
        self,
        i: int,
        j: int,
        ranks: dict[int, int],
        crowding: dict[int, float]
    ) -> int:
        """
        Crowded comparison operator.
        
        Returns:
            > 0 if i is better
            < 0 if j is better
            0 if equal
        """
        if ranks[i] < ranks[j]:
            return 1
        if ranks[i] > ranks[j]:
            return -1
        # Same rank: prefer higher crowding
        if crowding[i] > crowding[j]:
            return 1
        if crowding[i] < crowding[j]:
            return -1
        return 0
```

---

## Reference Point-Based Selection (NSGA-III Ready)

```python
def generate_reference_points(
    n_objectives: int,
    n_divisions: int
) -> np.ndarray:
    """
    Generate uniformly distributed reference points.
    
    Uses Das and Dennis's systematic approach for
    generating points on unit simplex.
    
    Args:
        n_objectives: Number of objectives
        n_divisions: Number of divisions along each axis
        
    Returns:
        Reference points, shape: (n_points, n_objectives)
    """
    from itertools import combinations_with_replacement
    
    points = []
    for combo in combinations_with_replacement(range(n_divisions + 1), n_objectives - 1):
        point = [0.0] * n_objectives
        prev = 0
        for i, val in enumerate(combo):
            point[i] = (val - prev) / n_divisions
            prev = val
        point[-1] = 1.0 - sum(point[:-1])
        points.append(point)
    
    return np.array(points)


class ReferencePointSelector:
    """
    Reference point-based selection for many-objective problems.
    
    Foundation for NSGA-III. Associates solutions with
    reference points and maintains diversity via niching.
    """
    
    def __init__(self, reference_points: np.ndarray):
        self.reference_points = reference_points
    
    def associate(
        self,
        fitnesses: Sequence[MultiObjectiveFitness]
    ) -> dict[int, int]:
        """
        Associate each individual with nearest reference point.
        
        Returns:
            Mapping from individual index to reference point index
        """
        # Normalize objectives to [0, 1]
        objs = np.array([f.objectives for f in fitnesses])
        ideal = objs.min(axis=0)
        nadir = objs.max(axis=0)
        normalized = (objs - ideal) / (nadir - ideal + 1e-10)
        
        associations = {}
        for i, norm_obj in enumerate(normalized):
            # Find closest reference point
            distances = np.linalg.norm(
                self.reference_points - norm_obj,
                axis=1
            )
            associations[i] = int(np.argmin(distances))
        
        return associations
```

---

## Hypervolume Indicator

```python
def hypervolume_2d(
    points: np.ndarray,
    reference: np.ndarray
) -> float:
    """
    Calculate hypervolume for 2D Pareto front.
    
    Efficient O(n log n) algorithm for bi-objective.
    
    Args:
        points: Pareto front points, shape: (n, 2)
        reference: Reference point (worse than all front points)
        
    Returns:
        Hypervolume value
    """
    # Sort by first objective (descending)
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]
    
    hv = 0.0
    prev_y = reference[1]
    
    for point in sorted_points:
        if point[1] > prev_y:
            width = reference[0] - point[0]
            height = point[1] - prev_y
            hv += width * height
            prev_y = point[1]
    
    return hv


def hypervolume_contribution(
    points: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    Calculate each point's exclusive hypervolume contribution.
    
    Useful for hypervolume-based selection.
    
    Args:
        points: Pareto front points
        reference: Reference point
        
    Returns:
        Contribution of each point
    """
    n = len(points)
    contributions = np.zeros(n)
    total_hv = hypervolume_2d(points, reference)
    
    for i in range(n):
        # HV without point i
        remaining = np.delete(points, i, axis=0)
        hv_without = hypervolume_2d(remaining, reference) if len(remaining) > 0 else 0
        contributions[i] = total_hv - hv_without
    
    return contributions
```
