# Evaluation Interfaces Contract

**Module**: `evolve.evaluation`  
**Purpose**: Define evaluator protocols and fitness computation interfaces

---

## Evaluator Protocol

```python
from typing import Protocol, TypeVar, Sequence, Generic
from dataclasses import dataclass

G = TypeVar('G', bound='Genome')

@dataclass(frozen=True)
class EvaluatorCapabilities:
    """
    Declares what an evaluator can do.
    
    Used by the engine to optimize evaluation strategy.
    """
    batchable: bool = True          # Can evaluate multiple at once
    stochastic: bool = False        # Results vary with RNG
    stateful: bool = False          # Has internal state
    n_objectives: int = 1           # Number of fitness objectives
    n_constraints: int = 0          # Number of constraints
    supports_diagnostics: bool = False  # Can return extra info


class Evaluator(Protocol[G]):
    """
    Computes fitness for batches of individuals.
    
    This is the PRIMARY ACCELERATION BOUNDARY.
    Evaluators may use GPU/JIT but must:
    - Have CPU reference implementation
    - Accept explicit seeds for reproducibility
    - Produce equivalent results across backends
    """
    
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """Declare evaluator capabilities."""
        ...
    
    def evaluate(
        self,
        individuals: Sequence['Individual[G]'],
        seed: int | None = None
    ) -> Sequence['Fitness']:
        """
        Evaluate batch of individuals.
        
        Args:
            individuals: Individuals to evaluate (may need decoding)
            seed: Random seed for stochastic evaluation
            
        Returns:
            Fitness values in same order as input
            
        Raises:
            EvaluationError: If evaluation fails
        """
        ...


class DiagnosticEvaluator(Protocol[G]):
    """Evaluator that can return additional information."""
    
    def evaluate_with_diagnostics(
        self,
        individuals: Sequence['Individual[G]'],
        seed: int | None = None
    ) -> tuple[Sequence['Fitness'], Sequence[dict]]:
        """
        Evaluate with diagnostic information.
        
        Returns:
            (fitness_values, diagnostics_per_individual)
            
        Diagnostics may include:
        - Per-objective breakdown
        - Intermediate values
        - Timing information
        - Gradient norms (for memetic)
        """
        ...
```

---

## Function Evaluator

```python
from typing import Callable
import numpy as np

class FunctionEvaluator(Generic[G]):
    """
    Evaluator wrapping a fitness function.
    
    The simplest evaluator type - applies a function
    to each individual's phenotype.
    """
    
    def __init__(
        self,
        fitness_fn: Callable[[Any], np.ndarray | float],
        decoder: 'Decoder[G]' | None = None,
        n_objectives: int = 1,
        n_constraints: int = 0,
        minimize: bool = True
    ) -> None:
        """
        Create function evaluator.
        
        Args:
            fitness_fn: Function mapping phenotype → fitness value(s)
            decoder: Optional genome→phenotype decoder
            n_objectives: Number of objectives (inferred from fn output if 1)
            n_constraints: Number of constraints
            minimize: If True, lower fitness is better
        """
        ...
    
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            n_objectives=self._n_objectives,
            n_constraints=self._n_constraints
        )
    
    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None
    ) -> Sequence[Fitness]:
        """
        Evaluate by applying fitness function.
        
        If decoder is provided, decodes genome first.
        """
        ...
```

---

## Batch Evaluator

```python
class BatchEvaluator(Protocol[G]):
    """
    Evaluator optimized for batch processing.
    
    Some evaluators (especially GPU-based) are more efficient
    when processing many individuals at once.
    """
    
    def evaluate_batch(
        self,
        phenotypes: Sequence[Any],  # Already decoded
        seed: int | None = None
    ) -> np.ndarray:
        """
        Evaluate batch of phenotypes efficiently.
        
        Args:
            phenotypes: Decoded phenotypes (tensors, arrays, etc.)
            seed: Random seed
            
        Returns:
            Array of shape (n_individuals, n_objectives)
        """
        ...
    
    @property
    def optimal_batch_size(self) -> int | None:
        """
        Suggested batch size for this evaluator.
        
        None means no preference.
        """
        ...
```

---

## Stochastic Evaluator

```python
class StochasticEvaluator(Protocol[G]):
    """
    Evaluator with stochastic fitness evaluation.
    
    Common in RL (episode rollouts) and noisy optimization.
    """
    
    def evaluate_n_times(
        self,
        individuals: Sequence['Individual[G]'],
        n_evaluations: int,
        seed: int
    ) -> Sequence['Fitness']:
        """
        Evaluate multiple times and aggregate.
        
        Args:
            individuals: Individuals to evaluate
            n_evaluations: Number of evaluations per individual
            seed: Base seed (derives n_evaluations seeds)
            
        Returns:
            Aggregated fitness (typically mean)
        """
        ...
    
    @property
    def aggregation(self) -> str:
        """How multiple evaluations are combined: 'mean', 'median', 'min'."""
        ...
```

---

## Reference Benchmark Functions

```python
# evolve/evaluation/reference/functions.py

def sphere(x: np.ndarray) -> float:
    """
    Sphere function (sum of squares).
    
    Minimum: f(0, 0, ..., 0) = 0
    """
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
    """
    Rastrigin function (highly multimodal).
    
    Minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (banana-shaped valley).
    
    Minimum: f(1, 1, ..., 1) = 0
    """
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


def ackley(x: np.ndarray) -> float:
    """
    Ackley function (many local minima).
    
    Minimum: f(0, 0, ..., 0) = 0
    """
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - 
                 np.exp(sum2 / n) + 20 + np.e)


# Multi-objective benchmarks

def zdt1(x: np.ndarray) -> np.ndarray:
    """
    ZDT1 benchmark (convex Pareto front).
    
    Pareto front: f2 = 1 - sqrt(f1)
    """
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])


def zdt2(x: np.ndarray) -> np.ndarray:
    """
    ZDT2 benchmark (non-convex Pareto front).
    
    Pareto front: f2 = 1 - (f1)^2
    """
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])


def zdt3(x: np.ndarray) -> np.ndarray:
    """
    ZDT3 benchmark (discontinuous Pareto front).
    """
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
    return np.array([f1, f2])
```

---

## Equivalence Testing

```python
def assert_evaluator_equivalence(
    reference: Evaluator[G],
    accelerated: Evaluator[G],
    test_individuals: Sequence[Individual[G]],
    seed: int,
    rtol: float = 1e-5,
    atol: float = 1e-10
) -> None:
    """
    Verify accelerated evaluator matches CPU reference.
    
    Args:
        reference: CPU reference evaluator
        accelerated: GPU/JIT accelerated evaluator
        test_individuals: Individuals to test
        seed: Random seed
        rtol: Relative tolerance
        atol: Absolute tolerance (for near-zero values)
        
    Raises:
        AssertionError: If results differ beyond tolerance
    """
    ref_fitness = reference.evaluate(test_individuals, seed)
    acc_fitness = accelerated.evaluate(test_individuals, seed)
    
    for i, (ref, acc) in enumerate(zip(ref_fitness, acc_fitness)):
        ref_vals = ref.values
        acc_vals = acc.values
        
        # Check for NaN
        if np.any(np.isnan(ref_vals)) or np.any(np.isnan(acc_vals)):
            raise AssertionError(f"NaN in fitness values at index {i}")
        
        # Relative tolerance check
        denom = np.maximum(np.abs(ref_vals), np.abs(acc_vals))
        denom = np.maximum(denom, atol)
        rel_diff = np.abs(ref_vals - acc_vals) / denom
        
        if np.any(rel_diff > rtol):
            raise AssertionError(
                f"Equivalence failed at index {i}: "
                f"max relative diff = {rel_diff.max():.2e}"
            )
```
