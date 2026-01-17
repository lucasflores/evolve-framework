"""
Evaluator protocols - Fitness computation interfaces.

This is the PRIMARY ACCELERATION BOUNDARY.
Evaluators may use GPU/JIT but must:
- Have CPU reference implementation
- Accept explicit seeds for reproducibility
- Produce equivalent results across backends
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, Sequence, TypeVar, runtime_checkable

from evolve.core.types import Fitness, Individual

G = TypeVar("G")


@dataclass(frozen=True)
class EvaluatorCapabilities:
    """
    Declares what an evaluator can do.
    
    Used by the engine to optimize evaluation strategy.
    
    Attributes:
        batchable: Can evaluate multiple individuals at once
        stochastic: Results vary with RNG (e.g., RL rollouts)
        stateful: Has internal state between evaluations
        n_objectives: Number of fitness objectives
        n_constraints: Number of constraint functions
        supports_diagnostics: Can return extra diagnostic info
        supports_gpu: Can run on GPU
        supports_jit: Can be JIT-compiled
    """

    batchable: bool = True
    stochastic: bool = False
    stateful: bool = False
    n_objectives: int = 1
    n_constraints: int = 0
    supports_diagnostics: bool = False
    supports_gpu: bool = False
    supports_jit: bool = False


class EvaluationError(Exception):
    """Raised when evaluation fails."""

    def __init__(self, message: str, individual_idx: int | None = None) -> None:
        super().__init__(message)
        self.individual_idx = individual_idx


@runtime_checkable
class Evaluator(Protocol[G]):
    """
    Computes fitness for batches of individuals.
    
    This is the PRIMARY ACCELERATION BOUNDARY.
    Evaluators may use GPU/JIT but must:
    - Have CPU reference implementation
    - Accept explicit seeds for reproducibility
    - Produce equivalent results across backends (within tolerance)
    
    Example:
        >>> class MyEvaluator:
        ...     @property
        ...     def capabilities(self) -> EvaluatorCapabilities:
        ...         return EvaluatorCapabilities(n_objectives=1)
        ...     
        ...     def evaluate(self, individuals, seed=None):
        ...         return [Fitness.scalar(f(ind.genome)) for ind in individuals]
    """

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """Declare evaluator capabilities."""
        ...

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
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


@runtime_checkable
class DiagnosticEvaluator(Protocol[G]):
    """
    Evaluator that can return additional diagnostic information.
    
    Useful for debugging, visualization, and understanding
    the fitness landscape.
    """

    def evaluate_with_diagnostics(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> tuple[Sequence[Fitness], Sequence[dict[str, Any]]]:
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


class FunctionEvaluator(Generic[G]):
    """
    Evaluator wrapping a fitness function.
    
    The simplest evaluator type - applies a function
    to each individual's phenotype.
    
    Example:
        >>> def sphere(x):
        ...     return np.sum(x ** 2)
        >>> evaluator = FunctionEvaluator(sphere)
        >>> fitness = evaluator.evaluate(individuals)
    """

    def __init__(
        self,
        fitness_fn: "Callable[[Any], float | np.ndarray]",  # type: ignore[name-defined]
        decoder: "Decoder[G, Any] | None" = None,  # type: ignore[name-defined]
        n_objectives: int = 1,
        n_constraints: int = 0,
        minimize: bool = True,
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
        self._fitness_fn = fitness_fn
        self._decoder = decoder
        self._n_objectives = n_objectives
        self._n_constraints = n_constraints
        self._minimize = minimize
        self._capabilities = EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            n_objectives=n_objectives,
            n_constraints=n_constraints,
        )

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """Return evaluator capabilities."""
        return self._capabilities

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate by applying fitness function.
        
        If decoder is provided, decodes genome first.
        """
        import numpy as np
        
        results: list[Fitness] = []
        
        for idx, individual in enumerate(individuals):
            try:
                # Decode genome if decoder provided
                if self._decoder is not None:
                    phenotype = self._decoder.decode(individual.genome)
                else:
                    # Assume genome has a 'genes' attribute or is directly usable
                    phenotype = getattr(individual.genome, "genes", individual.genome)
                
                # Evaluate
                raw_fitness = self._fitness_fn(phenotype)
                
                # Convert to Fitness object
                if isinstance(raw_fitness, (int, float)):
                    fitness = Fitness.scalar(float(raw_fitness))
                elif isinstance(raw_fitness, np.ndarray):
                    fitness = Fitness(values=raw_fitness.flatten())
                else:
                    fitness = Fitness.scalar(float(raw_fitness))
                
                results.append(fitness)
                
            except Exception as e:
                raise EvaluationError(
                    f"Evaluation failed for individual {idx}: {e}",
                    individual_idx=idx,
                ) from e
        
        return results


class BatchEvaluator(Generic[G]):
    """
    Evaluator that processes all individuals in a single batch.
    
    More efficient for vectorized operations (NumPy/GPU).
    
    Example:
        >>> def batch_sphere(genes_matrix):
        ...     return np.sum(genes_matrix ** 2, axis=1)
        >>> evaluator = BatchEvaluator(batch_sphere)
    """

    def __init__(
        self,
        batch_fn: "Callable[[np.ndarray], np.ndarray]",  # type: ignore[name-defined]
        n_objectives: int = 1,
    ) -> None:
        """
        Create batch evaluator.
        
        Args:
            batch_fn: Function mapping (N, D) array → (N,) or (N, M) array
            n_objectives: Number of objectives
        """
        self._batch_fn = batch_fn
        self._n_objectives = n_objectives
        self._capabilities = EvaluatorCapabilities(
            batchable=True,
            n_objectives=n_objectives,
        )

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """Return evaluator capabilities."""
        return self._capabilities

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """Evaluate all individuals in a single batch."""
        import numpy as np
        
        # Stack genes into matrix
        genes_list = [
            getattr(ind.genome, "genes", ind.genome)
            for ind in individuals
        ]
        genes_matrix = np.vstack(genes_list)
        
        # Batch evaluate
        raw_fitness = self._batch_fn(genes_matrix)
        
        # Convert to Fitness objects
        if raw_fitness.ndim == 1:
            return [Fitness.scalar(float(v)) for v in raw_fitness]
        else:
            return [Fitness(values=row) for row in raw_fitness]
