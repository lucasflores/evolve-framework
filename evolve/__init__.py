"""
Evolve Framework - Research-Grade Evolutionary Algorithms Experimentation

A modular, research-grade framework for evolutionary computation with:
- Model-agnostic architecture (no hard dependencies on ML frameworks)
- Deterministic reproducibility via explicit seeding
- Optional GPU/JIT acceleration
- Multi-objective optimization (NSGA-II)
- Neuroevolution and RL support
- Experiment tracking and checkpointing

Example:
    >>> from evolve import EvolutionEngine, EvolutionConfig, VectorGenome
    >>> from evolve.core.operators import TournamentSelection, UniformCrossover, GaussianMutation
    >>> from evolve.evaluation import FunctionEvaluator
    >>> 
    >>> engine = EvolutionEngine(
    ...     config=EvolutionConfig(population_size=100, max_generations=100),
    ...     evaluator=FunctionEvaluator(lambda g: sum(g.genes**2)),
    ...     selection=TournamentSelection(),
    ...     crossover=UniformCrossover(),
    ...     mutation=GaussianMutation(),
    ...     seed=42
    ... )
    >>> result = engine.run(initial_population)
    >>> print(f"Best fitness: {result.best.fitness}")
"""

__version__ = "0.1.0"
__author__ = "Evolve Framework Team"

# Core types
from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.core.population import Population
from evolve.core.engine import (
    EvolutionEngine,
    EvolutionConfig,
    EvolutionResult,
    create_initial_population,
)

# Operators
from evolve.core.operators import (
    TournamentSelection,
    RouletteSelection,
    RankSelection,
    ElitistSelection,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    SimulatedBinaryCrossover,
    GaussianMutation,
    PolynomialMutation,
)

# Representation
from evolve.representation.vector import VectorGenome

# Evaluation
from evolve.evaluation.evaluator import FunctionEvaluator, EvaluatorCapabilities

__all__ = [
    # Version
    "__version__",
    # Core types
    "Fitness",
    "Individual",
    "IndividualMetadata",
    "Population",
    # Engine
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "create_initial_population",
    # Selection
    "TournamentSelection",
    "RouletteSelection",
    "RankSelection",
    "ElitistSelection",
    # Crossover
    "SinglePointCrossover",
    "TwoPointCrossover",
    "UniformCrossover",
    "SimulatedBinaryCrossover",
    # Mutation
    "GaussianMutation",
    "PolynomialMutation",
    # Representation
    "VectorGenome",
    # Evaluation
    "FunctionEvaluator",
    "EvaluatorCapabilities",
]
