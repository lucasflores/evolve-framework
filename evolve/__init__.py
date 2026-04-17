"""
Evolve Framework - Research-Grade Evolutionary Algorithms Experimentation

A modular, research-grade framework for evolutionary computation with:
- Declarative experiment configuration via UnifiedConfig
- Model-agnostic architecture (no hard dependencies on ML frameworks)
- Deterministic reproducibility via explicit seeding
- Optional GPU/JIT acceleration
- Multi-objective optimization (NSGA-II)
- Evolvable Reproduction Protocols (ERP)
- Neuroevolution and RL support
- Experiment tracking and checkpointing

Quick Start:
    >>> from evolve import UnifiedConfig, create_engine
    >>> from evolve.factory import create_initial_population
    >>>
    >>> config = UnifiedConfig(
    ...     name="example",
    ...     population_size=100,
    ...     max_generations=50,
    ...     genome_type="vector",
    ...     genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
    ...     selection="tournament",
    ...     crossover="blend",
    ...     mutation="gaussian",
    ...     minimize=True,
    ...     seed=42,
    ... )
    >>> engine = create_engine(config, evaluator=my_fitness_fn)
    >>> population = create_initial_population(config)
    >>> result = engine.run(population)
    >>> print(f"Best fitness: {result.best.fitness}")
"""

__version__ = "0.1.0"
__author__ = "Evolve Framework Team"

# Core types
# Unified Configuration (new config system)
from evolve.config import (
    CallbackConfig,
    ConstraintSpec,
    ERPSettings,
    MetaEvolutionConfig,
    MultiObjectiveConfig,
    ObjectiveSpec,
    ParameterSpec,
    StoppingConfig,
    UnifiedConfig,
)
from evolve.core.engine import (
    EvolutionConfig,
    EvolutionEngine,
    EvolutionResult,
    create_initial_population,
)

# Operators
from evolve.core.operators import (
    ElitistSelection,
    GaussianMutation,
    PolynomialMutation,
    RankSelection,
    RouletteSelection,
    SimulatedBinaryCrossover,
    SinglePointCrossover,
    TournamentSelection,
    TwoPointCrossover,
    UniformCrossover,
)
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual, IndividualMetadata

# Evaluation
from evolve.evaluation.evaluator import EvaluatorCapabilities, FunctionEvaluator
from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig

# Factory (one-line engine creation)
from evolve.factory import (
    OperatorCompatibilityError,
    create_engine,
)
from evolve.factory import (
    create_initial_population as create_population_from_config,
)

# Meta-Evolution
from evolve.meta import (
    ConfigCodec,
    MetaEvaluator,
    MetaEvolutionResult,
    run_meta_evolution,
)

# Registry (operator, genome, and decoder lookup)
from evolve.registry import (
    get_decoder_registry,
    get_genome_registry,
    get_operator_registry,
)

# SCM (Structural Causal Model) Representation
from evolve.representation.scm import (
    AcyclicityMode,
    ConflictResolution,
    SCMAlphabet,
    SCMConfig,
    SCMGenome,
    scm_distance,
    scm_sequence_distance,
    scm_structural_distance,
)
from evolve.representation.scm_decoder import (
    DecodedSCM,
    SCMDecoder,
    SCMMetadata,
    to_string,
)

# Representation
from evolve.representation.vector import VectorGenome

# Reproduction (import submodule for ERP support)
# Note: ERPEngine must be imported separately to avoid circular imports:
#   from evolve.reproduction.engine import ERPConfig, ERPEngine
# Or use: import evolve.reproduction as erp

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
    # SCM Representation
    "SCMConfig",
    "SCMGenome",
    "SCMAlphabet",
    "ConflictResolution",
    "AcyclicityMode",
    "scm_distance",
    "scm_sequence_distance",
    "scm_structural_distance",
    "SCMDecoder",
    "DecodedSCM",
    "SCMMetadata",
    "to_string",
    # Evaluation
    "FunctionEvaluator",
    "EvaluatorCapabilities",
    "SCMEvaluator",
    "SCMFitnessConfig",
    # Unified Configuration
    "UnifiedConfig",
    "StoppingConfig",
    "CallbackConfig",
    "ERPSettings",
    "ObjectiveSpec",
    "ConstraintSpec",
    "MultiObjectiveConfig",
    "ParameterSpec",
    "MetaEvolutionConfig",
    # Factory
    "create_engine",
    "create_population_from_config",
    "OperatorCompatibilityError",
    # Registry
    "get_decoder_registry",
    "get_operator_registry",
    "get_genome_registry",
    # Meta-Evolution
    "ConfigCodec",
    "MetaEvaluator",
    "MetaEvolutionResult",
    "run_meta_evolution",
]
