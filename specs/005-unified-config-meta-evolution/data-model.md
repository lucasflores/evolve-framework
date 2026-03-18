# Data Model: Unified Configuration & Meta-Evolution Framework

**Date**: March 12, 2026  
**Branch**: `005-unified-config-meta-evolution`

---

## Overview

This document defines the data structures for the unified configuration and meta-evolution system. All entities use frozen dataclasses for immutability, hashability, and JSON serialization support.

---

## Core Configuration Entities

### UnifiedConfig

The top-level configuration class that encompasses all experiment parameters.

```python
@dataclass(frozen=True)
class UnifiedConfig:
    """
    Complete experiment specification covering standard evolution,
    ERP, multi-objective optimization, and meta-evolution.
    
    Serializable to/from JSON via to_dict()/from_dict().
    Immutable to ensure configuration integrity after creation.
    
    Fields:
        schema_version: Schema version for forward/backward compatibility
        name: Human-readable experiment identifier
        description: Optional description for documentation
        seed: Random seed for reproducibility
        
        population_size: Number of individuals in population
        max_generations: Maximum generations (stopping condition)
        elitism: Number of elite individuals preserved
        
        selection: Selection operator name (resolved via registry)
        selection_params: Parameters for selection operator
        crossover: Crossover operator name
        crossover_rate: Probability of applying crossover
        crossover_params: Parameters for crossover operator
        mutation: Mutation operator name  
        mutation_rate: Probability of mutation per individual
        mutation_params: Parameters for mutation operator
        
        genome_type: Genome representation type name
        genome_params: Parameters for genome initialization
        
        stopping: Stopping criteria configuration
        callbacks: Callback configuration for logging/checkpointing
        
        erp: ERP-specific configuration (None if ERP disabled)
        multiobjective: Multi-objective configuration (None if single-objective)
        meta: Meta-evolution configuration (None if not meta-evolving)
    
    Relationships:
        - References operators by string name → resolved via OperatorRegistry
        - References genome type by string name → resolved via GenomeRegistry
        - Contains nested config dataclasses for subsections
    
    Validation:
        - Validates operator compatibility with genome type at factory time
        - Validates required parameters present for chosen operators
        - Validates consistency (e.g., reference_point required if hypervolume tracked)
    """
    
    # Schema identification
    schema_version: str = "1.0.0"
    
    # Experiment identification
    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    
    # Random seed
    seed: int | None = None
    
    # Population settings
    population_size: int = 100
    max_generations: int = 100
    elitism: int = 1
    
    # Selection
    selection: str = "tournament"
    selection_params: dict[str, Any] = field(default_factory=dict)
    
    # Crossover
    crossover: str = "uniform"
    crossover_rate: float = 0.9
    crossover_params: dict[str, Any] = field(default_factory=dict)
    
    # Mutation
    mutation: str = "gaussian"
    mutation_rate: float = 1.0
    mutation_params: dict[str, Any] = field(default_factory=dict)
    
    # Representation
    genome_type: str = "vector"
    genome_params: dict[str, Any] = field(default_factory=dict)
    minimize: bool = True
    
    # Nested configurations (None = disabled)
    stopping: StoppingConfig | None = None
    callbacks: CallbackConfig | None = None
    erp: ERPSettings | None = None
    multiobjective: MultiObjectiveConfig | None = None
    meta: MetaEvolutionConfig | None = None
```

**State Transitions**: None (immutable configuration)

**Constraints**:
- `population_size` > 0
- `max_generations` > 0
- `elitism` >= 0 and <= `population_size`
- `crossover_rate` in [0.0, 1.0]
- `mutation_rate` in [0.0, 1.0]
- `schema_version` follows semantic versioning pattern

---

### StoppingConfig

Configuration for evolution stopping criteria.

```python
@dataclass(frozen=True)
class StoppingConfig:
    """
    Stopping criteria specification.
    
    Multiple criteria can be specified; evolution stops when ANY is met (OR semantics).
    
    Fields:
        max_generations: Stop after N generations (overrides UnifiedConfig.max_generations)
        fitness_threshold: Stop when best fitness reaches this value
        stagnation_generations: Stop after N generations with no improvement
        time_limit_seconds: Stop after this wall-clock duration
    
    Validation:
        - At least one criterion should be specified
        - Values must be positive when specified
    """
    
    max_generations: int | None = None
    fitness_threshold: float | None = None
    stagnation_generations: int | None = None
    time_limit_seconds: float | None = None
```

**Constraints**:
- `max_generations` > 0 when specified
- `stagnation_generations` > 0 when specified
- `time_limit_seconds` > 0 when specified

---

### CallbackConfig

Configuration for built-in callbacks.

```python
@dataclass(frozen=True)
class CallbackConfig:
    """
    Built-in callback configuration.
    
    Custom callbacks cannot be specified here (passed to factory separately).
    
    Fields:
        enable_logging: Whether to enable progress logging
        log_level: Logging verbosity level
        log_destination: Where to log ("console", "file", file path)
        
        enable_checkpointing: Whether to enable checkpointing
        checkpoint_dir: Directory for checkpoint files
        checkpoint_frequency: Save checkpoint every N generations
    
    Validation:
        - checkpoint_dir required if enable_checkpointing is True
        - checkpoint_frequency > 0 when checkpointing enabled
    """
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"  # Literal["DEBUG", "INFO", "WARNING"]
    log_destination: str = "console"
    
    # Checkpointing
    enable_checkpointing: bool = False
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 10
```

---

### ERPSettings

Evolvable Reproduction Protocol settings.

```python
@dataclass(frozen=True)
class ERPSettings:
    """
    Configuration for Evolvable Reproduction Protocols.
    
    When present in UnifiedConfig, the factory produces ERPEngine.
    
    Fields:
        step_limit: Maximum computation steps per protocol evaluation
        recovery_threshold: Success rate below which recovery triggers
        protocol_mutation_rate: Probability of mutating reproduction protocol
        enable_intent: Whether to evaluate intent policies
        enable_recovery: Whether to use recovery mechanisms
    
    Relationships:
        - Requires genome types that support protocol attachment
        - Uses existing ERP infrastructure (Intent, Matchability, Recovery)
    """
    
    step_limit: int = 1000
    recovery_threshold: float = 0.1
    protocol_mutation_rate: float = 0.1
    enable_intent: bool = True
    enable_recovery: bool = True
```

**Constraints**:
- `step_limit` > 0
- `recovery_threshold` in [0.0, 1.0]
- `protocol_mutation_rate` in [0.0, 1.0]

---

### MultiObjectiveConfig

Multi-objective optimization configuration.

```python
@dataclass(frozen=True)
class ObjectiveSpec:
    """
    Specification for a single optimization objective.
    
    Fields:
        name: Objective identifier (used in fitness dictionaries)
        direction: Whether to minimize or maximize this objective
        weight: Weight for weighted-sum scalarization (optional)
    """
    
    name: str
    direction: str = "minimize"  # Literal["minimize", "maximize"]
    weight: float = 1.0


@dataclass(frozen=True)
class ConstraintSpec:
    """
    Specification for a constraint function.
    
    Constraints return violation magnitude (0 = feasible, >0 = infeasible).
    
    Fields:
        name: Constraint identifier
        penalty_weight: Weight for penalty function approach (if used)
    
    Note: The actual constraint function is provided to the evaluator,
    not encoded in configuration (cannot serialize Python callables).
    """
    
    name: str
    penalty_weight: float = 1.0


@dataclass(frozen=True)
class MultiObjectiveConfig:
    """
    Multi-objective optimization settings.
    
    When present, the factory configures NSGA-II selection and 
    multi-objective fitness handling.
    
    Fields:
        objectives: Tuple of objective specifications
        reference_point: Reference point for hypervolume calculation
        constraints: Constraint specifications (for constraint dominance)
        constraint_handling: How to handle constraints in ranking
    
    Validation:
        - At least 2 objectives for true multi-objective
        - reference_point length must match objectives length
        - reference_point required if hypervolume is tracked
    """
    
    objectives: tuple[ObjectiveSpec, ...] = ()
    reference_point: tuple[float, ...] | None = None
    constraints: tuple[ConstraintSpec, ...] = ()
    constraint_handling: str = "dominance"  # Literal["dominance", "penalty"]
```

**Constraints**:
- `len(objectives)` >= 2 for multi-objective mode
- `len(reference_point)` == `len(objectives)` when specified
- `constraint_handling` in {"dominance", "penalty"}

---

## Registry Entities

### OperatorRegistry

Registry mapping operator names to implementations.

```python
class OperatorRegistry:
    """
    Singleton registry for evolutionary operators.
    
    Maps (category, name) tuples to operator classes.
    Tracks genome compatibility metadata for validation.
    
    Categories:
        - "selection": Selection operators
        - "crossover": Crossover operators  
        - "mutation": Mutation operators
    
    State:
        _operators: dict[tuple[str, str], type]
            Maps (category, name) to operator class
        _compatibility: dict[str, set[str]]
            Maps operator name to compatible genome types ({"*"} = all)
        _params_schema: dict[str, dict[str, Any]]
            Maps operator name to parameter schema (for validation)
    
    Access Pattern:
        Lazy singleton via get_operator_registry() accessor function.
        Built-in operators registered on first access.
    
    Methods:
        register(category, name, cls, compatible_genomes=None)
            Register operator with optional compatibility metadata
        get(category, name, **params) -> Operator
            Instantiate operator with given parameters
        is_compatible(operator_name, genome_type) -> bool
            Check if operator works with genome type
        list_operators(category) -> list[str]
            List registered operator names in category
    """
    
    # Implementation: see research.md for pattern
```

**Built-in Registrations**:

| Category | Name | Class | Compatible Genomes |
|----------|------|-------|-------------------|
| selection | tournament | TournamentSelection | * |
| selection | roulette | RouletteSelection | * |
| selection | rank | RankSelection | * |
| selection | crowded_tournament | CrowdedTournamentSelection | * |
| crossover | uniform | UniformCrossover | vector, sequence |
| crossover | single_point | SinglePointCrossover | vector, sequence |
| crossover | two_point | TwoPointCrossover | vector, sequence |
| crossover | sbx | SimulatedBinaryCrossover | vector |
| crossover | blend | BlendCrossover | vector |
| crossover | neat | NEATCrossover | graph |
| mutation | gaussian | GaussianMutation | vector |
| mutation | uniform | UniformMutation | vector |
| mutation | polynomial | PolynomialMutation | vector |
| mutation | boundary | UniformMutation | vector |
| mutation | neat | NEATMutation | graph |

---

### GenomeRegistry

Registry mapping genome type names to factory functions.

```python
class GenomeRegistry:
    """
    Singleton registry for genome types.
    
    Maps genome type names to factory functions that create genomes.
    
    State:
        _genome_factories: dict[str, Callable[..., Genome]]
            Maps type name to factory function
        _default_params: dict[str, dict[str, Any]]
            Maps type name to default parameters
    
    Access Pattern:
        Lazy singleton via get_genome_registry() accessor function.
        Built-in genomes registered on first access.
    
    Methods:
        register(name, factory_fn, default_params=None)
            Register genome type with factory function
        create(name, **params) -> Genome
            Create genome instance with parameters
        get_factory(name) -> Callable
            Get factory function for type
    """
```

**Built-in Registrations**:

| Name | Factory | Default Parameters |
|------|---------|-------------------|
| vector | VectorGenome.random | dimensions, bounds |
| sequence | SequenceGenome.random | length, alphabet |
| graph | GraphGenome.random | input_nodes, output_nodes |
| scm | SCMGenome.random | num_variables |

---

## Meta-Evolution Entities

### ParameterSpec

Specification for an evolvable parameter.

```python
@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification for a parameter to evolve in meta-evolution.
    
    Defines how a configuration parameter maps to/from vector genome dimensions.
    
    Fields:
        path: Dot-notation path to parameter (e.g., "mutation_params.sigma")
        param_type: Type of parameter for encoding strategy
        bounds: Min/max bounds for continuous/integer parameters
        choices: Valid choices for categorical parameters
        log_scale: Whether to use logarithmic scaling (for rates)
    
    Encoding:
        - continuous: Maps directly to float in [bounds[0], bounds[1]]
        - integer: Maps to float, rounded on decode
        - categorical: Maps to index in [0, len(choices)-1], rounded on decode
    
    Validation:
        - bounds required for continuous/integer
        - choices required for categorical
        - bounds[0] <= bounds[1]
    """
    
    path: str
    param_type: str = "continuous"  # Literal["continuous", "integer", "categorical"]
    bounds: tuple[float, float] | None = None
    choices: tuple[Any, ...] | None = None
    log_scale: bool = False
```

---

### MetaEvolutionConfig

Configuration for meta-evolution outer loop.

```python
@dataclass(frozen=True)
class MetaEvolutionConfig:
    """
    Configuration for meta-evolution (hyperparameter optimization).
    
    When present, the feature runs an outer evolutionary loop that
    evolves configurations and evaluates them via inner loops.
    
    Fields:
        evolvable_params: Parameters to evolve with their bounds
        outer_population_size: Population size for outer loop
        outer_generations: Generations for outer loop
        trials_per_config: Number of inner runs per configuration
        aggregation: How to aggregate fitness across trials
        inner_generations: Override inner loop generations (for speed)
    
    Relationships:
        - Uses ConfigCodec to encode/decode configurations
        - Uses MetaEvaluator to evaluate configurations
        - Produces MetaEvolutionResult
    
    Validation:
        - At least one evolvable parameter required
        - trials_per_config >= 1
        - outer_population_size > 0
        - outer_generations > 0
    """
    
    evolvable_params: tuple[ParameterSpec, ...] = ()
    outer_population_size: int = 20
    outer_generations: int = 10
    trials_per_config: int = 1
    aggregation: str = "mean"  # Literal["mean", "median", "best"]
    inner_generations: int | None = None  # Override for speed
```

---

### ConfigCodec

Encoder/decoder between configurations and vector genomes.

```python
class ConfigCodec:
    """
    Encodes UnifiedConfig to VectorGenome and decodes back.
    
    Works relative to a base configuration: only evolvable parameters
    are encoded; fixed parameters come from the base.
    
    State:
        base_config: UnifiedConfig
            Template configuration with fixed parameter values
        param_specs: tuple[ParameterSpec, ...]
            Parameters being evolved
        _dimension_mapping: list[ParameterSpec]
            Maps vector dimensions to parameter specs
    
    Methods:
        encode(config: UnifiedConfig) -> VectorGenome
            Extract evolvable parameters and encode to vector
        decode(genome: VectorGenome) -> UnifiedConfig
            Decode vector and merge with base configuration
        get_bounds() -> tuple[np.ndarray, np.ndarray]
            Return lower and upper bounds for vector genome
    
    Encoding Strategy:
        - Each ParameterSpec maps to one dimension
        - Continuous: value normalized to bounds
        - Integer: float value, rounded on decode
        - Categorical: index as float, rounded and looked up on decode
    
    Example:
        >>> codec = ConfigCodec(
        ...     base_config=base,
        ...     param_specs=(
        ...         ParameterSpec("population_size", "integer", bounds=(50, 500)),
        ...         ParameterSpec("mutation", "categorical", choices=("gaussian", "polynomial")),
        ...     )
        ... )
        >>> genome = codec.encode(config)
        >>> recovered = codec.decode(genome)
    """
```

---

### MetaEvaluator

Evaluator that runs inner evolutionary loops.

```python
class MetaEvaluator(Evaluator[VectorGenome]):
    """
    Evaluator for meta-evolution that treats configurations as individuals.
    
    Evaluates a configuration by:
    1. Decoding the vector genome to a configuration
    2. Running the inner evolutionary loop
    3. Extracting and caching the result
    4. Returning inner-loop performance as fitness
    
    State:
        codec: ConfigCodec
            Encodes/decodes configurations
        inner_evaluator: Evaluator
            Evaluator for the actual problem
        meta_config: MetaEvolutionConfig
            Meta-evolution settings
        _solution_cache: dict[str, Individual]
            Maps config hash to best solution found
    
    Methods:
        evaluate(genome: VectorGenome, rng: Random) -> Fitness
            Run inner loop and return performance
        get_cached_solution(config_hash: str) -> Individual | None
            Retrieve best solution for configuration
    
    Inner Loop Seeding:
        Seeds are computed deterministically from configuration hash
        and trial number: seed = hash(config_hash + trial_num) % 2^31
    
    Fitness Aggregation:
        For multi-trial evaluation, aggregates using specified method:
        - "mean": Average fitness across trials
        - "median": Median fitness across trials  
        - "best": Best fitness across trials
    """
```

---

### MetaEvolutionResult

Result of a meta-evolution run.

```python
@dataclass(frozen=True)
class MetaEvolutionResult:
    """
    Result of meta-evolution containing both configuration and solution.
    
    Fields:
        best_config: UnifiedConfig
            Best-performing configuration found
        best_solution: Individual
            Best solution found by the best configuration
        config_population: Population[VectorGenome]
            Final population of configuration genomes
        solution_cache: dict[str, Individual]
            All cached solutions indexed by config hash
        outer_history: list[dict[str, Any]]
            Metrics from each outer generation
        inner_history: list[dict[str, Any]] | None
            Metrics from best config's final inner run
        outer_generations: int
            Number of outer generations completed
        total_evaluations: int
            Total inner-loop evaluations performed
    
    Methods:
        get_pareto_configs() -> list[UnifiedConfig]
            For multi-objective meta, get Pareto-optimal configurations
        export_best_config(path: str)
            Save best configuration to JSON
    """
    
    best_config: UnifiedConfig
    best_solution: Individual
    config_population: Population
    solution_cache: dict[str, Individual]
    outer_history: list[dict[str, Any]]
    inner_history: list[dict[str, Any]] | None
    outer_generations: int
    total_evaluations: int
```

---

## Factory Function

### create_engine

Factory function that builds engines from unified configuration.

```python
def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable,
    seed: int | None = None,
    callbacks: Sequence[Callback] | None = None,
) -> EvolutionEngine | ERPEngine:
    """
    Create a ready-to-run engine from unified configuration.
    
    Args:
        config: Unified configuration specifying all parameters
        evaluator: Fitness evaluator or callable fitness function
        seed: Override configuration seed if provided
        callbacks: Custom callbacks (not specifiable in config)
    
    Returns:
        EvolutionEngine if config.erp is None
        ERPEngine if config.erp is specified
    
    Process:
        1. Resolve operators from registry by name
        2. Validate operator-genome compatibility
        3. Create genome factory from registry
        4. Instantiate callbacks from config
        5. Build stopping criteria from config
        6. Construct and return engine
    
    Raises:
        ValueError: If operator not found in registry
        ValueError: If operator incompatible with genome type
        ValueError: If required parameters missing
    """
```

---

## Entity Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             UnifiedConfig                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │StoppingConfig│  │CallbackConfig│  │ ERPSettings  │  │MultiObjectiveConf│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  │ ┌──────────────┐ │ │
│                                                         │ │ObjectiveSpec │ │ │
│  ┌──────────────────────────────────────────────────┐  │ │ConstraintSpec│ │ │
│  │             MetaEvolutionConfig                   │  │ └──────────────┘ │ │
│  │  ┌──────────────────────────────────────┐        │  └──────────────────┘ │
│  │  │ evolvable_params: tuple[ParameterSpec]│        │                       │
│  │  └──────────────────────────────────────┘        │                       │
│  └──────────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  OperatorRegistry   │    │   GenomeRegistry    │    │   ConfigCodec   │
│  ┌───────────────┐  │    │  ┌──────────────┐  │    │  ┌───────────┐  │
│  │selection: {...}│  │    │  │vector: fn()  │  │    │  │base_config│  │
│  │crossover: {...}│  │    │  │sequence: fn()│  │    │  │param_specs│  │
│  │mutation: {...} │  │    │  │graph: fn()   │  │    │  └───────────┘  │
│  └───────────────┘  │    │  │scm: fn()     │  │    └─────────────────┘
└─────────────────────┘    │  └──────────────┘  │             │
         │                 └─────────────────────┘             │
         │                          │                          │
         └─────────────┬────────────┘                          │
                       ▼                                       ▼
              ┌─────────────────┐                     ┌─────────────────┐
              │  create_engine  │                     │  MetaEvaluator  │
              └─────────────────┘                     │  ┌───────────┐  │
                       │                              │  │codec      │  │
                       ▼                              │  │evaluator  │  │
              ┌─────────────────┐                     │  │cache      │  │
              │ EvolutionEngine │                     │  └───────────┘  │
              │   or ERPEngine  │                     └─────────────────┘
              └─────────────────┘                              │
                                                               ▼
                                                    ┌───────────────────────┐
                                                    │  MetaEvolutionResult  │
                                                    │  ┌─────────────────┐  │
                                                    │  │best_config      │  │
                                                    │  │best_solution    │  │
                                                    │  │solution_cache   │  │
                                                    │  └─────────────────┘  │
                                                    └───────────────────────┘
```

---

## JSON Schema Example

```json
{
  "schema_version": "1.0.0",
  "name": "sphere_optimization",
  "description": "Optimize 10D Sphere function with meta-evolution",
  "seed": 42,
  
  "population_size": 100,
  "max_generations": 200,
  "elitism": 2,
  
  "selection": "tournament",
  "selection_params": {"tournament_size": 5},
  
  "crossover": "sbx",
  "crossover_rate": 0.9,
  "crossover_params": {"eta": 20.0},
  
  "mutation": "gaussian",
  "mutation_rate": 1.0,
  "mutation_params": {"sigma": 0.1},
  
  "genome_type": "vector",
  "genome_params": {
    "dimensions": 10,
    "bounds": [-5.12, 5.12]
  },
  "minimize": true,
  
  "stopping": {
    "fitness_threshold": 0.001,
    "stagnation_generations": 20
  },
  
  "callbacks": {
    "enable_logging": true,
    "log_level": "INFO",
    "enable_checkpointing": true,
    "checkpoint_dir": "./checkpoints",
    "checkpoint_frequency": 25
  },
  
  "multiobjective": null,
  "erp": null,
  "meta": null
}
```

---

## Validation Rules Summary

| Entity | Field | Rule |
|--------|-------|------|
| UnifiedConfig | population_size | > 0 |
| UnifiedConfig | max_generations | > 0 |
| UnifiedConfig | elitism | >= 0 and <= population_size |
| UnifiedConfig | crossover_rate | [0.0, 1.0] |
| UnifiedConfig | mutation_rate | [0.0, 1.0] |
| UnifiedConfig | schema_version | Semantic version pattern |
| StoppingConfig | max_generations | > 0 when specified |
| StoppingConfig | stagnation_generations | > 0 when specified |
| StoppingConfig | time_limit_seconds | > 0 when specified |
| CallbackConfig | checkpoint_dir | Required if checkpointing enabled |
| CallbackConfig | checkpoint_frequency | > 0 |
| ERPSettings | step_limit | > 0 |
| ERPSettings | recovery_threshold | [0.0, 1.0] |
| ERPSettings | protocol_mutation_rate | [0.0, 1.0] |
| MultiObjectiveConfig | objectives | len >= 2 for multi-objective |
| MultiObjectiveConfig | reference_point | len == len(objectives) |
| ParameterSpec | bounds | Required for continuous/integer |
| ParameterSpec | choices | Required for categorical |
| ParameterSpec | bounds | bounds[0] <= bounds[1] |
| MetaEvolutionConfig | evolvable_params | len >= 1 |
| MetaEvolutionConfig | trials_per_config | >= 1 |
| MetaEvolutionConfig | outer_population_size | > 0 |
| MetaEvolutionConfig | outer_generations | > 0 |
