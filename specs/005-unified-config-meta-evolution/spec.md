# Feature Specification: Unified Configuration & Meta-Evolution Framework

**Feature Branch**: `005-unified-config-meta-evolution`  
**Created**: March 12, 2026  
**Status**: Draft  
**Input**: User description: "Unify the evolve framework's configuration system into a single, JSON-serializable specification that can define and spawn any experiment type—standard evolution, ERP, multi-objective optimization, or meta-evolution—across all supported genome representations."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define Complete Experiment in JSON (Priority: P1)

A researcher wants to define a complete evolutionary optimization experiment—including population size, operators, and stopping criteria—in a single JSON file that can be loaded and executed without writing any Python orchestration code.

**Why this priority**: This is the foundational use case that enables all other features. Without declarative configuration, researchers must write boilerplate code for every experiment.

**Independent Test**: Can be tested by creating a JSON configuration file, loading it with the framework, and running evolution on a simple benchmark problem (e.g., Sphere function) to verify the configuration produces a working engine.

**Acceptance Scenarios**:

1. **Given** a JSON file specifying population_size=100, selection="tournament", crossover="sbx", mutation="gaussian", and a vector genome with 10 dimensions, **When** the researcher loads this configuration and calls the factory, **Then** a ready-to-run EvolutionEngine is returned with all operators instantiated correctly.

2. **Given** a valid JSON configuration, **When** the researcher modifies one parameter (e.g., mutation_probability) and saves, **Then** reloading produces an engine with the updated parameter without code changes.

3. **Given** a configuration with an invalid operator name (e.g., "nonexistent_mutation"), **When** the researcher attempts to build an engine, **Then** a descriptive error message identifies the unknown operator.

---

### User Story 2 - One-Line Engine Creation from Configuration (Priority: P1)

A researcher has a configuration object (loaded from JSON or created programmatically) and wants to create a ready-to-run engine with a single function call, without manual operator instantiation.

**Why this priority**: This directly addresses the pain point of manual orchestration and reduces the barrier to running experiments.

**Independent Test**: Can be tested by calling `create_engine(config, evaluator)` and verifying the returned engine runs successfully for one generation.

**Acceptance Scenarios**:

1. **Given** a unified configuration and a fitness function, **When** the researcher calls `create_engine(config, fitness_fn)`, **Then** a fully configured engine is returned ready to call `.run()`.

2. **Given** a configuration with ERP enabled, **When** calling the factory, **Then** an ERPEngine is returned instead of EvolutionEngine.

3. **Given** a configuration with multi-objective mode enabled and three objectives, **When** calling the factory, **Then** the engine uses NSGA-II selection and handles Pareto ranking automatically.

---

### User Story 3 - Register Custom Operators (Priority: P2)

A researcher has implemented a domain-specific mutation operator and wants to use it by name in configuration files, without modifying framework source code.

**Why this priority**: Extensibility is essential for researchers with custom problem domains, but the core declarative workflow (P1) must work first.

**Independent Test**: Can be tested by registering a custom operator, referencing it by name in a configuration, and verifying the engine uses the custom operator.

**Acceptance Scenarios**:

1. **Given** a custom mutation class `DomainMutation`, **When** the researcher calls `register_operator("mutation", "domain_specific", DomainMutation)`, **Then** configurations specifying `mutation: "domain_specific"` use this operator.

2. **Given** a registered custom operator with parameters, **When** the configuration includes `mutation_params: {rate: 0.1}`, **Then** the operator is instantiated with those parameters.

---

### User Story 4 - Switch Between Genome Representations (Priority: P2)

A researcher wants to switch between different genome representations (vector, sequence, graph, SCM) by changing configuration parameters, without restructuring experiment code.

**Why this priority**: The framework supports multiple representations, but configuration should abstract over them. This enables comparative studies across representations.

**Independent Test**: Can be tested by creating configs for the same experiment with different `genome_type` values and verifying each produces working experiments.

**Acceptance Scenarios**:

1. **Given** a configuration with `genome_type: "vector"` and `genome_params: {dimensions: 10, bounds: [-5, 5]}`, **When** building the engine, **Then** the population is initialized with 10-dimensional vector genomes.

2. **Given** a configuration with `genome_type: "graph"` and `genome_params: {input_nodes: 4, output_nodes: 2}`, **When** building the engine, **Then** graph genomes are used with appropriate topology mutation operators.

3. **Given** a configuration with an incompatible operator for the genome type (e.g., Gaussian mutation for graph genome), **When** building the engine, **Then** a validation error is raised before execution.

---

### User Story 5 - Meta-Evolution for Hyperparameter Optimization (Priority: P3)

A researcher wants to automatically find optimal hyperparameters (population size, mutation rate, operator choices) by evolving configurations themselves, with the framework handling inner evolutionary runs.

**Why this priority**: Meta-evolution builds on all previous functionality and is an advanced use case. It requires the full configuration system to work first.

**Independent Test**: Can be tested by specifying evolvable parameters with bounds, running meta-evolution on a benchmark problem, and verifying both a best configuration and best solution are returned.

**Acceptance Scenarios**:

1. **Given** a base configuration and parameter bounds specifying `population_size: [50, 500]` and `mutation_probability: [0.01, 0.3]` as evolvable, **When** running meta-evolution, **Then** the framework evolves these parameters and returns the best-performing configuration.

2. **Given** meta-evolution completes successfully, **When** accessing results, **Then** both `result.best_configuration` (optimal hyperparameters) and `result.best_solution` (solution found by best config) are available.

3. **Given** a meta-evolution run with 3 trials per configuration for robustness, **When** evaluating each configuration, **Then** fitness is aggregated across trials (mean by default).

---

### User Story 6 - Multi-Objective Configuration (Priority: P2)

A researcher wants to configure multi-objective optimization by declaring objective names, optimization directions, and reference points in configuration, with the framework automatically setting up NSGA-II selection.

**Why this priority**: Multi-objective optimization is implemented but lacks first-class configuration support. Many real problems are naturally multi-objective.

**Independent Test**: Can be tested by creating a configuration with multiple objectives and verifying the engine produces Pareto fronts.

**Acceptance Scenarios**:

1. **Given** a configuration with `objectives: [{name: "accuracy", direction: "maximize"}, {name: "complexity", direction: "minimize"}]`, **When** building the engine, **Then** NSGA-II selection is used and individuals have multi-objective fitness.

2. **Given** a multi-objective configuration with `reference_point: [1.0, 100.0]`, **When** evolution completes, **Then** hypervolume can be computed against the reference point.

---

### Edge Cases

- What happens when configuration specifies an operator with missing required parameters? The factory raises a validation error listing missing parameters.
- How does the system handle conflicting configuration flags (e.g., ERP enabled with incompatible genome type)? Validation catches and reports conflicts before engine construction.
- What happens when meta-evolution's inner loop fails (invalid configuration produces error)? The meta-evaluator assigns worst-case fitness and continues evolution.
- How does the system handle partial JSON (missing optional sections)? Defaults are applied; only missing required fields cause errors.

## Requirements *(mandatory)*

### Functional Requirements

**Configuration Schema**

- **FR-001**: System MUST provide a unified `UnifiedConfig` class that encompasses all parameters from `EvolutionConfig`, `ERPConfig`, and `ExperimentConfig`.
- **FR-002**: System MUST support JSON serialization via `to_json()` and deserialization via `from_json()` methods.
- **FR-003**: Configuration MUST compute a deterministic hash for experiment tracking and reproducibility.
- **FR-004**: Configuration MUST validate consistency (e.g., ERP parameters only required when ERP enabled).
- **FR-005**: Configuration MUST support creating modified copies via a `with_params(**kwargs)` method.
- **FR-006**: Configuration MUST include a schema version field following semantic versioning (major.minor.patch).
- **FR-007**: Framework MUST load older schema versions with deprecation warnings when breaking changes exist.
- **FR-008**: Framework MUST reject configurations with schema versions newer than the framework supports, with a clear version mismatch error.

**Stopping Criteria**

- **FR-009**: Configuration MUST support specifying a maximum generation limit as a stopping condition.
- **FR-010**: Configuration MUST support specifying a fitness threshold that stops evolution when reached.
- **FR-011**: Configuration MUST support stagnation detection that stops evolution after N generations with no fitness improvement.
- **FR-012**: Configuration MUST support a wall-clock time limit that stops evolution after a specified duration.
- **FR-013**: Multiple stopping criteria MAY be specified; evolution stops when ANY condition is met.

**Operator Registry**

- **FR-014**: System MUST provide an `OperatorRegistry` that maps string names to operator implementations.
- **FR-015**: Registry MUST use lazy initialization, populating built-in operators on first access.
- **FR-016**: Registry MUST pre-register all built-in selection operators (tournament, roulette, rank, crowded_tournament).
- **FR-017**: Registry MUST pre-register all built-in crossover operators (single_point, two_point, uniform, sbx, blend).
- **FR-018**: Registry MUST pre-register all built-in mutation operators (gaussian, uniform, polynomial, creep).
- **FR-019**: Registry MUST support runtime registration of custom operators via `register(category, name, cls)`.
- **FR-020**: Registry MUST instantiate operators with parameter dictionaries via `get(category, name, **params)`.
- **FR-021**: Registry MUST track genome type compatibility for operators that are representation-specific.

**Genome Registry**

- **FR-022**: System MUST provide a `GenomeRegistry` that maps genome type names to factory functions.
- **FR-023**: Registry MUST use lazy initialization, populating built-in genome types on first access.
- **FR-024**: Registry MUST support all four genome types: vector, sequence, graph, scm.
- **FR-025**: Registry MUST produce genome factory functions given type and parameters.
- **FR-026**: Registry MUST support registration of custom genome types at runtime.

**Engine Factory**

- **FR-027**: System MUST provide a `create_engine(config, evaluator, seed=None)` factory function.
- **FR-028**: Factory MUST resolve operators from registry based on configuration strings.
- **FR-029**: Factory MUST produce `ERPEngine` when `erp_enabled=True`, otherwise `EvolutionEngine`.
- **FR-030**: Factory MUST configure multi-objective selection when `multiobjective_enabled=True`.
- **FR-031**: Factory MUST validate operator-genome compatibility before constructing engine.
- **FR-032**: Factory MUST accept either callable fitness functions or Evaluator instances.
- **FR-033**: Factory MUST thread random seed through all components deterministically.

**Constraint Handling**

- **FR-034**: Configuration MAY specify constraints as named functions that return violation magnitude (0 = feasible).
- **FR-035**: When constraints are specified, constraint violation MUST be treated as an additional minimization objective.
- **FR-036**: In Pareto ranking, feasible solutions (zero total violation) MUST always dominate infeasible solutions.
- **FR-037**: Among infeasible solutions, those with lower total constraint violation MUST be ranked higher.

**Callbacks**

- **FR-038**: Configuration MAY specify built-in callbacks (logging, checkpointing, early stopping) with their parameters.
- **FR-039**: Factory MUST instantiate configured built-in callbacks and attach them to the engine.
- **FR-040**: Factory MUST accept additional custom callbacks as a separate parameter, not via configuration.
- **FR-041**: Built-in logging callback MUST support configurable log level and output destination.
- **FR-042**: Built-in checkpointing callback MUST support configurable checkpoint directory and frequency.

**Meta-Evolution**

- **FR-043**: System MUST provide a `ConfigCodec` that encodes configurations to vector genomes and decodes back.
- **FR-044**: Codec MUST support specifying which parameters are evolvable with their bounds.
- **FR-045**: Codec MUST handle both continuous parameters (mapped to float dimensions) and categorical parameters (mapped to discrete indices).
- **FR-046**: System MUST provide a `MetaEvaluator` that evaluates configurations by running inner evolutionary loops.
- **FR-047**: MetaEvaluator MUST cache the best solution found by each configuration.
- **FR-048**: MetaEvaluator MUST support running multiple trials per configuration and aggregating results.
- **FR-049**: MetaEvaluator MUST seed inner runs deterministically based on configuration hash and trial number.
- **FR-050**: System MUST provide a `MetaEvolutionResult` that includes both best configuration and best solution.

**Backward Compatibility**

- **FR-051**: Existing `EvolutionConfig`, `ERPConfig`, and engine APIs MUST remain functional without modification.
- **FR-052**: Unified configuration MUST be usable alongside existing configuration classes.

### Key Entities

- **UnifiedConfig**: Complete experiment specification including evolution parameters, operator names, genome type, multi-objective settings, ERP settings, and meta-evolution settings. Serializable to/from JSON.

- **OperatorRegistry**: Singleton registry mapping (category, name) tuples to operator classes. Categories include "selection", "crossover", "mutation". Tracks genome compatibility metadata.

- **GenomeRegistry**: Singleton registry mapping genome type names to factory functions and compatibility information.

- **ConfigCodec**: Encodes evolvable parameters of a configuration to a vector genome and decodes vector genomes back to configurations. Works relative to a base configuration.

- **MetaEvaluator**: Evaluator implementation that treats configuration genomes as individuals, runs inner loops, and returns inner-loop performance as fitness. Caches solutions.

- **MetaEvolutionResult**: Extended result type containing the outer loop Pareto front (configurations), the best configuration, and the best solution found by that configuration.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can define a complete experiment (representation, operators, parameters, objectives) in a single JSON file requiring no Python code for standard use cases.

- **SC-002**: Creating a ready-to-run engine from a configuration requires exactly one function call with no manual operator instantiation.

- **SC-003**: Switching from standard evolution to ERP or multi-objective requires only changing configuration flags—no code restructuring.

- **SC-004**: Setting up meta-evolution requires only specifying evolvable parameters and bounds; the framework handles encoding, inner loops, and result aggregation.

- **SC-005**: Meta-evolution results provide both the best configuration found and the best solution that configuration produced, accessible via distinct result attributes.

- **SC-006**: All four genome types (vector, sequence, graph, SCM) are configurable through the unified schema with appropriate representation-specific parameters.

- **SC-007**: All existing tests pass without modification after feature implementation.

- **SC-008**: Configuration JSON files are portable: a configuration saved on one machine can be loaded and executed on another machine with the same framework version.

## Assumptions

The following assumptions have been made where the feature description left details unspecified:

1. **Operator Parameters**: Operator parameters will be specified entirely in configuration. The registry will not maintain defaults; all required parameters must be explicit. This ensures configuration portability and reproducibility.

2. **Compatibility Enforcement**: Operator-genome compatibility will be validated at factory build time (not registration time). This provides early error detection while allowing flexible operator registration.

3. **Categorical Encoding**: Categorical parameters (operator choices) will use index mapping with bounds, as this is more compact and aligns with existing vector genome infrastructure.

4. **Multi-Objective Meta-Fitness**: When the inner loop is multi-objective, meta-evolution will optimize hypervolume by default. The reference point must be specified in configuration.

5. **ERP with Multi-Objective**: When ERP is combined with multi-objective, matchability will use Pareto rank (lower rank = better fitness) for compatibility decisions.

6. **Decoder Configuration**: Decoder type will be inferred from genome type by default. Graph genomes use network decoder, SCM genomes use SCM decoder. Custom decoders can be specified explicitly.

7. **Trial Aggregation**: Meta-evaluation will use mean aggregation across trials by default. Alternative aggregation methods (median, best) can be specified in configuration.

## Scope Boundaries

### In Scope

- Unified configuration schema covering all existing configuration types plus multi-objective and meta-evolution
- Operator registry with built-in operators pre-registered and extension points for custom operators
- Genome registry covering all existing representation types
- Configuration-to-engine factory that produces ready-to-run engines
- Configuration encoding/decoding for meta-evolution using existing vector genome infrastructure
- Meta-evaluator that runs inner loops and caches solutions
- Full JSON serialization and deserialization of configurations
- Backward compatibility with existing engine and configuration APIs

### Out of Scope

- Automatic hyperparameter bounds detection (researchers must specify ranges)
- Distributed or parallel meta-evolution (single-machine only for this feature)
- Neural architecture search beyond what graph genome already supports
- New operator implementations (this feature organizes existing operators)
- Breaking changes to existing public APIs
- GUI or web interface for configuration editing

## Constraints

- **Backward Compatibility**: Existing code using `EvolutionConfig`, `ERPConfig`, and `EvolutionEngine` must continue to work unchanged.
- **Explicit Randomness**: All randomness must flow through explicit random number generators with deterministic seeding.
- **Separation of Concerns**: Configuration knows nothing about evaluation; evaluators know nothing about configuration encoding.
- **No Framework Dependencies in Genomes**: No PyTorch/JAX types in configuration schemas.
- **Registry Extensibility**: Custom operators and genome types registrable at runtime without source modification.

## Dependencies

- Existing `EvolutionEngine` and `ERPEngine` implementations
- Existing operator implementations (selection, crossover, mutation)
- Existing genome implementations (VectorGenome, SequenceGenome, GraphGenome, SCMGenome)
- Existing multi-objective module (NSGA-II, Pareto ranking, crowding distance)
- Existing `ExperimentConfig` serialization infrastructure

## Clarifications

### Session 2026-03-12

- Q: How should the configuration system handle schema versioning for forward/backward compatibility? → A: Semantic versioning - configs include schema version; newer frameworks support older schemas with deprecation warnings; older frameworks reject newer schemas with clear version mismatch error.
- Q: Which stopping criteria should be declarable in the unified configuration? → A: Standard set - generation limit, fitness threshold, stagnation detection (no improvement for N generations), and wall-clock time limit.
- Q: How should the unified configuration handle constrained optimization problems? → A: Constraint dominance - constraint violation treated as additional minimization objective; feasible solutions always dominate infeasible ones in Pareto ranking.
- Q: When should the registries initialize and register built-in operators? → A: Lazy on first access - registries initialize and populate built-ins when first accessed (e.g., `get_operator_registry()`).
- Q: Should callbacks be declarable in the unified configuration? → A: Built-in only - built-in callbacks (logging, checkpointing, early stopping) configurable via config; custom callbacks passed to factory separately.
