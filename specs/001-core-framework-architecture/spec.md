# Feature Specification: Evolve Framework Core Architecture

**Feature Branch**: `001-core-framework-architecture`  
**Created**: 2026-01-13  
**Status**: Draft  
**Input**: Design a modular, extensible evolutionary algorithms experimentation framework suitable for research and benchmarking

## Overview

This specification defines a research-grade evolutionary algorithms experimentation framework designed around five independently extensible architectural layers. The framework supports classical genetic algorithms, neuroevolution, genetic programming, multi-objective optimization, and reinforcement learning while remaining model-agnostic and backend-agnostic by default.

The architecture enforces strict separation between evolutionary logic (which operates on abstract genomes), representation (which defines genotype-phenotype mappings), evaluation (where acceleration is permitted), execution backends (optional performance optimizations), and observability (experiment tracking and reproducibility).

## Clarifications

### Session 2026-01-13

- Q: How should numeric equivalence be validated between CPU and accelerated evaluators? → A: Relative tolerance only (values within 0.001% / 1e-5 of each other)
- Q: Should "Policy" be treated as a specialized type of Phenotype, or as a distinct concept? → A: Policy is a subtype of Phenotype (Policy extends/implements Phenotype interface)
- Q: What should happen when a configured accelerated backend (GPU/JAX) is unavailable? → A: Default behavior is to raise a clear `BackendUnavailableError` with installation instructions. Users may opt-in to automatic CPU fallback via `ExperimentConfig.fallback_to_cpu=True`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Classical Genetic Algorithm (Priority: P1)

A researcher wants to run a simple genetic algorithm to solve a continuous optimization problem (e.g., sphere function minimization) using only CPU computation. They configure population size, selection method, crossover rate, and mutation rate, then run the evolution loop and observe convergence.

**Why this priority**: This validates the fundamental Evolution Core and provides the simplest end-to-end test of the framework. All other use cases build upon this foundation.

**Independent Test**: Can be fully tested using pure Python with no external dependencies. Delivers a working evolutionary optimizer that solves benchmark problems.

**Acceptance Scenarios**:

1. **Given** a configured GA with tournament selection and Gaussian mutation, **When** evolution runs for 100 generations on the sphere function, **Then** the best fitness decreases monotonically on average and converges within expected bounds.
2. **Given** the same configuration and seed, **When** evolution runs twice, **Then** both runs produce identical fitness trajectories.
3. **Given** a configured GA, **When** the experiment completes, **Then** all generation metrics are logged and the final population can be checkpointed.

---

### User Story 2 - Multi-Objective Optimization with Pareto Ranking (Priority: P2)

A researcher wants to solve a multi-objective problem (e.g., ZDT benchmark) using NSGA-II-style Pareto ranking and crowding distance. They configure multiple fitness objectives, run evolution, and visualize the Pareto front.

**Why this priority**: Multi-objective optimization is a core differentiator and required for many real-world problems. Builds directly on the Evolution Core from US1.

**Independent Test**: Can be tested with pure Python using standard multi-objective benchmarks (ZDT1-6, DTLZ). Delivers Pareto front approximations comparable to reference implementations.

**Acceptance Scenarios**:

1. **Given** a ZDT1 problem with 2 objectives, **When** NSGA-II runs for 250 generations, **Then** the final population approximates the known Pareto front with hypervolume within 95% of reference.
2. **Given** a multi-objective configuration, **When** evolution completes, **Then** each individual has a Pareto rank and crowding distance computed.
3. **Given** individuals with constraint violations, **When** selection occurs, **Then** feasible individuals are preferred over infeasible ones.

---

### User Story 3 - Neuroevolution with Custom Neural Network Representation (Priority: P3)

A researcher wants to evolve neural network weights or topologies to solve a supervised learning task. They define a genome representing network parameters, a decoder that produces a neural network phenotype, and an evaluator that computes loss on a dataset.

**Why this priority**: Neuroevolution demonstrates the Representation Layer abstraction and the separation between framework-neutral genomes and backend-specific phenotypes.

**Independent Test**: Can be tested with a simple XOR problem using NumPy-based networks. GPU acceleration is optional. Delivers evolved networks that solve the target task.

**Acceptance Scenarios**:

1. **Given** a fixed-topology network genome and XOR dataset, **When** evolution runs, **Then** an individual emerges that correctly classifies all XOR patterns.
2. **Given** a genome decoded to a NumPy network, **When** the same genome is decoded to a PyTorch network (if available), **Then** both produce identical outputs for the same input.
3. **Given** a neuroevolution experiment, **When** checkpointing occurs, **Then** genomes are serialized without framework-specific objects.

---

### User Story 4 - Reinforcement Learning via Evolutionary Optimization (Priority: P4)

A researcher wants to evolve policies for an RL environment (e.g., CartPole). They configure an environment interface, define a policy genome, implement episode-based fitness evaluation, and run evolution.

**Why this priority**: RL is a major application domain for evolutionary methods. Demonstrates the Evaluation Layer abstraction where rollouts become fitness evaluations.

**Independent Test**: Can be tested with simple control tasks. Delivers evolved policies that achieve target episode returns.

**Acceptance Scenarios**:

1. **Given** a CartPole environment and policy genome, **When** evolution runs for 50 generations, **Then** the best policy achieves average episode return above 195.
2. **Given** stochastic rollouts, **When** multiple episodes are averaged for fitness, **Then** fitness variance decreases with more episodes.
3. **Given** vectorized environments, **When** evaluation uses batch rollouts, **Then** wall-clock time decreases proportionally.

---

### User Story 5 - Island Model with Migration (Priority: P5)

A researcher wants to run parallel populations (islands) with periodic migration to balance exploration and exploitation. They configure multiple islands with different selection pressures and define migration topology.

**Why this priority**: Island models are essential for scalability and diversity maintenance. Demonstrates the framework's support for structured populations.

**Independent Test**: Can be tested with multiple CPU processes. Delivers improved convergence compared to single-population baselines on deceptive fitness landscapes.

**Acceptance Scenarios**:

1. **Given** 4 islands with ring migration topology, **When** evolution runs, **Then** individuals migrate at configured intervals and genetic diversity is preserved across islands.
2. **Given** islands with different mutation rates, **When** evolution completes, **Then** best individuals from high-exploitation islands often originated from high-exploration islands.
3. **Given** deterministic migration scheduling, **When** the same experiment runs twice with identical seeds, **Then** migration events occur identically.

---

### User Story 6 - Speciation and Novelty Search (Priority: P6)

A researcher wants to maintain population diversity through speciation (grouping similar individuals) and novelty search (rewarding behavioral uniqueness). They configure distance metrics and novelty archives.

**Why this priority**: Diversity preservation mechanisms are crucial for avoiding premature convergence. Distinct from island models which use spatial separation.

**Independent Test**: Can be tested with deceptive maze navigation problems. Delivers solutions to problems where fitness-only approaches fail.

**Acceptance Scenarios**:

1. **Given** a deceptive maze, **When** novelty search runs, **Then** the archive grows with behaviorally diverse solutions and the goal is eventually reached.
2. **Given** NEAT-style speciation, **When** species form, **Then** individuals within a species have genotypic distance below threshold.
3. **Given** a novelty archive, **When** a new individual is evaluated, **Then** its novelty score is computed relative to archive contents.

---

### User Story 7 - GPU-Accelerated Batch Evaluation (Priority: P7)

A researcher wants to accelerate fitness evaluation by batching individuals and computing fitness on GPU. They implement an accelerated evaluator that maintains semantic equivalence with the CPU reference.

**Why this priority**: Performance acceleration is critical for large-scale experiments but must not compromise correctness. Demonstrates the optional acceleration pattern.

**Independent Test**: Can be tested by comparing GPU evaluator outputs against CPU reference. Delivers speedup while maintaining result equivalence.

**Acceptance Scenarios**:

1. **Given** identical genomes, **When** evaluated by CPU and GPU evaluators, **Then** fitness values match within floating-point tolerance.
2. **Given** a batch of 1000 individuals, **When** GPU evaluation runs, **Then** wall-clock time is at least 10x faster than sequential CPU evaluation.
3. **Given** an accelerated evaluator, **When** randomness is required, **Then** seeds are passed explicitly and results are reproducible.

---

### User Story 8 - Experiment Tracking and Reproducibility (Priority: P8)

A researcher wants to log all experiment configurations, track metrics across generations, checkpoint populations, and reproduce published results. They integrate with experiment tracking systems.

**Why this priority**: Reproducibility is a core constitution principle. Enables academic benchmarking and result verification.

**Independent Test**: Can be tested by running an experiment, checkpointing, resuming, and verifying continuity. Delivers complete experiment provenance.

**Acceptance Scenarios**:

1. **Given** an experiment configuration, **When** the experiment starts, **Then** all hyperparameters, seeds, and environment details are logged automatically.
2. **Given** a checkpoint file, **When** evolution resumes, **Then** it continues from the exact state (including RNG) as if uninterrupted.
3. **Given** logged experiment metadata and seed, **When** the experiment is re-run, **Then** identical results are produced.

---

### Edge Cases

- **Empty population**: What happens when population size is set to zero or all individuals are invalid? System MUST raise a clear error at configuration time.
- **Fitness NaN/Inf**: How does selection handle individuals with non-finite fitness? System MUST exclude them from selection or assign worst rank.
- **Genome serialization failure**: What happens when a genome cannot be serialized for checkpointing? System MUST raise an error with diagnostic information.
- **Backend unavailable**: What happens when a configured accelerated backend (GPU/JAX) is not available? System MUST raise `BackendUnavailableError` by default; if `fallback_to_cpu=True`, fall back gracefully to CPU reference with a warning logged.
- **Migration to empty island**: What happens when migration occurs but target island has no recipients? System MUST handle gracefully (skip or queue).
- **Constraint-only infeasible population**: What happens when all individuals violate constraints? System MUST continue evolution (repair or relaxation) rather than crash.
- **Seed collision across islands**: What happens when parallel islands use the same seed? System MUST derive unique seeds per island from a master seed.

## Requirements *(mandatory)*

### Functional Requirements

#### Evolution Core Layer

- **FR-001**: Evolution Core MUST define abstract interfaces for population initialization, selection, crossover, mutation, replacement, early stopping, migration, and speciation.
- **FR-002**: Evolution Core MUST NOT import or depend on PyTorch, JAX, TensorFlow, or any ML framework.
- **FR-003**: Evolution Core MUST NOT import or depend on any execution backend implementation.
- **FR-004**: Evolution Core MUST operate solely on abstract Genome types without knowledge of their internal structure.
- **FR-005**: Selection operators MUST support configurable elitism (preserving top N individuals unchanged).
- **FR-006**: The evolution loop MUST emit structured events at generation boundaries (start, end, statistics).
- **FR-007**: Early stopping MUST support pluggable criteria (fitness threshold, stagnation detection, generation limit).

#### Multi-Objective Optimization

- **FR-008**: Fitness MUST be representable as a vector of objective values (not just scalar).
- **FR-009**: System MUST provide Pareto dominance checking (individual A dominates B if better or equal on all objectives and strictly better on at least one).
- **FR-010**: System MUST provide Pareto ranking (non-dominated sorting into fronts).
- **FR-011**: System MUST provide crowding distance computation for diversity preservation within fronts.
- **FR-012**: Selection MUST support constraint handling where feasible individuals are preferred over infeasible.
- **FR-013**: Multi-objective components MUST be usable independently (Pareto ranking without requiring specific selection operator).

#### Representation Layer

- **FR-014**: System MUST define an abstract Genome interface that is framework-neutral (no PyTorch/JAX types).
- **FR-015**: System MUST define an abstract Phenotype interface representing decoded forms.
- **FR-016**: System MUST define a Decoder interface for genotype-to-phenotype mapping.
- **FR-017**: Genomes MUST be serializable using standard formats (pickle, JSON, or custom).
- **FR-018**: Decoders MAY produce backend-specific phenotypes (tensors, graphs) but this MUST NOT leak into Evolution Core.
- **FR-019**: System MUST provide reference implementations for common representations: fixed-length vectors, variable-length sequences, graph structures.

#### Evaluation Layer

- **FR-020**: System MUST define an abstract Evaluator interface accepting batches of individuals and returning fitness vectors.
- **FR-021**: Evaluators MUST declare their capabilities (batchable, stochastic, stateful).
- **FR-022**: Evaluators MUST accept explicit seed or PRNG key for reproducible stochastic evaluation.
- **FR-023**: All accelerated Evaluators (GPU, JIT) MUST have corresponding CPU reference implementations.
- **FR-024**: CPU and accelerated evaluators for the same task MUST produce equivalent results using relative tolerance validation (|a - b| / max(|a|, |b|) ≤ 1e-5).
- **FR-025**: Evaluators MAY return optional diagnostics (per-objective breakdown, intermediate values, timing).

#### Reinforcement Learning Support

- **FR-026**: System MUST define an abstract Environment interface for RL tasks.
- **FR-027**: System MUST define an abstract Policy interface as a subtype of Phenotype that maps observations to actions.
- **FR-028**: System MUST define a Rollout/Trajectory abstraction for episode data.
- **FR-029**: Episode-based fitness evaluation MUST support configurable number of episodes for variance reduction.
- **FR-030**: Environment interfaces MUST support optional vectorized execution for batch rollouts.

#### Memetic Algorithms Extension Points *(Deferred to Feature 002)*

> **Note**: Memetic algorithm implementation is deferred to a separate feature. This feature defines interface placeholders only; the `evolve/memetic/` module will contain stub files with Protocol definitions but no implementations.

- **FR-031**: System MUST define extension points for local search operators (interface only, implementation optional in core).
- **FR-032**: Local search operators MAY use gradient-based methods (PyTorch, JAX optimizers). *(Implementation deferred)*
- **FR-033**: Memetic components MUST be composable with standard evolutionary operators. *(Validation deferred)*

#### Island Models

- **FR-034**: System MUST support multiple semi-isolated populations (islands) evolving in parallel.
- **FR-035**: System MUST support configurable migration: interval, number of migrants, selection of migrants, replacement of recipients.
- **FR-036**: System MUST support configurable migration topology (ring, fully connected, custom).
- **FR-037**: Different islands MAY have different evolutionary parameters (selection pressure, mutation rate).

#### Speciation and Niching

- **FR-038**: System MUST support speciation via configurable genotypic or phenotypic distance metrics.
- **FR-039**: System MUST support fitness sharing within species/niches.
- **FR-040**: System MUST support novelty search with configurable novelty archives.
- **FR-041**: Speciation and island models MUST be usable simultaneously (species within islands).

#### Execution Backends

- **FR-042**: Execution backends MUST be optional, swappable components.
- **FR-043**: System MUST provide a default sequential CPU execution backend.
- **FR-044**: System MAY provide parallel execution backends (multiprocessing, distributed).
- **FR-045**: System MAY provide accelerated backends (GPU via PyTorch/JAX).
- **FR-046**: Backend selection MUST NOT require changes to Evolution Core code.

#### Experiment Management

- **FR-047**: System MUST automatically log all configuration parameters at experiment start.
- **FR-048**: System MUST log generation-level metrics (best fitness, average fitness, diversity metrics).
- **FR-049**: System MUST support checkpointing population state including RNG state.
- **FR-050**: System MUST support resuming evolution from checkpoint with exact continuity.
- **FR-051**: System MUST capture reproducibility metadata (seed, library versions, environment).
- **FR-052**: System MUST integrate with standard experiment tracking tools (MLflow, Weights & Biases).

#### Configuration

- **FR-053**: System MUST support declarative configuration of experiments (YAML, TOML, or dataclass).
- **FR-054**: Configuration MUST include: generations, population size, offspring count, selection method, crossover operator, mutation operator, elitism rate.
- **FR-055**: Configuration MUST support island-specific parameters when using island models.
- **FR-056**: Configuration MUST be serializable and hashable for versioning.

### Key Entities

- **Individual**: A candidate solution comprising a genome, fitness vector, and optional metadata (age, lineage, species ID).
- **Population**: An ordered collection of individuals with associated statistics (diversity, rank distribution).
- **Genome**: Framework-neutral representation of a candidate solution's genetic material (abstract interface).
- **Phenotype**: Decoded form of a genome that can be evaluated (may be backend-specific). Base interface for all decoded representations.
- **Policy**: Subtype of Phenotype specialized for RL; maps observations to actions.
- **Fitness**: Vector-valued measure of solution quality (supports single and multi-objective).
- **Evaluator**: Component that computes fitness for batches of individuals (acceleration boundary).
- **Operator**: Evolutionary operator (selection, crossover, mutation) implementing a specific strategy.
- **Island**: Semi-isolated population with its own evolutionary parameters and migration channels.
- **Species**: Group of similar individuals within a population for fitness sharing/protection.
- **Experiment**: Configuration, execution trace, and results of an evolutionary run.
- **Checkpoint**: Serialized state enabling exact resumption of an experiment.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can implement and run a classical GA solving standard benchmarks (sphere, Rastrigin, Rosenbrock) with fewer than 50 lines of configuration code.
- **SC-002**: Multi-objective experiments achieve hypervolume within 95% of published NSGA-II results on ZDT benchmarks.
- **SC-003**: Identical seeds produce byte-identical fitness trajectories across independent runs (100% determinism for CPU execution).
- **SC-004**: Neuroevolution experiments run without importing PyTorch/JAX when using CPU-only evaluation.
- **SC-005**: Accelerated evaluators produce results within 1e-5 relative tolerance of CPU reference implementations.
- **SC-006**: Checkpointed experiments resume and produce identical subsequent results as uninterrupted runs.
- **SC-007**: Island model experiments with 8 islands show improved final fitness compared to single population with equivalent total evaluation budget on deceptive problems.
- **SC-008**: Framework core modules have zero transitive dependencies on ML frameworks (verified by import analysis).
- **SC-009**: New evolutionary operators can be added without modifying Evolution Core (verified by implementing 3 novel operators as extensions).
- **SC-010**: Documentation enables a new user to run their first experiment within 15 minutes of installation.

## Assumptions

- Target users are researchers familiar with evolutionary computation concepts but not necessarily software engineering experts.
- Python 3.10+ is the primary development language; other language bindings are out of scope.
- NumPy is acceptable as a core dependency for numerical operations (considered framework-neutral).
- Standard library dependencies (dataclasses, typing, logging) are acceptable in all layers.
- Experiment tracking integration assumes users have access to MLflow or similar tools (graceful degradation if unavailable).
- GPU acceleration assumes CUDA availability; other accelerators (ROCm, Metal) are lower priority.
- Distributed execution assumes a shared filesystem or object storage for checkpoint/result aggregation.

## Dependencies

- Python standard library (typing, dataclasses, logging, pickle)
- NumPy (numerical operations, reference implementations)
- Optional: PyTorch (GPU evaluation, gradient-based local search)
- Optional: JAX (JIT compilation, vectorized evaluation)
- Optional: MLflow / Weights & Biases (experiment tracking)
- Optional: Ray / Dask (distributed execution)

## Out of Scope

- Automatic algorithm configuration / hyperparameter tuning (AutoML)
- Production deployment tooling (serving, monitoring, scaling)
- Visualization dashboards (beyond basic metric logging)
- Specific problem domain implementations (only abstract interfaces provided)
- Non-Python language bindings
- Real-time / online evolution (focus is batch experimentation)
