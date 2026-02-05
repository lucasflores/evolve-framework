# Feature Specification: Comprehensive Tutorial Notebooks for Evolve Framework

**Feature Branch**: `004-tutorial-notebooks`  
**Created**: 2026-01-31  
**Status**: Draft  
**Input**: Create comprehensive tutorial notebooks for the evolve framework representations, targeting ML/DS practitioners with strong statistics/linear algebra background but NO evolutionary algorithms experience.

## Overview

This specification defines a suite of 5 interactive Jupyter notebooks plus a shared utility module that teaches the Evolve framework's representation types through hands-on examples. Each notebook targets machine learning/data science practitioners who understand optimization, loss functions, and model training but have no prior exposure to evolutionary algorithms.

The tutorials bridge ML vocabulary to evolutionary terminology, provide end-to-end examples with synthetic data, demonstrate advanced features (island models, speciation, multi-objective optimization, GPU acceleration), and include visual diagnostics for understanding evolutionary dynamics.

## Clarifications

### Session 2026-01-31

- Q: Should mermaid rendering use subprocess to node, or pre-render all diagrams to SVG files checked into repo? → A: Dynamic rendering via beautiful-mermaid library at runtime (no pre-rendered SVGs, no Node.js subprocess)
- Q: Pareto visualization: 2D projections only, or interactive 3D for 3-objective SCM? → A: 2D pairwise projections plus interactive 3D option via plotly scatter3d for full Pareto surface exploration
- Q: Island model benchmarks: Recommended island count and population sizes for meaningful parallel speedup? → A: 4 islands with 50 individuals per island (200 total), balanced for typical 4-core machines
- Q: RL environment: CartPole (simple, fast) or LunarLander (more interesting, slower)? → A: CartPole-v1 as primary environment, LunarLander-v2 as optional advanced section
- Q: Speciation visualization: Phylogenetic tree style or species-count bar charts over time? → A: Stacked area chart as primary visualization (population composition over time), phylogenetic tree as optional advanced view
- Q: beautiful-mermaid theme: tokyo-night (dark) or github-light (light) as default? → A: github-light (light theme) for maximum readability, print-friendliness, and accessibility

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Learn VectorGenome for Continuous Optimization (Priority: P1)

An ML practitioner wants to understand how evolutionary algorithms work by solving a continuous optimization problem they already understand (e.g., function minimization). They follow a VectorGenome notebook that relates EA concepts to gradient-based optimization, generates synthetic test data, runs evolution, and visualizes convergence.

**Why this priority**: VectorGenome is the simplest representation and the best starting point. Continuous optimization maps directly to hyperparameter tuning and neural network weight optimization—familiar territory for ML practitioners.

**Independent Test**: Can be tested by running the notebook end-to-end on a fresh environment. Delivers a working optimization that finds near-optimal solutions to benchmark functions (Rastrigin, Rosenbrock) with clear visualizations.

**Acceptance Scenarios**:

1. **Given** a new user with NumPy/scikit-learn experience, **When** they complete the VectorGenome notebook, **Then** they can explain genome encoding, selection pressure, and mutation rate in ML terminology.
2. **Given** the notebook's synthetic data generation, **When** running with default parameters, **Then** evolution converges to within 1% of the known optimum within 100 generations.
3. **Given** the island model section, **When** parallel execution is enabled, **Then** multiple CPU processes run simultaneously and migration events are logged.
4. **Given** the GPU acceleration section, **When** a CUDA-capable GPU is available, **Then** benchmark comparison shows measurable speedup for large populations.

---

### User Story 2 - Learn SequenceGenome for Genetic Programming (Priority: P2)

An ML practitioner wants to understand genetic programming by evolving symbolic expressions. They follow a SequenceGenome notebook that generates synthetic regression data, evolves mathematical expressions, and compares results to symbolic regression baselines.

**Why this priority**: SequenceGenome demonstrates variable-length representations and tree-based operations, introducing the concept that "model structure" itself can evolve—a powerful insight for practitioners used to fixed architectures.

**Independent Test**: Can be tested by evolving symbolic expressions that rediscover known formulas (e.g., polynomial relationships, sine waves) from synthetic data.

**Acceptance Scenarios**:

1. **Given** synthetic data from a known polynomial function, **When** sequence evolution completes, **Then** the best individual represents an expression within 5% error of the true function.
2. **Given** the notebook's operator explanations, **When** users review crossover visualizations, **Then** they understand how subtree exchange differs from vector crossover.
3. **Given** the advanced sections, **When** bloat control is enabled, **Then** evolved expressions remain interpretable (under 20 nodes on average).

---

### User Story 3 - Learn GraphGenome/NEAT for Neuroevolution with Speciation (Priority: P3)

An ML practitioner wants to understand how neural network topologies can evolve (not just weights). They follow a GraphGenome/NEAT notebook that demonstrates topology evolution with full speciation implementation, protecting innovation through species-based selection.

**Why this priority**: NEAT is a landmark algorithm in neuroevolution. This notebook demonstrates evolving network structure—a concept that resonates with practitioners familiar with neural architecture search (NAS). Speciation is critical for understanding how EAs protect novel structures.

**Independent Test**: Can be tested by evolving networks that solve XOR (requires hidden nodes) and visualizing species formation over generations.

**Acceptance Scenarios**:

1. **Given** an XOR classification task, **When** NEAT evolution completes, **Then** at least one individual achieves 100% accuracy with an evolved topology containing hidden nodes.
2. **Given** the speciation visualization, **When** species are plotted over generations, **Then** users can observe species formation, growth, and extinction events.
3. **Given** compatibility threshold configuration, **When** threshold is tightened, **Then** more species form and population becomes more diverse.
4. **Given** the adjusted fitness explanation, **When** a novel topology emerges, **Then** it receives fitness sharing protection and survives initial low performance.

---

### User Story 4 - Learn Policy Evolution for Reinforcement Learning (Priority: P4)

An ML practitioner familiar with policy gradient methods wants to understand how evolution can optimize RL policies. They follow an RL notebook that evolves policies for Gymnasium control tasks, demonstrating episode rollouts as fitness evaluation.

**Why this priority**: RL practitioners understand reward signals and episode returns. This notebook reframes evolution as a derivative-free policy optimizer, demonstrating when evolution excels over gradient-based RL (sparse rewards, deceptive gradients).

**Independent Test**: Can be tested by evolving policies that solve CartPole-v1 (episode return > 475) within reasonable generations.

**Acceptance Scenarios**:

1. **Given** CartPole-v1 environment, **When** policy evolution completes, **Then** the best policy achieves average episode return > 475 over 10 evaluation episodes.
2. **Given** the fitness aggregation explanation, **When** users configure episode averaging, **Then** they understand the bias-variance tradeoff in noisy fitness.
3. **Given** the rollout visualization, **When** best policy is rendered, **Then** users see the evolved behavior in action.
4. **Given** the comparison section, **When** results are compared to random search, **Then** evolution shows statistically significant improvement.

---

### User Story 5 - Learn SCMGenome for Causal Discovery with Multi-Objective Optimization (Priority: P5)

An ML practitioner interested in causal inference wants to discover causal structure from observational data. They follow an SCMGenome notebook that evolves Structural Causal Models using full multi-objective optimization (NSGA-II), balancing data fit, sparsity, and structural simplicity.

**Why this priority**: Causal discovery is an emerging application area where EA methods shine due to the combinatorial structure space. This notebook demonstrates multi-objective optimization with Pareto fronts—essential for real-world problems with competing objectives.

**Independent Test**: Can be tested by generating synthetic data from a known causal graph, hiding the ground truth, and measuring rediscovery success.

**Acceptance Scenarios**:

1. **Given** synthetic data from a 5-variable causal DAG, **When** multi-objective evolution completes, **Then** the Pareto front contains solutions recovering >80% of true edges.
2. **Given** the Pareto front visualization, **When** users explore trade-offs, **Then** they can interactively select solutions balancing fit vs. sparsity.
3. **Given** latent variable scenario, **When** one variable is hidden from the data, **Then** evolution discovers proxy structures that approximate the hidden influence.
4. **Given** the comparison section, **When** weighted-sum approach is compared to true multi-objective, **Then** users understand why Pareto methods find diverse solutions.

---

### User Story 6 - Reuse Shared Tutorial Utilities (Priority: P2)

A tutorial user wants consistent data generation, visualization helpers, and terminology definitions across all notebooks. A shared tutorial_utils.py module provides synthetic data generators, plotting functions, mermaid diagram helpers, and a terminology glossary.

**Why this priority**: Consistency across tutorials reduces cognitive load and establishes patterns. The shared module also demonstrates software engineering best practices for EA experiments.

**Independent Test**: Can be tested by importing the module and verifying all data generators produce correctly shaped output with configurable noise.

**Acceptance Scenarios**:

1. **Given** the shared module imported, **When** users generate synthetic benchmark data, **Then** data dimensions, noise levels, and random seeds are configurable.
2. **Given** the mermaid helper functions, **When** users visualize evolutionary pipelines, **Then** diagrams render correctly in Jupyter using beautiful-mermaid.
3. **Given** the terminology glossary, **When** users query ML-to-EA mappings, **Then** they receive clear analogies (genome=model weights, fitness=-loss).

---

### Edge Cases

- **Missing GPU**: Notebooks gracefully degrade to CPU-only benchmarks with informative warnings
- **Missing Gymnasium**: RL notebook provides installation instructions and skips environment-dependent cells
- **Memory constraints**: Island model section includes guidance on population size limits for available RAM
- **Long runtimes**: All notebooks include expected runtime estimates and early-stopping guidance
- **Visualization failures**: Mermaid diagrams have static fallback images if library unavailable

## Requirements *(mandatory)*

### Functional Requirements

#### Shared Module (tutorial_utils.py)

- **FR-001**: Module MUST provide synthetic data generators for continuous optimization (sphere, Rastrigin, Rosenbrock, Ackley)
- **FR-002**: Module MUST provide synthetic data generators for symbolic regression (polynomials, trigonometric, composite)
- **FR-003**: Module MUST provide synthetic causal data generators with configurable DAG structure and noise
- **FR-004**: All data generators MUST accept random seed for reproducibility
- **FR-005**: All data generators MUST accept noise level parameter (0.0 = no noise, 1.0 = high noise)
- **FR-006**: Module MUST provide mermaid diagram rendering via beautiful-mermaid library with github-light theme default
- **FR-007**: Module MUST provide fitness-over-generations plotting with confidence intervals
- **FR-008**: Module MUST provide population diversity visualization (PCA/t-SNE projections)
- **FR-009**: Module MUST provide a terminology glossary mapping EA terms to ML concepts
- **FR-010**: Module MUST provide statistical summary functions (mean, std, percentiles) for population metrics

#### Notebook Structure (All Notebooks)

- **FR-011**: Each notebook MUST be standalone and executable end-to-end without external data files
- **FR-012**: Each notebook MUST include an "Evolutionary Algorithms Primer" section with ML analogies
- **FR-013**: Each notebook MUST explicitly state all configuration defaults with rationale
- **FR-014**: Each notebook MUST include at least 3 mermaid diagrams (genome to phenotype pipeline, evolutionary loop, population data flow)
- **FR-015**: Each notebook MUST include convergence visualizations (fitness over generations, best/mean/worst)
- **FR-016**: Each notebook MUST include population diversity metrics over generations (genotypic diversity via pairwise genome distance AND phenotypic spread in fitness space)
- **FR-017**: Each notebook MUST demonstrate callback usage for logging and early stopping
- **FR-018**: Each notebook MUST demonstrate checkpointing (save and resume evolution)
- **FR-019**: Each notebook MUST include "Extensions & Next Steps" section linking to other tutorials

#### Island Models (All Notebooks)

- **FR-020**: Each notebook MUST include island model section with actual parallel execution using multiprocessing
- **FR-021**: Island model section MUST demonstrate migration between populations
- **FR-022**: Island model section MUST support configurable topology (ring, star, fully-connected)
- **FR-023**: Island model section MUST compare single-population vs. island performance with statistical tests

#### GPU Acceleration (All Notebooks)

- **FR-024**: Each notebook MUST include GPU acceleration section with backend configuration
- **FR-025**: GPU section MUST include benchmark comparing CPU vs GPU execution times
- **FR-026**: GPU section MUST provide guidance on when acceleration provides benefit (population size thresholds, batch evaluation)

#### VectorGenome Notebook

- **FR-027**: Notebook MUST demonstrate continuous function optimization (minimization)
- **FR-028**: Notebook MUST visualize genome-to-phenotype mapping (identity for vectors)
- **FR-029**: Notebook MUST demonstrate Gaussian mutation with adaptive step sizes
- **FR-030**: Notebook MUST demonstrate multiple crossover operators (uniform, SBX) with visual comparisons

#### SequenceGenome Notebook

- **FR-031**: Notebook MUST demonstrate symbolic expression evolution for regression
- **FR-032**: Notebook MUST visualize expression trees and subtree crossover
- **FR-033**: Notebook MUST demonstrate bloat control techniques
- **FR-034**: Notebook MUST evaluate evolved expressions on held-out test data

#### GraphGenome/NEAT Notebook

- **FR-035**: Notebook MUST implement full NEAT speciation with genomic distance calculation
- **FR-036**: Notebook MUST demonstrate species formation with compatibility threshold tuning
- **FR-037**: Notebook MUST implement adjusted fitness (fitness sharing within species)
- **FR-038**: Notebook MUST visualize species over generations using stacked area chart (primary) with optional phylogenetic tree view
- **FR-039**: Notebook MUST evolve network topologies that solve XOR problem
- **FR-040**: Notebook MUST visualize evolved network architectures (nodes, connections, weights)

#### RL/Neuroevolution Notebook

- **FR-041**: Notebook MUST integrate with Gymnasium environments
- **FR-042**: Notebook MUST demonstrate policy decoding from genomes (weights to neural network)
- **FR-043**: Notebook MUST demonstrate episode rollout as fitness evaluation
- **FR-044**: Notebook MUST aggregate fitness across multiple episodes with configurable averaging
- **FR-045**: Notebook MUST solve CartPole-v1 as primary task with optional LunarLander-v2 advanced section
- **FR-046**: Notebook MUST render best policy behavior (video or animation)

#### SCMGenome Notebook

- **FR-047**: Notebook MUST implement full multi-objective optimization using NSGA-II or Pareto-based selection
- **FR-048**: Notebook MUST explain Pareto dominance with geometric intuition using 2D pairwise projections and interactive 3D plotly scatter3d
- **FR-049**: Notebook MUST explain crowding distance with visual examples
- **FR-050**: Notebook MUST visualize Pareto front evolution across generations
- **FR-051**: Notebook MUST demonstrate trade-off analysis between data fit, sparsity, and structural simplicity
- **FR-052**: Notebook MUST compare weighted-sum aggregation vs. true multi-objective optimization
- **FR-053**: Notebook MUST implement latent variable discovery scenario (hide observed columns, measure rediscovery)
- **FR-054**: Notebook MUST visualize causal graphs with ground truth comparison (edge-level accuracy metrics)

### Key Entities

- **Tutorial Notebook**: Jupyter notebook containing markdown explanations, code cells, and visualizations for one representation type
- **Shared Utils Module**: Python module providing data generation, visualization, and terminology support across all notebooks
- **Synthetic Dataset**: Configurable data generated from known ground truth for validation
- **Evolutionary Run**: Complete execution from initial population to termination with logged metrics
- **Benchmark Comparison**: Timed execution comparing different configurations (CPU/GPU, single/island, etc.)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users with no EA background can complete any single notebook in under 90 minutes
- **SC-002**: Each notebook executes end-to-end without errors on Python 3.10+ with documented dependencies
- **SC-003**: VectorGenome notebook converges to within 1% of known optima on benchmark functions
- **SC-004**: NEAT notebook evolves solutions achieving 100% accuracy on XOR within 150 generations
- **SC-005**: RL notebook evolves CartPole policies with average return > 475 within 100 generations
- **SC-006**: SCM notebook recovers >80% of causal edges from synthetic 5-variable DAG
- **SC-007**: Island model sections demonstrate >20% speedup using 4 parallel processes on suitable problems (population ≥200, evaluation time ≥10ms per individual where parallelism overhead is amortized)
- **SC-008**: GPU acceleration sections show measurable speedup for populations >1000 individuals
- **SC-009**: All notebooks pass accessibility check (clear headings, alt text for diagrams, code comments)
- **SC-010**: Each notebook includes at least 10 ML-to-EA terminology mappings with explanations

## Assumptions

- Users have Python experience and familiarity with Jupyter notebooks
- Users understand basic calculus (gradients, optimization) and linear algebra (matrices, vectors)
- Users have exposure to ML concepts (loss functions, model training, hyperparameters)
- The Evolve framework's core functionality (EvolutionEngine, operators, backends) is stable and documented
- beautiful-mermaid library is available for diagram rendering (fallback static images provided)
- Gymnasium library is available for RL examples (installation instructions provided)
- Optional GPU acceleration requires PyTorch or JAX with CUDA support

## Scope Boundaries

### In Scope

- 5 tutorial notebooks covering VectorGenome, SequenceGenome, GraphGenome/NEAT, RL/Neuroevolution, SCMGenome
- Shared tutorial_utils.py module with data generation, visualization, and terminology
- Full implementation of speciation in NEAT notebook
- Full implementation of NSGA-II multi-objective optimization in SCM notebook
- Actual parallel execution in island model sections
- Real CPU vs GPU benchmark comparisons

### Out of Scope

- Video tutorials or screencasts
- Integration with external datasets (all data is synthetic)
- Deployment or production guidance
- Advanced algorithm variants beyond core Evolve capabilities (e.g., CMA-ES, MOEA/D)
- Interactive web-based tutorial platform
