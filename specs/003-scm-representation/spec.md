# Feature Specification: SCM Representation for Causal Discovery

**Feature Branch**: `003-scm-representation`  
**Created**: 2026-01-30  
**Status**: Draft  
**Input**: User description: "Add a new evolutionary representation to the framework for causal discovery via evolution of Structural Causal Models (SCMs)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define and Evolve SCM Genomes (Priority: P1)

A researcher wants to discover causal relationships in observational data by evolving Structural Causal Models. They initialize an SCM population with their dataset's observed variables, configure a latent variable pool, and evolve genomes that encode potential causal equations.

**Why this priority**: Core functionality - without genome creation and evolution, no causal discovery can occur. This is the foundation all other features build upon.

**Independent Test**: Can be fully tested by creating an SCMGenome with a small alphabet (3 variables, basic operators), applying mutation, and verifying the genome maintains valid structure.

**Acceptance Scenarios**:

1. **Given** a dataset with observed variables [A, B, C], **When** I create an SCMConfig with max_latent_variables=2, **Then** the alphabet contains variable references (A, B, C, H1, H2), STORE genes (STORE_A, STORE_B, STORE_C, STORE_H1, STORE_H2), operators (+, -, *, /), and numeric constants.
2. **Given** an SCMGenome instance, **When** I call copy(), **Then** I receive an independent deep copy that can be mutated without affecting the original.
3. **Given** an SCMGenome, **When** I apply point mutation via the inner SequenceGenome, **Then** the genome structure remains valid and genes are replaced with alphabet-compatible symbols.

---

### User Story 2 - Decode Genomes into Causal Graphs (Priority: P1)

A researcher needs to interpret evolved genomes as actual causal models. They decode an SCMGenome into a DecodedSCM containing structural equations and a directed graph representing causal relationships.

**Why this priority**: Decoding is essential for fitness evaluation and interpretation. Without it, genomes are opaque gene sequences with no causal meaning.

**Independent Test**: Can be fully tested by creating a genome with known equation-producing genes, decoding it, and verifying the resulting equations and graph structure match expectations.

**Acceptance Scenarios**:

1. **Given** a genome encoding "A B + STORE_C" (postfix notation), **When** decoded, **Then** DecodedSCM contains equation "C = A + B" and graph has edges A→C and B→C.
2. **Given** a genome with multiple STORE_X genes for the same variable X, **When** decoded with conflict_resolution="first_wins", **Then** only the first equation is kept and later ones are recorded as junk.
3. **Given** a decoded SCM, **When** I access .graph, **Then** I receive a NetworkX DiGraph compatible with causal inference libraries.

---

### User Story 3 - Evaluate SCM Fitness Multi-Objectively (Priority: P1)

A researcher wants to guide evolution toward SCMs that fit data well, are sparse, and have simple equations. They configure the SCMEvaluator with their objectives and evaluate decoded SCMs against observed data.

**Why this priority**: Fitness evaluation drives selection. Without multi-objective evaluation, evolution cannot discover good causal models.

**Independent Test**: Can be fully tested by creating a known-structure SCM, generating synthetic data from it, and verifying that the evaluator assigns high fitness to the true model.

**Acceptance Scenarios**:

1. **Given** a DecodedSCM and observed data, **When** evaluated with default objectives (data_fit, sparsity, simplicity), **Then** I receive a tuple of three fitness values.
2. **Given** a cyclic SCM and acyclicity_mode="reject", **When** evaluated, **Then** fitness returns None and the individual is marked invalid.
3. **Given** an SCMEvaluator configured with constraint "conflict_free", **When** evaluating an SCM with unresolved conflicts, **Then** a penalty is applied per the configured conflict_penalty.

---

### User Story 4 - Configure Conflict Resolution Strategies (Priority: P2)

A researcher wants control over how conflicting equations (multiple definitions for the same variable) are handled during decoding.

**Why this priority**: Different experiments may require different semantics for conflicts - some may want deterministic first-wins, others may prefer detecting conflicts as a constraint violation.

**Independent Test**: Can be tested by creating genomes with known conflicts and verifying each resolution strategy produces expected decoded results.

**Acceptance Scenarios**:

1. **Given** conflict_resolution="last_wins", **When** a genome has two STORE_X genes, **Then** the second equation defines X and the first is junk.
2. **Given** conflict_resolution="all_junk", **When** conflicts exist, **Then** all conflicting equations are discarded and DecodedSCM.conflicts contains the conflict metadata.
3. **Given** any conflict resolution strategy, **When** decoding completes, **Then** DecodedSCM.metadata includes conflict_count and junk_gene_indices.

---

### User Story 5 - Handle Cyclic SCMs Gracefully (Priority: P2)

A researcher exploring causal discovery may encounter cyclic structures. They need configurable behavior for how cycles are detected and handled during evaluation.

**Why this priority**: Cyclic structures are common in exploratory evolution. The system must handle them gracefully rather than crashing or producing undefined behavior.

**Independent Test**: Can be tested by constructing cyclic genomes and verifying each handling mode produces expected fitness results and metadata.

**Acceptance Scenarios**:

1. **Given** acyclicity_mode="penalize" with strategy="acyclic_subgraph", **When** evaluating a cyclic SCM, **Then** fitness is computed on the maximal acyclic subgraph.
2. **Given** acyclicity_mode="penalize" with strategy="penalty_only", **When** evaluating a cyclic SCM, **Then** cycle_count * cycle_penalty_per_cycle is added to the penalty.
3. **Given** any acyclicity handling mode, **When** a cyclic SCM is decoded, **Then** DecodedSCM.is_cyclic returns True and DecodedSCM.cycles contains the detected cycles.

---

### User Story 6 - Serialize and Restore SCM Genomes (Priority: P2)

A researcher running long experiments needs to checkpoint and restore populations. SCMGenomes must support serialization for experiment persistence.

**Why this priority**: Checkpointing is essential for long-running experiments, but not blocking for initial functionality.

**Independent Test**: Can be tested by serializing a genome to dict, deserializing it back, and verifying equality.

**Acceptance Scenarios**:

1. **Given** an SCMGenome with ERC constants, **When** I call to_dict(), **Then** I receive a dictionary containing all genome state including ERC values.
2. **Given** a serialized genome dictionary, **When** I call SCMGenome.from_dict(data), **Then** I receive an equivalent genome that decodes to the same SCM.
3. **Given** a population of SCMGenomes, **When** checkpointed and restored via existing checkpoint infrastructure, **Then** all genomes are correctly restored.

---

### User Story 7 - Integrate with ERP for Matchability (Priority: P3)

A researcher using Evolvable Reproduction Protocol wants matchability to consider both sequence similarity and decoded causal structure when pairing parents.

**Why this priority**: Advanced feature building on existing ERP infrastructure. Valuable for sophisticated experiments but not required for basic SCM evolution.

**Independent Test**: Can be tested by computing matchability between genome pairs and verifying both sequence and structural similarity contribute to the score.

**Acceptance Scenarios**:

1. **Given** two SCMGenomes with similar sequences but different decoded graphs, **When** matchability is computed with structural_weight > 0, **Then** the structural difference reduces matchability.
2. **Given** SCMGenomes in an ERP-enabled evolution, **When** reproduction occurs, **Then** matchability computation completes without error and produces meaningful pairings.

---

### Edge Cases

- What happens when a genome produces no valid equations (all junk)? → DecodedSCM has empty equation set, graph has no edges, fitness reflects zero coverage.
- How does the system handle latent variables with no observed ancestors? → Decoder flags constraint violation, evaluator applies incomplete_coverage_penalty or rejects based on config.
- What happens when stack underflow occurs during decoding? → Underflow-causing gene is marked as junk, decoding continues with remaining genes (silent junk strategy).
- What happens when STORE_X is encountered with empty stack? → STORE_X is marked as junk, no equation produced, decoding continues (consistent with underflow handling).
- How are division-by-zero and undefined operations handled? → NaN propagation with optional penalty: NaN flows through MSE computation, configurable div_zero_penalty applied if any NaN in predictions.

## Requirements *(mandatory)*

### Functional Requirements

#### Genome Structure

- **FR-001**: System MUST provide `SCMGenome` class that wraps `SequenceGenome` and implements the `Genome` protocol (copy, __eq__, __hash__).
- **FR-002**: System MUST provide `SCMGenome` implementation of `SerializableGenome` protocol (to_dict, from_dict).
- **FR-003**: Genome alphabet MUST include: observed variable references, latent variable references (H1...Hn), STORE_* genes for all variables, arithmetic operators (+, -, *, /), discrete numeric constants (0, 1, 2, -1, 0.5, PI), and Ephemeral Random Constants (ERC).
- **FR-004**: ERCs MUST sample initial values from Gaussian distribution N(0, σ_init) at genome creation and support perturbation mutation that adds Gaussian noise N(0, σ_perturb), with configurable σ_init and σ_perturb.
- **FR-005**: SCMGenome MUST be compatible with existing mutation operators (point mutation, insertion, deletion) and crossover operators (one-point, two-point, uniform) applied to the inner SequenceGenome.

#### Configuration

- **FR-006**: System MUST provide `SCMConfig` dataclass with configurable parameters: observed_variables, max_latent_variables, conflict_resolution, acyclicity_mode, acyclicity_strategy, objectives, constraints, and penalty values. Default values MUST be:
  - max_latent_variables: 3
  - conflict_resolution: "first_wins"
  - acyclicity_mode: "reject"
  - acyclicity_strategy: "acyclic_subgraph" (used when mode="penalize")
  - objectives: ["data_fit", "sparsity", "simplicity"]
  - constraints: ["acyclicity"]
  - cycle_penalty_per_cycle: 1.0
  - incomplete_coverage_penalty: 10.0
  - conflict_penalty: 1.0
  - div_zero_penalty: 5.0
  - erc_sigma_init: 1.0
  - erc_sigma_perturb: 0.1
- **FR-007**: conflict_resolution MUST support modes: "first_wins", "last_wins", "all_junk".
- **FR-008**: acyclicity_mode MUST support modes: "reject", "penalize".
- **FR-009**: acyclicity_strategy (when mode="penalize") MUST support: "acyclic_subgraph", "parse_order", "penalty_only", "parent_inheritance", "composite".

#### Decoding

- **FR-010**: System MUST provide `SCMDecoder` implementing `Decoder[SCMGenome, DecodedSCM]` protocol.
- **FR-011**: Decoder MUST execute stack-based postfix evaluation: operands push to stack, operators pop operands and push results, STORE_X pops and creates equation "X = <expression>".
- **FR-012**: Decoder MUST build directed graph from equation dependencies (RHS variables point to LHS variable).
- **FR-013**: Decoding MUST be deterministic given the same genome.
- **FR-014**: Decoder MUST track and expose junk genes (genes that don't contribute to final equations).
- **FR-015**: Decoder MUST detect and report cycles in the decoded graph.

#### Decoded SCM

- **FR-016**: `DecodedSCM` MUST contain: equations (dict mapping variable to expression), graph (nx.DiGraph), metadata (conflict_count, junk_gene_indices, is_cyclic, cycles).
- **FR-017**: DecodedSCM.graph MUST be compatible with NetworkX and causal inference libraries (DoWhy).
- **FR-018**: DecodedSCM MUST provide is_cyclic property and cycles accessor for cycle information.

#### Evaluation

- **FR-019**: System MUST provide `SCMEvaluator` for multi-objective fitness evaluation.
- **FR-020**: Default objectives MUST include: data_fit (negative MSE/likelihood), sparsity (negative edge count), simplicity (negative total AST complexity).
- **FR-021**: Evaluator MUST support configurable additional objectives: coverage, latent_parsimony.
- **FR-022**: Evaluator MUST support configurable constraints: acyclicity, coverage, conflict_free.
- **FR-023**: Evaluator MUST apply configurable penalties: cycle_penalty_per_cycle, incomplete_coverage_penalty, conflict_penalty, div_zero_penalty.
- **FR-024**: When acyclicity_mode="reject", evaluator MUST return None for cyclic SCMs.
- **FR-025**: When acyclicity_mode="penalize", evaluator MUST apply the configured acyclicity_strategy.

#### Variable Handling

- **FR-026**: Observed variables MUST be fixed at experiment initialization (derived from dataset columns).
- **FR-027**: Latent variable pool size MUST be configurable via SCMConfig.max_latent_variables.
- **FR-028**: System MUST enforce constraint that latent variables have at least one observed ancestor (for marginalization).
- **FR-029**: Latent variables without observed ancestors MUST trigger constraint violation handling per configuration.

### Key Entities

- **SCMGenome**: Evolutionary unit encoding potential causal model. Wraps SequenceGenome with SCM-specific alphabet and semantics. Key attributes: genes (via inner sequence), alphabet, erc_values.
- **SCMConfig**: Configuration for SCM evolution. Key attributes: observed_variables, max_latent_variables, conflict_resolution, acyclicity_mode, acyclicity_strategy, objectives, constraints, penalties.
- **SCMDecoder**: Transforms genomes into phenotypes. Stateless transformer implementing Decoder protocol.
- **DecodedSCM**: Phenotype containing causal model. Key attributes: equations, graph, metadata, variables (observed + latent used).
- **SCMEvaluator**: Multi-objective fitness evaluator. Key attributes: objectives, constraints, penalty_config, data_reference.
- **SCMAlphabet**: Symbol set for genomes. Contains variable refs, store genes, operators, constants, ERC slots.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can create and evolve SCM populations of 1000+ individuals without memory issues or performance degradation.
- **SC-002**: Genome decoding completes in under 10ms for genomes up to 500 genes on standard hardware.
- **SC-003**: Given a synthetic dataset generated from a known 5-variable causal model, evolution discovers a structurally equivalent model within 100 generations in 80% of runs.
- **SC-004**: All existing tests pass after integration (zero regression).
- **SC-005**: Decoded SCMs successfully import into NetworkX and pass basic graph validity checks.
- **SC-006**: Serialization round-trip preserves genome equality: `genome == SCMGenome.from_dict(genome.to_dict())` for all test genomes.
- **SC-007**: Documentation includes working example that discovers causal structure from synthetic data.

## Clarifications

### Session 2026-01-30

- Q: What distribution for ERC initial sampling and perturbation mutation? → A: Gaussian for both - initial sampling from N(0, σ_init), perturbation adds N(0, σ_perturb)
- Q: How to handle stack underflow during decoding? → A: Silent junk - mark underflow-causing gene as junk, continue decoding remaining genes
- Q: How to handle STORE_X with empty stack (no expression to assign)? → A: Junk gene - mark STORE_X as junk, no equation produced, continue decoding
- Q: How to handle division by zero during evaluation? → A: NaN propagation with optional penalty - let NaN flow through MSE, apply configurable div_zero_penalty if any NaN in predictions
- Q: How should crossover recombine SCMGenome parents? → A: Direct sequence crossover - apply standard crossover operators on inner SequenceGenome (not SCM-aware)

## Assumptions

- NetworkX is an acceptable dependency for graph representation (already commonly used in causal inference).
- Stack-based postfix encoding provides sufficient expressiveness for the target causal discovery problems.
- Initial implementation focuses on linear/polynomial equations; transcendental functions are a future extension.
- Users have familiarity with basic causal inference concepts (DAGs, SCMs, interventions).
- Evaluation of data fit requires access to observed data during fitness computation (evaluator is stateful w.r.t. data).
- ERC initial sampling uses Gaussian distribution N(0, σ_init) with configurable σ_init (default: 1.0).
- ERC perturbation mutation adds Gaussian noise N(0, σ_perturb) with configurable σ_perturb (default: 0.1).

## Non-Goals (Explicit Exclusions)

- Cyclic/equilibrium SCM evaluation (equilibrium solving is complex; deferred to future extension)
- Latent confounders without observed ancestors (requires marginalization over unobserved variables)
- Symbolic algebra in the inner loop (Groebner bases, cylindrical algebraic decomposition)
- Transcendental functions in operator set (sin, cos, exp, log - future extension)
- GPU-accelerated decoding (CPU decoding is sufficient for initial implementation)
- Interactive visualization of causal graphs (users can use existing NetworkX/graphviz tools)
