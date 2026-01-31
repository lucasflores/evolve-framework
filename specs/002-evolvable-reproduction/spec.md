# Feature Specification: Evolvable Reproduction Protocols (ERP)

**Feature Branch**: `002-evolvable-reproduction`  
**Created**: January 28, 2026  
**Status**: Draft  
**Input**: User description: "Enable genomes/individuals to encode, evolve, and execute their own reproductive compatibility logic and offspring construction strategies"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Evolvable Mating Compatibility (Priority: P1)

As a researcher, I want individuals in my population to encode their own mate selection criteria so that reproductive compatibility emerges through evolution rather than being globally fixed.

**Why this priority**: This is the foundational capability that distinguishes ERP from traditional GAs. Without evolvable matchability, reproduction remains externally controlled, negating the core value proposition.

**Independent Test**: Can be fully tested by creating a population where individuals have different matchability functions, running selection, and verifying that mating pairs are determined by individual-level compatibility rather than global rules.

**Acceptance Scenarios**:

1. **Given** a population with individuals encoding different matchability criteria, **When** a reproduction event is attempted between Individual A and Individual B, **Then** both A's matchability function and B's matchability function are evaluated independently
2. **Given** Individual A accepts B but B rejects A (asymmetric compatibility), **When** reproduction is attempted, **Then** no offspring is produced
3. **Given** both individuals accept each other, **When** reproduction proceeds, **Then** offspring are generated using the appropriate crossover protocol
4. **Given** an individual's matchability function exceeds complexity limits, **When** evaluation is attempted, **Then** the system safely defaults to rejection without crashing

---

### User Story 2 - Evolve Crossover Strategies (Priority: P2)

As a researcher, I want offspring construction methods (crossover protocols) to be encoded within individuals and subject to evolution, so that effective reproduction strategies can emerge naturally.

**Why this priority**: Crossover is the mechanism by which genetic material combines. Evolving crossover strategies enables discovery of domain-appropriate mixing methods rather than relying on researcher intuition.

**Independent Test**: Can be tested by creating individuals with different crossover protocols (e.g., single-point vs. uniform), running evolution for multiple generations, and verifying that crossover protocols are inherited, mutated, and that certain protocols become more prevalent if they produce fitter offspring.

**Acceptance Scenarios**:

1. **Given** two compatible parents with different crossover protocols, **When** reproduction occurs, **Then** offspring receive a crossover protocol (inherited from one parent, combined, or mutated)
2. **Given** a crossover protocol specifies single-point crossover with a mutable crossover point, **When** the protocol is mutated, **Then** the crossover point parameter changes
3. **Given** a crossover protocol references an invalid strategy, **When** reproduction is attempted, **Then** the system falls back to cloning (no-op) without crashing
4. **Given** multiple generations of evolution, **When** examining protocol distribution, **Then** crossover protocols that produce fitter offspring increase in prevalence

---

### User Story 3 - Support Reproduction Intent Policies (Priority: P2)

As a researcher, I want individuals to encode policies governing when they attempt reproduction (beyond just compatibility), so that reproduction timing and resource allocation can evolve.

**Why this priority**: Separating "willingness to reproduce" from "compatibility with partner" enables richer evolutionary dynamics, including resource-budgeted reproduction and fitness-gated strategies.

**Independent Test**: Can be tested by creating individuals with different intent policies (always willing vs. fitness-threshold gated), running evolution, and verifying that intent is evaluated before matchability and affects reproductive frequency.

**Acceptance Scenarios**:

1. **Given** an individual with a fitness-threshold intent policy, **When** its fitness is below threshold, **Then** it does not attempt reproduction regardless of available compatible partners
2. **Given** an individual with an "always willing" intent policy, **When** compatible partners exist, **Then** it attempts reproduction on every opportunity
3. **Given** an individual with a resource-budgeted intent policy (e.g., max 3 offspring per generation), **When** the budget is exhausted, **Then** further reproduction attempts are declined
4. **Given** intent policies are heritable, **When** a parent reproduces, **Then** offspring inherit (possibly mutated) intent policies

---

### User Story 4 - Maintain System Stability Under Adversarial Protocols (Priority: P1)

As a researcher, I want the evolutionary system to remain stable even when individuals evolve degenerate, adversarial, or computationally expensive protocols, so that experiments complete reliably.

**Why this priority**: Without robust safety guarantees, evolvable protocols could crash experiments, corrupt state, or consume unbounded resources. Stability is essential for practical use.

**Independent Test**: Can be tested by deliberately injecting malformed protocols (infinite loops, invalid references, excessive complexity) and verifying the system continues operating with appropriate fallbacks.

**Acceptance Scenarios**:

1. **Given** an individual with a matchability function that would loop infinitely, **When** evaluation begins, **Then** the system terminates evaluation within the step limit and returns rejection
2. **Given** an individual's protocol attempts to access global mutable state, **When** execution is attempted, **Then** the access is blocked and the protocol fails safely
3. **Given** an entire population develops degenerate protocols (all reject everyone), **When** reproduction phase runs, **Then** the system handles zero successful matings gracefully (e.g., via elitism or immigration)
4. **Given** a crossover protocol produces invalid offspring genomes, **When** offspring construction completes, **Then** invalid offspring are discarded or repaired before entering the population

---

### User Story 5 - Support Neutral Drift via Junk Code (Priority: P3)

As a researcher, I want reproduction protocol genomes to support inactive logic and dormant strategies, so that neutral drift can occur and latent capabilities can emerge through mutation.

**Why this priority**: Junk code enables evolutionary exploration by providing raw material for mutation to activate. This is important for long-term evolvability but not critical for initial function.

**Independent Test**: Can be tested by creating protocols with inactive regions, running evolution, and verifying that mutations can activate previously dormant logic.

**Acceptance Scenarios**:

1. **Given** a protocol genome with an inactive matchability rule, **When** the protocol executes, **Then** the inactive rule does not affect behavior
2. **Given** a mutation activates a previously dormant crossover strategy, **When** the individual reproduces, **Then** the newly activated strategy is used
3. **Given** protocol genomes of varying sizes (due to junk code), **When** fitness is evaluated, **Then** genome size alone does not affect fitness (no parsimony pressure on protocols by default)

---

### User Story 6 - Integrate with Multi-Objective Evolution (Priority: P2)

As a researcher using multi-objective optimization, I want ERP to work seamlessly with Pareto ranking, elitism, and crowding distance, so that evolvable reproduction enhances rather than conflicts with established MOEA mechanisms.

**Why this priority**: Multi-objective evolution is a core use case. ERP must integrate cleanly to be useful in realistic research scenarios.

**Independent Test**: Can be tested by running NSGA-II style evolution with ERP enabled, verifying that selection still operates on Pareto fronts while ERP governs mate pairing.

**Acceptance Scenarios**:

1. **Given** a multi-objective population with Pareto ranking, **When** reproduction occurs, **Then** ERP matchability determines who CAN mate, while selection determines who DOES mate based on rank
2. **Given** an individual's matchability function biases toward diverse partners (different crowding region), **When** reproduction occurs, **Then** offspring tend to maintain population diversity
3. **Given** elitism rules preserve top individuals, **When** ERP protocols would eliminate all elites from mating pool, **Then** elites still survive to next generation (selection > ERP for survival)

---

### Edge Cases

- What happens when no individuals in the population are mutually compatible? The system injects immigrants (5-10% of population) with default accept-all protocols to restore connectivity.
- How does the system handle circular protocol references (protocol A references B references A)? Safe termination with default behavior.
- What if crossover protocols from two parents are incompatible for combination? Fall back to inheriting one parent's protocol randomly.
- What happens when all intent policies say "never reproduce"? Elitism preserves population; optional immigration can inject new individuals.
- How are protocols initialized for the first generation? Configurable default protocols or random initialization within valid bounds.

## Requirements *(mandatory)*

### Functional Requirements

**Matchability System**

- **FR-001**: System MUST allow each individual to encode a Matchability Function that evaluates potential mates
- **FR-002**: Matchability evaluation MUST be asymmetric by default (A to B independent of B to A)
- **FR-003**: Matchability functions MUST support both boolean (accept/reject) and probabilistic (0-1) outputs
- **FR-004**: Matchability functions MUST have access to partner genome statistics (distance, similarity, fitness rank, niche label)
- **FR-005**: Matchability evaluation MUST complete within configurable step/complexity limits (default: 1,000 steps). Configuration via `ERPConfig.step_limit: int` parameter
- **FR-006**: Matchability functions that exceed limits or fail MUST safely default to rejection

**Reproduction Intent**

- **FR-007**: System MUST support optional Reproduction Intent Policies separate from matchability
- **FR-008**: Intent policies MUST be able to reference individual internal state (fitness, age, offspring count)
- **FR-009**: Intent policies MUST be heritable and mutable
- **FR-010**: Intent policy evaluation MUST occur before matchability evaluation

**Crossover Protocols**

- **FR-011**: System MUST allow each individual to encode or reference a Crossover Protocol
- **FR-012**: Crossover protocols MUST support parameterized operations (crossover point, mixing rate, etc.)
- **FR-013**: Offspring MUST receive a crossover protocol via inheritance, combination, or mutation (default: random single-parent inheritance with 50/50 probability)
- **FR-014**: System MUST support multiple crossover protocol classes: single-point, multi-point, uniform, grammar-aware, module exchange, and no-op (cloning)
- **FR-015**: Invalid or failing crossover protocols MUST fall back to cloning

**Safety and Sandboxing**

- **FR-016**: Protocol execution MUST be sandboxed with no access to global mutable state
- **FR-017**: All protocol execution MUST have explicit step/complexity limits (default: 1,000 steps)
- **FR-018**: Protocol failures MUST default to "no reproduction" without crashing the evolutionary loop
- **FR-019**: Offspring produced by protocols MUST be validated before entering the population
- **FR-019a**: When zero successful matings occur in a generation, system MUST inject immigrants (5-10% of population size) with default accept-all protocols

**Junk Code and Neutral Drift**

- **FR-020**: Protocol genomes MUST support inactive/unreferenced logic regions
- **FR-021**: Mutations MUST be able to activate dormant logic
- **FR-022**: Protocol genome size MUST NOT directly affect fitness evaluation

**Multi-Objective Integration**

- **FR-023**: System-level selection (Pareto ranking, elitism) MUST remain authoritative over survival decisions
- **FR-024**: ERP MUST only influence mating pair formation, not survival selection
- **FR-025**: Matchability functions MUST be able to access crowding distance and diversity metrics

**Extensibility**

- **FR-026**: Design MUST allow future addition of RL-trained policies without core refactoring
- **FR-027**: Design MUST allow future addition of memetic local search post-crossover
- **FR-028**: Design MUST allow protocol-aware migration in island models

### Key Entities

- **Individual**: An evolutionary unit consisting of a primary genome (problem-specific) and a Reproduction Protocol Genome (RPG)
- **Reproduction Protocol Genome (RPG)**: Structured, evolvable encoding containing matchability function, intent policy, and crossover protocol
- **Matchability Function**: Evolvable logic determining whether another individual is an acceptable mate
- **Reproduction Intent Policy**: Evolvable logic determining when an individual attempts reproduction
- **Crossover Protocol**: Evolvable specification for how offspring genomes are constructed from parents
- **Reproduction Event**: An occurrence where two mutually compatible individuals with willing intent produce offspring

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Reproduction logic is no longer globally fixed - different individuals demonstrably employ different mating strategies within the same population
- **SC-002**: Over 100 generations, observable variation exists in matchability acceptance rates, intent policy behaviors, and crossover protocol usage across the population
- **SC-003**: Reproductive strategies show heritability - offspring mating behaviors correlate with parent mating behaviors at r > 0.5. Measured via: (1) track parent-offspring pairs over 100 generations, (2) compute Pearson correlation between parent acceptance rate and offspring acceptance rate, (3) verify r > 0.5 with p < 0.01
- **SC-004**: System remains stable for 10,000+ generations under adversarial protocol injection (no crashes, no hangs, no state corruption)
- **SC-005**: Protocol evaluation overhead adds less than 20% to total reproduction phase time compared to fixed-protocol baseline
- **SC-006**: New reproduction mechanisms (e.g., new crossover type) can be added by implementing a single interface without modifying core engine code
- **SC-007**: Multi-objective test suite (NSGA-II benchmarks) passes with ERP enabled, achieving equivalent or better hypervolume compared to fixed reproduction

## Clarifications

### Session 2026-01-28

- Q: What should the default step limit be for protocol execution? → A: Moderate limit (1,000 steps) - balanced safety and expressiveness
- Q: When two parents have different crossover protocols, how should the offspring's protocol be determined? → A: Random single-parent inheritance (50/50 chance from either parent)
- Q: When reproduction fails completely (zero successful matings), what recovery mechanism should be used? → A: Small immigration (5-10% of population) with default accept-all protocols

## Assumptions

- The existing evolve framework provides sufficient hooks for customizing reproduction behavior at the individual level
- Protocol genomes can be represented using the existing genome/representation infrastructure
- Computational overhead of protocol evaluation is acceptable given the research value of evolvable reproduction
- Users are comfortable with non-deterministic mating outcomes that depend on evolved protocols
- Initial protocol genome initialization will use sensible defaults (e.g., accept-all matchability, always-willing intent, single-point crossover)

## Dependencies

- Core framework architecture (spec 001) must provide extensible reproduction hooks
- Representation system must support composite genomes (primary + protocol)
- Multi-objective module must expose crowding/diversity metrics to protocol evaluation context

## Out of Scope

- Guaranteeing population connectivity (preventing reproductive isolation)
- Preventing reproductive dead-ends or speciation collapse
- Solving global optimization via reproduction alone
- GUI visualization of protocol evolution
- Automatic protocol optimization or meta-learning
