# Feature Specification: Symbiogenetic Merge Operator

**Feature Branch**: `013-symbiogenetic-merge`  
**Created**: 2026-04-17  
**Status**: Draft  
**Input**: User description: "Add a symbiogenetic merge operator — a new evolutionary mechanism that permanently absorbs one individual (the symbiont) into another (the host), producing a single offspring that is compositionally more complex than either parent."

## Clarifications

### Session 2026-04-17

- Q: What happens to the symbiont individual in the population after a merge? → A: Configurable — user chooses whether symbiont is consumed (removed) or survives.
- Q: What does merge rate apply to — per generation toggle, per individual, or fixed count? → A: Per-individual probability, consistent with crossover_rate/mutation_rate semantics.
- Q: How are hosts selected for merge — random, fitness-biased, or via the selection operator? → A: Random (uniform) from the population.
- Q: How is "genome complexity" defined for non-graph genome types? → A: Gene count (length) — nodes+connections for GraphGenome, sequence length for SequenceGenome, array length for VectorGenome, token count for EmbeddingGenome.
- Q: How are interface connections split between host→symbiont and symbiont→host directions? → A: Equal split — half in each direction, remainder randomly assigned.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Merge Two Graph Genomes via Symbiogenesis (Priority: P1)

A researcher running a NEAT-based experiment wants to trigger a symbiogenetic merge between two individuals from different species. The merge absorbs the symbiont's subgraph into the host's graph, producing a single offspring with preserved internal topology from both parents and new interface connections bridging the two subgraphs.

**Why this priority**: The GraphGenome implementation is the most complex and highest-value case. NEAT graph genomes have innovation tracking, speciation, and topological structure that demand careful handling — getting this right validates the entire merge concept.

**Independent Test**: Can be fully tested by creating two GraphGenome individuals with known topologies, merging them, and verifying the resulting genome contains all nodes and connections from both parents plus the expected interface connections. Delivers a working complexity-increasing operator for NEAT experiments.

**Acceptance Scenarios**:

1. **Given** a host GraphGenome with 5 nodes and 7 connections and a symbiont GraphGenome with 3 nodes and 4 connections, **When** a symbiogenetic merge is performed, **Then** the resulting genome contains 8 nodes and at least 11 connections (original connections plus interface connections).
2. **Given** a host and symbiont with overlapping node IDs, **When** the merge is performed, **Then** the symbiont's node IDs are remapped to a fresh range with no collisions.
3. **Given** a host and symbiont with overlapping innovation numbers, **When** the merge is performed, **Then** the symbiont's innovation numbers are remapped to a fresh range above the current global counter.
4. **Given** a configurable number of interface connections (e.g., 4), **When** the merge is performed, **Then** exactly that many new connections are created bridging host and symbiont subgraphs with fresh innovation numbers and random weights.
5. **Given** a merged offspring, **When** its distance to un-merged individuals is computed, **Then** the distance is large enough that NEAT speciation places it in a new species.

---

### User Story 2 - Configure and Trigger Merge Events in the Engine (Priority: P1)

A researcher configures their experiment with a symbiogenetic merge rate and symbiont source strategy. During evolution, the engine automatically triggers merge events at the configured rate after the standard crossover and mutation phases, selecting host-symbiont pairs according to the chosen sourcing strategy.

**Why this priority**: Without engine integration, the operator exists in isolation and cannot be used in experiments. This is co-priority with the operator itself.

**Independent Test**: Can be tested by running a short evolution with merge enabled at a high rate and verifying that merge events occur, offspring appear in the population, and generation metrics include merge-related statistics.

**Acceptance Scenarios**:

1. **Given** a UnifiedConfig with merge rate set to 0.1, **When** the engine runs a generation, **Then** each individual independently has a 10% chance of being selected as a host for a merge event (per-individual probability, consistent with crossover_rate/mutation_rate semantics).
2. **Given** a symbiont source strategy set to "cross_species", **When** a merge event is triggered, **Then** the symbiont is selected from a different species than the host.
3. **Given** a symbiont source strategy set to "archive", **When** a merge event is triggered, **Then** the symbiont is drawn from a hall-of-fame archive of past high-fitness individuals.
4. **Given** merge events occurring during a run, **When** generation metrics are logged, **Then** merge-specific metrics (merge count, average genome complexity, complexity change) are reported.

---

### User Story 3 - Define and Register a Custom Merge Strategy (Priority: P2)

A researcher working with a custom genome type wants to define their own merge strategy by implementing the SymbiogeneticMerge protocol and registering it in the operator registry. The engine automatically discovers and uses the registered strategy when the corresponding genome type is configured.

**Why this priority**: Extensibility is important but secondary to the core operator and engine integration working end-to-end.

**Independent Test**: Can be tested by implementing a trivial custom merge strategy for VectorGenome (concatenation), registering it, and running a short experiment that triggers merge events with VectorGenome individuals.

**Acceptance Scenarios**:

1. **Given** a user-defined class that implements the SymbiogeneticMerge protocol, **When** it is registered in the operator registry under the "merge" category, **Then** the engine can look it up by name and use it for merge events.
2. **Given** a registered merge strategy for a specific genome type, **When** the engine is configured with that genome type, **Then** the engine selects the compatible merge strategy automatically.
3. **Given** no merge strategy is registered for a genome type but merge is enabled, **When** the engine initializes, **Then** a clear error message indicates that no compatible merge strategy is available.

---

### User Story 4 - Merge with Non-Graph Genome Types (Priority: P3)

A researcher running soft-prompt evolution (EmbeddingGenome) or sequence-based evolution (SequenceGenome) wants to use symbiogenetic merge to combine two individuals — for example, vertically stacking two embedding matrices or concatenating two sequences.

**Why this priority**: Extends value to additional genome types, but the core GraphGenome implementation and engine integration must be stable first. Implementation strategies for these types are simpler (concatenation/stacking).

**Independent Test**: Can be tested by merging two SequenceGenome individuals and verifying the result is the concatenation of both sequences, or by merging two EmbeddingGenome individuals and verifying the result is the vertical stack of their matrices.

**Acceptance Scenarios**:

1. **Given** two SequenceGenome individuals, **When** a merge is performed, **Then** the resulting genome contains the concatenated sequences from both parents.
2. **Given** two EmbeddingGenome individuals with the same embedding dimension, **When** a merge is performed, **Then** the resulting genome's token count equals the sum of both parents' token counts.
3. **Given** two VectorGenome individuals, **When** a merge is performed, **Then** the resulting genome's gene array is the concatenation of both parents' gene arrays.

---

### User Story 5 - Track Merge Events and Complexity Metrics (Priority: P2)

A researcher wants observability into how symbiogenetic merges affect population complexity over generations. The tracking system logs merge-specific metrics — number of merges per generation, genome complexity before and after merge, and symbiont source — so the researcher can analyze whether merges are driving complexity transitions.

**Why this priority**: Observability is essential for understanding whether the operator is working as intended, but the operator and engine must function first.

**Independent Test**: Can be tested by running an experiment with tracking enabled and merge events configured, then querying the tracking backend for merge-related metrics and verifying they are present and accurate.

**Acceptance Scenarios**:

1. **Given** tracking is enabled and a merge event occurs, **When** generation metrics are collected, **Then** the metrics include `merge_count`, `mean_genome_complexity`, and `complexity_delta`.
2. **Given** a merged offspring is created, **When** its metadata is inspected, **Then** it includes the origin "symbiogenetic_merge", the host and symbiont parent IDs, and the symbiont source strategy used.
3. **Given** multiple generations with merge events, **When** metrics are plotted over time, **Then** trends in genome complexity and merge frequency are visible.

---

### Edge Cases

- What happens when the population has only one species and cross-species symbiont sourcing is configured? The operator should fall back gracefully (skip the merge or use an alternative source) and log a warning.
- What happens when the hall-of-fame archive is empty (e.g., first generation) and archive-based sourcing is configured? The operator should skip the merge for that generation.
- What happens when the host and symbiont are the same individual? The operator must ensure host and symbiont are distinct individuals.
- What happens when a merged genome exceeds a configurable maximum complexity threshold (node or connection count)? The merge should be rejected to prevent unbounded growth.
- What happens when the configured number of interface connections exceeds available nodes? The operator should create as many as possible and log the shortfall.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a `SymbiogeneticMerge` protocol generic over genome type `G`, with signature `merge(host: G, symbiont: G, rng: Random, **kwargs) -> G` returning a single merged genome.
- **FR-002**: System MUST provide a GraphGenome merge implementation that remaps the symbiont's node IDs to a fresh range that does not collide with the host's node IDs.
- **FR-003**: System MUST remap the symbiont's innovation numbers to a fresh range above the current global innovation counter when merging GraphGenomes.
- **FR-004**: System MUST take the union of all nodes and connections from both host and symbiont genomes, preserving the symbiont's internal topology and weights intact.
- **FR-005**: System MUST create a configurable number of interface connections bridging host and symbiont subgraphs — split equally between host→symbiont and symbiont→host directions (remainder randomly assigned for odd counts) — each with a fresh innovation number and a random weight.
- **FR-006**: System MUST support configurable weight initialization for interface connections via a strategy selector (`weight_method`: mean, host_biased, or random) and, when using random initialization, configurable Gaussian parameters (`weight_mean` default 0.0, `weight_std` default 1.0).
- **FR-007**: System MUST support symbiont sourcing from the same population across different species ("cross_species" strategy).
- **FR-008**: System MUST support symbiont sourcing from an archive of past high-fitness individuals ("archive" strategy).
- **FR-009**: System MUST integrate symbiogenetic merge as a separate phase in the evolutionary loop, executing after selection, crossover, and mutation.
- **FR-010**: System MUST support configurable merge rate as a per-individual probability — each individual independently has `merge_rate` chance of being selected as a host for a merge event, consistent with how `crossover_rate` and `mutation_rate` operate in the engine.
- **FR-011**: System MUST allow the merge operator to be registered and discovered through the existing operator registry under a "merge" category.
- **FR-012**: System MUST provide merge implementations for SequenceGenome (concatenation), VectorGenome (concatenation), and EmbeddingGenome (vertical stacking of embedding matrices).
- **FR-013**: System MUST record merge-specific metadata on merged offspring, including origin type, host parent ID, symbiont parent ID, and symbiont source strategy.
- **FR-014**: System MUST log merge-related observability metrics per generation: merge count, mean genome complexity (measured as gene count: nodes+connections for GraphGenome, sequence length for SequenceGenome, array length for VectorGenome, token count for EmbeddingGenome), and complexity change.
- **FR-015**: System MUST enforce that host and symbiont are distinct individuals.
- **FR-016**: System MUST handle edge cases gracefully — single-species populations with cross-species sourcing, empty archives, and complexity limits — by skipping the merge and logging a warning.
- **FR-017**: System MUST expose merge configuration through UnifiedConfig fields: merge operator name, merge rate, symbiont source strategy, number of interface connections, interface weight distribution parameters, optional maximum complexity threshold, and symbiont fate (consumed or survives).
- **FR-018**: System MUST support a configurable symbiont fate policy: "consumed" (symbiont is removed from the population after merge) or "survives" (symbiont remains in the population unchanged). Default is "consumed".
- **FR-019**: System MUST select hosts for merge events uniformly at random from the population.

### Key Entities

- **SymbiogeneticMerge**: The protocol defining the merge operator contract — generic over genome type, takes a host and symbiont, returns a single merged genome.
- **MergeConfig**: The configuration entity capturing all merge parameters — rate, symbiont source strategy, interface connection count, weight distribution, complexity threshold, and symbiont fate policy.
- **Merged Offspring**: An Individual whose metadata records its symbiogenetic origin, linking back to both host and symbiont parents.
- **Symbiont Source**: A strategy that determines where the symbiont comes from — cross-species selection or historical archive.
- **Interface Connection**: A new connection bridging the host and symbiont subgraphs within a merged GraphGenome.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A symbiogenetic merge of two graph-based genomes completes successfully, producing a single offspring whose structural complexity (node + connection count) exceeds that of either parent.
- **SC-002**: Merged offspring are automatically placed in new species by the existing speciation mechanism due to their structural novelty, without any changes to the speciation logic.
- **SC-003**: Researchers can enable symbiogenetic merge in their experiments by adding configuration fields, with no code changes required beyond configuration.
- **SC-004**: At least three genome types (GraphGenome, SequenceGenome, VectorGenome) have working merge implementations that pass all acceptance tests.
- **SC-005**: Merge events are visible in the tracking system with per-generation metrics, enabling researchers to correlate merges with fitness and complexity trends.
- **SC-006**: Users can register custom merge strategies for new genome types using the standard registry pattern, verified by a working custom registration test.
- **SC-007**: The merge operator handles all identified edge cases without errors, falling back gracefully and logging appropriate warnings.

## Assumptions

- The existing NEAT speciation mechanism correctly assigns merged organisms to new species based on structural distance — no changes to speciation logic are needed.
- The existing NEAT crossover correctly handles excess innovation numbers from symbiont genes (inheriting them only from the fitter parent) — no changes to crossover logic are needed.
- Subsequent NEAT mutations (add_node, add_connection, weight perturbation) naturally refine interface connections over generations — no special post-merge adaptation phase is required.
- The hall-of-fame archive for symbiont sourcing is maintained by an existing or new callback — the merge operator consumes the archive but does not manage its lifecycle.
- Interface connection weight distribution defaults to Gaussian with mean 0.0 and standard deviation 1.0, consistent with existing weight initialization patterns in the framework.
- The default number of interface connections is 4 (2 host-to-symbiont, 2 symbiont-to-host), providing a minimal bidirectional interface. For odd counts, the remainder connection's direction is randomly assigned.
- Merge rate defaults to 0.0 (disabled), ensuring backward compatibility — existing experiments are unaffected unless merge is explicitly configured.
- The maximum complexity threshold is optional and defaults to no limit, allowing unbounded growth unless the researcher opts into complexity constraints.
- SequenceGenome and VectorGenome merge strategies are simple concatenation; more sophisticated strategies (e.g., interleaving, alignment-based merging) are deferred to future work.
- EmbeddingGenome merge requires both genomes to have the same embedding dimension; mismatched dimensions are an error.
