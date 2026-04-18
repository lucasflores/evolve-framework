# Research: Symbiogenetic Merge Operator

**Feature**: 013-symbiogenetic-merge  
**Date**: 2026-04-17  
**Phase**: 0 — Outline & Research

## R1: How to integrate a new operator phase into the engine loop

**Decision**: Insert merge phase after offspring creation (post-crossover, post-mutation) and before population replacement in `EvolutionEngine._step()`.

**Rationale**: The engine loop in `evolve/core/engine.py` follows a clear sequence: select → crossover → mutate → create offspring → replace population → evaluate. The merge phase operates on the newly created offspring pool, selecting hosts uniformly at random and sourcing symbionts from the configured strategy. This placement mirrors how mutation operates as a post-crossover refinement, and keeps the merge phase decoupled from the reproduction pipeline.

**Alternatives considered**:
- *Before crossover*: Would merge parents before reproduction, conflating merge with selection. Rejected because merge is a reproductive event, not a selection event.
- *After evaluation*: Would require a second evaluation pass for merged offspring. Rejected because it doubles evaluation cost and complicates the generation boundary.
- *As a callback*: Callbacks are observational, not transformational. Rejected because merge modifies the population.

## R2: How to add a new operator category to the registry

**Decision**: Add `"merge"` to the existing `OperatorRegistry` categories and register builtin merge operators in `_register_builtin_operators()`.

**Rationale**: The registry uses `(category, name)` tuples as keys. Adding "merge" requires no structural changes — just registration calls. The `compatible_genomes` metadata on each registration enables automatic compatibility checking when the factory resolves the merge operator.

**Alternatives considered**:
- *Separate MergeRegistry*: Would fragment the operator system. Rejected because the existing OperatorRegistry already supports arbitrary categories.
- *Protocol-only (no registry)*: Would require manual wiring. Rejected because it violates Constitution III (Declarative Completeness).

## R3: How to handle innovation number remapping for GraphGenome merge

**Decision**: Use the existing `InnovationTracker` to reserve a contiguous range of innovation numbers for the symbiont's remapped connections. Call `InnovationTracker.get_innovation()` for each interface connection to get fresh numbers.

**Rationale**: The `InnovationTracker` in `evolve/representation/graph.py` already manages global innovation number assignment. Reserving a range for the symbiont's connections ensures no collisions with existing or future innovations. Interface connections get their own fresh innovation numbers via the standard `get_innovation()` path, making them visible to NEAT crossover's matching/disjoint/excess gene classification.

**Alternatives considered**:
- *Offset-based remapping without tracker*: Would risk collisions with future innovations. Rejected because it breaks NEAT crossover invariants.
- *Keep symbiont innovation numbers unchanged*: Would cause false matching with host genes. Rejected because innovation numbers must be unique per structural topology.

## R4: How to handle node ID remapping for GraphGenome merge

**Decision**: Compute `max_node_id = max(node.id for node in host.nodes)` and remap all symbiont node IDs by adding `max_node_id + 1` as an offset. Use `InnovationTracker.reserve_node_ids()` to reserve the range.

**Rationale**: Node IDs in GraphGenome are integers used for connection routing (`from_node`, `to_node`). Simple offset remapping is deterministic, O(N) in symbiont size, and guarantees no collisions. Reserving IDs in the tracker prevents future `get_new_node_id()` calls from reusing remapped IDs.

**Alternatives considered**:
- *Hash-based remapping*: Non-deterministic, harder to debug. Rejected.
- *UUID node IDs*: Would break the integer-based NEAT infrastructure. Rejected.

## R5: How to structure MergeConfig as a frozen dataclass

**Decision**: Create `MergeConfig` as a frozen dataclass in `evolve/config/merge.py`, following the exact pattern of `ERPSettings` and `MetaEvolutionConfig`. Add `merge: MergeConfig | None = None` to `UnifiedConfig`.

**Rationale**: All advanced config sections in the framework use the same pattern: frozen dataclass with sensible defaults, optional field on UnifiedConfig (None = disabled). This ensures backward compatibility (existing configs are unaffected) and follows Constitution III.

**Alternatives considered**:
- *Flat fields on UnifiedConfig*: Would clutter the top-level config with 7+ merge-specific fields. Rejected for cohesion.
- *External config file*: Would break the single-source-of-truth principle. Rejected.

## R6: How to define the SymbiogeneticMerge protocol

**Decision**: Define as a `@runtime_checkable Protocol[G]` with a single method `merge(host: G, symbiont: G, rng: Random, **kwargs) -> G`. Place in `evolve/core/operators/merge.py` alongside the existing operator protocols.

**Rationale**: The existing CrossoverOperator, MutationOperator, and SelectionOperator protocols all follow this pattern: `@runtime_checkable`, generic over `G`, explicit `rng` parameter. The merge protocol returns a single `G` (not a pair like crossover), reflecting the asymmetric absorption semantics.

**Alternatives considered**:
- *ABC instead of Protocol*: Rejected per `agents.md` — "Protocols over ABCs."
- *Returning Individual instead of G*: Would couple the operator to the Individual type. Rejected because operators work at the genome level.

## R7: How to implement symbiont sourcing strategies

**Decision**: Implement sourcing as simple functions (not a protocol/registry) called by the engine during the merge phase. Two strategies: `cross_species` (select from a different NEAT species) and `archive` (select from a hall-of-fame list maintained by a callback).

**Rationale**: Sourcing is a small, closed set of strategies tightly coupled to the engine's population management. A full protocol/registry would be over-engineering for two strategies. The engine can switch on the string name from MergeConfig.

**Alternatives considered**:
- *Full SymbiontSource protocol + registry*: Over-engineering for 2 strategies. Can be refactored later if more strategies emerge.
- *Sourcing inside the merge operator*: Would couple the operator to population structure. Rejected because operators should be genome-level.

## R8: How to track merge-specific metrics

**Decision**: Create a `MergeMetricCollector` in `evolve/experiment/collectors/merge.py` following the `NEATMetricCollector` pattern. Add a `SYMBIOGENESIS` value to the `MetricCategory` enum. The collector tracks: `merge_count`, `mean_genome_complexity`, `complexity_delta`, and `merge_sources` (breakdown by source strategy).

**Rationale**: The existing collector pattern provides a clean separation between metric collection and tracking backend integration. A new `MetricCategory` enum value allows users to opt into merge metrics via `TrackingConfig.categories`.

**Alternatives considered**:
- *Embed metrics in engine loop*: Would violate separation of concerns. Rejected.
- *Use Fitness.metadata*: Would conflate per-individual metadata with population-level metrics. Rejected.

## R9: How to implement the hall-of-fame archive for symbiont sourcing

**Decision**: Implement a `HallOfFameCallback` that maintains a bounded list of the best individuals seen across all generations. The merge engine phase reads this archive when `symbiont_source="archive"` is configured. The callback is automatically added when merge with archive sourcing is enabled.

**Rationale**: The callback system already provides `on_generation_end` hooks with access to the population. A hall-of-fame is a standard evolutionary computation pattern. The callback owns the archive lifecycle; the merge phase only reads it.

**Alternatives considered**:
- *Archive managed by the engine*: Would add state to the engine that only merge uses. Rejected.
- *External file-based archive*: Over-engineering for in-memory use. Rejected.

## R10: How non-graph genome merges work (concatenation/stacking)

**Decision**: Implement merge for each genome type using its natural concatenation operation:
- **SequenceGenome**: `SequenceGenome(genes=host.genes + symbiont.genes, alphabet=host.alphabet, max_length=None)`
- **VectorGenome**: `VectorGenome(genes=np.concatenate([host.genes, symbiont.genes]), bounds=concatenated_bounds)`
- **EmbeddingGenome**: `EmbeddingGenome(embeddings=np.vstack([host.embeddings, symbiont.embeddings]), model_id=host.model_id)`

**Rationale**: These are the simplest, most natural operations for each type. They directly increase genome size (the goal of symbiogenesis) without requiring complex alignment or structural analysis.

**Alternatives considered**:
- *Interleaving for SequenceGenome*: More complex, unclear benefit. Deferred to future work.
- *Weighted averaging for VectorGenome*: Does not increase complexity. Rejected.
- *Horizontal stacking for EmbeddingGenome*: Would change embed_dim, breaking model compatibility. Rejected.
