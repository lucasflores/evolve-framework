# Tasks: Symbiogenetic Merge Operator

**Input**: Design documents from `/specs/013-symbiogenetic-merge/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included per project TDD requirements (`agents.md` specifies Red-Green-Refactor).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the new files and module structure for the symbiogenetic merge feature.

- [X] T001 Create merge operator module at `evolve/core/operators/merge.py` with module docstring and imports
- [X] T002 Create merge config module at `evolve/config/merge.py` with module docstring and imports
- [X] T003 [P] Create merge metric collector module at `evolve/experiment/collectors/merge.py` with module docstring and imports
- [X] T004 [P] Create test files: `tests/unit/core/operators/test_merge.py`, `tests/unit/config/test_merge_config.py`, `tests/unit/experiment/collectors/test_merge_collector.py`, `tests/integration/test_engine_merge.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Config, registry, and enum changes that ALL user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Add `SYMBIOGENESIS = "symbiogenesis"` to `MetricCategory` enum in `evolve/config/tracking.py`
- [X] T006 Implement `MergeConfig` frozen dataclass with validation in `evolve/config/merge.py` — fields: `operator`, `merge_rate`, `symbiont_source`, `symbiont_fate`, `archive_size`, `interface_count`, `interface_ratio`, `weight_method`, `weight_mean`, `weight_std`, `max_complexity`, `operator_params` per data-model.md
- [X] T007 Add `merge: MergeConfig | None = None` field to `UnifiedConfig` in `evolve/config/unified.py` and update `compute_hash()` to include it
- [X] T008 Add `"merge"` to `OperatorRegistry.CATEGORIES` tuple in `evolve/registry/operators.py`
- [X] T009 Define `SymbiogeneticMerge` protocol (generic over `G`) in `evolve/core/operators/merge.py` with method signature `merge(host: G, symbiont: G, rng: Random, **kwargs: Any) -> G`
- [X] T010 Write unit tests for `MergeConfig` validation (merge_rate bounds, interface_count > 0, interface_ratio bounds, archive_size > 0, weight_std > 0, max_complexity > 0 when set) in `tests/unit/config/test_merge_config.py`
- [X] T011 Write unit test verifying `"merge"` category exists in `OperatorRegistry` in `tests/unit/core/operators/test_merge.py`
- [X] T012 Write unit test verifying `SymbiogeneticMerge` protocol is `@runtime_checkable` and can be satisfied by a minimal implementation in `tests/unit/core/operators/test_merge.py`

**Checkpoint**: Foundation ready — MergeConfig, protocol, registry category, and MetricCategory all in place.

---

## Phase 3: User Story 1 — Merge Two Graph Genomes via Symbiogenesis (Priority: P1) 🎯 MVP

**Goal**: Implement the `GraphSymbiogeneticMerge` operator that absorbs a symbiont GraphGenome into a host GraphGenome, remapping node IDs and innovation numbers, taking the union of all genes, and generating interface connections.

**Independent Test**: Create two GraphGenome individuals with known topologies, merge them, verify the result has all nodes/connections from both plus interface connections with no ID collisions.

### Tests for User Story 1

- [X] T013 [P] [US1] Write test: merged genome contains all host and symbiont nodes (remapped) in `tests/unit/core/operators/test_merge.py`
- [X] T014 [P] [US1] Write test: symbiont node IDs are remapped to non-colliding range in `tests/unit/core/operators/test_merge.py`
- [X] T015 [P] [US1] Write test: symbiont innovation numbers are remapped via InnovationTracker in `tests/unit/core/operators/test_merge.py`
- [X] T016 [P] [US1] Write test: interface connections are created bridging host and symbiont subgraphs with correct directional split in `tests/unit/core/operators/test_merge.py`
- [X] T017 [P] [US1] Write test: host and symbiont internal topology and weights are preserved in `tests/unit/core/operators/test_merge.py`
- [X] T018 [P] [US1] Write test: ValueError raised when host is symbiont (same instance) in `tests/unit/core/operators/test_merge.py`
- [X] T019 [P] [US1] Write test: merge is deterministic given same RNG seed in `tests/unit/core/operators/test_merge.py`
- [X] T066 [P] [US1] Write test: merged offspring's structural distance (via `neat_distance`) exceeds speciation threshold, verifying SC-002 in `tests/unit/core/operators/test_merge.py`

### Implementation for User Story 1

- [X] T020 [US1] Add `InnovationTracker.reserve_node_ids(count)` helper method in `evolve/representation/graph.py` to reserve a contiguous range of node IDs
- [X] T021 [US1] Implement `GraphSymbiogeneticMerge` dataclass in `evolve/core/operators/merge.py` — fields: `interface_count`, `interface_ratio`, `weight_method`, `weight_mean`, `weight_std`; method: `merge()` with node ID remapping, innovation remapping, union of genes, and interface connection generation per research.md R3/R4
- [X] T022 [US1] Register `GraphSymbiogeneticMerge` as `"graph_symbiogenetic"` in `OperatorRegistry` builtin registrations in `evolve/registry/operators.py` with `compatible_genomes=["GraphGenome"]`
- [X] T023 [US1] Verify all US1 tests pass

**Checkpoint**: GraphGenome merge operator works standalone — can merge two graph genomes and produce a structurally valid offspring.

---

## Phase 4: User Story 2 — Configure and Trigger Merge Events in the Engine (Priority: P1)

**Goal**: Integrate the merge phase into `EvolutionEngine._step()` after crossover/mutation and before population replacement. Implement symbiont sourcing strategies and symbiont fate handling.

**Independent Test**: Run a short evolution with merge enabled at a high rate. Verify merge events occur, offspring have `origin="symbiogenetic_merge"`, and the population changes reflect merges.

### Tests for User Story 2

- [X] T024 [P] [US2] Write test: engine runs a generation with merge enabled and produces merged offspring with `origin="symbiogenetic_merge"` in `tests/integration/test_engine_merge.py`
- [X] T025 [P] [US2] Write test: merge_rate=0.0 results in no merge events in `tests/integration/test_engine_merge.py`
- [X] T026 [P] [US2] Write test: merge_rate=1.0 results in all eligible individuals being merge candidates in `tests/integration/test_engine_merge.py`
- [X] T027 [P] [US2] Write test: symbiont is consumed (removed from population) when `symbiont_fate="consumed"` in `tests/integration/test_engine_merge.py`
- [X] T028 [P] [US2] Write test: symbiont survives when `symbiont_fate="survives"` in `tests/integration/test_engine_merge.py`
- [X] T056 [P] [US2] Write test: merged offspring metadata includes origin, host_id, symbiont_id, and source_strategy in `tests/unit/core/operators/test_merge.py`
- [X] T029 [P] [US2] Write test: cross_species sourcing selects symbiont from a different species in `tests/integration/test_engine_merge.py`
- [X] T030 [P] [US2] Write test: single-species population with cross_species sourcing gracefully skips merge and emits warning in `tests/integration/test_engine_merge.py`

### Implementation for User Story 2

- [X] T031 [US2] Add `"symbiogenetic_merge"` as a valid `origin` value in `evolve/core/types.py`
- [X] T032 [US2] Implement `HallOfFameCallback` in `evolve/core/callbacks.py` — maintains bounded archive of best individuals, updated `on_generation_end`
- [X] T033 [US2] Implement merge phase in `EvolutionEngine._step()` in `evolve/core/engine.py` — after offspring creation, before population replacement: iterate offspring at merge_rate probability, source symbiont, call merge operator, create Individual with merge metadata, handle symbiont fate
- [X] T034 [US2] Implement cross_species symbiont sourcing logic in engine merge phase (select from different species) in `evolve/core/engine.py`
- [X] T035 [US2] Implement archive symbiont sourcing logic in engine merge phase (select from HallOfFameCallback.archive) in `evolve/core/engine.py`
- [X] T036 [US2] Update `create_engine()` factory in `evolve/factory/engine.py` to resolve merge operator from registry when `config.merge` is not None, and auto-add `HallOfFameCallback` when `symbiont_source="archive"`
- [X] T037 [US2] Verify all US2 tests pass

**Checkpoint**: Full engine integration — merge events fire automatically during evolution with proper symbiont sourcing and fate handling.

---

## Phase 5: User Story 3 — Define and Register a Custom Merge Strategy (Priority: P2)

**Goal**: Verify the registry-based extensibility by registering a user-defined merge operator and using it in an experiment.

**Independent Test**: Implement a trivial custom merge for VectorGenome, register it, configure an engine to use it, and verify merge events use the custom operator.

### Tests for User Story 3

- [X] T038 [P] [US3] Write test: user-defined class satisfying SymbiogeneticMerge protocol can be registered and retrieved from OperatorRegistry in `tests/unit/core/operators/test_merge.py`
- [X] T039 [P] [US3] Write test: engine selects compatible merge operator based on genome type in `tests/integration/test_engine_merge.py`
- [X] T040 [P] [US3] Write test: clear error when no compatible merge operator registered for genome type in `tests/integration/test_engine_merge.py`

### Implementation for User Story 3

- [X] T041 [US3] Add compatibility check in `create_engine()` factory in `evolve/factory/engine.py` — verify registered merge operator is compatible with configured genome type, raise clear error if not
- [X] T042 [US3] Verify all US3 tests pass

**Checkpoint**: Custom merge operators can be registered, discovered, and used — full extensibility validated.

---

## Phase 6: User Story 4 — Merge with Non-Graph Genome Types (Priority: P3)

**Goal**: Provide merge implementations for SequenceGenome (concatenation), VectorGenome (concatenation), and EmbeddingGenome (vertical stacking).

**Independent Test**: Merge two SequenceGenome individuals and verify concatenation. Merge two EmbeddingGenome individuals and verify token count equals sum.

### Tests for User Story 4

- [X] T043 [P] [US4] Write test: SequenceGenome merge concatenates genes from both parents in `tests/unit/core/operators/test_merge.py`
- [X] T044 [P] [US4] Write test: SequenceGenome merge raises error on alphabet mismatch in `tests/unit/core/operators/test_merge.py`
- [X] T045 [P] [US4] Write test: VectorGenome merge concatenates gene arrays and bounds in `tests/unit/core/operators/test_merge.py`
- [X] T046 [P] [US4] Write test: EmbeddingGenome merge vertically stacks matrices, token count equals sum in `tests/unit/core/operators/test_merge.py`
- [X] T047 [P] [US4] Write test: EmbeddingGenome merge raises error on embed_dim mismatch in `tests/unit/core/operators/test_merge.py`

### Implementation for User Story 4

- [X] T048 [P] [US4] Implement `SequenceSymbiogeneticMerge` dataclass in `evolve/core/operators/merge.py` — concatenation of `host.genes + symbiont.genes`
- [X] T049 [P] [US4] Implement `VectorSymbiogeneticMerge` dataclass in `evolve/core/operators/merge.py` — `np.concatenate` of gene arrays and bounds
- [X] T050 [P] [US4] Implement `EmbeddingSymbiogeneticMerge` dataclass in `evolve/core/operators/merge.py` — `np.vstack` of embedding matrices, validate same `model_id` and `embed_dim`
- [X] T051 [US4] Register all three operators in `OperatorRegistry` builtin registrations in `evolve/registry/operators.py`: `"sequence_symbiogenetic"`, `"vector_symbiogenetic"`, `"embedding_symbiogenetic"` with appropriate `compatible_genomes`
- [X] T052 [US4] Verify all US4 tests pass

**Checkpoint**: All four genome types have working merge implementations.

---

## Phase 7: User Story 5 — Track Merge Events and Complexity Metrics (Priority: P2)

**Goal**: Implement `MergeMetricCollector` to report `merge/count`, `merge/mean_genome_complexity`, and `merge/complexity_delta` per generation. Merged offspring metadata includes origin, parent IDs, and source strategy.

**Independent Test**: Run an experiment with tracking and merge enabled, verify metrics are emitted and contain expected values.

### Tests for User Story 5

- [X] T053 [P] [US5] Write test: MergeMetricCollector reports `merge/count` matching actual merge events in `tests/unit/experiment/collectors/test_merge_collector.py`
- [X] T054 [P] [US5] Write test: MergeMetricCollector reports `merge/mean_genome_complexity` as average gene count in `tests/unit/experiment/collectors/test_merge_collector.py`
- [X] T055 [P] [US5] Write test: MergeMetricCollector reports `merge/complexity_delta` as mean complexity increase in `tests/unit/experiment/collectors/test_merge_collector.py`

### Implementation for User Story 5

- [X] T057 [US5] Implement `MergeMetricCollector` class in `evolve/experiment/collectors/merge.py` following NEATMetricCollector pattern — collect `merge/count`, `merge/mean_genome_complexity`, `merge/complexity_delta` per generation
- [X] T058 [US5] Wire `MergeMetricCollector` activation to `MetricCategory.SYMBIOGENESIS` in tracking integration — auto-enable when category is in `TrackingConfig.categories`
- [X] T059 [US5] Verify all US5 tests pass

**Checkpoint**: Full observability — merge metrics visible in tracking backend.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Edge case hardening, documentation, and validation.

- [X] T060 [P] Write test: archive-based sourcing with empty archive skips merge and emits warning in `tests/integration/test_engine_merge.py`
- [X] T061 [P] Write test: host and symbiont are always distinct individuals during engine merge phase in `tests/integration/test_engine_merge.py`
- [X] T067 [P] Write test: merge is skipped when merged genome would exceed `max_complexity` threshold, emitting warning in `tests/integration/test_engine_merge.py`
- [X] T068 [P] [US1] Write test: interface_count exceeding available nodes creates as many connections as possible and logs warning in `tests/unit/core/operators/test_merge.py`
- [X] T069 [P] [US2] Write test: merged offspring metadata includes `source_strategy` field matching configured symbiont_source in `tests/integration/test_engine_merge.py`
- [X] T070 [P] [US2] Write test: host selection for merge is uniform random (statistical verification with high merge_rate) in `tests/integration/test_engine_merge.py`
- [X] T062 [P] Add `__all__` exports to `evolve/core/operators/merge.py`, `evolve/config/merge.py`, `evolve/experiment/collectors/merge.py`
- [X] T063 Update `evolve/core/operators/__init__.py` to re-export merge protocol and implementations
- [X] T064 Run `quickstart.md` code examples as validation — verify config creation, engine creation, and merge execution work end-to-end
- [X] T065 Run full test suite (`pytest tests/`) to confirm no regressions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 — GraphGenome merge operator
- **US2 (Phase 4)**: Depends on Phase 2 + Phase 3 (needs at least one merge operator registered)
- **US3 (Phase 5)**: Depends on Phase 2 + Phase 4 (needs engine integration)
- **US4 (Phase 6)**: Depends on Phase 2 only (independent operator implementations)
- **US5 (Phase 7)**: Depends on Phase 2 + Phase 4 (needs engine to produce merge events)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1 (P1)**: After Foundational — no cross-story deps
- **US2 (P1)**: After US1 (needs a registered merge operator to integrate)
- **US3 (P2)**: After US2 (needs working engine integration to validate)
- **US4 (P3)**: After Foundational — can parallelize with US1/US2 (different files, independent operators)
- **US5 (P2)**: After US2 (needs engine producing merge events to collect metrics from)

### Within Each User Story

- Tests MUST be written and FAIL before implementation (TDD)
- Protocol/config before implementations
- Implementations before registry registrations
- Engine integration after standalone operators work
- Story complete before moving to next priority

### Parallel Opportunities

- T003/T004 (Setup) can run in parallel
- T005/T006/T007/T008/T009 (Foundational) — T005 and T008 can parallel; T006 before T007
- All US1 tests (T013–T019) can run in parallel
- US4 implementation (T048/T049/T050) can run in parallel — different genome types, same file but independent classes
- US5 tests (T053–T056) can run in parallel
- US4 (Phase 6) can parallelize with US2 (Phase 4) — different files entirely

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (write first, expect failures):
Task T013: test merged genome contains all nodes
Task T014: test node ID remapping
Task T015: test innovation number remapping
Task T016: test interface connections
Task T017: test topology preservation
Task T018: test host==symbiont ValueError
Task T019: test determinism

# Then implement sequentially:
Task T020: InnovationTracker.reserve_node_ids() helper
Task T021: GraphSymbiogeneticMerge implementation
Task T022: Registry registration
Task T023: Verify all pass
```

## Parallel Example: User Story 4

```bash
# Launch all US4 tests together (write first, expect failures):
Task T043: SequenceGenome merge test
Task T044: SequenceGenome alphabet mismatch test
Task T045: VectorGenome merge test
Task T046: EmbeddingGenome merge test
Task T047: EmbeddingGenome embed_dim mismatch test

# Launch all US4 implementations in parallel (independent classes):
Task T048: SequenceSymbiogeneticMerge
Task T049: VectorSymbiogeneticMerge
Task T050: EmbeddingSymbiogeneticMerge

# Then register all:
Task T051: Registry registrations
Task T052: Verify all pass
```

---

## Implementation Strategy

### MVP First (User Story 1 + 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: US1 — GraphGenome merge operator
4. Complete Phase 4: US2 — Engine integration
5. **STOP and VALIDATE**: Run `pytest` + quickstart validation
6. Ship MVP: researchers can use symbiogenetic merge in NEAT experiments

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 (GraphGenome merge) → Test standalone → Core operator works
3. US2 (Engine integration) → Test end-to-end → **MVP shipped** 🎯
4. US4 (Non-graph genomes) → Test independently → Broader genome support
5. US3 (Custom registration) → Test extensibility → User-defined strategies
6. US5 (Metrics tracking) → Test observability → Full research workflow
7. Polish → Edge cases, docs, regression check

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: US1 (GraphGenome merge) → US2 (Engine integration)
   - Developer B: US4 (Non-graph genome merges) — fully independent
3. After US2 complete:
   - Developer A: US3 (Custom registration) → US5 (Metrics)
   - Developer B: Polish phase
