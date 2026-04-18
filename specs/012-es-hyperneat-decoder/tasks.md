# Tasks: ES-HyperNEAT Decoder (CPPN-to-Network Indirect Encoding)

**Input**: Design documents from `/specs/012-es-hyperneat-decoder/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/decoder.md, quickstart.md

**Tests**: Not explicitly requested — test tasks are omitted. Tests should be added alongside implementation per dev-stack commit policy.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. Since the user requested sequential execution, tasks are ordered for strict top-to-bottom execution.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create new files and directory scaffolding required by the feature.

- [ ] T001 Create empty module file `evolve/representation/cppn_decoder.py` with module docstring
- [ ] T002 Create empty test file `tests/unit/representation/test_cppn_decoder.py`
- [ ] T003 Create empty test file `tests/unit/representation/test_activations.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Activation function additions and internal data structures that ALL user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Add `sin_activation` function (`np.sin(x)`) and register it as `"sin"` in `ACTIVATIONS` dict in `evolve/representation/network.py`
- [ ] T005 Add `abs_activation` function (`np.abs(x)`) and register it as `"abs"` in `ACTIVATIONS` dict in `evolve/representation/network.py`
- [ ] T006 Implement `DecodeStats` frozen dataclass in `evolve/representation/cppn_decoder.py` with fields: `neurons_discovered`, `connections_before_pruning`, `neurons_pruned`, `connections_after_pruning`, `neurons_final`
- [ ] T007 Implement `QuadTreeNode` dataclass in `evolve/representation/cppn_decoder.py` with fields: `x`, `y`, `half_size`, `depth`, `children`

**Checkpoint**: Foundation ready — activation functions available, data structures defined.

---

## Phase 3: User Story 1 — Decode a CPPN into a Neural Network with Discovered Topology (Priority: P1) 🎯 MVP

**Goal**: A researcher can call `decoder.decode(genome)` on a CPPN `GraphGenome` and receive a callable `NEATNetwork` with automatically discovered hidden neurons and CPPN-determined connection weights.

**Independent Test**: Construct a simple CPPN GraphGenome with known outputs, decode it, verify the resulting network has neurons in information-dense regions, connections above threshold, and produces correct forward-pass outputs.

### Implementation for User Story 1

- [ ] T008 [US1] Implement `CPPNToNetworkDecoder.__init__()` in `evolve/representation/cppn_decoder.py` — store all parameters (`input_positions`, `output_positions`, `weight_threshold`, `variance_threshold`, `max_quadtree_depth`, `distance_input`, `hidden_activation`), validate non-empty positions (raise `ValueError`), validate `hidden_activation` via `get_activation()` (catch `KeyError`, re-raise as `ValueError`), validate numeric ranges per FR-006 (`weight_threshold > 0`, `variance_threshold > 0`, `max_quadtree_depth >= 1` — raise `ValueError`), initialize `_last_decode_stats = None`
- [ ] T009 [US1] Implement `CPPNToNetworkDecoder._build_cppn()` private method in `evolve/representation/cppn_decoder.py` — use existing `GraphToNetworkDecoder.decode()` to convert the CPPN `GraphGenome` into a callable `NEATNetwork`; validate CPPN input count matches expected (4 or 5 depending on `distance_input`), raise `ValueError` on mismatch
- [ ] T010 [US1] Implement `CPPNToNetworkDecoder._discover_hidden_neurons()` private method in `evolve/representation/cppn_decoder.py` — quadtree decomposition over `[-1,1]×[-1,1]`; sample CPPN at four quadrant centers of each leaf, compute variance, subdivide if variance > `variance_threshold` and depth < `max_quadtree_depth`; return list of `(x, y)` positions from qualifying leaf centers
- [ ] T011 [US1] Implement `CPPNToNetworkDecoder._query_connections()` private method in `evolve/representation/cppn_decoder.py` — iterate all (source, target) neuron pairs; query CPPN with `(x1, y1, x2, y2)` or `(x1, y1, x2, y2, d)` if `distance_input=True`; create connection where `|output| > weight_threshold`; return dict mapping `(src_id, tgt_id)` to weight
- [ ] T012 [US1] Implement `CPPNToNetworkDecoder._prune_disconnected()` private method in `evolve/representation/cppn_decoder.py` — forward BFS from input neuron IDs, backward BFS from output neuron IDs; keep only hidden neurons in the intersection of both reachable sets; remove connections involving pruned neurons
- [ ] T013 [US1] Implement `CPPNToNetworkDecoder.decode()` public method in `evolve/representation/cppn_decoder.py` — orchestrate the three-phase pipeline: (1) `_build_cppn`, (2) `_discover_hidden_neurons`, (3) `_query_connections` + `_prune_disconnected`; assign neuron IDs (inputs 0..N-1, outputs N..N+M-1, hidden from N+M onward); build `NEATNetwork` with topological sort; populate `DecodeStats` and store in `_last_decode_stats`; emit structured decode event via callback/tracking system
- [ ] T014 [US1] Implement `last_decode_stats` property on `CPPNToNetworkDecoder` in `evolve/representation/cppn_decoder.py`
- [ ] T015 [US1] Write unit tests in `tests/unit/representation/test_cppn_decoder.py` covering: decode simple CPPN → verify network callable; decode uniform-output CPPN → no hidden neurons; decode high-variance CPPN → hidden neurons in expected region; deterministic decoding (same input → same output); all weights below threshold → no connections; duplicate coordinates in positions handled; minimal CPPN (no hidden nodes) decodes successfully; init with empty `input_positions` raises `ValueError`; init with empty `output_positions` raises `ValueError`; init with invalid `hidden_activation` raises `ValueError`; init with `weight_threshold=0` raises `ValueError`; init with `max_quadtree_depth=0` raises `ValueError`
- [ ] T015b [US1] Write observability test in `tests/unit/representation/test_cppn_decoder.py` — verify `decode()` emits structured decode event with `DecodeStats` fields (neurons_discovered, connections_before_pruning, neurons_pruned, connections_after_pruning, neurons_final) via the callback/tracking system
- [ ] T015c [US1] Write scale test in `tests/unit/representation/test_cppn_decoder.py` — construct a CPPN with `max_quadtree_depth=6`, decode it, verify decode completes without error and the resulting network with thousands of connections is callable (SC-005)
- [ ] T016 [US1] Write unit tests in `tests/unit/representation/test_activations.py` verifying `get_activation("sin")` and `get_activation("abs")` return correct callables

**Checkpoint**: Core ES-HyperNEAT decoding is functional and tested. A user can programmatically construct a decoder, decode a CPPN, and get a working network.

---

## Phase 4: User Story 2 — Configure and Run ES-HyperNEAT via UnifiedConfig and create_engine() (Priority: P2)

**Goal**: A user sets `decoder="cppn_to_network"` in `UnifiedConfig`, calls `create_engine()`, and gets an engine with a correctly wired `CPPNToNetworkDecoder`.

**Independent Test**: Create a `UnifiedConfig` with `decoder="cppn_to_network"` and `decoder_params`, call `create_engine()`, verify the engine contains a `CPPNToNetworkDecoder` with the specified parameters.

### Implementation for User Story 2

- [ ] T017 [US2] Register `"cppn_to_network"` factory in `evolve/registry/decoders.py` — add `create_cppn_to_network_decoder(**kwargs)` factory function that imports and instantiates `CPPNToNetworkDecoder`; register it in `_register_builtin_decoders()` under the name `"cppn_to_network"`
- [ ] T018 [US2] Verify `create_engine()` in `evolve/factory/engine.py` correctly resolves `decoder="cppn_to_network"` with `decoder_params` — inspect existing wiring logic and add handling if needed to pass `decoder_params` to the registry factory
- [ ] T019 [US2] Write integration test in `tests/integration/test_cppn_engine.py` — test `UnifiedConfig(decoder="cppn_to_network", decoder_params={...})` → `create_engine()` produces engine with `CPPNToNetworkDecoder`; test missing required `decoder_params` raises clear error; test `"cppn_to_network"` appears in `DecoderRegistry` listing
- [ ] T020 [US2] Add `"cppn_to_network"` registry test case to `tests/unit/test_decoder_registry.py`

**Checkpoint**: Full declarative configuration path works end-to-end.

---

## Phase 5: User Story 3 — CPPN Activation Functions Available for GraphGenome Evolution (Priority: P3)

**Goal**: `get_activation("sin")`, `get_activation("abs")`, `get_activation("gaussian")`, and `get_activation("linear")` all return valid callables usable by NEAT mutation for CPPN node genes.

**Independent Test**: Verify all four activation functions are retrievable and produce mathematically correct outputs; verify NEATMutation can assign them to new nodes.

> Note: The activation function implementations (T004, T005) and tests (T016) were completed in Phase 2 and Phase 3. This phase verifies integration with the mutation operator.

### Implementation for User Story 3

- [ ] T021 [US3] Verify `gaussian` and `linear` activations already exist in `ACTIVATIONS` dict in `evolve/representation/network.py` — confirm presence, add if missing
- [ ] T022 [US3] Add test in `tests/unit/representation/test_activations.py` verifying NEATMutation can assign `sin`, `abs`, `gaussian`, `linear`, and `sigmoid` activations to new nodes when configured with the CPPN activation set

**Checkpoint**: CPPN activation functions fully available for evolution.

---

## Phase 6: User Story 4 — Pruning Disconnected Neurons from Decoded Network (Priority: P3)

**Goal**: After CPPN connection thresholding, hidden neurons with no input-to-output path are automatically pruned from the decoded network.

**Independent Test**: Construct a scenario where thresholding leaves some discovered neurons isolated; verify they are absent from the returned network.

> Note: The pruning implementation (`_prune_disconnected`) was completed in T012 as part of US1. This phase adds dedicated pruning-focused tests.

### Implementation for User Story 4

- [ ] T023 [US4] Write targeted pruning tests in `tests/unit/representation/test_cppn_decoder.py` — test neuron with no path to any output is pruned; test neuron with no path from any input is pruned; test neuron on a valid input→output path survives; test all connections involving pruned neurons are removed

**Checkpoint**: Pruning correctness is independently verified.

---

## Phase 7: User Story 5 — Optional Euclidean Distance Input to CPPN (Priority: P4)

**Goal**: When `distance_input=True`, the CPPN receives a 5th input (Euclidean distance) for each connection query.

**Independent Test**: Decode the same CPPN with `distance_input=True` vs `False` and verify the CPPN receives 5 vs 4 inputs respectively.

> Note: Distance input handling is implemented in T009 (input validation) and T011 (query construction) as part of US1. This phase adds targeted tests.

### Implementation for User Story 5

- [ ] T024 [US5] Write distance-input tests in `tests/unit/representation/test_cppn_decoder.py` — test `distance_input=True` passes 5 inputs to CPPN (x1, y1, x2, y2, d=√((x2-x1)²+(y2-y1)²)); test `distance_input=False` passes 4 inputs; test CPPN input count mismatch raises `ValueError`

**Checkpoint**: Distance input configuration is verified.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and final cleanup.

- [ ] T025 Add module-level `__all__` exports to `evolve/representation/cppn_decoder.py` (`CPPNToNetworkDecoder`, `DecodeStats`)
- [ ] T026 Run all existing tests to confirm no regressions (`pytest tests/`)
- [ ] T027 Validate quickstart.md code snippets execute correctly against the implementation
- [ ] T028 Add `cppn_decoder` to Sphinx API docs in `docs/api/` (create `evolve.representation.cppn_decoder.rst`)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — activation functions and data structures
- **Phase 3 (US1 — Core Decoding)**: Depends on Phase 2 — uses activations, `DecodeStats`, `QuadTreeNode`
- **Phase 4 (US2 — Registry/Config)**: Depends on Phase 3 — needs `CPPNToNetworkDecoder` class to exist
- **Phase 5 (US3 — Activation Integration)**: Depends on Phase 2 — verifies activations with mutation
- **Phase 6 (US4 — Pruning Tests)**: Depends on Phase 3 — tests pruning logic from US1
- **Phase 7 (US5 — Distance Input Tests)**: Depends on Phase 3 — tests distance logic from US1
- **Phase 8 (Polish)**: Depends on all prior phases

### Sequential Execution Order

```
T001 → T002 → T003 → T004 → T005 → T006 → T007 →
T008 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016 →
T017 → T018 → T019 → T020 →
T021 → T022 →
T023 →
T024 →
T025 → T026 → T027 → T028
```

### Within Each User Story

- Data structures before methods that use them
- Private methods before the public `decode()` method that orchestrates them
- Implementation before tests
- Core implementation before integration

---

## Summary

| Metric | Value |
|--------|-------|
| **Total tasks** | 28 |
| **Phase 1 (Setup)** | 3 tasks |
| **Phase 2 (Foundational)** | 4 tasks |
| **Phase 3 (US1 — Core Decoding)** | 9 tasks |
| **Phase 4 (US2 — Registry/Config)** | 4 tasks |
| **Phase 5 (US3 — Activations)** | 2 tasks |
| **Phase 6 (US4 — Pruning)** | 1 task |
| **Phase 7 (US5 — Distance Input)** | 1 task |
| **Phase 8 (Polish)** | 4 tasks |
| **MVP scope** | Phases 1–3 (US1): 16 tasks |
| **Files created** | `evolve/representation/cppn_decoder.py`, `tests/unit/representation/test_cppn_decoder.py`, `tests/unit/representation/test_activations.py`, `tests/integration/test_cppn_engine.py`, `docs/api/evolve.representation.cppn_decoder.rst` |
| **Files modified** | `evolve/representation/network.py`, `evolve/registry/decoders.py`, `tests/unit/test_decoder_registry.py` |
