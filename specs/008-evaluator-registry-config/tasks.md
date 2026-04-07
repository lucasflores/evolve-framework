# Tasks: Evaluator Registry & UnifiedConfig Declarative Completeness

**Input**: Design documents from `/specs/008-evaluator-registry-config/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/public-api.md, quickstart.md

**Tests**: Included — dev-stack enforces TDD (Red-Green-Refactor). Write tests in the same commit as the implementation they cover.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No setup tasks required — existing project with established structure, dependencies, and test infrastructure.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the two new registries and extend `UnifiedConfig` with new fields. These components are required by ALL user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T001 [P] Create EvaluatorRegistry with singleton pattern, lazy initialization, register/get/is_registered/list_evaluators methods, and 7 built-in registrations (benchmark, function, llm_judge, ground_truth, scm, rl, meta) with deferred imports for ML deps in evolve/registry/evaluators.py
- [X] T002 [P] Create CallbackRegistry with singleton pattern, lazy initialization, register/get/is_registered/list_callbacks methods, and 4 built-in registrations (logging, checkpoint, print, history) in evolve/registry/callbacks.py
- [X] T003 [P] Add evaluator (str | None, default None), evaluator_params (dict, default {}), and custom_callbacks (tuple[dict], default ()) fields with __post_init__ validation to UnifiedConfig in evolve/config/unified.py
- [X] T004 Export get_evaluator_registry, reset_evaluator_registry, EvaluatorRegistry, get_callback_registry, reset_callback_registry, CallbackRegistry from evolve/registry/__init__.py

**Checkpoint**: Registries exist and UnifiedConfig accepts new fields — user story implementation can now begin.

---

## Phase 3: User Story 1 — Declare Evaluator in Config and Run (Priority: P1) 🎯 MVP

**Goal**: A researcher declares `evaluator="benchmark"` and `evaluator_params` in config, calls `create_engine(config)` with no evaluator argument, and gets a working engine.

**Independent Test**: Create a `UnifiedConfig` with `evaluator="benchmark"` and `evaluator_params={"function_name": "sphere", "dimensions": 10}`, call `create_engine(config)` with no evaluator argument, verify the engine runs a generation successfully.

### Tests for User Story 1

- [X] T005 [P] [US1] Write unit tests for EvaluatorRegistry: singleton behavior, built-in evaluator resolution (benchmark with sphere, function), KeyError with available list for unknown names, register/overwrite semantics, TypeError when registering non-callable factory, unknown evaluator_params produce context-rich error, deferred import guarantee (importing evaluators module does NOT trigger ML dependency imports) in tests/unit/test_evaluator_registry.py
- [X] T006 [P] [US1] Write integration tests for declarative engine creation: config with evaluator="benchmark" resolves and runs, explicit evaluator overrides config, missing evaluator raises ValueError with available list, full combined flow test (config with evaluator + custom_callbacks + all new fields → create_engine → run a generation) in tests/integration/test_declarative_engine.py

### Implementation for User Story 1

- [X] T007 [US1] Implement evaluator resolution logic in create_engine: resolve from EvaluatorRegistry when config.evaluator is set and no explicit evaluator passed, add runtime_overrides parameter that merges with evaluator_params before factory invocation, raise ValueError when neither config.evaluator nor explicit evaluator is provided in evolve/factory/engine.py

**Checkpoint**: `create_engine(config)` resolves evaluator declaratively — core MVP functional.

---

## Phase 4: User Story 2 — Register and Use a Custom Evaluator (Priority: P2)

**Goal**: A domain researcher registers a custom evaluator factory at startup, references it by name in config, and `create_engine` resolves it.

**Independent Test**: Register a custom evaluator factory under `"my_domain_eval"`, create a config referencing that name, verify `create_engine` resolves and instantiates it correctly.

### Tests for User Story 2

- [X] T008 [P] [US2] Write tests for custom evaluator registration and usage: register user factory, resolve via config, overwrite built-in name, list shows custom alongside built-ins in tests/unit/test_evaluator_registry.py

### Implementation for User Story 2

> No new implementation required — US2 acceptance scenarios are satisfied by EvaluatorRegistry.register() (T001) and create_engine resolution (T007). T008 validates the end-to-end custom evaluator workflow.

**Checkpoint**: Custom evaluator registration and declarative usage verified.

---

## Phase 5: User Story 4 — Experiment Hash Reflects Full Specification (Priority: P2)

**Goal**: Two configs that differ only in evaluator or callback configuration produce different hashes. Legacy configs produce identical hashes to before.

**Independent Test**: Create two configs differing only in `evaluator` or `evaluator_params`, compute their hashes, verify they differ.

### Tests for User Story 4

- [X] T009 [P] [US4] Write tests for compute_hash() with new fields: hashes differ by evaluator name, differ by evaluator_params, differ by custom_callbacks, backward-compatible hash when all new fields are defaults in tests/unit/test_unified_config_ext.py

### Implementation for User Story 4

- [X] T010 [US4] Update compute_hash() to conditionally include evaluator, evaluator_params, and custom_callbacks in hash dict only when non-default (None, {}, () respectively) to preserve backward compatibility in evolve/config/unified.py

**Checkpoint**: Experiment tracking hashes are complete and backward-compatible.

---

## Phase 6: User Story 6 — Full Serialization Roundtrip (Priority: P2)

**Goal**: A config with all new fields survives to_dict/from_dict, to_json/from_json, and to_file/from_file roundtrips. Legacy JSON files load with defaults.

**Independent Test**: Create a config with all new fields populated, serialize to JSON, deserialize, assert field equality and hash match.

### Tests for User Story 6

- [X] T011 [P] [US6] Write tests for serialization roundtrip: to_dict/from_dict preserves evaluator, evaluator_params, custom_callbacks; to_json/from_json roundtrip; to_file/from_file roundtrip; legacy JSON missing new fields loads with defaults in tests/unit/test_unified_config_ext.py

### Implementation for User Story 6

- [X] T012 [US6] Update to_dict() to serialize custom_callbacks tuple as list of dicts, update from_dict() to convert custom_callbacks list back to tuple and handle missing keys with defaults in evolve/config/unified.py

**Checkpoint**: Configs are fully shareable as JSON — the "hand someone a file" promise works.

---

## Phase 7: User Story 3 — Declare Custom Callbacks in Config (Priority: P3)

**Goal**: A researcher declares custom callbacks by name in config, and the factory resolves them alongside standard CallbackConfig callbacks in the correct execution order (Config → Custom → Explicit).

**Independent Test**: Register a callback under `"my_tracker"`, add it to `custom_callbacks` in config, call `create_engine`, verify the callback's `on_generation_end` fires during a run.

### Tests for User Story 3

- [X] T013 [P] [US3] Write unit tests for CallbackRegistry: singleton behavior, 4 built-in callbacks resolve, KeyError for unknown names, register/overwrite in tests/unit/test_callback_registry.py
- [X] T014 [P] [US3] Write integration tests for callback wiring: custom_callbacks resolved from registry, execution order is Config → Custom → Explicit, empty custom_callbacks is no-op, unregistered callback name raises error, duplicate callback name in custom_callbacks both resolve and fire in tests/integration/test_declarative_engine.py

### Implementation for User Story 3

- [X] T015 [US3] Implement custom_callbacks resolution in create_engine: resolve each entry from CallbackRegistry, merge with CallbackConfig-derived callbacks (first) and explicitly passed callbacks (last) in declared order in evolve/factory/engine.py

**Checkpoint**: Callback wiring is fully declarative — experiments are reproducible via config alone.

---

## Phase 8: User Story 5 — Genome Params Validation (Priority: P3)

**Goal**: Unrecognized `genome_params` keys cause a clear validation error at factory time instead of being silently ignored.

**Independent Test**: Create a config with `genome_type="vector"` and `genome_params={"dimensons": 10}` (typo), call `create_engine`, verify a validation error is raised naming the unrecognized key.

### Tests for User Story 5

- [X] T016 [P] [US5] Write tests for genome_params validation: reject unrecognized keys with clear error, accept valid keys, skip validation for factories with **kwargs signatures, remove injected params (rng) from validation in tests/unit/test_genome_params_validation.py

### Implementation for User Story 5

- [X] T017 [US5] Implement signature introspection validation in GenomeRegistry.create(): use inspect.signature() on factory callable, collect accepted param names, exclude injected params (rng), detect VAR_KEYWORD to skip validation, raise ValueError listing unknown keys and accepted params in evolve/registry/genomes.py

**Checkpoint**: Genome params are validated — no silent misconfiguration.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final validation and cleanup across all stories.

- [X] T018 [P] Add __all__ exports to evolve/registry/evaluators.py and evolve/registry/callbacks.py
- [X] T019 Validate quickstart.md examples execute correctly against the implementation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: No dependencies — start immediately. BLOCKS all user stories.
- **US1 (Phase 3)**: Depends on Phase 2 completion. BLOCKS US2 (needs create_engine resolution).
- **US2 (Phase 4)**: Depends on Phase 3 (registry + create_engine resolution must exist).
- **US4 (Phase 5)**: Depends on Phase 2 (needs new UnifiedConfig fields). Independent of US1.
- **US6 (Phase 6)**: Depends on Phase 2 (needs new UnifiedConfig fields). Independent of US1, US4.
- **US3 (Phase 7)**: Depends on Phase 2. Independent of US1 (separate registry, separate engine path).
- **US5 (Phase 8)**: Depends on Phase 2. Fully independent of all other stories.
- **Polish (Phase 9)**: Depends on all user stories being complete.

### User Story Independence Matrix

| Story | Depends On | Independent Of |
|-------|-----------|----------------|
| US1 (P1) | Phase 2 | US3, US4, US5, US6 |
| US2 (P2) | Phase 2, US1 | US3, US4, US5, US6 |
| US4 (P2) | Phase 2 | US1, US2, US3, US5, US6 |
| US6 (P2) | Phase 2 | US1, US2, US3, US4, US5 |
| US3 (P3) | Phase 2 | US1, US2, US4, US5, US6 |
| US5 (P3) | Phase 2 | US1, US2, US3, US4, US6 |

**⚠️ File-level coordination notes**:
- US1 (T007) and US3 (T015) both modify `evolve/factory/engine.py`. Logically independent but sequential execution avoids merge conflicts.
- US4 (T010) and US6 (T012) both modify `evolve/config/unified.py`. Sequential execution (US4 → US6) recommended.

### Within Each User Story

- Tests MUST be written first and FAIL before implementation (TDD per dev-stack)
- Tests and implementation go in the same commit
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 2**: T001 ‖ T002 ‖ T003 (three different files), then T004
- **After Phase 2**: US4, US6, US3, US5 can all start in parallel (different files, independent stories)
- **US1 must complete before US2** (US2 validates custom evaluator flow built in US1)
- **Within each story**: Test tasks marked [P] can run in parallel

---

## Parallel Example: After Phase 2

```
# These four stories can execute in parallel (different files, no cross-deps):
Stream A: US1 → US2 (sequential, US2 depends on US1)
Stream B: US4 (hash changes in unified.py)
Stream C: US6 (serialization changes in unified.py) — NOTE: coordinate with Stream B on unified.py
Stream D: US3 (callback registry + engine.py callback merge)
Stream E: US5 (genomes.py validation)
```

**Coordination note**: US4 (T010) and US6 (T012) both modify `evolve/config/unified.py`. If parallelized, merge carefully. Sequential execution (US4 → US6) avoids conflicts.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (4 tasks)
2. Complete Phase 3: US1 — Declare Evaluator in Config (3 tasks)
3. **STOP and VALIDATE**: `create_engine(config)` resolves evaluator declaratively
4. This alone delivers the highest-value capability

### Incremental Delivery

1. Phase 2 → Foundation ready
2. US1 → Declarative evaluator works → **MVP!**
3. US2 → Custom evaluators validated
4. US4 → Hashes are complete
5. US6 → Serialization roundtrip works → **Full reproducibility!**
6. US3 → Declarative callbacks → **Complete declarative spec!**
7. US5 → Genome params validated → **Bulletproof configs!**
8. Polish → Quickstart validated

### Recommended Order (Single Developer)

Phase 2 → US1 → US2 → US4 → US6 → US3 → US5 → Polish

This follows priority order (P1 → P2 → P3) and avoids unified.py merge conflicts by doing US4 and US6 sequentially.

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable after Phase 2
- Tests and implementation go in the same commit per dev-stack TDD rules
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
