# Tasks: Evolutionary Soft-Prompt Optimization (ESPO)

**Input**: Design documents from `/specs/007-llm-soft-prompt-evolution/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project structure and shared configuration for ESPO modules

- [X] T001 Create ESPO package directories: `evolve/meta/soft_prompt/`, `evolve/config/`, `tests/unit/`, `tests/integration/` with `__init__.py` files
- [X] T002 [P] Implement `DimensionalityStrategy` enum and `EmbeddingGenomeConfig` dataclass in `evolve/representation/embedding_config.py` (FR-002)
- [X] T003 [P] Implement `TaskSpec` and `RubricCriterion` dataclasses in `evolve/evaluation/task_spec.py` (FR-015)
- [X] T004 [P] Implement `ESPOConfig` aggregating all ESPO settings in `evolve/config/espo.py`
- [X] T042 [P] Add unit tests for `TaskSpec` and `RubricCriterion` in `tests/unit/test_task_spec.py` — verify validation, serialization, rubric criteria construction (FR-015)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core genome type that ALL user stories depend on — MUST complete before any story work

**⚠️ CRITICAL**: No user story work can begin until EmbeddingGenome is implemented and passing protocol compliance

- [X] T005 Implement `EmbeddingGenome` frozen dataclass in `evolve/representation/embedding.py` — stores 2D `np.ndarray` of shape `(n_tokens, embed_dim)`, `model_id`, `seed_text`, `strategy`; implements `Genome` and `SerializableGenome` protocols with `copy()`, `__eq__`, `__hash__`, `to_dict()`, `from_dict()` (FR-001, FR-016)
- [X] T006 Implement flat-vector adapters on `EmbeddingGenome`: `flat()` property returning 1D view, `to_vector_genome()` and `@classmethod from_vector_genome()` in `evolve/representation/embedding.py` (FR-009)
- [X] T007 Register `"embedding"` genome type in `evolve/registry/genomes.py` `_register_builtin_genomes()` with factory function
- [X] T008 Add ESPO fixtures to `tests/conftest.py` — `embedding_genome_config`, `sample_embedding_genome`, `sample_task_spec`

**Checkpoint**: EmbeddingGenome passes framework protocol compliance — `isinstance(genome, Genome)`, `isinstance(genome, SerializableGenome)`, round-trip serialization, immutability

---

## Phase 3: User Story 2 — Create and Configure an Embedding Genome (Priority: P1) 🎯 MVP

**Goal**: Researchers can create, configure, serialize, and manage EmbeddingGenome objects with full framework protocol compliance

**Independent Test**: Create an EmbeddingGenome, verify shape, serialize/deserialize, confirm protocol requirements

### Implementation for User Story 2

- [X] T009 [P] [US2] Add protocol compliance tests in `tests/unit/test_embedding_genome.py` — verify `Genome` and `SerializableGenome` protocol satisfaction, immutability, equality, hashing, round-trip serialization, flat accessor, validation errors for bad inputs
- [X] T010 [P] [US2] Add unit tests for `DimensionalityStrategy` and `EmbeddingGenomeConfig` validation in `tests/unit/test_embedding_genome.py`
- [X] T011 [US2] Verify genome module import boundary (SC-009): add static analysis test in `tests/unit/test_embedding_genome.py` that asserts `evolve/representation/embedding.py` imports no ML frameworks

**Checkpoint**: User Story 2 complete — genome is the foundation for all subsequent stories

---

## Phase 4: User Story 1 — Evolve a Soft Prompt on a Benchmark Task (Priority: P1)

**Goal**: End-to-end ESPO experiment on a QA benchmark produces a best individual with fitness ≥ seed baseline

**Independent Test**: Run ESPO on a small QA dataset (50 questions) with a local model, verify best accuracy > seed accuracy

### Implementation for User Story 1

- [X] T012 [US1] Implement `SoftPromptDecoder` in `evolve/meta/soft_prompt/decoder.py` — lazy model/tokenizer loading, `decode(genome, task_input)` via embedding-layer injection, `embed_text()` for seed embedding, model ID validation (FR-003, FR-004)
- [X] T013 [US1] Implement `GroundTruthEvaluator` in `evolve/evaluation/benchmark.py` — implements `Evaluator[EmbeddingGenome]` protocol, computes accuracy/F1/exact-match/pass@k from ground truth, deterministic with explicit seed (FR-005)
- [X] T014 [US1] Implement noise-based `PopulationInitializer.noise_init()` in `evolve/meta/soft_prompt/initializer.py` — embeds seed text, pads/truncates to `n_tokens`, perturbs with calibrated Gaussian noise (FR-010)
- [X] T015 [US1] Implement `ESPOCallback` in `evolve/meta/soft_prompt/callback.py` — logs per-generation metrics (best/mean/worst fitness, diversity L2 stats, infeasibility rate, mutation magnitude, best decoded text) to MLflow via existing `MLflowTracker` (FR-018)
- [X] T016 [US1] Add integration test for end-to-end ESPO pipeline in `tests/integration/test_espo_pipeline.py` — configures engine with EmbeddingGenome, decoder mock, benchmark evaluator, runs 5 generations, verifies fitness improvement and MLflow logging
- [X] T043 [P] [US1] Add integration tests for `GroundTruthEvaluator` in `tests/integration/test_benchmark_evaluator.py` — verify deterministic scoring (accuracy, F1, exact-match), reproducibility across runs (SC-006), edge cases (empty inputs, all-correct, all-wrong)

**Checkpoint**: Core ESPO loop works end-to-end — User Story 1 is the MVP

---

## Phase 5: User Story 3 — Apply Token-Aware Evolutionary Operators (Priority: P2)

**Goal**: Token-aware mutation and crossover operators that respect the 2D token structure of embedding genomes

**Independent Test**: Apply token-aware mutation → verify only selected tokens perturbed; apply token-level crossover → verify offspring tokens are whole-token copies from parents

### Implementation for User Story 3

- [X] T017 [P] [US3] Implement `TokenAwareMutator` in `evolve/core/operators/token_mutation.py` — per-token Gaussian mutation with configurable rate, sigma, and optional coherence radius clamping; implements `MutationOperator[EmbeddingGenome]` protocol (FR-007)
- [X] T018 [P] [US3] Implement `TokenLevelCrossover` in `evolve/core/operators/token_crossover.py` — single-point and two-point token-level crossover swapping whole tokens; implements `CrossoverOperator[EmbeddingGenome]` protocol (FR-008)
- [X] T019 [US3] Register token operators in `evolve/registry/operators.py` `_register_builtin_operators()` — `"token_gaussian"` mutation, `"token_single_point"` and `"token_two_point"` crossover with `compatible_genomes={"embedding"}`
- [X] T020 [P] [US3] Add unit tests for `TokenAwareMutator` in `tests/unit/test_token_mutation.py` — verify per-token selection, noise magnitude, coherence radius clamping, unmutated tokens unchanged, determinism with seed
- [X] T021 [P] [US3] Add unit tests for `TokenLevelCrossover` in `tests/unit/test_token_crossover.py` — verify whole-token preservation, parent shape match validation, offspring shape correctness, determinism with seed
- [X] T022 [US3] Add integration test for flat-vector operator compatibility in `tests/integration/test_espo_pipeline.py` — apply `GaussianMutation` and `BlendCrossover` via `to_vector_genome()`/`from_vector_genome()` adapter, verify valid offspring (FR-009)

**Checkpoint**: Both token-aware and flat-vector operators work with EmbeddingGenome

---

## Phase 6: User Story 4 — Initialize a Population from Seed Text (Priority: P2)

**Goal**: Two population initialization strategies (noise-based and LLM-discovered) produce diverse, valid populations from seed text

**Independent Test**: Initialize population via each strategy, verify valid shapes, mutual distinctness, and coherent outputs

### Implementation for User Story 4

- [X] T023 [US4] Implement `PopulationInitializer.llm_variation_init()` in `evolve/meta/soft_prompt/initializer.py` — prompts LLM for paraphrase variants, embeds each, falls back to noise-based if unavailable (FR-011)
- [X] T024 [US4] Add pad/truncate logic in `PopulationInitializer` for seed texts with different token counts than configured `n_tokens`
- [X] T025 [P] [US4] Add unit tests for both initialization strategies in `tests/unit/test_initializer.py` — verify population size, genome shape, mutual distinctness (pairwise L2 > 0), pad/truncate behavior

**Checkpoint**: Population initialization produces diverse, valid genomes from seed text

---

## Phase 7: User Story 5 — Decode Soft Prompts via Model Injection (Priority: P2)

**Goal**: Decode embedding genomes into model output text via embedding-layer injection

**Independent Test**: Create genome, decode with trivial input on a local model, verify output is valid text

### Implementation for User Story 5

- [X] T026 [US5] Add model mismatch error handling and embed_dim validation in `SoftPromptDecoder.decode()` in `evolve/meta/soft_prompt/decoder.py` (FR-004)
- [X] T027 [P] [US5] Add unit tests for `SoftPromptDecoder` in `tests/integration/test_soft_prompt_decoder.py` — verify decode output is string, model mismatch raises ValueError, embed_dim mismatch raises ValueError, lazy loading behavior

**Checkpoint**: Decoding from genome → model output is reliable with proper error boundaries

---

## Phase 8: User Story 6 — Defend Against Coherence Collapse (Priority: P3)

**Goal**: Three-layer coherence defense prevents wasted evaluations on incoherent genomes

**Independent Test**: Create extreme mutations, verify each layer (independently toggled) handles infeasible individuals

### Implementation for User Story 6

- [X] T028 [US6] Implement `CoherenceDefense` in `evolve/meta/soft_prompt/coherence.py` — Layer 1: mutation L2 norm clamping; Layer 2: perplexity-based feasibility check using framework `Fitness.constraints`; Layer 3: fitness-based selection integration; each layer independently toggleable (FR-012, FR-013)
- [X] T029 [US6] Implement all-infeasible recovery in `ESPOCallback` — detect when all individuals are infeasible, restore previous generation, reduce mutation magnitude by 50%, retry (FR-012)
- [X] T030 [P] [US6] Add unit tests for `CoherenceDefense` in `tests/unit/test_coherence_defense.py` — verify each layer independently, toggle on/off behavior, L2 clamping math, infeasibility marking via `Fitness.constraints`, all-infeasible recovery logic

**Checkpoint**: Coherence defense layers work independently and in combination

---

## Phase 9: User Story 7 — Evaluate with LLM-as-Judge (Priority: P3)

**Goal**: LLM-as-judge evaluator returns multi-dimensional fitness from rubric criteria for open-ended tasks

**Independent Test**: Configure rubric with two criteria, evaluate genome output, verify fitness has one value per criterion

### Implementation for User Story 7

- [X] T031 [US7] Implement `LLMJudgeEvaluator` in `evolve/evaluation/llm_judge.py` — implements `Evaluator[EmbeddingGenome]` protocol, formats judge prompt with rubric, parses per-criterion scores as JSON, returns multi-dimensional `Fitness`, handles malformed responses with retry/minimum-score fallback (FR-006)
- [X] T032 [P] [US7] Add unit tests for `LLMJudgeEvaluator` in `tests/integration/test_llm_judge_evaluator.py` — verify multi-objective fitness shape matches rubric criteria count, score clamping, malformed JSON handling, integration with NSGA-II selection (SC-007)

**Checkpoint**: LLM-as-judge returns correct multi-objective fitness for open-ended evaluation

---

## Phase 10: User Story 8 — Transfer Evolved Prompts Across Models (Priority: P3)

**Goal**: Text-mediated transfer converts evolved soft prompt to hard prompt text for use on different models

**Independent Test**: Evolve on model A, transfer to text, verify text is usable as standard prompt seed on model B

### Implementation for User Story 8

- [X] T033 [US8] Implement `text_mediated_transfer()` in `evolve/meta/soft_prompt/transfer.py` — decodes best genome to text interpretation suitable as hard prompt (FR-014)
- [X] T034 [US8] Add re-seeding support: accept transferred text as `seed_text` for new `EmbeddingGenomeConfig` on a different model (FR-017)
- [X] T035 [P] [US8] Add unit tests for text-mediated transfer in `tests/unit/test_transfer.py` — verify transfer produces non-empty text, text can be used as seed for new config on different model_id

**Checkpoint**: Cross-model transfer via text works end-to-end

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Quality improvements affecting multiple user stories

- [X] T036 [P] Add `evolve-framework[llm]` optional dependency group in `pyproject.toml` for `torch`, `transformers`
- [X] T037 [P] Update `evolve/representation/__init__.py` to export `EmbeddingGenome`, `EmbeddingGenomeConfig`, `DimensionalityStrategy`
- [X] T038 [P] Update `evolve/core/operators/__init__.py` to export `TokenAwareMutator`, `TokenLevelCrossover`
- [X] T039 [P] Update `evolve/evaluation/__init__.py` to export `GroundTruthEvaluator`, `LLMJudgeEvaluator`, `TaskSpec`
- [X] T040 Run full test suite and verify all tests pass
- [X] T041 Run quickstart.md validation — execute code snippets against a mock/small model
- [X] T044 [P] Add `@pytest.mark.slow` performance benchmark in `tests/integration/test_espo_pipeline.py` — measure wall-clock time per individual evaluation, assert ≤60s on single GPU with 7–8B model (SC-010)
- [X] T045 [P] Add `@pytest.mark.slow` long-form validation in `tests/integration/test_espo_pipeline.py` — run 50-generation ESPO experiment on mock QA data, verify best individual accuracy exceeds seed baseline (SC-001)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US2 (Phase 3)**: Depends on Phase 2 — genome tests validate the foundation
- **US1 (Phase 4)**: Depends on Phase 2 — needs EmbeddingGenome, adds decoder + evaluator + initializer
- **US3 (Phase 5)**: Depends on Phase 2 — needs EmbeddingGenome for operators
- **US4 (Phase 6)**: Depends on T012 (decoder) from US1 — needs `embed_text()` for initialization
- **US5 (Phase 7)**: Depends on T012 (decoder) from US1 — tests decoder error handling
- **US6 (Phase 8)**: Depends on T017 (mutation) from US3 and T015 (ESPOCallback) from US1 — coherence defense wraps mutation; all-infeasible recovery lives in ESPOCallback
- **US7 (Phase 9)**: Depends on T012 (decoder) from US1 — judge evaluator uses decoder
- **US8 (Phase 10)**: Depends on T012 (decoder) from US1 — transfer uses decoder
- **Polish (Phase 11)**: Depends on all stories complete

### User Story Dependencies

```text
Phase 1 (Setup)
    │
    ▼
Phase 2 (Foundational: EmbeddingGenome)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Phase 3 (US2)  Phase 4 (US1)  Phase 5 (US3)
  Genome Tests   Decoder+Eval   Operators
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
Phase 6 (US4)  Phase 7 (US5)  Phase 9 (US7)  Phase 10 (US8)
  Initializer    Decode Errs    LLM Judge      Transfer
                                    
Phase 5 (US3) ──▶ Phase 8 (US6)
  Operators        Coherence
Phase 4 (US1) ──▶ Phase 8 (US6)  [T015 ESPOCallback needed for T029]
                     
All ──▶ Phase 11 (Polish)
```

### Within Each User Story

- Models/dataclasses before services
- Services before integration
- Core implementation before error handling
- Story complete before moving to next priority

### Parallel Opportunities

**After Phase 2 completes, these can run in parallel:**
- Phase 3 (US2 — genome tests) ‖ Phase 4 (US1 — decoder/evaluator) ‖ Phase 5 (US3 — operators)

**After Phase 4 (US1 decoder) completes:**
- Phase 6 (US4) ‖ Phase 7 (US5) ‖ Phase 9 (US7) ‖ Phase 10 (US8) — all need decoder

**Within phases, [P] tasks can run in parallel:**
- T002 ‖ T003 ‖ T004 (Setup)
- T017 ‖ T018 (Operators)
- T020 ‖ T021 (Operator tests)

---

## Implementation Strategy

### MVP First (User Story 1 + 2)

1. Phase 1: Setup → Phase 2: Foundational (EmbeddingGenome)
2. Phase 3: US2 (genome protocol compliance tests)
3. Phase 4: US1 (decoder + benchmark evaluator + noise init + MLflow callback)
4. **STOP and VALIDATE**: Run 5-generation ESPO experiment on mock QA data
5. Verify fitness improvement and MLflow logging ← this is the MVP

### Incremental Delivery

1. Setup + Foundational → EmbeddingGenome ready
2. US2 + US1 → End-to-end ESPO works (MVP!)
3. US3 → Token-aware operators for better search
4. US4 → LLM-discovered initialization
5. US5 → Decoder error boundaries
6. US6 → Coherence defense
7. US7 → LLM-as-judge multi-objective
8. US8 → Cross-model transfer

---

## Summary

| Metric | Count |
|--------|-------|
| **Total tasks** | 45 |
| **Setup tasks** | 5 (T001–T004, T042) |
| **Foundational tasks** | 4 (T005–T008) |
| **US1 tasks** | 6 (T012–T016, T043) |
| **US2 tasks** | 3 (T009–T011) |
| **US3 tasks** | 6 (T017–T022) |
| **US4 tasks** | 3 (T023–T025) |
| **US5 tasks** | 2 (T026–T027) |
| **US6 tasks** | 3 (T028–T030) |
| **US7 tasks** | 2 (T031–T032) |
| **US8 tasks** | 3 (T033–T035) |
| **Polish tasks** | 8 (T036–T041, T044–T045) |
| **Parallelizable tasks** | 24 (marked [P]) |
| **MVP scope** | Phases 1–4 (18 tasks) |
