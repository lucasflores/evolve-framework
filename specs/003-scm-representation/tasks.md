# Tasks: SCM Representation for Causal Discovery

**Input**: Design documents from `/specs/003-scm-representation/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

**Tests**: Included as specified in plan.md (unit, property, integration tests)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Exact file paths included in descriptions

## Path Conventions

- **Source**: `evolve/representation/`, `evolve/evaluation/`
- **Tests**: `tests/unit/`, `tests/property/`, `tests/integration/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and core type definitions

- [X] T001 Add NetworkX dependency to pyproject.toml if not present
- [X] T002 [P] Create evolve/representation/scm.py with module docstring and imports
- [X] T003 [P] Create evolve/representation/scm_decoder.py with module docstring and imports
- [X] T004 [P] Create evolve/evaluation/scm_evaluator.py with module docstring and imports
- [X] T005 [P] Create tests/unit/test_scm_genome.py with imports and fixtures
- [X] T006 [P] Create tests/unit/test_scm_decoder.py with imports and fixtures
- [X] T007 [P] Create tests/unit/test_scm_evaluator.py with imports and fixtures

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core types that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Implement ConflictResolution enum in evolve/representation/scm.py
- [X] T009 [P] Implement AcyclicityMode enum in evolve/representation/scm.py
- [X] T010 [P] Implement AcyclicityStrategy enum in evolve/representation/scm.py
- [X] T011 Implement SCMConfig dataclass with validation in evolve/representation/scm.py
- [X] T012 Implement SCMAlphabet with from_config() factory in evolve/representation/scm.py
- [X] T013 Implement Expression AST nodes (Var, Const, BinOp) in evolve/representation/scm_decoder.py
- [X] T014 [P] Implement expression utilities (complexity, variables, evaluate, to_string) in evolve/representation/scm_decoder.py
- [X] T015 [P] Implement expr_to_dict/expr_from_dict serialization in evolve/representation/scm_decoder.py
- [X] T016 Unit tests for SCMConfig validation in tests/unit/test_scm_genome.py
- [X] T017 [P] Unit tests for SCMAlphabet generation in tests/unit/test_scm_genome.py
- [X] T018 [P] Unit tests for Expression AST and utilities in tests/unit/test_scm_decoder.py

**Checkpoint**: Foundation ready - SCMConfig, SCMAlphabet, Expression types all functional ✓

---

## Phase 3: User Story 1 - Define and Evolve SCM Genomes (Priority: P1) 🎯 MVP

**Goal**: Researchers can create SCM genomes from observed variables, copy them, and apply mutation via inner SequenceGenome

**Independent Test**: Create SCMGenome with 3 variables, apply point mutation, verify structure validity

### Tests for User Story 1

- [X] T019 [P] [US1] Unit test for SCMGenome.random() creation in tests/unit/test_scm_genome.py
- [X] T020 [P] [US1] Unit test for SCMGenome.copy() independence in tests/unit/test_scm_genome.py
- [X] T021 [P] [US1] Unit test for SCMGenome Genome protocol (__eq__, __hash__) in tests/unit/test_scm_genome.py
- [X] T022 [P] [US1] Unit test for ERC Gaussian sampling and mutation in tests/unit/test_scm_genome.py
- [X] T023 [P] [US1] Unit test for alphabet compatibility with SequenceGenome in tests/unit/test_scm_genome.py

### Implementation for User Story 1

- [X] T024 [US1] Implement SCMGenome dataclass with inner SequenceGenome composition in evolve/representation/scm.py
- [X] T025 [US1] Implement SCMGenome.copy() returning independent deep copy in evolve/representation/scm.py
- [X] T026 [US1] Implement SCMGenome.__eq__ and __hash__ in evolve/representation/scm.py
- [X] T027 [US1] Implement SCMGenome.genes property delegating to inner in evolve/representation/scm.py
- [X] T028 [US1] Implement SCMGenome.random() with ERC Gaussian N(0, σ_init) sampling in evolve/representation/scm.py
- [X] T029 [US1] Implement ERC perturbation mutation adding N(0, σ_perturb) in evolve/representation/scm.py
- [X] T030 [US1] Export SCMGenome, SCMConfig, SCMAlphabet from evolve/representation/__init__.py

**Checkpoint**: User Story 1 complete - SCMGenome creation, copying, and mutation fully functional ✓

---

## Phase 4: User Story 2 - Decode Genomes into Causal Graphs (Priority: P1)

**Goal**: Researchers can decode SCMGenome into DecodedSCM with equations and NetworkX DiGraph

**Independent Test**: Create genome encoding "A B + STORE_C", decode it, verify equation C = A + B and edges A→C, B→C

### Tests for User Story 2

- [X] T031 [P] [US2] Unit test for stack-based postfix decoding in tests/unit/test_scm_decoder.py
- [X] T032 [P] [US2] Unit test for STORE_X equation creation in tests/unit/test_scm_decoder.py
- [X] T033 [P] [US2] Unit test for junk gene detection (underflow, empty stack STORE) in tests/unit/test_scm_decoder.py
- [X] T034 [P] [US2] Unit test for NetworkX graph construction from equations in tests/unit/test_scm_decoder.py
- [X] T035 [P] [US2] Unit test for decoding determinism (same genome → same result) in tests/unit/test_scm_decoder.py

### Implementation for User Story 2

- [X] T036 [US2] Implement SCMMetadata dataclass in evolve/representation/scm_decoder.py
- [X] T037 [US2] Implement DecodedSCM dataclass with equations, graph, metadata in evolve/representation/scm_decoder.py
- [X] T038 [US2] Implement SCMDecoder class skeleton with config in evolve/representation/scm_decoder.py
- [X] T039 [US2] Implement SCMDecoder._decode_stack_machine() core loop in evolve/representation/scm_decoder.py
- [X] T040 [US2] Implement operand push logic (variables, constants, ERCs) in evolve/representation/scm_decoder.py
- [X] T041 [US2] Implement operator pop-push logic (+, -, *, /) in evolve/representation/scm_decoder.py
- [X] T042 [US2] Implement STORE_X handling creating equations in evolve/representation/scm_decoder.py
- [X] T043 [US2] Implement stack underflow → junk gene handling in evolve/representation/scm_decoder.py
- [X] T044 [US2] Implement empty stack STORE_X → junk gene handling in evolve/representation/scm_decoder.py
- [X] T045 [US2] Implement NetworkX DiGraph construction from equations in evolve/representation/scm_decoder.py
- [X] T046 [US2] Implement cycle detection using nx.is_directed_acyclic_graph() and nx.simple_cycles() in evolve/representation/scm_decoder.py
- [X] T047 [US2] Implement SCMDecoder.decode() composing all steps in evolve/representation/scm_decoder.py
- [X] T048 [US2] Export SCMDecoder, DecodedSCM, Expression types from evolve/representation/__init__.py

**Checkpoint**: User Story 2 complete - Genomes decode to DecodedSCM with graph and metadata ✓

---

## Phase 5: User Story 3 - Evaluate SCM Fitness Multi-Objectively (Priority: P1)

**Goal**: Researchers can evaluate DecodedSCMs against observed data with configurable objectives and penalties

**Independent Test**: Create known-structure SCM, generate synthetic data, verify true model gets high fitness

### Tests for User Story 3

- [X] T049 [P] [US3] Unit test for data_fit objective (negative MSE) in tests/unit/test_scm_evaluator.py
- [X] T050 [P] [US3] Unit test for sparsity objective (negative edge count) in tests/unit/test_scm_evaluator.py
- [X] T051 [P] [US3] Unit test for simplicity objective (negative AST complexity) in tests/unit/test_scm_evaluator.py
- [X] T052 [P] [US3] Unit test for cycle penalty application in tests/unit/test_scm_evaluator.py
- [X] T053 [P] [US3] Unit test for div_zero_penalty with NaN propagation in tests/unit/test_scm_evaluator.py
- [X] T054 [P] [US3] Unit test for acyclicity_mode=reject returning None in tests/unit/test_scm_evaluator.py

### Implementation for User Story 3

- [X] T055 [US3] Implement SCMFitnessConfig dataclass in evolve/evaluation/scm_evaluator.py
- [X] T056 [US3] Implement SCMEvaluationResult dataclass in evolve/evaluation/scm_evaluator.py
- [X] T057 [US3] Implement SCMEvaluator class skeleton with data and decoder in evolve/evaluation/scm_evaluator.py
- [X] T058 [US3] Implement SCMEvaluator.capabilities property in evolve/evaluation/scm_evaluator.py
- [X] T059 [US3] Implement _compute_predictions() evaluating equations on data in evolve/evaluation/scm_evaluator.py
- [X] T060 [US3] Implement _objective_data_fit() computing negative MSE in evolve/evaluation/scm_evaluator.py
- [X] T061 [US3] Implement _objective_sparsity() counting edges in evolve/evaluation/scm_evaluator.py
- [X] T062 [US3] Implement _objective_simplicity() summing AST complexity in evolve/evaluation/scm_evaluator.py
- [X] T063 [US3] Implement _objective_coverage() computing variable coverage in evolve/evaluation/scm_evaluator.py
- [X] T064 [US3] Implement _objective_latent_parsimony() counting latent variables in evolve/evaluation/scm_evaluator.py
- [X] T064a [US3] Implement _validate_latent_ancestors() checking nx.ancestors() ∩ observed_variables ≠ ∅ for each latent variable in evolve/evaluation/scm_evaluator.py
- [X] T064b [US3] Apply latent_ancestor_penalty when latent variables lack observed ancestors in evolve/evaluation/scm_evaluator.py
- [X] T065 [US3] Implement _check_constraints() for acyclicity, coverage, conflict_free in evolve/evaluation/scm_evaluator.py
- [X] T066 [US3] Implement _compute_penalties() aggregating all penalties in evolve/evaluation/scm_evaluator.py
- [X] T067 [US3] Implement SCMEvaluator.evaluate() composing decode → objectives → constraints → fitness in evolve/evaluation/scm_evaluator.py
- [X] T068 [US3] Export SCMEvaluator, SCMFitnessConfig from evolve/evaluation/__init__.py

**Checkpoint**: User Story 3 complete - Multi-objective evaluation with penalties fully functional ✓

---

## Phase 6: User Story 4 - Configure Conflict Resolution Strategies (Priority: P2)

**Goal**: Researchers can control how conflicting equations are handled during decoding

**Independent Test**: Create genomes with known conflicts, verify each resolution strategy produces expected results

### Tests for User Story 4

- [X] T069 [P] [US4] Unit test for conflict_resolution=first_wins in tests/unit/test_scm_decoder.py
- [X] T070 [P] [US4] Unit test for conflict_resolution=last_wins in tests/unit/test_scm_decoder.py
- [X] T071 [P] [US4] Unit test for conflict_resolution=all_junk in tests/unit/test_scm_decoder.py
- [X] T072 [P] [US4] Unit test for conflict metadata in DecodedSCM.metadata in tests/unit/test_scm_decoder.py

### Implementation for User Story 4

- [X] T073 [US4] Implement conflict detection in SCMDecoder tracking multiple STORE_X in evolve/representation/scm_decoder.py
- [X] T074 [US4] Implement first_wins resolution keeping first equation in evolve/representation/scm_decoder.py
- [X] T075 [US4] Implement last_wins resolution keeping last equation in evolve/representation/scm_decoder.py
- [X] T076 [US4] Implement all_junk resolution discarding all conflicts in evolve/representation/scm_decoder.py
- [X] T077 [US4] Populate metadata.conflicts with variable → count mapping in evolve/representation/scm_decoder.py

**Checkpoint**: User Story 4 complete - All three conflict resolution strategies working ✓

---

## Phase 7: User Story 5 - Handle Cyclic SCMs Gracefully (Priority: P2)

**Goal**: Researchers can configure how cycles are detected and handled during evaluation

**Independent Test**: Create cyclic genomes, verify each handling mode produces expected fitness results

### Tests for User Story 5

- [X] T078 [P] [US5] Unit test for acyclicity_strategy=acyclic_subgraph in tests/unit/test_scm_evaluator.py
- [X] T079 [P] [US5] Unit test for acyclicity_strategy=parse_order in tests/unit/test_scm_evaluator.py
- [X] T080 [P] [US5] Unit test for acyclicity_strategy=penalty_only in tests/unit/test_scm_evaluator.py
- [X] T080a [P] [US5] Unit test for acyclicity_strategy=parent_inheritance in tests/unit/test_scm_evaluator.py
- [X] T080b [P] [US5] Unit test for acyclicity_strategy=composite in tests/unit/test_scm_evaluator.py
- [X] T081 [P] [US5] Unit test for cycle metadata (is_cyclic, cycles) in tests/unit/test_scm_decoder.py

### Implementation for User Story 5

- [X] T082 [US5] Implement _extract_acyclic_subgraph() finding maximal DAG in evolve/evaluation/scm_evaluator.py
- [X] T083 [US5] Implement _break_cycles_by_parse_order() using decode order in evolve/evaluation/scm_evaluator.py
- [X] T083a [US5] Implement _apply_parent_inheritance() using ERP-aware cycle breaking in evolve/evaluation/scm_evaluator.py
- [X] T083b [US5] Implement _apply_composite() combining subgraph + proportional penalty in evolve/evaluation/scm_evaluator.py
- [X] T084 [US5] Implement _apply_acyclicity_strategy() dispatching to strategy in evolve/evaluation/scm_evaluator.py
- [X] T085 [US5] Update evaluate() to apply strategy when mode=penalize in evolve/evaluation/scm_evaluator.py

**Checkpoint**: User Story 5 complete - All acyclicity handling strategies implemented and tested ✓

---

## Phase 8: User Story 6 - Serialize and Restore SCM Genomes (Priority: P2)

**Goal**: Researchers can checkpoint and restore SCMGenome populations

**Independent Test**: Serialize genome with ERCs to dict, deserialize, verify equality

### Tests for User Story 6

- [X] T086 [P] [US6] Unit test for SCMGenome.to_dict() serialization in tests/unit/test_scm_genome.py
- [X] T087 [P] [US6] Unit test for SCMGenome.from_dict() deserialization in tests/unit/test_scm_genome.py
- [X] T088 [P] [US6] Property test for serialization round-trip in tests/property/test_scm_properties.py

### Implementation for User Story 6

- [X] T089 [US6] Implement SCMGenome.to_dict() including inner genome and ERC values in evolve/representation/scm.py
- [X] T090 [US6] Implement SCMGenome.from_dict() reconstructing genome state in evolve/representation/scm.py
- [X] T091 [US6] Verify compatibility with existing checkpoint infrastructure

**Checkpoint**: User Story 6 complete - Serialization and checkpoint compatibility verified ✓

---

## Phase 9: User Story 7 - Integrate with ERP for Matchability (Priority: P3)

**Goal**: Matchability considers both sequence similarity and decoded causal structure

**Independent Test**: Compute matchability between genome pairs, verify both factors contribute

### Tests for User Story 7

- [X] T092 [P] [US7] Unit test for sequence-only matchability in tests/unit/test_scm_genome.py
- [X] T093 [P] [US7] Unit test for structural matchability contribution in tests/unit/test_scm_genome.py
- [X] T094 [P] [US7] Integration test with ERP-enabled evolution in tests/integration/test_scm_discovery.py

### Implementation for User Story 7

- [X] T095 [US7] Implement SCMMatchability computing sequence similarity in evolve/representation/scm.py
- [X] T096 [US7] Implement structural similarity using graph edit distance in evolve/representation/scm.py
- [X] T097 [US7] Implement weighted combination (structural_weight parameter) in evolve/representation/scm.py
- [X] T098 [US7] Register SCMMatchability with ERP infrastructure in evolve/representation/scm.py

**Checkpoint**: User Story 7 complete - ERP matchability considers both sequence and structure

---

## Phase 10: Property & Integration Tests

**Purpose**: Cross-cutting verification and end-to-end testing

- [X] T099 [P] Create tests/property/test_scm_properties.py with hypothesis strategies
- [X] T100 [P] Property test for decoding determinism (same genome → same DecodedSCM) in tests/property/test_scm_properties.py
- [X] T101 [P] Property test for graph validity (edges match equation dependencies) in tests/property/test_scm_properties.py
- [X] T102 Create tests/integration/test_scm_discovery.py with synthetic data fixtures
- [X] T103 Integration test for end-to-end SCM discovery on 3-variable synthetic model
- [X] T104 Integration test for population evolution (1000 individuals, 50 generations)
- [X] T104a Memory profiling test verifying 1000+ population evolution without memory issues (SC-001)
- [X] T105 Performance test for decoding 500-gene genome under 10ms

**Checkpoint**: All property invariants hold, integration tests pass

---

## Phase 11: Polish & Cross-Cutting Concerns ✓

**Purpose**: Documentation, cleanup, and final validation

- [X] T106 [P] Add module docstrings to all new files
- [X] T107 [P] Add inline documentation for complex algorithms (stack machine, cycle detection)
- [X] T108 Verify all existing tests pass (zero regression)
- [X] T109 Run mypy type checking on new modules
- [X] T110 Run quickstart.md validation (all examples execute)
- [X] T110a Document end-to-end SCM discovery example in quickstart.md showing causal structure discovery from synthetic data (SC-007)
- [X] T111 Update docs/api/representation.md with SCM documentation
- [X] T112 Update evolve/__init__.py to expose public SCM API

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ─────────────┐
                             ▼
Phase 2 (Foundational) ──────┼──▶ BLOCKS ALL USER STORIES
                             │
     ┌───────────────────────┴───────────────────────┐
     ▼                       ▼                       ▼
Phase 3 (US1)          Phase 4 (US2)           Phase 5 (US3)
  P1 MVP               P1 Decoding             P1 Evaluation
     │                       │                       │
     └───────────────────────┼───────────────────────┘
                             ▼
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
Phase 6 (US4)          Phase 7 (US5)           Phase 8 (US6)
P2 Conflicts          P2 Cycles              P2 Serialization
     │                       │                       │
     └───────────────────────┴───────────────────────┘
                             │
                             ▼
                      Phase 9 (US7)
                      P3 ERP Integration
                             │
                             ▼
                      Phase 10 (Property/Integration)
                             │
                             ▼
                      Phase 11 (Polish)
```

### User Story Dependencies

| Story | Depends On | Can Parallel With |
|-------|------------|-------------------|
| US1 (Genome) | Foundational | US2, US3 (after foundation) |
| US2 (Decoding) | Foundational | US1, US3 |
| US3 (Evaluation) | US2 (needs decoder) | US1 |
| US4 (Conflicts) | US2 | US5, US6 |
| US5 (Cycles) | US2, US3 | US4, US6 |
| US6 (Serialization) | US1 | US4, US5 |
| US7 (ERP) | US1, US2 | - |

### Parallel Opportunities Per Phase

**Phase 2 (Foundational)**:
- T008, T009, T010 (enums) in parallel
- T013, T014, T015 (Expression types) in parallel
- T016, T017, T018 (tests) in parallel

**Phase 3-5 (P1 Stories)** after foundation complete:
- US1 tests (T019-T023) all parallel
- US2 tests (T031-T035) all parallel
- US3 tests (T049-T054) all parallel

**Phase 6-8 (P2 Stories)** after P1 stories complete:
- All three P2 phases can run in parallel
- Tests within each phase are parallel

---

## Implementation Strategy

### MVP Scope (First Deliverable)

Complete Phases 1-5 for minimal viable SCM discovery:

1. ✓ SCMConfig + SCMAlphabet (foundation)
2. ✓ SCMGenome creation and mutation (US1)
3. ✓ SCMDecoder with basic stack machine (US2)
4. ✓ SCMEvaluator with default objectives (US3)

**MVP Test**: Create population, evolve 10 generations, verify fitness improves

### Incremental Delivery

1. **MVP**: US1 + US2 + US3 (core functionality)
2. **Robustness**: US4 + US5 (conflict/cycle handling)
3. **Persistence**: US6 (serialization)
4. **Advanced**: US7 (ERP integration)

---

## Task Summary

| Phase | Task Count | Parallel Tasks | Blocking |
|-------|------------|----------------|----------|
| 1 - Setup | 7 | 6 | No |
| 2 - Foundational | 11 | 7 | YES |
| 3 - US1 (Genome) | 12 | 5 | No |
| 4 - US2 (Decoding) | 18 | 5 | No |
| 5 - US3 (Evaluation) | 20 | 6 | No |
| 6 - US4 (Conflicts) | 9 | 4 | No |
| 7 - US5 (Cycles) | 8 | 4 | No |
| 8 - US6 (Serialization) | 6 | 3 | No |
| 9 - US7 (ERP) | 7 | 3 | No |
| 10 - Property/Integration | 7 | 4 | No |
| 11 - Polish | 7 | 2 | No |
| **TOTAL** | **112** | **49** | |

**Independent Test Criteria per Story**:
- US1: Create genome, mutate, verify structure
- US2: Decode known genome, verify equations and graph
- US3: Evaluate known SCM against synthetic data
- US4: Decode with conflicts, verify resolution strategy
- US5: Decode cyclic genome, verify handling mode
- US6: Serialize/deserialize, verify equality
- US7: Compute matchability, verify weighted combination
