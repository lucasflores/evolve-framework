# Tasks: Comprehensive Documentation Refactor Centered on UnifiedConfig

**Input**: Design documents from `/specs/011-docs-unifiedconfig-refactor/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Preparation and information gathering before any documentation changes

- [x] T001 Audit current UnifiedConfig API surface — read evolve/config/unified.py, evolve/factory/engine.py, and all registry modules (evolve/registry/operators.py, evolve/registry/evaluators.py, evolve/registry/callbacks.py, evolve/registry/genomes.py) to build a complete reference table of: all UnifiedConfig fields, all registered operator/evaluator/callback/genome names, and create_engine() factory behavior
- [x] T002 Audit current tutorial notebooks — read all 7 notebooks in docs/tutorials/ to identify every instance of: (a) hand-rolled operator/selection/crossover/mutation code, (b) manual EvolutionEngine/ERPEngine construction, (c) code that should use framework API instead; produce a per-notebook change list
- [x] T003 Audit current examples — read all 4 .py files in examples/ to identify manual construction patterns and determine what can vs. cannot be expressed declaratively via UnifiedConfig

---

## Phase 2: Foundational (Tutorial Renumbering)

**Purpose**: Rename tutorial files to new numbering before any content changes. MUST complete before user story phases.

**⚠️ CRITICAL**: All tutorial content changes depend on renumbering being complete first.

- [x] T004 Rename docs/tutorials/07_unified_config.ipynb → docs/tutorials/01_unified_config.ipynb
- [x] T005 Rename docs/tutorials/01_vector_genome.ipynb → docs/tutorials/02_vector_genome.ipynb
- [x] T006 Rename docs/tutorials/02_sequence_genome.ipynb → docs/tutorials/03_sequence_genome.ipynb
- [x] T007 Rename docs/tutorials/03_graph_genome_neat.ipynb → docs/tutorials/04_graph_genome_neat.ipynb
- [x] T008 Rename docs/tutorials/04_rl_neuroevolution.ipynb → docs/tutorials/05_rl_neuroevolution.ipynb
- [x] T009 Rename docs/tutorials/05_scm_multiobjective.ipynb → docs/tutorials/06_scm_multiobjective.ipynb
- [x] T010 Rename docs/tutorials/06_evolvable_reproduction_protocols.ipynb → docs/tutorials/07_evolvable_reproduction.ipynb

**Checkpoint**: All tutorials renumbered. Content changes can now begin.

---

## Phase 3: User Story 1 — First-Time User Runs Experiment via UnifiedConfig (Priority: P1) 🎯 MVP

**Goal**: README quickstart leads users directly to UnifiedConfig + create_engine() as the canonical pattern.

**Independent Test**: A new user can copy the README quickstart and run a complete experiment.

### Implementation for User Story 1

- [x] T011 [US1] Rewrite the quickstart/getting-started section of README.md with a complete UnifiedConfig + create_engine() example that runs end-to-end (using genome_type="vector", a simple benchmark evaluator, and seed=42)
- [x] T012 [US1] Add a "Representations" section to README.md listing all available genome types (vector, sequence, graph, scm, network) with their genome_type string and representative genome_params dict
- [x] T013 [US1] Add a "Built-in Registry" section to README.md listing all registered operators (selection: tournament/roulette/rank/crowded_tournament; crossover: single_point/two_point/uniform/sbx/blend; mutation: gaussian/uniform/polynomial/creep), evaluators (benchmark/function/llm_judge/ground_truth/scm/rl), callbacks (logging/checkpoint/print/history), and genomes (vector/sequence/graph/scm/network)
- [x] T014 [US1] Remove or update any existing README sections that show manual EvolutionEngine construction as the primary pattern, replacing them with UnifiedConfig + create_engine() references
- [x] T015 [US1] Validate README quickstart code by extracting and running it as a standalone Python script

**Checkpoint**: README quickstart is UnifiedConfig-first and runs correctly.

---

## Phase 4: User Story 2 — Notebook Tutorials Teach Concepts via Framework Methods (Priority: P1)

**Goal**: Every tutorial uses UnifiedConfig + create_engine() and framework API operators. Concept explanations stay in markdown cells; all code uses framework API.

**Independent Test**: Each rewritten notebook runs end-to-end without errors using framework API only.

### Implementation for User Story 2

- [x] T016 [US2] Rewrite docs/tutorials/01_unified_config.ipynb — ensure it serves as the foundational tutorial: covers UnifiedConfig fields, create_engine(), running an experiment, config serialization/hashing, and sub-configs (stopping, tracking, ERP, multiobjective). Add a prerequisites cell.
- [x] T017 [US2] Rewrite docs/tutorials/02_vector_genome.ipynb — replace all hand-rolled operator code with UnifiedConfig (genome_type="vector", genome_params with bounds/size) + create_engine(). Keep markdown cells explaining continuous optimization concepts. Add prerequisites cell.
- [x] T018 [US2] Rewrite docs/tutorials/03_sequence_genome.ipynb — replace manual EvolutionEngine construction with UnifiedConfig (genome_type="sequence", genome_params with alphabet/lengths) + create_engine(). Keep markdown explaining GP/sequence concepts. Add prerequisites cell.
- [x] T019 [US2] Rewrite docs/tutorials/04_graph_genome_neat.ipynb — replace manual construction with UnifiedConfig (genome_type="graph", genome_params for NEAT) + create_engine(). Keep markdown explaining topology evolution concepts. Add prerequisites cell with note about optional dependencies.
- [x] T020 [US2] Rewrite docs/tutorials/05_rl_neuroevolution.ipynb — replace manual construction with UnifiedConfig (evaluator="rl", evaluator_params for environment) + create_engine(). Keep markdown explaining neuroevolution concepts. Add prerequisites cell with note about RL optional dependencies (gymnasium).
- [x] T021 [US2] Rewrite docs/tutorials/06_scm_multiobjective.ipynb — replace manual construction with UnifiedConfig (genome_type="scm").with_multiobjective() + create_engine(). Keep markdown explaining causal discovery and NSGA-II concepts. Add prerequisites cell.
- [x] T022 [US2] Rewrite docs/tutorials/07_evolvable_reproduction.ipynb — replace manual ERPEngine construction with UnifiedConfig.with_erp() + create_engine() for all declaratively-expressible features. Add a clearly-labeled "Advanced: Manual Override" section for custom matchability functions and individual-level protocol assignment that cannot be expressed declaratively. Keep markdown explaining sexual selection and speciation concepts. Add prerequisites cell.
- [x] T023 [US2] Update docs/tutorials/README.md — rewrite the learning path to list 01_unified_config as the starting point, update all notebook descriptions to match rewritten content, update the numbering, and ensure the ML-to-EA terminology table is still accurate.
- [x] T024 [US2] Run all 7 tutorial notebooks via jupyter nbconvert --execute to verify they execute without errors

**Checkpoint**: All tutorials use UnifiedConfig + create_engine(), retain concept explanations in markdown, and execute cleanly.

---

## Phase 5: User Story 3 — Examples All Use UnifiedConfig (Priority: P2)

**Goal**: All example scripts use UnifiedConfig as their configuration mechanism.

**Independent Test**: Each example runs without errors using `python examples/<name>.py`.

### Implementation for User Story 3

- [x] T025 [P] [US3] Rewrite examples/sexual_selection.py — convert to UnifiedConfig.with_erp() + create_engine(), add "Advanced: Manual Override" section for custom protocol assignment that can't be expressed declaratively
- [x] T026 [P] [US3] Rewrite examples/protocol_evolution.py — convert to UnifiedConfig.with_erp() + create_engine(), add "Advanced: Manual Override" section for protocol tracking that requires imperative code
- [x] T027 [P] [US3] Rewrite examples/speciation_demo.py — convert to UnifiedConfig.with_erp() + create_engine(), add "Advanced: Manual Override" section for cosine similarity matchability
- [x] T028 [US3] Review examples/mlflow_tracking_demo.py — already uses UnifiedConfig; verify it's current, update docstring if needed, ensure seed value is set
- [x] T029 [US3] Run all 4 examples to verify they execute without errors

**Checkpoint**: All examples use UnifiedConfig and run correctly.

---

## Phase 6: User Story 4 — Guides Cover Advanced Declarative Patterns (Priority: P2)

**Goal**: Updated ERP guide + new Advanced Configuration Guide with declarative patterns for all features.

**Independent Test**: All code snippets in guides can be extracted and executed.

### Implementation for User Story 4

- [x] T030 [P] [US4] Rewrite docs/guides/erp-best-practices.md — convert primary code examples to UnifiedConfig + ERPSettings. Move existing manual ERPEngine examples to one "Advanced: Direct ERPEngine Usage" appendix section for power users. Update all advice to reference declarative config.
- [x] T031 [P] [US4] Create docs/guides/advanced-configuration.md with sections for: (1) Custom Evaluators — how to register and reference in UnifiedConfig via EvaluatorRegistry, (2) Custom Callbacks — how to register and reference in UnifiedConfig via CallbackRegistry, (3) Multi-Objective Configuration — UnifiedConfig.with_multiobjective() with MultiObjectiveConfig, (4) Meta-Evolution Configuration — MetaEvolutionConfig in UnifiedConfig, (5) MLflow Tracking — TrackingConfig in UnifiedConfig with all tracking categories. Each section must have a complete runnable code example using UnifiedConfig.
- [x] T032 [US4] Review all guide code snippets for correctness — verify import paths, config field names, and registry names match current framework API

**Checkpoint**: Guides provide comprehensive declarative patterns for all advanced features.

---

## Phase 7: User Story 5 — Docstrings Reference UnifiedConfig (Priority: P3)

**Goal**: All public API docstrings include UnifiedConfig usage examples and/or registry name references.

**Independent Test**: Docstring code examples can be validated as correct Python.

### Implementation for User Story 5

- [x] T033 [P] [US5] Update docstring in evolve/core/engine.py (EvolutionEngine class) — primary example should show create_engine(config) pattern, not manual construction
- [x] T034 [P] [US5] Update docstring in evolve/factory/engine.py (create_engine function) — ensure example shows full UnifiedConfig usage with all major config fields
- [x] T035 [P] [US5] Update docstrings in evolve/representation/vector.py (VectorGenome) — add genome_type="vector" and representative genome_params dict for UnifiedConfig
- [x] T036 [P] [US5] Update docstrings in evolve/representation/sequence.py (SequenceGenome) — add genome_type="sequence" and representative genome_params dict
- [x] T037 [P] [US5] Update docstrings in evolve/representation/graph.py (GraphGenome) — add genome_type="graph" and representative genome_params dict
- [x] T038 [P] [US5] Update docstrings in evolve/representation/scm.py (SCMGenome) — add genome_type="scm" and representative genome_params dict
- [x] T039 [P] [US5] Update docstrings in evolve/core/operators/selection.py — add registry name strings (tournament, roulette, rank, crowded_tournament) to each operator class docstring
- [x] T040 [P] [US5] Update docstrings in evolve/core/operators/crossover.py or evolve/core/operators/token_crossover.py — add registry name strings (single_point, two_point, uniform, sbx, blend) to each operator class
- [x] T041 [P] [US5] Update docstrings in evolve/core/operators/mutation.py or evolve/core/operators/token_mutation.py — add registry name strings (gaussian, uniform, polynomial, creep) to each operator class
- [x] T042 [P] [US5] Update docstrings in evolve/registry/operators.py, evolve/registry/evaluators.py, evolve/registry/callbacks.py, evolve/registry/genomes.py — ensure each registry class docstring lists all built-in entries with their registration names

**Checkpoint**: All public API docstrings reference UnifiedConfig patterns and registry names.

---

## Phase 8: User Story 6 — API Reference Docs Build and Are Current (Priority: P3)

**Goal**: Sphinx docs build cleanly with all public modules documented.

**Independent Test**: `cd docs && make html` succeeds with zero errors and zero warnings.

### Implementation for User Story 6

- [x] T043 [US6] Run Sphinx build (cd docs && make html) and capture all errors/warnings
- [x] T044 [US6] Fix any missing module references in docs/api/*.rst files — ensure all public modules in evolve/ have corresponding .rst entries
- [x] T045 [US6] Fix any broken cross-references or missing imports in Sphinx configuration (docs/conf.py) — handle optional-dependency modules with autodoc mock imports
- [x] T046 [US6] Re-run Sphinx build and verify zero errors and zero warnings

**Checkpoint**: API docs build cleanly and all public modules are documented.

---

## Phase 9: User Story 7 — Tutorials README Learning Path (Priority: P2)

**Goal**: Tutorials README accurately indexes all renumbered notebooks.

**Independent Test**: Every notebook referenced in README exists and description matches content.

*Note: This is largely covered by T023 in Phase 4. This phase handles any remaining cross-references.*

### Implementation for User Story 7

- [x] T047 [US7] Final review of docs/tutorials/README.md — verify every referenced notebook file exists with correct filename, every description accurately reflects rewritten content, and the learning path starts with 01_unified_config
- [x] T048 [US7] Check for any other files that reference tutorial filenames by number (docs/conf.py, docs/index.rst, etc.) and update those references to match new numbering

**Checkpoint**: Tutorials README is fully consistent with actual notebook files.

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final validation across all documentation artifacts

- [x] T049 Review and update tutorial utility files (docs/tutorials/tutorial_utils.py, docs/tutorials/mermaid_renderer.py, docs/tutorials/erp_test_data.py) — ensure they don't reference old patterns or tutorial numbers
- [x] T050 Check CONTRIBUTING.md and docs/index.rst for references to old tutorial numbers or manual construction patterns; update if found
- [x] T051 Run all 7 tutorial notebooks via jupyter nbconvert --execute to confirm they all pass
- [x] T052 Run all 4 examples (python examples/*.py) to confirm they all pass
- [x] T053 Run Sphinx build (cd docs && make html) to confirm zero errors/warnings
- [x] T054 Run existing test suite (pytest tests/) to confirm no regressions from docstring changes
- [x] T055 Final cross-check: search all documentation files for any remaining references to manual EvolutionEngine/ERPEngine construction outside of "Advanced" sections — flag and fix any found

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 audit completion — BLOCKS all content changes
- **US1 README (Phase 3)**: Depends on Phase 1 audit (T001)
- **US2 Tutorials (Phase 4)**: Depends on Phase 2 renumbering AND Phase 1 audit (T001, T002)
- **US3 Examples (Phase 5)**: Depends on Phase 1 audit (T001, T003)
- **US4 Guides (Phase 6)**: Depends on Phase 1 audit (T001)
- **US5 Docstrings (Phase 7)**: Depends on Phase 1 audit (T001)
- **US6 API Docs (Phase 8)**: Depends on Phase 7 docstring updates
- **US7 Tutorials README (Phase 9)**: Depends on Phase 4 tutorial rewrites
- **Polish (Phase 10)**: Depends on all prior phases

### User Story Dependencies

- **US1 (README)**: Independent after Phase 1
- **US2 (Tutorials)**: Depends on Phase 2 renumbering
- **US3 (Examples)**: Independent after Phase 1 — can run parallel with US1, US2, US4
- **US4 (Guides)**: Independent after Phase 1 — can run parallel with US1, US2, US3
- **US5 (Docstrings)**: Independent after Phase 1 — can run parallel with US1-US4
- **US6 (API Docs)**: Depends on US5 (docstrings must be updated first)
- **US7 (Tutorials README)**: Depends on US2 (tutorials must be rewritten first)

### Parallel Opportunities

After Phase 2 (renumbering), these can all proceed in parallel:
- Phase 3 (README), Phase 4 (Tutorials), Phase 5 (Examples), Phase 6 (Guides), Phase 7 (Docstrings)

Within phases, all tasks marked [P] can be done in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Audit
2. Complete Phase 2: Renumber tutorials
3. Complete Phase 3: README quickstart (US1)
4. **STOP and VALIDATE**: New user can run quickstart

### Incremental Delivery

1. Setup + Foundational → Renumbering done
2. US1 (README) → Quickstart works (MVP!)
3. US2 (Tutorials) → All tutorials use UnifiedConfig
4. US3 (Examples) + US4 (Guides) → Complete declarative docs
5. US5 (Docstrings) + US6 (API Docs) → Polished API surface
6. US7 (Tutorials README) + Polish → Everything consistent

---

## Notes

- This is a documentation-only feature — no framework source code changes except docstrings
- All code examples must use seed=42 (or another explicit seed) per Constitution V
- ERP "Advanced: Manual Override" sections document known declarative limitations, not bugs
- Tutorial concept explanations stay in markdown cells; only code cells are refactored
- Commit after each completed phase for clean git history
