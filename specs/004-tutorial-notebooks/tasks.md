# Tasks: Tutorial Notebooks for Evolve Framework

**Input**: Design documents from `/specs/004-tutorial-notebooks/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

**Tests**: Included per FR-001 to FR-054 requirements (tutorial_utils unit tests + notebook execution validation)

**Organization**: Tasks grouped by user story to enable independent implementation and testing

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US6)
- Exact file paths included in descriptions

---

## Phase 1: Setup (Project Infrastructure)

**Purpose**: Create directory structure and initialize shared infrastructure

- [X] T001 Create tutorial directory structure: `docs/tutorials/`, `docs/tutorials/utils/`, `tests/tutorials/`
- [X] T002 [P] Create `docs/tutorials/utils/__init__.py` with module exports
- [X] T003 [P] Create `tests/tutorials/__init__.py` for test discovery
- [X] T004 [P] Add tutorial dependencies to `pyproject.toml` (beautiful-mermaid, plotly, gymnasium, papermill)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core shared module infrastructure that ALL notebooks depend on

**⚠️ CRITICAL**: No notebook work can begin until this phase is complete

### Data Structures (from data-model.md)

- [X] T005 [P] Implement `BenchmarkFunction` dataclass in `docs/tutorials/utils/tutorial_utils.py`
- [X] T006 [P] Implement `SymbolicRegressionData` dataclass in `docs/tutorials/utils/tutorial_utils.py`
- [X] T007 [P] Implement `CausalDAGData` dataclass with `edge_accuracy()` method in `docs/tutorials/utils/tutorial_utils.py`
- [X] T008 [P] Implement `EvolutionHistory` dataclass with `from_callback_logs()` method in `docs/tutorials/utils/tutorial_utils.py`
- [X] T009 [P] Implement `SpeciesHistory` dataclass with `to_stacked_area_data()` method in `docs/tutorials/utils/tutorial_utils.py`
- [X] T010 [P] Implement `ParetoFront` dataclass with `dominates()` and `crowding_distances()` methods in `docs/tutorials/utils/tutorial_utils.py`
- [X] T011 [P] Implement `TerminologyEntry` dataclass and `TERMINOLOGY_GLOSSARY` constant in `docs/tutorials/utils/tutorial_utils.py`
- [X] T012 [P] Implement `IslandConfig` and `MigrationEvent` dataclasses in `docs/tutorials/utils/tutorial_utils.py`
- [X] T013 [P] Implement `BenchmarkResult` dataclass with `speedup_vs()` method in `docs/tutorials/utils/tutorial_utils.py` (used by T064, T081, T101, T121, T143 for GPU benchmarks)

### Terminology Glossary (FR-009)

- [X] T014 Implement `get_glossary()` function returning glossary dict in `docs/tutorials/utils/tutorial_utils.py`
- [X] T015 Implement `print_glossary_table()` for Jupyter display in `docs/tutorials/utils/tutorial_utils.py`

### Mermaid Rendering (FR-006)

- [X] T016 Implement `render_mermaid()` function with github-light theme default in `docs/tutorials/utils/tutorial_utils.py`
- [X] T017 [P] Define `EVOLUTIONARY_LOOP_DIAGRAM` constant (standard EA loop) in `docs/tutorials/utils/tutorial_utils.py`
- [X] T018 [P] Define `GENOME_PHENOTYPE_DIAGRAM` constant (encoding/decoding pipeline) in `docs/tutorials/utils/tutorial_utils.py`
- [X] T019 [P] Define `ISLAND_MODEL_DIAGRAM` constant (island topology) in `docs/tutorials/utils/tutorial_utils.py`

### Statistical Utilities (FR-010)

- [X] T020 Implement `compute_population_stats()` for mean, std, percentiles in `docs/tutorials/utils/tutorial_utils.py`

**Checkpoint**: Core infrastructure ready - user story implementation can now begin

---

## Phase 3: User Story 6 - Shared Tutorial Utilities (Priority: P2) 🎯 FOUNDATION

**Goal**: Complete `tutorial_utils.py` module with all data generators and visualization functions

**Independent Test**: Import module and verify all data generators produce correctly shaped output with configurable noise and seeds

### Tests for User Story 6

- [X] T021 [P] [US6] Create test file `tests/tutorials/test_tutorial_utils.py` with pytest fixtures
- [X] T022 [P] [US6] Test benchmark functions (sphere, rastrigin, rosenbrock, ackley) produce correct optima in `tests/tutorials/test_tutorial_utils.py`
- [X] T023 [P] [US6] Test polynomial/trigonometric/composite data generators respect seed and noise params in `tests/tutorials/test_tutorial_utils.py`
- [X] T024 [P] [US6] Test causal DAG generator produces valid acyclic graphs in `tests/tutorials/test_tutorial_utils.py`
- [X] T025 [P] [US6] Test EvolutionHistory/SpeciesHistory/ParetoFront dataclass methods in `tests/tutorials/test_tutorial_utils.py`
- [X] T025a [P] [US6] Add visualization smoke tests: verify plot functions return valid Figure objects without errors in `tests/tutorials/test_tutorial_utils.py`
- [X] T025b [P] [US6] Test `BenchmarkResult.speedup_vs()` method computes correct speedup ratios in `tests/tutorials/test_tutorial_utils.py`

### Implementation: Benchmark Functions (FR-001)

- [X] T026 [US6] Implement `sphere_function()` in `docs/tutorials/utils/tutorial_utils.py`
- [X] T027 [P] [US6] Implement `rastrigin_function()` in `docs/tutorials/utils/tutorial_utils.py`
- [X] T028 [P] [US6] Implement `rosenbrock_function()` in `docs/tutorials/utils/tutorial_utils.py`
- [X] T029 [P] [US6] Implement `ackley_function()` in `docs/tutorials/utils/tutorial_utils.py`
- [X] T030 [US6] Implement `get_benchmark()` factory function in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Symbolic Regression Data (FR-002)

- [X] T031 [US6] Implement `generate_polynomial_data()` with train/test split in `docs/tutorials/utils/tutorial_utils.py`
- [X] T032 [P] [US6] Implement `generate_trigonometric_data()` in `docs/tutorials/utils/tutorial_utils.py`
- [X] T033 [P] [US6] Implement `generate_composite_data()` with complexity levels in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Causal Data Generation (FR-003, FR-004, FR-005)

- [X] T034 [US6] Implement `generate_causal_dag_data()` with hidden_fraction support in `docs/tutorials/utils/tutorial_utils.py`
- [X] T035 [P] [US6] Implement `generate_chain_dag_data()` for simple causal chains in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Fitness Plots (FR-007)

- [X] T036 [US6] Implement `plot_fitness_history()` with best/mean/worst lines and std band in `docs/tutorials/utils/tutorial_utils.py`
- [X] T037 [P] [US6] Implement `plot_fitness_comparison()` for multiple runs in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Diversity Visualization (FR-008)

- [X] T038 [US6] Implement `plot_population_diversity()` with PCA/t-SNE options in `docs/tutorials/utils/tutorial_utils.py`
- [X] T039 [P] [US6] Implement `plot_diversity_over_generations()` in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Pareto Visualization (FR-048, FR-049, FR-050)

- [X] T040 [US6] Implement `plot_pareto_2d_projections()` with 3 subplots in `docs/tutorials/utils/tutorial_utils.py`
- [X] T041 [P] [US6] Implement `plot_pareto_3d_interactive()` with plotly scatter3d in `docs/tutorials/utils/tutorial_utils.py`
- [X] T042 [P] [US6] Implement `plot_pareto_evolution()` for animated front progression in `docs/tutorials/utils/tutorial_utils.py`
- [X] T043 [US6] Implement `plot_crowding_distance_visual()` for FR-049 intuition in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Speciation Visualization (FR-038)

- [X] T044 [US6] Implement `plot_species_stacked_area()` with max_species_shown limit in `docs/tutorials/utils/tutorial_utils.py`
- [X] T045 [P] [US6] Implement `plot_species_phylogeny()` optional advanced view in `docs/tutorials/utils/tutorial_utils.py`

### Implementation: Causal Graph Visualization (FR-054)

- [X] T046 [US6] Implement `plot_causal_graph_comparison()` with ground truth overlay in `docs/tutorials/utils/tutorial_utils.py`

**Checkpoint**: tutorial_utils.py complete - all notebooks can now use shared utilities

---

## Phase 4: User Story 1 - VectorGenome Tutorial (Priority: P1) 🎯 MVP

**Goal**: ML practitioner learns continuous optimization with EA, relates to gradient-based optimization

**Independent Test**: Run notebook end-to-end, verify Rastrigin/Rosenbrock converge to within 1% of optima within 100 generations

### Tests for User Story 1

- [X] T047 [P] [US1] Create notebook validation test `tests/tutorials/test_notebook_execution.py` with papermill
- [X] T048 [P] [US1] Add VectorGenome notebook execution test to `tests/tutorials/test_notebook_execution.py`

### Implementation for User Story 1

- [X] T049 [US1] Create `docs/tutorials/01_vector_genome.ipynb` skeleton with section headers
- [X] T050 [US1] Write "Introduction & Learning Objectives" section in `docs/tutorials/01_vector_genome.ipynb`
- [X] T051 [US1] Write "EA Primer with ML Analogies" section (FR-012) with terminology table in `docs/tutorials/01_vector_genome.ipynb`
- [X] T052 [US1] Add 3+ mermaid diagrams (FR-014): genome-phenotype, evolutionary loop, population flow in `docs/tutorials/01_vector_genome.ipynb`
- [X] T053 [US1] Write "Problem Setup" section: Rastrigin/Rosenbrock benchmark intro in `docs/tutorials/01_vector_genome.ipynb`
- [X] T054 [US1] Implement "Genome-to-Phenotype Mapping" section (FR-028) showing identity mapping in `docs/tutorials/01_vector_genome.ipynb`
- [X] T055 [US1] Implement "Mutation Operators" section (FR-029) with adaptive Gaussian mutation in `docs/tutorials/01_vector_genome.ipynb`
- [X] T056 [US1] Implement "Crossover Operators" section (FR-030) comparing uniform vs SBX with visuals in `docs/tutorials/01_vector_genome.ipynb`
- [X] T057 [US1] Implement "Running Evolution" section (FR-027) with explicit config defaults (FR-013) in `docs/tutorials/01_vector_genome.ipynb`
- [X] T058 [US1] Add convergence visualization (FR-015) using plot_fitness_history() in `docs/tutorials/01_vector_genome.ipynb`
- [X] T059 [US1] Add population diversity metrics (FR-016) using plot_population_diversity() in `docs/tutorials/01_vector_genome.ipynb`
- [X] T060 [US1] Implement callback usage for logging/early stopping (FR-017) in `docs/tutorials/01_vector_genome.ipynb`
- [X] T061 [US1] Implement checkpointing section (FR-018) save/resume evolution in `docs/tutorials/01_vector_genome.ipynb`
- [X] T062 [US1] Write "Island Model Parallelism" section (FR-020, FR-021, FR-022) with 4×50 config in `docs/tutorials/01_vector_genome.ipynb`
- [X] T063 [US1] Add statistical comparison single vs island (FR-023) in `docs/tutorials/01_vector_genome.ipynb`
- [X] T064 [US1] Write "GPU Acceleration" section (FR-024, FR-025, FR-026) with benchmarks in `docs/tutorials/01_vector_genome.ipynb`
- [X] T065 [US1] Write "Extensions & Next Steps" section (FR-019) linking to other tutorials in `docs/tutorials/01_vector_genome.ipynb`

**Checkpoint**: User Story 1 complete - VectorGenome tutorial fully functional as MVP

---

## Phase 5: User Story 2 - SequenceGenome Tutorial (Priority: P2)

**Goal**: ML practitioner learns genetic programming, evolves symbolic expressions

**Independent Test**: Evolve expressions that rediscover known polynomials within 5% error on test data

### Tests for User Story 2

- [X] T066 [P] [US2] Add SequenceGenome notebook execution test to `tests/tutorials/test_notebook_execution.py`

### Implementation for User Story 2

- [X] T067 [US2] Create `docs/tutorials/02_sequence_genome.ipynb` skeleton with section headers
- [X] T068 [US2] Write "Introduction & Learning Objectives" section in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T069 [US2] Write "EA Primer: Variable-Length Representations" section (FR-012) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T070 [US2] Add 3+ mermaid diagrams (FR-014): expression tree, subtree crossover, evaluation pipeline in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T071 [US2] Write "Problem Setup" section: symbolic regression intro with synthetic data in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T072 [US2] Implement "Expression Tree Visualization" section (FR-032) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T073 [US2] Implement "Subtree Crossover" section (FR-032) with visual comparison to vector crossover in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T074 [US2] Implement "Bloat Control" section (FR-033) parsimony pressure, depth limits in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T075 [US2] Implement "Running Evolution" section (FR-031) with config defaults in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T076 [US2] Add convergence visualization (FR-015) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T077 [US2] Add test set evaluation (FR-034) showing generalization in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T078 [US2] Implement callback usage (FR-017) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T079 [US2] Implement checkpointing section (FR-018) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T080 [US2] Write "Island Model" section (FR-020, FR-021, FR-022, FR-023) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T081 [US2] Write "GPU Acceleration" section (FR-024, FR-025, FR-026) in `docs/tutorials/02_sequence_genome.ipynb`
- [X] T082 [US2] Write "Extensions & Next Steps" section (FR-019) in `docs/tutorials/02_sequence_genome.ipynb`

**Checkpoint**: User Story 2 complete - SequenceGenome tutorial fully functional

---

## Phase 6: User Story 3 - GraphGenome/NEAT Tutorial (Priority: P3)

**Goal**: ML practitioner learns topology evolution with NEAT speciation

**Independent Test**: Evolve networks solving XOR (100% accuracy) with visible species formation

### Tests for User Story 3

- [X] T083 [P] [US3] Add GraphGenome/NEAT notebook execution test to `tests/tutorials/test_notebook_execution.py`

### Implementation for User Story 3

- [X] T084 [US3] Create `docs/tutorials/03_graph_genome_neat.ipynb` skeleton with section headers
- [X] T085 [US3] Write "Introduction & Learning Objectives" section in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T086 [US3] Write "EA Primer: Evolving Structure" section (FR-012) with NAS analogy in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T087 [US3] Add 3+ mermaid diagrams (FR-014): NEAT encoding, speciation, adjusted fitness in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T088 [US3] Write "Problem Setup" section: XOR classification requiring hidden nodes in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T089 [US3] Implement "NEAT Encoding" section: nodes, connections, innovation numbers in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T090 [US3] Implement "Speciation with Genomic Distance" section (FR-035) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T091 [US3] Implement "Compatibility Threshold Tuning" section (FR-036) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T092 [US3] Implement "Adjusted Fitness (Fitness Sharing)" section (FR-037) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T093 [US3] Add species stacked area visualization (FR-038) using plot_species_stacked_area() in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T094 [US3] Add optional phylogenetic tree view (FR-038) using plot_species_phylogeny() in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T095 [US3] Implement "Running NEAT Evolution" section (FR-039) with XOR evaluation in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T096 [US3] Add network architecture visualization (FR-040) showing nodes, edges, weights in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T097 [US3] Add convergence visualization (FR-015) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T098 [US3] Implement callback usage (FR-017) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T099 [US3] Implement checkpointing section (FR-018) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T100 [US3] Write "Island Model" section (FR-020, FR-021, FR-022, FR-023) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T101 [US3] Write "GPU Acceleration" section (FR-024, FR-025, FR-026) in `docs/tutorials/03_graph_genome_neat.ipynb`
- [X] T102 [US3] Write "Extensions & Next Steps" section (FR-019) in `docs/tutorials/03_graph_genome_neat.ipynb`

**Checkpoint**: User Story 3 complete - NEAT tutorial with full speciation functional

---

## Phase 7: User Story 4 - RL/Neuroevolution Tutorial (Priority: P4)

**Goal**: RL practitioner learns evolution as derivative-free policy optimizer

**Independent Test**: Evolve CartPole-v1 policy achieving average return >475 over 10 episodes

### Tests for User Story 4

- [X] T103 [P] [US4] Add RL notebook execution test to `tests/tutorials/test_notebook_execution.py`

### Implementation for User Story 4

- [X] T104 [US4] Create `docs/tutorials/04_rl_neuroevolution.ipynb` skeleton with section headers
- [X] T105 [US4] Write "Introduction & Learning Objectives" section in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T106 [US4] Write "EA Primer: Evolution vs Gradients" section (FR-012) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T107 [US4] Add 3+ mermaid diagrams (FR-014): policy decoding, episode rollout, fitness aggregation in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T108 [US4] Write "Problem Setup" section: CartPole-v1 environment intro in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T109 [US4] Implement "Gymnasium Integration" section (FR-041) with env setup in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T110 [US4] Implement "Policy Decoding" section (FR-042) genome-to-neural-network in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T111 [US4] Implement "Episode Rollout as Fitness" section (FR-043) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T112 [US4] Implement "Fitness Aggregation" section (FR-044) bias-variance tradeoff in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T113 [US4] Implement "Running Policy Evolution" section (FR-045) for CartPole-v1 in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T114 [US4] Add policy behavior rendering/video (FR-046) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T115 [US4] Add comparison vs random search with statistical significance in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T116 [US4] Add optional LunarLander-v2 advanced section (FR-045) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T117 [US4] Add convergence visualization (FR-015) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T118 [US4] Implement callback usage (FR-017) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T119 [US4] Implement checkpointing section (FR-018) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T120 [US4] Write "Island Model" section (FR-020, FR-021, FR-022, FR-023) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T121 [US4] Write "GPU Acceleration" section (FR-024, FR-025, FR-026) in `docs/tutorials/04_rl_neuroevolution.ipynb`
- [X] T122 [US4] Write "Extensions & Next Steps" section (FR-019) in `docs/tutorials/04_rl_neuroevolution.ipynb`

**Checkpoint**: User Story 4 complete - RL/Neuroevolution tutorial fully functional

---

## Phase 8: User Story 5 - SCMGenome Multi-Objective Tutorial (Priority: P5)

**Goal**: ML practitioner learns causal discovery with NSGA-II multi-objective optimization

**Independent Test**: Recover >80% of causal edges from 5-variable synthetic DAG

### Tests for User Story 5

- [X] T123 [P] [US5] Add SCM notebook execution test to `tests/tutorials/test_notebook_execution.py`

### Implementation for User Story 5

- [X] T124 [US5] Create `docs/tutorials/05_scm_multiobjective.ipynb` skeleton with section headers
- [X] T125 [US5] Write "Introduction & Learning Objectives" section in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T126 [US5] Write "EA Primer: Multi-Objective Optimization" section (FR-012) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T127 [US5] Add 3+ mermaid diagrams (FR-014): SCM encoding, NSGA-II flow, Pareto concepts in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T128 [US5] Write "Problem Setup" section: causal discovery motivation in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T129 [US5] Implement "Pareto Dominance" section (FR-048) with 2D projections + 3D plotly in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T130 [US5] Implement "Crowding Distance" section (FR-049) with visual examples in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T131 [US5] Implement "NSGA-II Algorithm" section (FR-047) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T132 [US5] Implement "Three Objectives" section (FR-051): data fit, sparsity, simplicity in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T133 [US5] Implement "Running Multi-Objective Evolution" section with 5-variable DAG in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T134 [US5] Add Pareto front evolution visualization (FR-050) using plot_pareto_evolution() in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T135 [US5] Add interactive 3D Pareto exploration (FR-048) using plot_pareto_3d_interactive() in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T136 [US5] Implement "Weighted-Sum vs Multi-Objective" comparison (FR-052) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T137 [US5] Implement "Latent Variable Discovery" section (FR-053) with hidden columns in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T138 [US5] Add causal graph comparison visualization (FR-054) with edge-level accuracy in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T139 [US5] Add convergence visualization (FR-015) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T140 [US5] Implement callback usage (FR-017) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T141 [US5] Implement checkpointing section (FR-018) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T142 [US5] Write "Island Model" section (FR-020, FR-021, FR-022, FR-023) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T143 [US5] Write "GPU Acceleration" section (FR-024, FR-025, FR-026) in `docs/tutorials/05_scm_multiobjective.ipynb`
- [X] T144 [US5] Write "Extensions & Next Steps" section (FR-019) in `docs/tutorials/05_scm_multiobjective.ipynb`

**Checkpoint**: User Story 5 complete - SCM multi-objective tutorial fully functional

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, validation, and final integration

- [X] T145 [P] Create `docs/tutorials/README.md` with learning path and notebook index
- [X] T146 [P] Add runtime estimates to each notebook's introduction section
- [X] T147 [P] Add edge case handling: missing GPU warnings, Gymnasium install instructions, memory guidance
- [X] T147a [P] Implement GPU graceful degradation pattern: `check_gpu_available()` utility with fallback logic and informative warnings in `docs/tutorials/utils/tutorial_utils.py`
- [X] T148 [P] Add accessibility check: clear headings, alt text for diagrams, code comments (SC-009)
- [X] T149 [P] Verify all notebooks include 10+ ML-to-EA terminology mappings (SC-010)
- [X] T149a Validate notebook import independence: verify each notebook executes without importing from other tutorial notebooks (FR-011)
- [X] T150 Run all notebook execution tests via papermill
- [X] T151 Run quickstart.md validation workflow
- [X] T152 Final review: verify all FR-001 to FR-054 requirements are implemented

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (data structures, helpers)
    ↓
Phase 3: US6 - tutorial_utils.py (BLOCKS all notebooks)
    ↓
┌─────────────────────────────────────────────────────┐
│  Notebooks can proceed in parallel after Phase 3    │
├─────────────────────────────────────────────────────┤
│ Phase 4: US1 - VectorGenome (P1) 🎯 MVP            │
│ Phase 5: US2 - SequenceGenome (P2)                 │
│ Phase 6: US3 - GraphGenome/NEAT (P3)               │
│ Phase 7: US4 - RL/Neuroevolution (P4)              │
│ Phase 8: US5 - SCMGenome (P5)                      │
└─────────────────────────────────────────────────────┘
    ↓
Phase 9: Polish (after all desired notebooks complete)
```

### User Story Dependencies

- **User Story 6 (tutorial_utils.py)**: MUST complete before any notebook
- **User Story 1 (VectorGenome)**: Independent after US6 - validates all shared code
- **User Story 2 (SequenceGenome)**: Independent after US6
- **User Story 3 (GraphGenome/NEAT)**: Independent after US6
- **User Story 4 (RL/Neuroevolution)**: Independent after US6
- **User Story 5 (SCMGenome)**: Independent after US6

### Within Each User Story

1. Tests written first (verify they fail)
2. Infrastructure/setup tasks
3. Core content implementation
4. Visualization integration
5. Advanced sections (island model, GPU)
6. Extensions section

### Parallel Opportunities

All tasks marked `[P]` within the same phase can run in parallel:

- Phase 2: All dataclass implementations (T005-T013) can run in parallel
- Phase 3: Data generator tests (T021-T025) can run in parallel
- Phase 3: Benchmark functions (T026-T029) can run in parallel
- Phase 3: Visualization functions can mostly run in parallel
- **Cross-Phase**: After Phase 3, all 5 notebooks can be developed in parallel

---

## Parallel Example: User Story 6 (tutorial_utils.py)

```bash
# Launch all dataclass implementations together (Phase 2):
T005: BenchmarkFunction dataclass
T006: SymbolicRegressionData dataclass
T007: CausalDAGData dataclass
T008: EvolutionHistory dataclass
T009: SpeciesHistory dataclass
T010: ParetoFront dataclass

# Launch all benchmark functions together (Phase 3):
T026-T029: sphere, rastrigin, rosenbrock, ackley

# Launch visualization functions in parallel:
T040-T042: Pareto visualization functions
T036-T037: Fitness plot functions
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (dataclasses)
3. Complete Phase 3: US6 (tutorial_utils.py)
4. Complete Phase 4: US1 (VectorGenome)
5. **STOP and VALIDATE**: Run notebook end-to-end, verify convergence criteria
6. Demo/review if ready

### Incremental Delivery

1. Setup + Foundational + tutorial_utils → Foundation ready
2. Add VectorGenome notebook → Test independently → **MVP Demo!**
3. Add SequenceGenome notebook → Test independently → Incremental release
4. Add GraphGenome/NEAT notebook → Test independently → Incremental release
5. Add RL/Neuroevolution notebook → Test independently → Incremental release
6. Add SCMGenome notebook → Test independently → Full release
7. Polish phase → Final documentation and validation

### Parallel Team Strategy

With multiple developers after Phase 3 (tutorial_utils.py) completes:

- Developer A: User Story 1 (VectorGenome) - P1
- Developer B: User Story 2 (SequenceGenome) - P2
- Developer C: User Story 3 (GraphGenome/NEAT) - P3
- Developer D: User Story 4 (RL/Neuroevolution) - P4
- Developer E: User Story 5 (SCMGenome) - P5

---

## Notes

- `[P]` tasks = different files or independent code sections, no dependencies
- `[Story]` label maps task to specific user story for traceability
- FR references link tasks to functional requirements in spec.md
- Each notebook independently completable and testable after tutorial_utils.py
- Commit after each logical task group
- Verify success criteria (SC-001 to SC-010) at each checkpoint
