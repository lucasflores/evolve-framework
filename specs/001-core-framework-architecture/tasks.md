# Tasks: Evolve Framework Core Architecture

**Input**: Design documents from `/specs/001-core-framework-architecture/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2...)
- All paths relative to repository root

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create project structure and configure development environment

- [X] T001 Create package structure: `evolve/__init__.py`, `evolve/core/__init__.py`, `evolve/representation/__init__.py`, `evolve/evaluation/__init__.py`
- [X] T002 [P] Create `pyproject.toml` with Python 3.10+ requirement, NumPy dependency, optional extras for [pytorch], [jax], [mlflow]
- [X] T003 [P] Create `tests/` directory structure mirroring `evolve/`: `tests/unit/core/`, `tests/unit/representation/`, `tests/integration/`
- [X] T004 [P] Configure pytest with hypothesis in `pyproject.toml` and create `conftest.py` with shared fixtures
- [X] T005 [P] Create `.github/workflows/ci.yml` for automated testing on push
- [X] T006 [P] Create `evolve/utils/__init__.py` and `evolve/utils/random.py` with seeded RNG utilities

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that ALL user stories depend on - MUST complete before any story work

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Implement `Fitness` dataclass in `evolve/core/types.py` per contracts/core.md (value, metadata, comparison)
- [X] T008 Implement `Individual` dataclass in `evolve/core/types.py` per contracts/core.md (genome, fitness, generation)
- [X] T009 Implement `Population` container in `evolve/core/population.py` per contracts/core.md (individuals, statistics, iteration)
- [X] T010 [P] Define `Genome` Protocol in `evolve/representation/genome.py` per contracts/representation.md (copy, eq, hash)
- [X] T011 [P] Define `Phenotype` Protocol in `evolve/representation/phenotype.py` per contracts/representation.md
- [X] T012 [P] Define `Decoder` Protocol in `evolve/representation/phenotype.py` per contracts/representation.md
- [X] T013 [P] Define `Evaluator` Protocol and `EvaluatorCapabilities` in `evolve/evaluation/evaluator.py` per contracts/evaluation.md
- [X] T014 [P] Define `SelectionOperator` Protocol in `evolve/core/operators/selection.py` per contracts/operators.md
- [X] T015 [P] Define `CrossoverOperator` Protocol in `evolve/core/operators/crossover.py` per contracts/operators.md
- [X] T016 [P] Define `MutationOperator` Protocol in `evolve/core/operators/mutation.py` per contracts/operators.md
- [X] T017 Implement `Callback` Protocol and `StoppingCriterion` Protocol in `evolve/core/callbacks.py` per contracts/core.md
- [X] T018 Create `evolve/core/operators/__init__.py` re-exporting all operator protocols

**Checkpoint**: Foundation protocols ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Classical Genetic Algorithm (Priority: P1) 🎯 MVP

**Goal**: Run a simple GA to solve continuous optimization problems using only CPU

**Independent Test**: Optimize sphere/Rastrigin functions, verify deterministic reproduction with same seed

### Implementation for User Story 1

- [X] T019 [P] [US1] Implement `VectorGenome` in `evolve/representation/vector.py` per contracts/representation.md (genes, bounds, copy, clip_to_bounds)
- [X] T020 [P] [US1] Implement `VectorIdentityDecoder` in `evolve/representation/vector.py` (genome.genes as phenotype)
- [X] T021 [P] [US1] Implement `TournamentSelection` in `evolve/core/operators/selection.py` per contracts/operators.md
- [X] T022 [P] [US1] Implement `UniformCrossover` in `evolve/core/operators/crossover.py` per contracts/operators.md
- [X] T023 [P] [US1] Implement `GaussianMutation` in `evolve/core/operators/mutation.py` per contracts/operators.md
- [X] T024 [US1] Implement `FunctionEvaluator` in `evolve/evaluation/evaluator.py` per contracts/evaluation.md
- [X] T025 [P] [US1] Implement reference benchmark functions in `evolve/evaluation/reference/functions.py` per contracts/evaluation.md (sphere, rastrigin, rosenbrock)
- [X] T026 [US1] Implement `EvolutionEngine` in `evolve/core/engine.py` per contracts/core.md (initialize, step, run, best property)
- [X] T027 [US1] Add elitism support to `EvolutionEngine` (preserve top N individuals unchanged)
- [X] T028 [US1] Add generation-boundary event emission to `EvolutionEngine` (on_generation_start, on_generation_end callbacks)
- [X] T029 [US1] Implement `GenerationLimitStopping` criterion in `evolve/core/stopping.py`
- [X] T030 [US1] Implement `FitnessThresholdStopping` criterion in `evolve/core/stopping.py`
- [X] T031 [US1] Implement `StagnationStopping` criterion in `evolve/core/stopping.py`
- [X] T032 [US1] Create integration test `tests/integration/test_simple_ga.py` verifying sphere optimization converges
- [X] T033 [US1] Create property test `tests/property/test_determinism.py` verifying identical seeds produce identical trajectories

**Checkpoint**: Classical GA functional - can solve benchmark problems with full determinism ✅

---

## Phase 4: User Story 2 - Multi-Objective Optimization (Priority: P2)

**Goal**: Solve multi-objective problems using NSGA-II-style Pareto ranking and crowding distance

**Independent Test**: Optimize ZDT1 benchmark, verify Pareto front hypervolume within 95% of reference

### Implementation for User Story 2

- [X] T034 [P] [US2] Implement `MultiObjectiveFitness` in `evolve/multiobjective/fitness.py` per contracts/multiobjective.md (objectives array, constraint_violations)
- [X] T035 [P] [US2] Implement `dominates()` function in `evolve/multiobjective/dominance.py` per contracts/multiobjective.md (Pareto dominance with constraint handling)
- [X] T036 [US2] Implement `pareto_front()` function in `evolve/multiobjective/dominance.py` per contracts/multiobjective.md
- [X] T037 [US2] Implement `fast_non_dominated_sort()` in `evolve/multiobjective/ranking.py` per contracts/multiobjective.md (NSGA-II algorithm)
- [X] T038 [US2] Implement `crowding_distance()` in `evolve/multiobjective/crowding.py` per contracts/multiobjective.md
- [X] T039 [US2] Implement `NSGA2Selector` in `evolve/multiobjective/selection.py` per contracts/multiobjective.md (rank + crowding selection)
- [X] T040 [US2] Implement `CrowdedTournamentSelection` in `evolve/multiobjective/selection.py` per contracts/multiobjective.md
- [X] T041 [P] [US2] Implement ZDT benchmark functions in `evolve/evaluation/reference/functions.py` per contracts/evaluation.md (zdt1, zdt2, zdt3)
- [X] T042 [P] [US2] Implement `hypervolume_2d()` in `evolve/multiobjective/metrics.py` per contracts/multiobjective.md
- [X] T043 [US2] Create `evolve/multiobjective/__init__.py` re-exporting public API
- [X] T044 [US2] Create integration test `tests/integration/test_nsga2.py` verifying ZDT1 Pareto front approximation
- [X] T045 [US2] Create property test `tests/property/test_pareto.py` verifying dominance relation properties (transitivity, antisymmetry)

**Checkpoint**: Multi-objective optimization functional - NSGA-II produces valid Pareto fronts ✅

---

## Phase 5: User Story 3 - Neuroevolution (Priority: P3)

**Goal**: Evolve neural network weights/topologies for supervised learning tasks

**Independent Test**: Evolve network to solve XOR using NumPy-only evaluation

### Implementation for User Story 3

- [X] T046 [P] [US3] Implement `GraphGenome` (NEAT-style) in `evolve/representation/graph.py` per contracts/representation.md (NodeGene, ConnectionGene, innovation numbers)
- [X] T047 [US3] Implement `SequenceGenome` in `evolve/representation/sequence.py` per contracts/representation.md (variable-length, alphabet constraint)
- [X] T048 [P] [US3] Implement `NumpyNetwork` phenotype in `evolve/representation/network.py` per contracts/representation.md (weights, biases, activations)
- [X] T049 [US3] Implement `GraphToNetworkDecoder` in `evolve/representation/decoder.py` per contracts/representation.md (topological sort, network construction)
- [X] T050 [US3] Implement `NEATCrossover` in `evolve/core/operators/crossover.py` (gene alignment by innovation number)
- [X] T051 [US3] Implement `NEATMutation` in `evolve/core/operators/mutation.py` (add node, add connection, weight perturbation)
- [X] T052 [US3] Implement `InnovationTracker` in `evolve/representation/graph.py` (global innovation counter for NEAT)
- [X] T053 [US3] Implement genome serialization `to_dict()`/`from_dict()` for all genome types per contracts/representation.md
- [X] T054 [US3] Create integration test `tests/integration/test_neuroevolution.py` verifying XOR problem can be solved

**Checkpoint**: Neuroevolution functional - can evolve network topologies with NumPy phenotypes ✅

---

## Phase 6: User Story 4 - Reinforcement Learning (Priority: P4)

**Goal**: Evolve policies for RL environments (e.g., CartPole)

**Independent Test**: Evolve policy achieving >195 average return on CartPole

### Implementation for User Story 4

- [X] T055 [P] [US4] Implement `Space` specification in `evolve/rl/environment.py` per contracts/rl.md (box, discrete)
- [X] T056 [P] [US4] Define `Environment` Protocol in `evolve/rl/environment.py` per contracts/rl.md (reset, step, observation_space, action_space)
- [X] T057 [US4] Implement `GymAdapter` in `evolve/rl/environment.py` per contracts/rl.md (wraps gymnasium.Env)
- [X] T058 [P] [US4] Define `Policy` Protocol (extends Phenotype) in `evolve/rl/policy.py` per contracts/rl.md
- [X] T059 [P] [US4] Implement `LinearPolicy` in `evolve/rl/policy.py` per contracts/rl.md
- [X] T060 [P] [US4] Implement `MLPPolicy` in `evolve/rl/policy.py` per contracts/rl.md
- [X] T061 [US4] Implement `RecurrentPolicy` in `evolve/rl/policy.py` per contracts/rl.md (stateful with reset)
- [X] T062 [US4] Implement `RolloutResult` dataclass in `evolve/rl/rollout.py` per contracts/rl.md
- [X] T063 [US4] Implement `StandardRollout` in `evolve/rl/rollout.py` per contracts/rl.md (episode execution)
- [X] T064 [US4] Implement `evaluate_policy()` multi-episode aggregation in `evolve/rl/rollout.py` per contracts/rl.md
- [X] T065 [US4] Implement `RLEvaluator` in `evolve/rl/evaluator.py` per contracts/rl.md (decoder + env + rollout → fitness)
- [X] T066 [US4] Create `evolve/rl/__init__.py` re-exporting public API
- [X] T067 [US4] Create integration test `tests/integration/test_rl_evolution.py` (skip if gymnasium not installed)

**Checkpoint**: RL evolution functional - can evolve policies for Gym environments

---

## Phase 7: User Story 5 - Island Model (Priority: P5)

**Goal**: Run parallel populations with periodic migration for improved exploration

**Independent Test**: 4 islands with ring topology show genetic diversity preservation

### Implementation for User Story 5

- [X] T068 [P] [US5] Implement `Island` dataclass in `evolve/diversity/islands/island.py` per contracts/diversity.md (id, population, topology, migration_rate)
- [X] T069 [P] [US5] Implement `ring_topology()` in `evolve/diversity/islands/topology.py` per contracts/diversity.md
- [X] T070 [P] [US5] Implement `fully_connected_topology()` in `evolve/diversity/islands/topology.py` per contracts/diversity.md
- [X] T071 [P] [US5] Implement `hypercube_topology()` in `evolve/diversity/islands/topology.py` per contracts/diversity.md
- [X] T072 [US5] Define `MigrationPolicy` Protocol in `evolve/diversity/islands/migration.py` per contracts/diversity.md
- [X] T073 [US5] Implement `BestMigration` policy in `evolve/diversity/islands/migration.py` per contracts/diversity.md
- [X] T074 [US5] Implement `RandomMigration` policy in `evolve/diversity/islands/migration.py` per contracts/diversity.md
- [X] T075 [US5] Implement `MigrationController` in `evolve/diversity/islands/migration.py` per contracts/diversity.md (synchronous, deterministic)
- [X] T076 [US5] Implement `IslandEvolutionEngine` in `evolve/diversity/islands/engine.py` (multi-population evolution with migration)
- [X] T077 [US5] Create `evolve/diversity/islands/__init__.py` re-exporting public API
- [X] T078 [US5] Create integration test `tests/integration/test_islands.py` verifying migration and diversity preservation
- [X] T079 [US5] Create property test verifying deterministic migration with identical seeds

**Checkpoint**: Island model functional - parallel populations with configurable migration

---

## Phase 8: User Story 6 - Speciation and Novelty Search (Priority: P6)

**Goal**: Maintain diversity through speciation and novelty-based selection

**Independent Test**: Novelty search solves deceptive problem where fitness-only fails

### Implementation for User Story 6

- [X] T080 [P] [US6] Define `DistanceFunction` Protocol in `evolve/diversity/speciation.py` per contracts/diversity.md
- [X] T081 [P] [US6] Implement `euclidean_distance()` in `evolve/diversity/speciation.py` per contracts/diversity.md
- [X] T082 [P] [US6] Implement `neat_distance()` in `evolve/diversity/speciation.py` per contracts/diversity.md (disjoint, excess, weight diff)
- [X] T083 [US6] Implement `Species` dataclass in `evolve/diversity/speciation.py` per contracts/diversity.md (representative, members, stagnation)
- [X] T084 [US6] Define `Speciator` Protocol in `evolve/diversity/speciation.py` per contracts/diversity.md
- [X] T085 [US6] Implement `ThresholdSpeciator` in `evolve/diversity/speciation.py` per contracts/diversity.md
- [X] T086 [US6] Implement `explicit_fitness_sharing()` in `evolve/diversity/niching.py` per contracts/diversity.md
- [X] T087 [US6] Implement `NoveltyArchive` in `evolve/diversity/novelty.py` per contracts/diversity.md (behaviors, novelty score, threshold-based add)
- [X] T088 [US6] Define `BehaviorCharacterization` Protocol in `evolve/diversity/novelty.py` per contracts/diversity.md
- [X] T089 [US6] Implement `QDArchive` (MAP-Elites) in `evolve/diversity/novelty.py` per contracts/diversity.md (grid cells, coverage)
- [X] T090 [US6] Create `evolve/diversity/__init__.py` re-exporting public API
- [X] T091 [US6] Create integration test `tests/integration/test_speciation.py` verifying species formation
- [X] T092 [US6] Create integration test `tests/integration/test_novelty.py` verifying novelty archive growth

**Checkpoint**: Diversity mechanisms functional - speciation and novelty search operational ✅

---

## Phase 9: User Story 7 - GPU-Accelerated Evaluation (Priority: P7)

**Goal**: Accelerate fitness evaluation via GPU while maintaining CPU equivalence

**Independent Test**: GPU evaluator matches CPU reference within 1e-5 relative tolerance, achieves 10x speedup

### Implementation for User Story 7

- [X] T093 [P] [US7] Create `evolve/backends/__init__.py` with backend detection utilities
- [X] T094 [P] [US7] Define `ExecutionBackend` Protocol in `evolve/backends/base.py`
- [X] T095 [US7] Implement `SequentialBackend` (default CPU) in `evolve/backends/sequential.py`
- [X] T096 [US7] Implement `ParallelBackend` (multiprocessing) in `evolve/backends/parallel.py` with seed derivation
- [X] T097 [P] [US7] Create `evolve/backends/accelerated/__init__.py` with optional import handling
- [X] T098 [US7] Implement `TorchEvaluator` in `evolve/backends/accelerated/torch_evaluator.py` (batch GPU evaluation)
- [X] T099 [US7] Implement `JaxEvaluator` in `evolve/backends/accelerated/jax_evaluator.py` (JIT-compiled evaluation)
- [X] T100 [US7] Implement `assert_evaluator_equivalence()` test utility in `evolve/evaluation/testing.py` per contracts/evaluation.md
- [X] T101 [US7] Create benchmark test `tests/benchmarks/test_cpu_gpu_equivalence.py` verifying relative tolerance ≤1e-5
- [X] T102 [US7] Create benchmark test `tests/benchmarks/test_scaling.py` verifying speedup metrics

**Checkpoint**: Accelerated evaluation functional - GPU evaluators match CPU with verified speedup ✅

---

## Phase 10: User Story 8 - Experiment Tracking (Priority: P8)

**Goal**: Full experiment reproducibility with configuration, checkpointing, and metric tracking

**Independent Test**: Checkpoint, kill, resume experiment and verify identical continuation

### Implementation for User Story 8

- [X] T103 [P] [US8] Implement `ExperimentConfig` dataclass in `evolve/experiment/config.py` per contracts/experiment.md (all hyperparameters, validation, hash)
- [X] T104 [US8] Implement `Checkpoint` dataclass in `evolve/experiment/checkpoint.py` per contracts/experiment.md (population, RNG state, generation)
- [X] T105 [US8] Implement `CheckpointManager` in `evolve/experiment/checkpoint.py` per contracts/experiment.md (save, load_latest, prune)
- [X] T106 [US8] Define `MetricTracker` Protocol in `evolve/experiment/metrics.py` per contracts/experiment.md
- [X] T107 [US8] Implement `LocalTracker` (CSV + JSON) in `evolve/experiment/metrics.py` per contracts/experiment.md
- [X] T108 [P] [US8] Create `evolve/experiment/tracking/__init__.py` with optional import handling
- [X] T109 [US8] Implement `MLflowTracker` in `evolve/experiment/tracking/mlflow.py` per contracts/experiment.md
- [X] T110 [US8] Implement `WandbTracker` in `evolve/experiment/tracking/wandb.py` (similar pattern to MLflow)
- [X] T111 [US8] Implement `ExperimentRunner` in `evolve/experiment/runner.py` per contracts/experiment.md (setup, run, resume)
- [X] T112 [US8] Add RNG state capture/restore to `EvolutionEngine` for checkpoint resumption
- [X] T113 [US8] Create `evolve/experiment/__init__.py` re-exporting public API
- [X] T114 [US8] Create integration test `tests/integration/test_checkpointing.py` verifying exact resume
- [X] T115 [US8] Create property test verifying checkpoint + resume produces identical results as uninterrupted run

**Checkpoint**: Experiment management functional - full reproducibility achieved ✅

---

## Phase 11: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and validation across all user stories

- [X] T116 [P] Create `README.md` with installation, quickstart example, and feature overview
- [X] T117 [P] Create `docs/api/` directory with module-level documentation
- [X] T118 [P] Create `docs/tutorials/` with examples from quickstart.md
- [X] T119 [P] Create `CONTRIBUTING.md` with development guidelines
- [X] T120 Verify zero ML imports in core modules: `evolve/core/`, `evolve/representation/`, `evolve/evaluation/` (SC-008)
- [X] T121 Run `quickstart.md` examples as integration tests
- [X] T122 Add type annotations validation with mypy
- [X] T123 Code cleanup: remove unused imports, fix linting issues
- [X] T124 Create `evolve/__init__.py` with version and top-level re-exports

**Checkpoint**: Polish complete - documentation and validation done ✅

---

## Phase 12: Memetic Extension Points (Deferred Implementation)

**Purpose**: Define interface stubs for future memetic algorithm support (FR-031 to FR-033)

**Note**: These tasks create Protocol definitions only. Full implementation deferred to Feature 002.

- [ ] T125 Create `evolve/memetic/__init__.py` with module docstring explaining deferred status
- [ ] T126 [P] Define `LocalSearchOperator` Protocol stub in `evolve/memetic/protocols.py` (FR-031)
- [ ] T127 [P] Define `MemeticPipeline` Protocol stub in `evolve/memetic/protocols.py` demonstrating composability with standard operators (FR-033)

**Checkpoint**: Extension points defined - implementation tracked in Feature 002

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ──────────────────────────────────────────┐
                                                          │
Phase 2 (Foundational) ◀──────────────────────────────────┘
      │
      │ BLOCKS ALL USER STORIES
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    USER STORIES (can parallelize)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3 (US1: GA)        ──────▶ MVP DELIVERY POINT                │
│  Phase 4 (US2: MO)        ──────▶ Adds multi-objective              │
│  Phase 5 (US3: Neuro)     ──────▶ Adds neuroevolution               │
│  Phase 6 (US4: RL)        ──────▶ Adds RL evolution                 │
│  Phase 7 (US5: Islands)   ──────▶ Adds parallel populations         │
│  Phase 8 (US6: Diversity) ──────▶ Adds speciation/novelty           │
│  Phase 9 (US7: GPU)       ──────▶ Adds acceleration                 │
│  Phase 10 (US8: Tracking) ──────▶ Adds reproducibility              │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
Phase 11 (Polish) ◀─── All desired stories complete
```

### User Story Dependencies

| Story | Can Start After | Dependencies on Other Stories |
|-------|-----------------|------------------------------|
| US1 (GA) | Phase 2 | None - independent |
| US2 (MO) | Phase 2 | None - uses different Fitness type |
| US3 (Neuro) | Phase 2 | None - uses different Genome types |
| US4 (RL) | Phase 2 | None - uses different Evaluator |
| US5 (Islands) | Phase 2 | Integrates with US1 engine patterns |
| US6 (Diversity) | Phase 2 | Uses US3 GraphGenome for NEAT distance |
| US7 (GPU) | Phase 2 | Requires US1 FunctionEvaluator for comparison |
| US8 (Tracking) | Phase 2 | Integrates with US1 EvolutionEngine |

### Parallel Opportunities

**Within Setup (Phase 1)**:
```
T002, T003, T004, T005, T006 can run in parallel
```

**Within Foundational (Phase 2)**:
```
T010, T011, T012, T013, T014, T015, T016 can run in parallel (protocols)
```

**Within Each User Story**:
- Tasks marked [P] can run in parallel
- Models/protocols before implementations
- Implementations before integration tests

**Across User Stories** (with team):
```
Developer A: US1 (MVP) → US5 (Islands) → US7 (GPU)
Developer B: US2 (MO) → US6 (Diversity) → US8 (Tracking)
Developer C: US3 (Neuro) → US4 (RL)
```

---

## Implementation Strategy

### MVP First (Recommended)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational)
2. Complete Phase 3 (US1: Classical GA)
3. **STOP & VALIDATE**: Run integration tests, verify sphere optimization
4. Deploy/demo MVP with basic GA capability

### Incremental Delivery

| Increment | Stories | Capability Added |
|-----------|---------|------------------|
| MVP | US1 | Basic genetic algorithm |
| +MO | US1, US2 | Multi-objective optimization |
| +Neuro | US1-US3 | Neural network evolution |
| +RL | US1-US4 | Policy evolution for games |
| +Scale | US1-US5 | Parallel island populations |
| +Diversity | US1-US6 | Speciation and novelty |
| +Perf | US1-US7 | GPU acceleration |
| Full | US1-US8 | Complete experiment tracking |

---

## Summary

| Phase | Task Count | Parallel Tasks |
|-------|------------|----------------|
| Setup | 6 | 5 |
| Foundational | 12 | 7 |
| US1 (GA) | 15 | 7 |
| US2 (MO) | 12 | 4 |
| US3 (Neuro) | 9 | 3 |
| US4 (RL) | 13 | 5 |
| US5 (Islands) | 12 | 5 |
| US6 (Diversity) | 13 | 4 |
| US7 (GPU) | 10 | 3 |
| US8 (Tracking) | 13 | 2 |
| Polish | 9 | 4 |
| Memetic (stubs) | 3 | 2 |
| **Total** | **127** | **51** |

**MVP Scope**: Phases 1-3 (33 tasks) delivers a working evolutionary optimizer
**Suggested First Milestone**: US1 + US2 (45 tasks) provides single and multi-objective optimization
