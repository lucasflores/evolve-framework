# Data Model: Evolve Framework Core Architecture

**Feature**: 001-core-framework-architecture  
**Date**: 2026-01-13  
**Source**: [spec.md](spec.md) Key Entities section

---

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVOLUTION CORE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │  Experiment  │────────▶│   Island     │────────▶│  Population  │       │
│   │              │  1..*   │              │   1     │              │       │
│   └──────────────┘         └──────────────┘         └──────────────┘       │
│          │                        │                        │                │
│          │                        │ migration              │ contains       │
│          ▼                        ▼                        ▼                │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │  Checkpoint  │         │   Species    │◀────────│  Individual  │       │
│   │              │         │              │  0..1   │              │       │
│   └──────────────┘         └──────────────┘         └──────────────┘       │
│                                                            │                │
│                                                            │ has            │
│                                                            ▼                │
│                                                     ┌──────────────┐       │
│                                                     │   Fitness    │       │
│                                                     │              │       │
│                                                     └──────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPRESENTATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │  Individual  │────────▶│    Genome    │────────▶│  Phenotype   │       │
│   │              │   1     │   (abstract) │ decode  │  (abstract)  │       │
│   └──────────────┘         └──────────────┘         └──────────────┘       │
│                                   △                        △                │
│                                   │                        │                │
│                    ┌──────────────┼──────────────┐         │                │
│                    │              │              │         │                │
│             ┌──────┴─────┐ ┌──────┴─────┐ ┌──────┴─────┐   │                │
│             │   Vector   │ │  Sequence  │ │   Graph    │   │                │
│             │   Genome   │ │   Genome   │ │   Genome   │   │                │
│             └────────────┘ └────────────┘ └────────────┘   │                │
│                                                            │                │
│                                                     ┌──────┴─────┐          │
│                                                     │   Policy   │          │
│                                                     │ (RL only)  │          │
│                                                     └────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│   │  Evaluator   │────────▶│   Fitness    │         │ Environment  │       │
│   │  (abstract)  │ returns │              │         │  (RL only)   │       │
│   └──────────────┘         └──────────────┘         └──────────────┘       │
│          △                                                 │                │
│          │                                                 │ rollout        │
│   ┌──────┴──────────────────────┐                         ▼                │
│   │              │              │                  ┌──────────────┐        │
│ ┌─┴────────┐ ┌───┴──────┐ ┌─────┴────┐            │  Trajectory  │        │
│ │ Function │ │  Torch   │ │   JAX    │            │              │        │
│ │Evaluator │ │Evaluator │ │Evaluator │            └──────────────┘        │
│ └──────────┘ └──────────┘ └──────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Entities

### Individual

A candidate solution in the population.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | `UUID` | Unique identifier | Auto-generated |
| `genome` | `Genome` | Genetic representation | Required, immutable after creation |
| `fitness` | `Fitness \| None` | Evaluated fitness | None until evaluated |
| `metadata` | `IndividualMetadata` | Optional tracking info | Optional |
| `created_at` | `int` | Generation of creation | ≥ 0 |

**IndividualMetadata**:
| Field | Type | Description |
|-------|------|-------------|
| `age` | `int` | Generations survived |
| `parent_ids` | `tuple[UUID, ...] \| None` | Lineage tracking |
| `species_id` | `int \| None` | Species assignment |
| `origin` | `str` | "init" / "crossover" / "mutation" / "migration" |

**Validation Rules**:
- `genome` must be serializable
- `fitness` is `None` for unevaluated individuals
- `age` increments each generation the individual survives

**State Transitions**:
```
Created (fitness=None) → Evaluated (fitness set) → Selected/Discarded
                                                 → Mutated → New Individual
                                                 → Crossed → New Individual
```

---

### Population

Ordered collection of individuals with statistics.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `individuals` | `Sequence[Individual]` | Member individuals | Non-empty |
| `generation` | `int` | Current generation | ≥ 0 |
| `statistics` | `PopulationStatistics` | Computed metrics | Computed on demand |

**PopulationStatistics**:
| Field | Type | Description |
|-------|------|-------------|
| `size` | `int` | Number of individuals |
| `best_fitness` | `Fitness` | Best fitness in population |
| `mean_fitness` | `Fitness` | Mean fitness across population |
| `diversity` | `float` | Genotypic/phenotypic diversity measure |
| `species_count` | `int` | Number of distinct species |
| `front_sizes` | `list[int]` | Individuals per Pareto front (multi-obj) |

**Validation Rules**:
- `individuals` must not be empty (FR: empty population raises error)
- All individuals must have compatible genome types
- Statistics recomputed when population changes

---

### Genome (Abstract)

Framework-neutral genetic representation.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| — | — | *Abstract interface* | Implementation-dependent |

**Required Methods** (Protocol):
```python
class Genome(Protocol):
    def copy(self) -> Self: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
```

**Concrete Implementations**:

#### VectorGenome
| Field | Type | Description |
|-------|------|-------------|
| `genes` | `np.ndarray` | Fixed-length float vector |
| `bounds` | `tuple[np.ndarray, np.ndarray]` | (lower, upper) bounds |

#### SequenceGenome
| Field | Type | Description |
|-------|------|-------------|
| `genes` | `list[Any]` | Variable-length sequence |
| `alphabet` | `set[Any] \| None` | Valid gene values |

#### GraphGenome (NEAT-style)
| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `dict[int, NodeGene]` | Node ID → node data |
| `connections` | `dict[int, ConnectionGene]` | Innovation # → connection |
| `inputs` | `list[int]` | Input node IDs |
| `outputs` | `list[int]` | Output node IDs |

---

### Phenotype (Abstract)

Decoded form of a genome that can be evaluated.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| — | — | *Abstract interface* | May be backend-specific |

**Required Methods** (Protocol):
```python
class Phenotype(Protocol):
    def __call__(self, inputs: Any) -> Any: ...
```

#### Policy (extends Phenotype)

Phenotype specialized for RL that maps observations to actions.

| Field | Type | Description |
|-------|------|-------------|
| `action_space` | `ActionSpace` | Valid action specification |
| `observation_space` | `ObservationSpace` | Expected input specification |

**Additional Methods**:
```python
class Policy(Phenotype, Protocol):
    def act(self, observation: np.ndarray, rng: Random) -> np.ndarray: ...
    def reset(self) -> None: ...  # For stateful policies
```

---

### Fitness

Vector-valued measure of solution quality.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `values` | `np.ndarray` | Objective values | Shape: (n_objectives,) |
| `constraints` | `np.ndarray \| None` | Constraint violations | Shape: (n_constraints,), ≤ 0 = feasible |
| `is_feasible` | `bool` | All constraints satisfied | Computed property |

**Validation Rules**:
- `values` must not contain NaN (FR: NaN fitness excluded from selection)
- `values` may contain Inf (assigned worst rank)
- `constraints` where value ≤ 0 means constraint satisfied

**Comparison**:
- Single-objective: direct comparison of `values[0]`
- Multi-objective: Pareto dominance (see `dominance.py`)
- Constrained: feasible individuals always preferred (FR-012)

---

### Evaluator (Abstract)

Component that computes fitness for batches of individuals.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `capabilities` | `EvaluatorCapabilities` | Declared features | Required |

**EvaluatorCapabilities**:
| Field | Type | Description |
|-------|------|-------------|
| `batchable` | `bool` | Can evaluate multiple at once |
| `stochastic` | `bool` | Results vary with RNG |
| `stateful` | `bool` | Has internal state |
| `n_objectives` | `int` | Number of fitness objectives |
| `n_constraints` | `int` | Number of constraints |

**Required Methods** (Protocol):
```python
class Evaluator(Protocol[G]):
    @property
    def capabilities(self) -> EvaluatorCapabilities: ...
    
    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None
    ) -> Sequence[Fitness]: ...
```

---

### Operator (Abstract)

Evolutionary operator implementing a specific strategy.

**Selection Operator**:
```python
class SelectionOperator(Protocol[G]):
    def select(
        self,
        population: Population[G],
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]: ...
```

**Crossover Operator**:
```python
class CrossoverOperator(Protocol[G]):
    def crossover(
        self,
        parent1: G,
        parent2: G,
        rng: Random
    ) -> tuple[G, G]: ...
```

**Mutation Operator**:
```python
class MutationOperator(Protocol[G]):
    def mutate(
        self,
        genome: G,
        rng: Random
    ) -> G: ...
```

---

### Island

Semi-isolated population with its own parameters.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | `int` | Island identifier | Unique within model |
| `population` | `Population` | Island's population | Required |
| `config` | `IslandConfig` | Island-specific params | Required |
| `rng` | `Random` | Island's RNG (derived from master seed) | Deterministic |

**IslandConfig**:
| Field | Type | Description |
|-------|------|-------------|
| `selection` | `SelectionOperator` | Selection strategy |
| `crossover` | `CrossoverOperator` | Crossover strategy |
| `mutation` | `MutationOperator` | Mutation strategy |
| `mutation_rate` | `float` | Probability of mutation |
| `crossover_rate` | `float` | Probability of crossover |
| `elitism` | `int` | Number of elites preserved |

---

### Species

Group of similar individuals within a population.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | `int` | Species identifier | Unique |
| `representative` | `Genome` | Species prototype | Required |
| `members` | `set[UUID]` | Member individual IDs | Non-empty |
| `age` | `int` | Generations since formation | ≥ 0 |
| `best_fitness` | `Fitness` | Best fitness ever in species | For stagnation |
| `stagnation` | `int` | Generations without improvement | For extinction |

**Validation Rules**:
- Species with 0 members are removed
- Representative updated each generation (usually fittest member)

---

### Experiment

Configuration, execution trace, and results of an evolutionary run.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | `UUID` | Experiment identifier | Auto-generated |
| `config` | `ExperimentConfig` | Full configuration | Required, immutable |
| `status` | `ExperimentStatus` | Current state | Enum |
| `start_time` | `datetime` | When started | Set on start |
| `end_time` | `datetime \| None` | When completed | Set on completion |
| `seed` | `int` | Master random seed | Required |

**ExperimentConfig**:
| Field | Type | Description |
|-------|------|-------------|
| `generations` | `int` | Max generations |
| `population_size` | `int` | Individuals per population |
| `offspring_count` | `int` | Offspring per generation |
| `selection` | `SelectionOperator` | Selection strategy |
| `crossover` | `CrossoverOperator` | Crossover operator |
| `mutation` | `MutationOperator` | Mutation operator |
| `evaluator` | `Evaluator` | Fitness evaluator |
| `elitism` | `int` | Elites preserved |
| `early_stopping` | `StoppingCriterion \| None` | Optional stopping |
| `islands` | `list[IslandConfig] \| None` | Island model config |
| `speciation` | `SpeciationConfig \| None` | Speciation config |
| `tracking` | `TrackingConfig \| None` | Experiment tracking |

**ExperimentStatus** (Enum):
- `PENDING` - Created but not started
- `RUNNING` - Evolution in progress
- `PAUSED` - Checkpointed, can resume
- `COMPLETED` - Finished successfully
- `FAILED` - Error during execution

---

### Checkpoint

Serialized state enabling exact resumption.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `experiment_id` | `UUID` | Parent experiment | Required |
| `generation` | `int` | Generation number | ≥ 0 |
| `population_state` | `bytes` | Serialized population(s) | Required |
| `rng_state` | `bytes` | RNG state for resumption | Required |
| `metrics` | `dict` | Metrics up to this point | Required |
| `timestamp` | `datetime` | When created | Auto-set |
| `version` | `str` | Framework version | For compatibility |

**Validation Rules**:
- Checkpoint version must match framework version for safe resumption
- RNG state must include all island RNGs for deterministic resume

---

## Relationships Summary

| Entity | Relates To | Cardinality | Description |
|--------|------------|-------------|-------------|
| Experiment | Island | 1..* | Experiment has 1+ islands (1 for simple GA) |
| Island | Population | 1..1 | Each island has exactly one population |
| Population | Individual | 1..* | Population contains 1+ individuals |
| Individual | Genome | 1..1 | Individual has exactly one genome |
| Individual | Fitness | 0..1 | Fitness is None until evaluated |
| Individual | Species | 0..1 | Optional species assignment |
| Genome | Phenotype | 1..1 | Genome decodes to one phenotype |
| Policy | Phenotype | is-a | Policy is a specialized Phenotype |
| Evaluator | Fitness | produces | Evaluator produces fitness values |
| Experiment | Checkpoint | 1..* | Experiment can have multiple checkpoints |
