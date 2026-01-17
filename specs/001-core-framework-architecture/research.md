# Research: Evolve Framework Core Architecture

**Feature**: 001-core-framework-architecture  
**Date**: 2026-01-13  
**Purpose**: Resolve technical decisions and document best practices for evolutionary algorithms framework design

## Research Questions

Based on the Technical Context, the following areas required investigation:

1. Interface design patterns for evolutionary operators
2. Deterministic parallel execution strategies
3. Multi-objective optimization algorithm choices
4. Genome serialization approaches
5. Experiment tracking integration patterns

---

## 1. Interface Design for Evolutionary Operators

### Decision: Protocol-based interfaces with explicit type parameters

**Rationale**: Python's `typing.Protocol` provides structural subtyping without inheritance hierarchies. This matches the framework's extensibility goals—users can implement custom operators without inheriting from base classes.

**Alternatives Considered**:
- Abstract Base Classes (ABC): Requires explicit inheritance, less flexible for duck-typing
- Plain duck typing: No static type checking, harder to document contracts
- Generic base classes: Overly complex for simple operator signatures

**Implementation Pattern**:
```python
from typing import Protocol, TypeVar, Sequence

G = TypeVar('G', bound='Genome')

class SelectionOperator(Protocol[G]):
    def select(
        self, 
        population: Sequence[Individual[G]], 
        n: int,
        rng: Random
    ) -> Sequence[Individual[G]]: ...
```

**Key Insight**: Type parameters (`G`) allow operators to be generic over genome types while maintaining type safety. The `rng: Random` parameter enforces determinism.

---

## 2. Deterministic Parallel Execution

### Decision: Order-independent aggregation with reproducible work distribution

**Rationale**: Parallel fitness evaluation is the primary scalability mechanism. Determinism requires:
1. Fixed mapping of individuals to workers (not dynamic scheduling)
2. Order-independent fitness aggregation (results don't depend on completion order)
3. Per-worker RNG streams derived from master seed + worker ID

**Alternatives Considered**:
- Dynamic work stealing: Higher throughput but non-deterministic
- Global RNG with locks: Deterministic but serializes randomness
- Timestamp-based ordering: Fragile, depends on clock precision

**Implementation Pattern**:
```python
def parallel_evaluate(
    individuals: Sequence[Individual],
    evaluator: Evaluator,
    n_workers: int,
    master_seed: int
) -> Sequence[Fitness]:
    # Fixed chunking ensures same individual -> same worker
    chunks = partition(individuals, n_workers)
    worker_seeds = [derive_seed(master_seed, i) for i in range(n_workers)]
    
    # Results aggregated by original index, not completion order
    results = parallel_map(
        lambda chunk, seed: evaluator.evaluate(chunk, seed),
        chunks, worker_seeds
    )
    return flatten_by_index(results)
```

**Key Insight**: `derive_seed(master_seed, worker_id)` creates deterministic per-worker streams. SplitMix64 or similar PRNG splitting algorithms ensure independence.

---

## 3. Multi-Objective Optimization: NSGA-II vs NSGA-III

### Decision: Implement NSGA-II as primary, design for NSGA-III extensibility

**Rationale**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the de facto standard for 2-3 objective problems. NSGA-III extends this to many-objective (4+) problems using reference points. Starting with NSGA-II provides immediate value while the interface design accommodates NSGA-III.

**Alternatives Considered**:
- SPEA2: Similar performance, more complex archive management
- MOEA/D: Decomposition-based, different paradigm (good for extension)
- SMS-EMOA: Hypervolume-based, computationally expensive

**NSGA-II Components**:
1. **Non-dominated sorting**: O(MN²) where M=objectives, N=population
2. **Crowding distance**: Diversity within fronts
3. **Binary tournament**: Selection based on rank, then crowding

**Interface Design** (supports NSGA-III extension):
```python
class RankingStrategy(Protocol):
    def rank(self, population: Population) -> Sequence[int]: ...
    
class DiversityMetric(Protocol):
    def compute(self, front: Sequence[Individual]) -> Sequence[float]: ...

# NSGA-II uses crowding distance; NSGA-III uses reference point association
```

**Key Insight**: Separating ranking from diversity metrics allows swapping crowding distance (NSGA-II) for reference point association (NSGA-III) without changing the selection logic.

---

## 4. Genome Serialization

### Decision: Pickle with JSON fallback, custom serializer protocol

**Rationale**: Pickle handles arbitrary Python objects but isn't portable. JSON is portable but limited to basic types. A protocol allows genomes to declare their preferred serialization.

**Alternatives Considered**:
- Pickle only: Not portable across Python versions, security concerns
- JSON only: Can't serialize NumPy arrays, graphs, custom objects
- Protocol buffers: Overkill for research framework, requires schema management
- MessagePack: Good compromise but adds dependency

**Implementation Pattern**:
```python
class Serializable(Protocol):
    def to_dict(self) -> dict: ...
    
    @classmethod
    def from_dict(cls, data: dict) -> Self: ...

def serialize_genome(genome: Genome) -> bytes:
    if isinstance(genome, Serializable):
        return json.dumps(genome.to_dict()).encode()
    return pickle.dumps(genome)
```

**Key Insight**: Genomes that implement `Serializable` get portable JSON; others fall back to pickle. This balances flexibility with portability.

---

## 5. Experiment Tracking Integration

### Decision: Abstract tracker interface with MLflow as reference implementation

**Rationale**: Experiment tracking tools (MLflow, W&B, Neptune) have similar concepts (runs, parameters, metrics, artifacts) but different APIs. An abstraction allows swapping implementations and graceful degradation when no tracker is available.

**Alternatives Considered**:
- MLflow-only: Vendor lock-in, not all users have MLflow
- Direct file logging: No aggregation, harder to query
- Custom tracking DB: Reinventing the wheel

**Interface Pattern**:
```python
class ExperimentTracker(Protocol):
    def start_run(self, config: ExperimentConfig) -> RunContext: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, path: Path, name: str) -> None: ...
    def end_run(self) -> None: ...

class NullTracker:
    """No-op implementation for when tracking is disabled."""
    def start_run(self, config): return NullContext()
    def log_metrics(self, metrics, step): pass
    def log_artifact(self, path, name): pass
    def end_run(self): pass
```

**Key Insight**: `NullTracker` implements the same interface, allowing code to run without tracking infrastructure. This is essential for unit tests and quick experiments.

---

## 6. Island Model Synchronization

### Decision: Synchronous migration with configurable intervals

**Rationale**: Asynchronous migration introduces non-determinism. Synchronous migration at fixed generation intervals maintains reproducibility while still enabling diversity exchange.

**Alternatives Considered**:
- Asynchronous migration: Better performance but non-deterministic
- Continuous migration: Blurs island boundaries, loses isolation benefits
- Adaptive intervals: Interesting research direction but adds complexity

**Implementation Pattern**:
```python
@dataclass
class MigrationPolicy:
    interval: int  # Migrate every N generations
    n_migrants: int  # Number to send per migration
    selection: Callable  # How to choose migrants (e.g., best, random)
    replacement: Callable  # How to replace recipients (e.g., worst, random)

class IslandModel:
    def step(self, generation: int):
        for island in self.islands:
            island.evolve_one_generation()
        
        if generation % self.policy.interval == 0:
            self._synchronous_migrate()
```

**Key Insight**: All islands complete their generation before migration. This synchronization point ensures deterministic results.

---

## 7. Speciation Distance Metrics

### Decision: Pluggable distance with NEAT-style compatibility as reference

**Rationale**: Different representations need different distance metrics. NEAT uses a weighted sum of disjoint genes, excess genes, and weight differences. Fixed-length vectors use Euclidean distance. The interface should accommodate both.

**NEAT Compatibility Distance**:
```
δ = c₁(E/N) + c₂(D/N) + c₃·W̄
```
Where E=excess genes, D=disjoint genes, N=normalizer, W̄=avg weight diff

**Interface Pattern**:
```python
class DistanceMetric(Protocol[G]):
    def distance(self, g1: G, g2: G) -> float: ...

class NEATDistance(DistanceMetric[NEATGenome]):
    def __init__(self, c1: float, c2: float, c3: float): ...
    
class EuclideanDistance(DistanceMetric[VectorGenome]):
    def distance(self, g1, g2) -> float:
        return np.linalg.norm(g1.genes - g2.genes)
```

---

## 8. CPU/GPU Equivalence Testing

### Decision: Relative tolerance with special handling for edge cases

**Rationale**: Floating-point operations on GPU may use different precision (FP16, TF32) or operation ordering. Relative tolerance handles scale differences; absolute tolerance handles near-zero values.

**Formula** (from spec clarification):
```
equivalent iff |a - b| / max(|a|, |b|, ε) ≤ 1e-5
```
Where ε=1e-10 prevents division by zero.

**Edge Cases**:
- Both zero: equivalent
- One zero, one non-zero: use absolute tolerance fallback
- NaN: never equivalent (NaN ≠ NaN)
- Inf: equivalent only if both same sign Inf

**Implementation Pattern**:
```python
def assert_equivalent(cpu: np.ndarray, gpu: np.ndarray, rtol=1e-5, atol=1e-10):
    if np.any(np.isnan(cpu)) or np.any(np.isnan(gpu)):
        raise AssertionError("NaN values detected")
    
    denom = np.maximum(np.abs(cpu), np.abs(gpu))
    denom = np.maximum(denom, atol)  # Prevent div by zero
    rel_diff = np.abs(cpu - gpu) / denom
    
    if not np.all(rel_diff <= rtol):
        raise AssertionError(f"Max relative diff: {rel_diff.max()}")
```

---

## Summary of Decisions

| Topic | Decision | Key Rationale |
|-------|----------|---------------|
| Interfaces | Protocol-based with type parameters | Structural subtyping, no inheritance required |
| Parallelism | Fixed work distribution, seed derivation | Determinism without serializing RNG |
| Multi-objective | NSGA-II primary, extensible to NSGA-III | Standard algorithm, modular design |
| Serialization | Pickle + JSON fallback with protocol | Flexibility with portability option |
| Tracking | Abstract interface, NullTracker fallback | No vendor lock-in, graceful degradation |
| Islands | Synchronous migration at intervals | Deterministic, clear synchronization points |
| Speciation | Pluggable distance metrics | Different representations need different metrics |
| Equivalence | Relative tolerance with edge case handling | Handles GPU precision differences |

---

## References

- Deb, K., et al. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (2002)
- Stanley, K.O., Miikkulainen, R. "Evolving Neural Networks through Augmenting Topologies" (2002)
- Fortin, F.-A., et al. "DEAP: Evolutionary Algorithms Made Easy" (2012)
- Salimans, T., et al. "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" (2017)
