# Data Model: Symbiogenetic Merge Operator

**Feature**: 013-symbiogenetic-merge  
**Date**: 2026-04-17  
**Phase**: 1 — Design & Contracts

---

## Entities

### 1. SymbiogeneticMerge (Protocol)

**Module**: `evolve/core/operators/merge.py`  
**Kind**: `@runtime_checkable Protocol[G]`

| Field/Method | Type | Description |
|-------------|------|-------------|
| `merge()` | method | Absorb symbiont into host, producing a single offspring genome |

**Method signature**:
```python
def merge(self, host: G, symbiont: G, rng: Random, **kwargs: Any) -> G
```

**Constraints**:
- Must return a **new** genome instance (immutability invariant)
- Must accept explicit `rng` for determinism (Constitution V)
- `host` and `symbiont` must be the same genome type
- Host ≠ symbiont identity enforced by operator (`ValueError`) (FR-015)

**Relationships**: Registered in `OperatorRegistry` under category `"merge"`. Resolved by `create_engine()` factory via `MergeConfig.operator`.

---

### 2. MergeConfig (Frozen Dataclass)

**Module**: `evolve/config/merge.py`  
**Kind**: `@dataclass(frozen=True)`

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `operator` | `str` | `"graph_symbiogenetic"` | Must exist in registry under `"merge"` category |
| `merge_rate` | `float` | `0.0` | 0.0 ≤ merge_rate ≤ 1.0 |
| `symbiont_source` | `Literal["cross_species", "archive"]` | `"cross_species"` | — |
| `symbiont_fate` | `Literal["consumed", "survives"]` | `"consumed"` | — |
| `archive_size` | `int` | `50` | > 0; only relevant when symbiont_source="archive" |
| `interface_count` | `int` | `4` | > 0; number of interface connections to create |
| `interface_ratio` | `float` | `0.5` | 0.0 ≤ interface_ratio ≤ 1.0; fraction of interface connections host→symbiont vs symbiont→host |
| `weight_method` | `Literal["mean", "host_biased", "random"]` | `"mean"` | — |
| `weight_mean` | `float` | `0.0` | Mean for Gaussian weight init (used when weight_method="random") |
| `weight_std` | `float` | `1.0` | > 0.0; std dev for Gaussian weight init (used when weight_method="random") |
| `max_complexity` | `int \| None` | `None` | If set, > 0; merge rejected when offspring gene count would exceed this |
| `operator_params` | `dict[str, Any]` | `field(default_factory=dict)` | Passed as `**kwargs` to merge operator |

**Relationships**: Optional field on `UnifiedConfig` (`merge: MergeConfig | None = None`). When `None`, merge phase is disabled.

**Validation rules** (in `__post_init__`):
- `merge_rate` must be in `[0.0, 1.0]`
- `interface_count` must be `> 0`
- `interface_ratio` must be in `[0.0, 1.0]`
- `archive_size` must be `> 0`
- `weight_std` must be `> 0.0`
- `max_complexity` must be `> 0` if not `None`

---

### 3. GraphSymbiogeneticMerge (Dataclass)

**Module**: `evolve/core/operators/merge.py`  
**Kind**: `@dataclass(frozen=True)`, implements `SymbiogeneticMerge[GraphGenome]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interface_count` | `int` | `4` | Number of interface connections to create |
| `interface_ratio` | `float` | `0.5` | Fraction of interface connections that are host→symbiont (rest are symbiont→host) |
| `weight_method` | `Literal["mean", "host_biased", "random"]` | `"mean"` | How to set weights on interface connections |
| `weight_mean` | `float` | `0.0` | Mean for Gaussian weight init (used when weight_method="random") |
| `weight_std` | `float` | `1.0` | Std dev for Gaussian weight init (used when weight_method="random") |

**Behavior**:
1. Remap symbiont node IDs (offset by `max(host.node_ids) + 1`)
2. Remap symbiont connection innovation numbers via `InnovationTracker`
3. Union host + remapped symbiont nodes and connections
4. Generate interface connections between host output-adjacent nodes and symbiont input-adjacent nodes (and vice versa per `interface_ratio`)
5. Assign weights to interface connections per `weight_method`
6. Return new `GraphGenome` with merged structure

**Registry name**: `"graph_symbiogenetic"`  
**Compatible genomes**: `["GraphGenome"]`

---

### 4. SequenceSymbiogeneticMerge (Dataclass)

**Module**: `evolve/core/operators/merge.py`  
**Kind**: `@dataclass(frozen=True)`, implements `SymbiogeneticMerge[SequenceGenome]`

**Behavior**: Concatenate `host.genes + symbiont.genes`. Alphabet must match.

**Registry name**: `"sequence_symbiogenetic"`  
**Compatible genomes**: `["SequenceGenome"]`

---

### 5. VectorSymbiogeneticMerge (Dataclass)

**Module**: `evolve/core/operators/merge.py`  
**Kind**: `@dataclass(frozen=True)`, implements `SymbiogeneticMerge[VectorGenome]`

**Behavior**: Concatenate `np.concatenate([host.genes, symbiont.genes])`. Bounds concatenated.

**Registry name**: `"vector_symbiogenetic"`  
**Compatible genomes**: `["VectorGenome"]`

---

### 6. EmbeddingSymbiogeneticMerge (Dataclass)

**Module**: `evolve/core/operators/merge.py`  
**Kind**: `@dataclass(frozen=True)`, implements `SymbiogeneticMerge[EmbeddingGenome]`

**Behavior**: Vertically stack `np.vstack([host.embeddings, symbiont.embeddings])`. `model_id` must match.

**Registry name**: `"embedding_symbiogenetic"`  
**Compatible genomes**: `["EmbeddingGenome"]`

---

### 7. HallOfFameCallback (Dataclass)

**Module**: `evolve/core/callbacks.py` (or `evolve/core/operators/merge.py`)  
**Kind**: `@dataclass`, implements `Callback`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_size` | `int` | `50` | Maximum archive capacity |
| `archive` | `list[Individual]` | `[]` | Sorted best-first; mutable (updated each generation) |
| `priority` | `int` | `100` | Callback execution priority |

**Behavior**: On `on_generation_end`, inserts new top performers into the archive (bounded by `max_size`). Engine reads `archive` during merge phase when `symbiont_source="archive"`.

---

### 8. MergeMetricCollector

**Module**: `evolve/experiment/collectors/merge.py`  
**Kind**: Class implementing metric collector protocol

| Metric | Type | Description |
|--------|------|-------------|
| `merge_count` | `int` | Number of merges performed this generation |
| `mean_genome_complexity` | `float` | Average gene count of merged offspring |
| `complexity_delta` | `float` | Mean (offspring_complexity - host_complexity) |

**Activation**: Enabled when `MetricCategory.SYMBIOGENESIS` is in `TrackingConfig.categories`.

---

### 9. Updated enums/types

**MetricCategory** (in `evolve/config/tracking.py`):
- Add: `SYMBIOGENESIS = "symbiogenesis"`

**IndividualMetadata.origin** (in `evolve/core/types.py`):
- Add: `"symbiogenetic_merge"` to valid origin values

**OperatorRegistry.CATEGORIES** (in `evolve/registry/operators.py`):
- Add: `"merge"` to the categories tuple

---

## Entity Relationship Diagram

```text
UnifiedConfig
  └── merge: MergeConfig | None
        ├── operator → OperatorRegistry["merge", name]
        │                  └── SymbiogeneticMerge[G] (Protocol)
        │                        ├── GraphSymbiogeneticMerge
        │                        ├── SequenceSymbiogeneticMerge
        │                        ├── VectorSymbiogeneticMerge
        │                        └── EmbeddingSymbiogeneticMerge
        ├── symbiont_source → Engine merge phase logic
        │                        ├── "cross_species" → Speciator
        │                        └── "archive" → HallOfFameCallback.archive
        └── symbiont_fate → Engine merge phase logic

TrackingConfig
  └── categories: frozenset[MetricCategory]
        └── SYMBIOGENESIS → MergeMetricCollector

Individual
  └── metadata: IndividualMetadata
        └── origin: "symbiogenetic_merge"
        └── parent_ids: (host_id, symbiont_id)
```

---

## State Transitions

### Merge Phase (per generation)

```text
[Offspring Pool] ──(for each offspring at merge_rate probability)──►
  ┌─ Select host (uniform random from eligible offspring)
  ├─ Source symbiont (cross_species | archive)
  ├─ Verify host ≠ symbiont
  ├─ Call merge_operator.merge(host, symbiont, rng)
  ├─ Create Individual with origin="symbiogenetic_merge"
  ├─ If symbiont_fate="consumed": remove symbiont from pool
  └─ Replace host in offspring pool with merged offspring
──► [Modified Offspring Pool]
```

### Symbiont Lifecycle (when fate="consumed")

```text
[Active Population Member] ──(selected as symbiont)──►
  [Absorbed into Host] ──(removed from population)──►
  [Lineage preserved in offspring metadata]
```
