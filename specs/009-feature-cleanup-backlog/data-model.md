# Data Model: Feature & Cleanup Backlog

## Modified Entities

### PopulationStatistics (frozen dataclass)

**Current fields**: size, best_fitness, worst_fitness, mean_fitness, std_fitness, diversity, species_count, front_sizes, evaluated_count

**Added fields**:
- `minimize: bool` — optimization direction flag (default True)
- `median_fitness: Fitness | None` — median fitness value
- `q1_fitness: float | None` — 25th percentile fitness
- `q3_fitness: float | None` — 75th percentile fitness
- `min_fitness: Fitness | None` — raw minimum fitness (direction-independent)
- `max_fitness: Fitness | None` — raw maximum fitness (direction-independent)
- `fitness_range: float | None` — max_val - min_val
- `unique_fitness_count: int | None` — count of distinct fitness values

**Behavioral change**: `best_fitness`/`worst_fitness` computed using `minimize` flag (argmin for minimize=True, argmax for minimize=False).

### Genome Protocol

**Added method**: `distance(self, other: Self) -> float` — representation-aware distance between two genomes.

### VectorGenome

**Implements**: `distance()` using L2 (Euclidean) norm: `np.linalg.norm(self.genes - other.genes)`

### SequenceGenome

**Implements**: `distance()` using Levenshtein edit distance (dynamic programming, no external dependency).

### UnifiedConfig (frozen dataclass)

**Added fields**:
- `training_data: DatasetConfig | None` — training dataset specification (default None)
- `validation_data: DatasetConfig | None` — validation/test dataset specification (default None)

### DatasetConfig (new frozen dataclass)

**Fields**:
- `name: str` — human-readable dataset name
- `path: str | None` — filesystem path or URI to the data (default None)
- `data: Any | None` — in-memory data reference (default None, not serialized)
- `context: str` — MLflow context string (default "training" or "validation")

**Methods**: `to_dict()`, `from_dict()` (excludes `data` field from serialization)

### Callback Protocol

**Added property**: `priority: int` — execution order (lower = earlier, default 0)

### TrackingCallback

**Changed defaults**: `priority = 1000` (runs after user callbacks)

**New behavior in `on_run_start()`**:
- Calls `mlflow.set_tags(tags_dict)` when tags present in config
- Calls `mlflow.log_input()` for training_data/validation_data when present

### EvolutionEngine

**Changed `run()` signature**: `callbacks` parameter merged with `self._callbacks` instead of replacing.

**New in `_compute_metrics()`**: Extended fitness distribution, genome diversity (via `distance()` protocol), and search movement metrics. Gated by `TrackingConfig.categories`.

**New state**: `_prev_centroid`, `_prev_best_genome` for cross-generation movement metrics.

### MetaEvaluator

**New behavior**: Creates parent MLflow run, inner trials as nested child runs with tags.

## Entity Relationships

```
UnifiedConfig ──has──> DatasetConfig (training_data, validation_data)
UnifiedConfig ──has──> TrackingConfig ──has──> MetricCategory (gates metrics)
EvolutionEngine ──uses──> Population ──computes──> PopulationStatistics
EvolutionEngine ──dispatches──> Callback[] (sorted by priority)
EvolutionEngine ──computes──> metrics dict ──consumed by──> TrackingCallback
MetaEvaluator ──creates──> parent MLflow run ──nests──> child MLflow runs
Genome ──implements──> distance() ──used by──> EvolutionEngine (diversity metrics)
```
