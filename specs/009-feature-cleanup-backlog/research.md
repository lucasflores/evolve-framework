# Research: Feature & Cleanup Backlog

## R1: MLflow Nested Run API for Meta-Evolution

**Decision**: Use `mlflow.start_run(nested=True)` to create child runs within a parent meta-evolution run.

**Rationale**: MLflow natively supports nested runs. The parent run ID is automatically set. Child runs appear indented under the parent in the MLflow UI. Tags can be set on child runs for additional filtering.

**Alternatives considered**:
- Flat runs with tag-based linking: simpler but loses UI hierarchy
- Experiment-per-meta-run: too much overhead, fragments results

## R2: MLflow Native Dataset Logging

**Decision**: Use `mlflow.log_input()` with `mlflow.data.from_numpy()`, `mlflow.data.from_pandas()`, or `mlflow.data.from_dict()` depending on the data type. Context parameter distinguishes "training" vs "validation" vs "test".

**Rationale**: MLflow's native dataset API supports numpy arrays, pandas DataFrames, and dict-based data. The `context` parameter maps directly to data split semantics.

**Alternatives considered**:
- Custom artifact logging: loses MLflow UI integration for Datasets tab
- Path-only logging: doesn't capture schema/digest metadata

## R3: MLflow Native Tags vs Parameters

**Decision**: Use `mlflow.set_tags(tags_dict)` for native tag population. Continue logging as flattened parameters for backward compatibility.

**Rationale**: MLflow tags are string→string. UnifiedConfig tags are already string→string. Direct mapping. Tags appear in the MLflow UI Tags column and are searchable via `mlflow.search_runs(filter_string="tags.key = 'value'")`.

**Alternatives considered**:
- Tags only (drop parameter logging): breaks backward compatibility
- Parameters only (status quo): misses native Tags field

## R4: Genome Distance Protocol Design

**Decision**: Add `distance(self, other: Self) -> float` method to the `Genome` protocol. Provide implementations for `VectorGenome` (L2/Euclidean), `SequenceGenome` (Levenshtein edit distance). For genome types that don't implement `distance()`, the engine's diversity metrics computation should skip diversity metrics gracefully (log a warning once, set metrics to None).

**Rationale**: Protocol method is Pythonic, avoids external dispatch, and aligns with existing Genome protocol pattern. L2 is standard for continuous spaces; edit distance is standard for sequences.

**Alternatives considered**:
- External DistanceRegistry: adds indirection, harder to discover
- Abstract base class: too rigid for plugin genomes

## R5: Callback Priority Mechanism

**Decision**: Add `priority: int` property to the Callback protocol with default 0. The engine sorts callbacks by priority (ascending, stable) before dispatching events. TrackingCallback defaults to priority 1000.

**Rationale**: Numeric priorities are simple, familiar (CSS z-index, task scheduler patterns), and composable. Stable sort preserves registration order for equal priorities.

**Alternatives considered**:
- Dependency graph: over-engineered for current needs
- Phase splitting: requires protocol changes, breaks backward compatibility

## R6: Population Statistics Minimize Flag

**Decision**: Add `minimize: bool` field to `PopulationStatistics` (frozen dataclass). `_compute_statistics()` accepts `minimize` parameter and uses `np.argmin` or `np.argmax` accordingly. Default `minimize=True` preserves backward compatibility.

**Rationale**: Minimal change. Field name preservation avoids breaking API consumers. The flag makes the dataclass self-describing.

**Alternatives considered**:
- Rename to min_fitness/max_fitness: breaking change for all consumers
- Both fields: redundant, confusing API surface

## R7: Pairwise Distance Sampling Strategy

**Decision**: For populations > 100 individuals, sample 100 random pairs for mean pairwise distance calculation. Use explicit RNG for reproducibility.

**Rationale**: O(n²) pairwise distance is prohibitive for large populations. Sampling provides a statistically valid estimate with O(k) complexity where k is the sample size.

**Alternatives considered**:
- Full O(n²): too slow for populations > 1000
- No sampling (skip metric): loses valuable information
