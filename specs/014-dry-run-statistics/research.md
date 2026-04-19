# Research: Dry-Run Statistics Tool

**Feature**: 014-dry-run-statistics  
**Date**: 2026-04-18  
**Status**: Complete

## Research Tasks

### RT-1: Micro-Benchmark Strategy for Atomic Operations

**Context**: The dry-run tool must time a single invocation of each atomic operation and multiply by structural constants. Need to determine how to set up each micro-benchmark without running a full engine.

**Decision**: Create a small set of sample individuals (2–3) using the existing `GenomeRegistry.create()` with `config.genome_params`, then invoke each operator once:
- **Evaluation**: Call `evaluator.evaluate([individual], seed=seed)` — a batch of 1 for sequential, or a minimal batch for parallel/GPU backends.
- **Selection**: Call `selection.select(sample_population, n=2, rng=rng)`.
- **Crossover**: Call `crossover.crossover(parent1_genome, parent2_genome, rng)`.
- **Mutation**: Call `mutation.mutate(genome, rng)`.
- **Merge** (if enabled): Call `merge.merge(individual1, individual2, rng)`.
- **Initialization**: Time `GenomeRegistry.create()` × 1.

**Rationale**: Each operator is a pure function on small inputs. The existing `timing_context()` utility provides wall-clock + CPU time for each call. No full engine setup needed — operators and evaluator can be resolved directly from the registry/factory.

**Alternatives considered**:
- Running sample generations (rejected: unnecessarily expensive, doesn't exploit structural symmetry).
- Analytical-only estimation (rejected: no real timing data; accuracy depends on complexity models that may not match actual implementations).

---

### RT-2: Backend-Aware Evaluation Benchmarking

**Context**: For parallel/accelerated backends, a single-individual evaluation doesn't capture real batch behavior (process pool overhead, GPU kernel launch, JIT compilation). Need to benchmark through the actual backend.

**Decision**: Use the backend's native batch interface:
- **Sequential**: Evaluate 1 individual, multiply by `population_size`.
- **Parallel**: Evaluate a minimal batch of `2 × n_workers` individuals via `ParallelBackend.map_evaluate()`. Compute per-individual time = total / batch_size, then multiply by `population_size`. This captures process pool startup, serialization overhead, and real parallelism.
- **JAX/PyTorch**: Evaluate a minimal batch (e.g., 32 or `population_size` if small) via the accelerated evaluator. This captures JIT warm-up (JAX) and GPU kernel launch overhead. Report JIT compilation time separately if detectable.

**Rationale**: The evaluator's `evaluate()` method is already a batch interface (`Sequence[Individual] → Sequence[Fitness]`). Benchmarking through it with a small batch captures real backend behavior without running a full generation.

**Alternatives considered**:
- Benchmark single individual, apply theoretical scaling (rejected: doesn't capture parallelism overhead, GPU kernel launch costs, or JIT warm-up).

---

### RT-3: Resource Auto-Detection Best Practices

**Context**: Need to detect CPU count, memory, and GPU without adding new external dependencies.

**Decision**: Compose from existing codebase patterns:
- **CPU count**: Reuse `_get_cpu_count()` from `evolve/backends/parallel.py` (already container-aware via cgroup inspection).
- **Memory**: Use `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` on Linux, `os.sysconf('SC_PHYS_PAGES')` on macOS. Fallback: None if unavailable.
- **GPU**: Use `evolve/utils/dependencies.py` `check_dependency()` to probe for torch/jax, then conditionally import and query device info using existing patterns from backends (e.g., `torch.cuda.is_available()`, `jax.devices()`).
- **No psutil**: Avoid new external deps — stdlib is sufficient for CPU/memory; GPU detection reuses existing optional dep patterns.

**Rationale**: All patterns already exist in the codebase. Composing them avoids new dependencies and ensures consistency.

**Alternatives considered**:
- Adding `psutil` as a dependency (rejected: violates minimal dependency principle; stdlib covers our needs).
- Skipping resource detection (rejected: spec requires it; hardware context is essential for interpreting estimates).

---

### RT-4: Structural Constants Derivation from UnifiedConfig

**Context**: Need to know exactly what multipliers to derive from the config for each phase.

**Decision**: The structural constants per generation are:
- **Evaluation**: `population_size` evaluations per generation (all individuals evaluated, including elites re-evaluated in initial population).
- **Selection**: `n_parents = (population_size - elitism) * 2` parent selections per generation.
- **Crossover**: `n_offspring = population_size - elitism` offspring (one crossover per pair = `n_offspring` crossover calls, applied with probability `crossover_rate`).
- **Mutation**: `n_offspring` mutation calls (applied with probability `mutation_rate`).
- **Merge**: `population_size * merge_rate` merge operations per generation (if merge enabled).
- **Total generations**: `max_generations` from config.
- **Initialization**: One-shot cost of `population_size` genome creations.

These are directly derivable from `EvolutionConfig` fields: `population_size`, `max_generations`, `elitism`, `crossover_rate`, `mutation_rate`, `merge_rate`.

**Rationale**: These constants mirror the exact loop structure in `EvolutionEngine._step()`. The math is transparent and verifiable.

---

### RT-5: Memory Estimation Approach

**Context**: Need to estimate peak memory usage without running the full pipeline.

**Decision**: Measure the byte size of a single sample genome (via `sys.getsizeof` or numpy array `.nbytes` for vector genomes), then compute:
- **Population memory**: `genome_bytes × population_size × 2` (current + offspring overlap during replacement).
- **Individual overhead**: Add per-individual metadata (UUID, fitness, metadata dict) × population_size.
- **History memory**: `metrics_dict_size × max_generations` for the history list.
- Report as an order-of-magnitude estimate with the measured per-genome size disclosed.

**Rationale**: Genome size is the dominant memory factor. Measuring one genome's actual size and multiplying is more accurate than analytical models, especially for variable-size genomes (graph, SCM).

**Alternatives considered**:
- Full `tracemalloc` profiling of a sample generation (rejected: too expensive for a dry-run; overkill for an estimate).

---

### RT-6: ASCII Table Output Format

**Context**: The report must include a human-readable ASCII table.

**Decision**: Use stdlib `str` formatting — no external deps (no `rich`, `tabulate`). Format:

```
╔══════════════════╦═══════════════╦═════════╦════════════╗
║ Phase            ║ Est. Time     ║ % Total ║ Bottleneck ║
╠══════════════════╬═══════════════╬═════════╬════════════╣
║ Initialization   ║     0.12s     ║   0.1%  ║            ║
║ Evaluation       ║   312.50s     ║  72.3%  ║     ★      ║
║ Selection        ║    23.40s     ║   5.4%  ║            ║
║ Variation        ║    89.00s     ║  20.6%  ║            ║
║ Merge            ║     7.10s     ║   1.6%  ║            ║
╠══════════════════╬═══════════════╬═════════╬════════════╣
║ TOTAL            ║   432.12s     ║ 100.0%  ║            ║
╚══════════════════╩═══════════════╩═════════╩════════════╝
```

Implement as a `__str__()` or `summary()` method on `DryRunReport`.

**Rationale**: Box-drawing characters render correctly in modern terminals, notebooks, and log files. No external dependency needed.

**Alternatives considered**:
- Plain text with alignment (rejected: less readable).
- `rich` library (rejected: new dependency for a single table).

---

### RT-7: Meta-Evolution Cost Estimation

**Context**: Meta-evolution (`config.meta`) runs an outer loop that evolves configurations, where each candidate config is evaluated by running a complete inner evolutionary run. This is an exponential cost multiplier that the dry-run tool must surface.

**Decision**: When `config.is_meta_evolution` is True:
1. Estimate the cost of a single inner evolutionary run by applying the standard dry-run estimation to the inner config (with `config.meta.inner_generations` overriding `max_generations` if set).
2. Multiply inner run cost by `outer_generations × trials_per_config` to get total meta-evolution cost.
3. Report as a hierarchical breakdown: inner run estimate → outer multiplication → total.
4. Check `MetaEvaluator`'s config hash caching — if enabled, note that re-evaluation of identical configs is avoided, but worst-case assumes no cache hits.

**Rationale**: Meta-evolution transforms the cost model from linear (generations × per-gen cost) to multiplicative (outer × trials × inner run cost). The inner run cost uses the same estimation logic recursively. Caching makes actual cost variable, so we report worst-case.

**Alternatives considered**:
- Ignoring meta-evolution and letting users multiply manually (rejected: defeats the purpose of the tool; meta-evolution is the easiest cost to miscalculate).
- Benchmarking a full inner trial (rejected: defeats the purpose of a dry-run; the inner estimate should use micro-benchmarks).

---

### RT-8: ERP Overhead Benchmarking

**Context**: ERP (`config.erp`) adds per-generation overhead: intent evaluation for each candidate parent, and matchability checking between potential mating pairs. Both involve policy evaluation and genetic distance computation.

**Decision**: When `config.is_erp_enabled` is True:
1. **Intent evaluation**: Benchmark one intent policy evaluation call. Multiply by `population_size` (every individual evaluated for intent per generation).
2. **Matchability**: Benchmark one matchability check (includes genome distance computation). Multiply by estimated attempted matings — conservatively `population_size / 2` pairs (bidirectional check means `population_size` matchability evaluations).
3. **Protocol mutation**: Included in variation cost (same complexity class as regular mutation).
4. **Recovery**: Not benchmarked — recovery is conditional (triggers only below `recovery_threshold`). Report as a caveat: "ERP recovery overhead not included; triggers only if mating success rate < {threshold}."

**Rationale**: Intent and matchability are the dominant ERP costs. Recovery is unpredictable and conditional. Protocol mutation is O(1) per offspring, negligible vs. intent/matchability.

**Alternatives considered**:
- Including worst-case recovery cost (rejected: recovery is rare and highly variable; including it would overestimate for typical runs).
- Benchmarking genetic distance separately (rejected: matchability check already includes distance computation).

---

### RT-9: NSGA-II Ranking Overhead Benchmarking

**Context**: Multi-objective configurations (`config.multiobjective`) replace single-objective selection with NSGA-II environmental selection: fast non-dominated sort O(M×N²) + crowding distance O(M×N log N).

**Decision**: When `config.is_multiobjective` is True:
1. Create a sample population with random multi-objective fitness values (M objectives from config).
2. Benchmark one call to `fast_non_dominated_sort()` with the full population size — cost is O(M×N²) so using the actual N is important for accuracy.
3. Benchmark one call to `crowding_distance()` on the first front.
4. Report as a "ranking" phase with per-generation cost = sort_time + crowding_time.
5. If constraints are configured, benchmark constraint violation computation and include in ranking cost.

**Rationale**: NSGA-II sort cost is superlinear in N, so benchmarking with the actual population size is necessary. Using a smaller sample would underestimate.

**Alternatives considered**:
- Analytical estimation only (rejected: constant factors in the Python implementation matter; O(N²) with a large constant can be significant).
- Benchmarking with a small sample and scaling (rejected: O(N²) scaling makes small-sample benchmarks unreliable for large populations).

---

### RT-10: Decoder Overhead Benchmarking

**Context**: When `config.decoder` is configured, every individual must be decoded (genome → phenotype) before evaluation. For graph genomes, this involves topological sort O(nodes+edges) per individual.

**Decision**: When `config.decoder` is set:
1. Resolve the decoder from the registry.
2. Benchmark one decode operation on a sample genome.
3. Report decode cost as part of the evaluation phase: per-individual eval cost = decode_time + fitness_eval_time.
4. Structural multiplier: decode_time × population_size × generations.

**Rationale**: Decoding happens per-individual, before evaluation. It's a direct multiplicative cost on the evaluation phase. For simple genomes (vectors), decode is trivial or absent. For complex genomes (graphs, SCMs), it can add 20-50% to evaluation cost.

**Alternatives considered**:
- Reporting decode as a separate phase (rejected: decode always pairs with evaluation; separating them would fragment the bottleneck analysis and make the table confusing).
- Ignoring decode for vector genomes (accepted: if no decoder is configured, this step is skipped automatically).

---

### RT-11: Tracking and Metric Collector Overhead

**Context**: MLflow tracking (`config.tracking`) adds per-generation overhead for logging metrics, and metric collectors compute derived analytics (hypervolume, selection pressure, entropy).

**Decision**:
1. **Local MLflow tracking**: If `config.tracking.enabled` and backend is local, benchmark one `log_metrics()` call. Multiply by `generations / log_interval`. Report as "tracking overhead".
2. **Remote MLflow tracking**: Cannot benchmark network latency. Report a caveat: "Remote MLflow tracking configured — per-generation overhead depends on network latency (typically 50-500ms per logged generation). Not included in estimate."
3. **Metric collectors**: If `config.tracking.categories` enables expensive collectors (e.g., multi-objective metrics with hypervolume), benchmark one collector pass. Multiply by generations. Report as part of tracking overhead.
4. **Overall approach**: Group tracking + collectors as "overhead" in the report — low priority for detailed benchmarking since they're typically <5% of total cost for local tracking.

**Rationale**: Tracking overhead is usually negligible for local backends but can be significant for remote servers. Since remote latency is unbenchmarkable, we report it as a caveat rather than a fabricated estimate.

**Alternatives considered**:
- Ignoring tracking overhead entirely (rejected: remote tracking can add 10%+ overhead for short generations).
- Benchmarking with a test HTTP request to the tracking server (rejected: adds network complexity; server latency varies).
