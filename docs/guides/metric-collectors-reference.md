# Metric Collectors Reference Guide

This document is the authoritative reference for all metric collectors in `evolve/experiment/collectors/`.
Each section describes one collector: how to enable it, what metrics it produces, the mathematical
formula for each metric, a plain-language intuition, an evolutionary interpretation of high and low
values, and a degenerate-case note explaining what is returned (and why) when the formula is undefined
or the denominator is zero.

> **Maintenance note**: This file is hand-maintained. When adding or changing a collector, update this
> guide to keep it in sync with the source.

---

## Summary Table

| Collector | Enabling Mechanism | Primary Metric Keys |
|-----------|-------------------|---------------------|
| `DerivedAnalyticsCollector` | `MetricCategory.DERIVED` | `selection_pressure`, `fitness_improvement_velocity`, `population_entropy` |
| `ERPMetricCollector` | `MetricCategory.ERP` (auto-enabled when ERP reproduction active) | `mating_success_rate`, `attempted_matings`, `successful_matings`, per-protocol rates |
| `FitnessMetadataCollector` | `MetricCategory.METADATA` | Dynamic: `meta_<field>_mean`, `meta_<field>_std`, etc. |
| `IslandsMetricCollector` | Auto-enabled when island model active | `inter_island_variance`, `intra_island_variance`, `migration_events` |
| `MergeMetricCollector` | `MetricCategory.SYMBIOGENESIS` engine guard (manual instantiation) | `merge/count`, `merge/mean_genome_complexity`, `merge/complexity_delta` |
| `MultiObjectiveMetricCollector` | `MetricCategory.MULTIOBJECTIVE` (auto-enabled when MO active) | `pareto_front_size`, `hypervolume`, `crowding_diversity`, `spread` |
| `NEATMetricCollector` | Manual instantiation (no `MetricCategory` gate) | `average_node_count`, `average_connection_count`, `topology_innovations` |
| `SpeciationMetricCollector` | `MetricCategory.SPECIATION` | `species_count`, `average_species_size`, `species_births`, `species_extinctions`, `stagnation_count` |
| `EnsembleMetricCollector` | `MetricCategory.ENSEMBLE` (explicit only, not auto-enabled) | `ensemble/gini_coefficient`, `ensemble/participation_ratio`, `ensemble/top_k_concentration`, `ensemble/expert_turnover`\*, `ensemble/specialization_index`\* |

\* conditional on optional context fields

---

## DerivedAnalyticsCollector

**Enabling mechanism**: Add `MetricCategory.DERIVED` (or `"derived"`) to `TrackingConfig.categories`.
The engine instantiates this collector automatically.

**Class**: `evolve.experiment.collectors.derived.DerivedAnalyticsCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `selection_pressure` | $f_{\text{best}} / f_{\text{mean}}$ | How much better the best individual is than average | **High**: strong directional selection, risk of premature convergence. **Low**: weak selection, population drifts randomly. | Returns `0.0` when `mean_fitness` is zero or unavailable. |
| `fitness_improvement_velocity` | $\Delta f_{\text{best}} / \Delta t$ over a sliding window | How fast the best fitness is improving per generation | **High**: active progress. **Low**: plateau or stagnation; consider mutation rate increase or diversity injection. | Returns `0.0` when fewer generations than the window have elapsed or no prior best exists. |
| `population_entropy` | $H = -\sum_b p_b \log_2 p_b$ over histogram bins of fitness values | Shannon entropy of the fitness distribution | **High**: diverse fitness landscape, good exploration. **Low**: population collapsed to a single fitness value; diversity has been lost. | Returns `0.0` when all individuals have identical fitness (single bin with $p = 1$). |

---

## ERPMetricCollector

**Enabling mechanism**: Add `MetricCategory.ERP` (or `"erp"`) to `TrackingConfig.categories`.
Auto-enabled by the engine when ERP reproduction is active.

**Class**: `evolve.experiment.collectors.erp.ERPMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `attempted_matings` | Count of mating attempts | Raw throughput of the mating system | **High**: many mating opportunities. **Low**: reproductive bottleneck. | Returns `0.0` when no mating stats are provided in context. |
| `successful_matings` | Count of matings producing valid offspring | How productive the mating system is | **High**: high compatibility across protocols. **Low**: poor protocol matching or excessive rejection. | Returns `0.0` when no mating stats are provided. |
| `mating_success_rate` | $\text{successful} / \text{attempted}$ | Fraction of mating attempts that succeed | **High (near 1.0)**: protocols well-matched to population. **Low (near 0.0)**: widespread incompatibility; tune protocol compatibility thresholds. | Returns `0.0` when `attempted_matings` is zero — no attempts means no success. |
| `<protocol>_success_rate` | $\text{successes}_p / \text{attempts}_p$ per protocol $p$ | Per-protocol mating efficiency | **High**: this protocol is compatible with the current population. **Low**: this protocol is mismatched; consider removing or rebalancing. | Returns `0.0` when `attempts_p` is zero for a given protocol. |

---

## FitnessMetadataCollector

**Enabling mechanism**: Add `MetricCategory.METADATA` (or `"metadata"`) to `TrackingConfig.categories`.
The engine instantiates this collector automatically. Metric keys are discovered dynamically from
`Fitness.metadata` fields present in the population.

**Class**: `evolve.experiment.collectors.metadata.FitnessMetadataCollector`

| Metric Key Pattern | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-------------------|---------|-----------|----------------------------|--------------------------|
| `meta_<field>_mean` | $\bar{v} = \frac{1}{n}\sum_i v_i$ for field `<field>` | Population-average of a domain-specific scalar | Tracks the evolution of any measurable property of the solutions. | Returns nothing (field omitted) when fewer than `threshold` fraction of individuals have the field. |
| `meta_<field>_std` | $\sigma = \sqrt{\frac{1}{n}\sum_i (v_i - \bar{v})^2}$ | Spread of domain-specific values | **Low**: population converging on a uniform property value. **High**: diverse properties across solutions. | Returns `0.0` when all values are identical (variance is zero). |
| `meta_<field>_best` | $\max_i v_i$ (or $\min_i v_i$ for minimize) | Best domain-specific value in the population | Tracks the frontier of improvement on user-defined metrics. | Returns the single value when only one individual has the field. |

---

## IslandsMetricCollector

**Enabling mechanism**: Auto-enabled by the engine when the island model is active
(i.e., `context.island_populations` is not `None`). No explicit `MetricCategory` required.

**Class**: `evolve.experiment.collectors.islands.IslandsMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `inter_island_variance` | $\text{Var}(\bar{f}_1, \bar{f}_2, \ldots, \bar{f}_k)$ across island mean fitnesses | How different the islands' fitness levels are from each other | **High**: islands diverging — good for diversity. **Low**: islands converging — migration may be too frequent. | Returns `0.0` when fewer than 2 islands are present. |
| `intra_island_variance` | $\frac{1}{k}\sum_{i=1}^{k}\text{Var}(f_{ij})$ averaged within each island | Average within-island diversity | **High**: healthy within-island exploration. **Low**: all islands are individually converged. | Returns `0.0` when all islands contain a single individual or have uniform fitness. |
| `migration_events` | Count of individuals migrated this generation | Volume of genetic exchange between islands | **High**: frequent mixing; may reduce inter-island diversity. **Low**: islands evolving independently. | Returns `0` when migration tracking is disabled or no migrations occurred. |

---

## MergeMetricCollector

**Enabling mechanism**: Manual instantiation by the engine, guarded by
`MetricCategory.SYMBIOGENESIS` (`"symbiogenesis" in config.metric_categories`)
and requires a `SymbiogeneticMerge` operator to be configured.
This collector has a stateful `record_merge()` call-site in the engine;
it is not a pure-context collector.

**Class**: `evolve.experiment.collectors.merge.MergeMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `merge/count` | Integer count of merge operations per generation | How many symbiogenetic events occurred | **High**: active symbiogenesis phase. **Low (0)**: no merges triggered; check `merge_rate` config. | Returns `0` when no merges were recorded (`record_merge()` never called). |
| `merge/mean_genome_complexity` | $\frac{1}{n}\sum_i \lvert\text{genes}(m_i)\rvert$ over merged offspring | Average size of merged offspring genomes | **Growing**: merges consistently increase genome complexity. **Stable**: host and symbiont contribute balanced gene counts. | Returns `0.0` when no merges occurred. |
| `merge/complexity_delta` | $\frac{1}{n}\sum_i (\lvert\text{genes}(m_i)\rvert - \lvert\text{genes}(h_i)\rvert)$ | Mean complexity increase caused by each merge | **High**: each merge substantially grows the genome. **Near 0**: symbiont genes largely duplicate host genes. | Returns `0.0` when no merges occurred. |

---

## MultiObjectiveMetricCollector

**Enabling mechanism**: Add `MetricCategory.MULTIOBJECTIVE` (or `"multiobjective"`) to
`TrackingConfig.categories`. Auto-enabled when multi-objective fitness is active.
Requires `context.pareto_front` to be populated.

**Class**: `evolve.experiment.collectors.multiobjective.MultiObjectiveMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `pareto_front_size` | $\lvert\mathcal{F}_0\rvert$ (cardinality of front) | How many non-dominated solutions exist | **Growing**: increasing solution diversity. **Shrinking**: convergence to a narrow region of the Pareto front. | Returns `0` when no Pareto front is provided or it is empty. |
| `hypervolume` | Dominated volume $\lambda\bigl(\bigcup_{x \in \mathcal{F}_0} [x, r]\bigr)$ relative to reference point $r$ | Total quality of the front as a single scalar | **High**: front dominates a large objective space volume. **Low**: front near the reference point; quality and spread are poor. | Returns `0.0` when the front is empty or all front points are dominated by the reference point. |
| `crowding_diversity` | $\frac{1}{\lvert\mathcal{F}_0\rvert}\sum_{i} d_i^{\text{crowd}}$ mean crowding distance | How evenly spread the front solutions are | **High**: well-distributed front. **Low**: solutions clustered in one region. | Returns `0.0` when the front has fewer than 3 solutions (crowding distance is undefined). |
| `spread` | $\Delta = \frac{d_f + d_l + \sum_{i=1}^{n-1}\lvert d_i - \bar{d}\rvert}{d_f + d_l + (n-1)\bar{d}}$ | Uniformity of distribution along the front | **Near 0**: perfect spread. **Near 1**: clustered, poor coverage of extreme solutions. | Returns `0.0` when fewer than 2 solutions exist on the front. |

---

## NEATMetricCollector

**Enabling mechanism**: Manual instantiation — there is no `MetricCategory` gate for NEAT.
Instantiate `NEATMetricCollector` directly and add it to your tracking setup.
Works with graph-based genomes that expose `nodes` and `connections` attributes.

**Class**: `evolve.experiment.collectors.neat.NEATMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `average_node_count` | $\frac{1}{n}\sum_i \lvert\text{nodes}(g_i)\rvert$ | Mean network size across all genomes | **Growing**: NEAT is discovering increasingly complex networks. **Stable**: complexity is bounded or structural innovations are being pruned. | Returns `0.0` when no individuals have a genome with a `nodes` attribute. |
| `average_connection_count` | $\frac{1}{n}\sum_i \lvert\text{connections}(g_i)\rvert$ | Mean connectivity density across genomes | **High**: dense networks with many interaction paths. **Low**: sparse, efficient topologies. | Returns `0.0` when no individuals have a genome with a `connections` attribute. |
| `topology_innovations` | Count of structurally new genes (innovation numbers) seen for the first time | How many new structural mutations occurred | **High**: active exploration of the topology space. **Low**: population has converged structurally; no new innovations being introduced. | Returns `0` when `track_innovations=False` or no individuals have novel genes. |

---

## SpeciationMetricCollector

**Enabling mechanism**: Add `MetricCategory.SPECIATION` (or `"speciation"`) to
`TrackingConfig.categories`. Requires `context.species_info` to be populated by the speciation
component.

**Class**: `evolve.experiment.collectors.speciation.SpeciationMetricCollector`

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `species_count` | $\lvert\{s : s \in \text{species\_info}\}\rvert$ | Number of active species | **High**: rich speciation; diverse niches being explored. **Low (1)**: population collapsed to a single species — diversity mechanism inactive. | Returns `0.0` when `species_info` is `None` or empty. |
| `average_species_size` | $\frac{N}{|\text{species}|}$ | Mean individuals per species | **High**: few large species. **Low**: many small, specialised species. | Returns `0.0` when there are no species. |
| `species_births` | Count of species IDs not present in the previous generation | How many new niches were discovered | **High**: active speciation, new niches being opened. **Low**: speciation is stable or converged. | Returns `0` on the first generation (no prior species to compare against). |
| `species_extinctions` | Count of species IDs from previous generation no longer present | How many niches were lost | **High**: volatile speciation; many niches collapsing. **Low**: stable species ecosystem. | Returns `0` on the first generation. |
| `stagnation_count` | Count of species whose best fitness has not improved in recent generations | Number of species making no progress | **High**: many species are stuck; increase mutation rate or apply fitness sharing. **Low**: most species are actively improving. | Returns `0` when `track_dynamics=False` or on the first generation. |

---

## EnsembleMetricCollector

**Enabling mechanism**: Add `MetricCategory.ENSEMBLE` (or `"ensemble"`) to
`TrackingConfig.categories`. This category is **not** auto-enabled — it must be explicitly added.
Unlike other categories derived from `UnifiedConfig`, `ENSEMBLE` has no corresponding structural
prerequisite; it is always safe to enable.

```python
from evolve.config.tracking import TrackingConfig, MetricCategory

tracking = TrackingConfig(
    categories=frozenset({MetricCategory.CORE, MetricCategory.ENSEMBLE})
)
```

**Class**: `evolve.experiment.collectors.ensemble.EnsembleMetricCollector`

**Parameters**:
- `top_k_percent: float = 10.0` — Percentage of top individuals used for top-k concentration and (when `elite_size` is `None`) expert turnover. Must be in `(0.0, 100.0]`.
- `elite_size: int | None = None` — Fixed elite set size for expert turnover. When `None`, derived as `ceil(top_k_percent / 100 × N)`.

### Always-Present Metrics

| Metric Key | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|---------|-----------|----------------------------|--------------------------|
| `ensemble/gini_coefficient` | $G = \dfrac{\sum_i\sum_j \lvert f_i - f_j\rvert}{2N\sum_i f_i}$, range $[0, 1]$ | Fitness inequality across the population (borrowed from economics) | **High (→ 1)**: one or few individuals dominate fitness — severe fitness concentration, risk of early convergence. **Low (→ 0)**: all individuals contribute equally, good ensemble diversity. | Returns `0.0` when total fitness is zero — uniform-zero is the maximum-equality limit, Gini = 0 is the correct semantic interpretation. |
| `ensemble/participation_ratio` | $PR = \dfrac{(\sum_i f_i)^2}{\sum_i f_i^2}$, range $[1, N]$ | Effective number of contributing individuals (borrowed from statistical physics / inverse participation ratio) | **High (≈ N)**: all individuals contribute roughly equally. **Low (≈ 1)**: a single individual dominates all fitness; ensemble is degenerate. | Returns `float(population_size)` when total fitness is zero — when all individuals are identically zero, all N contribute equally in the uniform limit, so PR = N is the correct semantic answer. |
| `ensemble/top_k_concentration` | $C_k = \dfrac{\sum_{i \in \text{top-}k} f_i}{\sum_i f_i}$, range $[0, 1]$ | Fraction of total fitness held by the top-$k$% | **High (→ 1)**: the top fraction monopolises fitness, likely indicating star topology or runaway selection. **Low (→ 0)**: fitness is dispersed across the population. | Returns `0.0` when total fitness is zero — when no individual has nonzero fitness, no individual concentrates it. |

### Conditionally-Present Metrics

| Metric Key | Condition | Formula | Intuition | Evolutionary Interpretation | Degenerate-Case Behavior |
|-----------|-----------|---------|-----------|----------------------------|--------------------------|
| `ensemble/expert_turnover` | `context.previous_elites is not None` | $T = \dfrac{\lvert\text{elite}_t \setminus \text{elite}_{t-1}\rvert}{\lvert\text{elite}_t\rvert}$, range $[0, 1]$, where elite membership is compared by stable individual key (`ind.id` when present, otherwise `id(ind)`) | Fraction of the elite set that is new this generation | **High (→ 1)**: elite membership is volatile; fitness landscape may be rugged or the search is still exploring. **Low (→ 0)**: elite is stable; the search has found a robust set of high-fitness solutions. | Returns `1.0` when `previous_elites` is an empty list — all current elite members are new by definition (no prior elite to compare against). Key is **omitted** when `previous_elites is None`. |
| `ensemble/specialization_index` | `context.species_info is not None` | $\eta^2 = SS_{\text{between}} / SS_{\text{total}}$; $SS_{\text{between}} = \sum_s n_s(\bar{f}_s - \bar{f})^2$; $SS_{\text{total}} = \sum_i (f_i - \bar{f})^2$; range $[0, 1]$ | Fraction of total fitness variance explained by between-species differences (eta-squared from ANOVA) | **High (→ 1)**: species have highly divergent fitness levels — strong specialization across niches. **Low (→ 0)**: species have similar fitness distributions — no niche specialization, speciation is not contributing to fitness diversity. | Returns `0.0` when $SS_{\text{total}} = 0$ — a fully converged population (all individuals identical fitness) has no variance for species membership to explain; zero specialization is the correct semantic interpretation. Key is **present** when `species_info` is not `None`, even if the value is `0.0`. |
