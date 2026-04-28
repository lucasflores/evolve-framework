# Feature Specification: Ensemble Metric Collector & Reference Guide

**Feature Branch**: `016-ensemble-metric-collector`  
**Created**: 2026-04-27  
**Status**: Draft  
**Input**: User description: "Add a new EnsembleMetricCollector to the evolve-framework experiment tracking system that treats the current generation as a mixture-of-experts ensemble for observability purposes only — the evolutionary process is not affected. The collector computes five new population-level metrics derived solely from each individual's fitness values and existing CollectionContext fields: Gini coefficient, Participation Ratio, Top-k Concentration, Expert Turnover, and Specialization Index. Gated by a new MetricCategory.ENSEMBLE enum value. Includes a reference guide documentation artifact covering all metric collectors."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Enable ensemble observability for a running experiment (Priority: P1)

A researcher running an evolutionary experiment wants richer insight into how fitness is distributed across the population — specifically whether a small clique of high-fitness individuals dominates, whether the "expert set" is stable or churning, and whether species specialization explains variance. They add `MetricCategory.ENSEMBLE` to their `TrackingConfig` and rerun without touching any other part of the config. All five new metrics appear in the MLflow run alongside existing metrics.

**Why this priority**: This is the entire value proposition of the feature. Everything else serves this scenario.

**Independent Test**: Can be fully tested by constructing a `CollectionContext` with a synthetic population and calling `EnsembleMetricCollector().collect(context)`. Delivers a populated `dict[str, float]` with the expected metric keys.

**Acceptance Scenarios**:

1. **Given** a `CollectionContext` with a population of N individuals each having a scalar fitness value, **When** `EnsembleMetricCollector().collect(context)` is called, **Then** the returned dict contains `ensemble/gini_coefficient`, `ensemble/participation_ratio`, `ensemble/top_k_concentration`, and all values are finite floats in [0, 1].
2. **Given** a `CollectionContext` where `previous_elites` is `None`, **When** `collect()` is called, **Then** `ensemble/expert_turnover` is absent from the result (not a `KeyError`, just omitted).
3. **Given** a `CollectionContext` where `species_info` is `None`, **When** `collect()` is called, **Then** `ensemble/specialization_index` is absent from the result.
4. **Given** a population where all individuals have identical fitness, **When** `collect()` is called, **Then** `ensemble/gini_coefficient` is `0.0` and `ensemble/participation_ratio` equals population size.
5. **Given** a population where one individual has all the fitness and the rest have zero, **When** `collect()` is called, **Then** `ensemble/gini_coefficient` approaches `(N-1)/N` and `ensemble/top_k_concentration` equals `1.0` when top-k% covers that individual.

---

### User Story 2 — Consult the metric collector reference guide (Priority: P2)

A developer or researcher new to the framework wants to understand which metrics are tracked, what they mean mathematically, and which `MetricCategory` to enable. They open the reference guide doc, find any collector by name, and read: the formula, the intuition, the interpretation in an evolutionary context, and the enabling category — all in one place.

**Why this priority**: Without the reference guide, the new metrics (and all existing ones) remain undiscoverable unless the user reads source code. The guide is the primary discoverability surface.

**Independent Test**: The reference guide file exists at `docs/guides/metric-collectors-reference.md`, contains a section for every collector in `evolve/experiment/collectors/`, and each section includes formula, intuition, interpretation, and enabling category.

**Acceptance Scenarios**:

1. **Given** the reference guide is published, **When** a user searches for `EnsembleMetricCollector`, **Then** they find a section listing all five metrics with formulas, plain-language intuition, evolutionary interpretation, and `MetricCategory.ENSEMBLE` as the enabling category.
2. **Given** the reference guide is published, **When** a user searches for any existing collector (e.g., `DerivedAnalyticsCollector`, `SpeciationMetricCollector`, `ERPMetricCollector`), **Then** they find a matching section with the same four fields.
3. **Given** a new collector is added to the framework in the future, **When** a maintainer looks at the reference guide, **Then** the existing structure provides a clear template for adding a new section.

---

### User Story 3 — Verify Specialization Index activates only with speciation (Priority: P3)

A researcher using speciation wants to know how much of the fitness variance is "explained" by species membership. They enable `MetricCategory.ENSEMBLE`. The `ensemble/specialization_index` metric appears only when `species_info` is present in the context, and its value increases as inter-species fitness divergence grows.

**Why this priority**: This is a conditional metric with clear guard behavior; incorrect activation would produce misleading values.

**Independent Test**: Two `CollectionContext` instances — one with `species_info` populated with divergent species fitnesses, one without — are used to confirm presence/absence of `ensemble/specialization_index` and that values increase monotonically with inter-species divergence.

**Acceptance Scenarios**:

1. **Given** `species_info` maps two species with highly divergent mean fitnesses, **When** `collect()` is called, **Then** `ensemble/specialization_index` is close to `1.0`.
2. **Given** `species_info` maps two species with identical fitness distributions, **When** `collect()` is called, **Then** `ensemble/specialization_index` is close to `0.0`.
3. **Given** `species_info` is `None`, **When** `collect()` is called, **Then** `ensemble/specialization_index` is not present in the result dict.

---

### Edge Cases

- What happens when the population is empty? Collector returns an empty dict without raising.
- What happens when all fitness values are zero? Gini = 0.0; Participation Ratio = `float(population_size)` (all N individuals contribute equally — consistent with the uniform-fitness limit); Top-k Concentration = 0.0 (top-k individuals hold zero of zero total fitness, which is defined as 0.0).
- What happens when `previous_elites` is an empty list (not `None`)? Expert Turnover is 1.0 (all current elites are new, no prior elite to match). Convention documented and tested.
- What happens when top-k% rounds to zero individuals? The collector clamps to at least one individual.
- What happens when a population has only one individual? Gini = 0.0, Participation Ratio = 1.0, Specialization Index = 0.0 if species_info is present (zero total variance → return 0.0 per Q3 convention), omitted if species_info is absent.
- What happens when fitness values are negative? Values are shifted by `abs(min(fitness))` before all computations; a single debug-level warning is logged per generation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The framework MUST expose a new `MetricCategory.ENSEMBLE` enum value (`"ensemble"`) in `evolve.config.tracking.MetricCategory` that can be added to `TrackingConfig.categories` like any existing category.
- **FR-002**: The framework MUST provide an `EnsembleMetricCollector` class in `evolve/experiment/collectors/ensemble.py` that satisfies the `MetricCollector` protocol (implements `collect(context: CollectionContext) -> dict[str, float]` and `reset() -> None`).
- **FR-003**: `EnsembleMetricCollector.collect()` MUST compute and return `ensemble/gini_coefficient` — a value in [0, 1] measuring fitness inequality across the current generation, where 0 = perfect equality and approaches 1 as one individual monopolises all fitness.
- **FR-004**: `EnsembleMetricCollector.collect()` MUST compute and return `ensemble/participation_ratio` — the effective number of contributing individuals, computed as $(\sum_i f_i)^2 / \sum_i f_i^2$ over non-negative fitness values. Equals population size when fitnesses are uniform; approaches 1 when one individual dominates.
- **FR-005**: `EnsembleMetricCollector.collect()` MUST compute and return `ensemble/top_k_concentration` — the fraction of total population fitness held by the top-`k`% of individuals, where `k` is configurable at construction time and defaults to 10. Returns a value in [0, 1].
- **FR-006**: `EnsembleMetricCollector.collect()` MUST compute and return `ensemble/expert_turnover` when `context.previous_elites` is not `None` — the fraction of the current elite set (top-k by fitness) whose identity is not present in `previous_elites`. Returns a value in [0, 1].
- **FR-007**: `EnsembleMetricCollector.collect()` MUST compute and return `ensemble/specialization_index` when `context.species_info` is not `None` — eta-squared: the fraction of total fitness variance explained by between-species variance. Returns a value in [0, 1].
- **FR-008**: When required optional context (`previous_elites` or `species_info`) is absent, the corresponding metric MUST be omitted from the returned dict rather than returning `NaN` or raising an exception.
- **FR-009**: All five metrics MUST be computed solely from `context.population` fitness values and existing `CollectionContext` fields. No changes to `Population`, `Individual`, `Fitness`, selection, reproduction, or evaluators are permitted.
- **FR-010**: `EnsembleMetricCollector` MUST be registered in `evolve/experiment/collectors/__init__.py` and exported from the collectors package `__all__`.
- **FR-011**: The framework MUST include a Markdown reference guide at `docs/guides/metric-collectors-reference.md` covering all ten metric collectors (nine existing + `EnsembleMetricCollector`) with a consistent section format per collector. Each section MUST document the enabling mechanism for that collector: its `MetricCategory` value, whether it is auto-enabled by engine configuration, or whether it requires manual instantiation (as is the case for `MergeMetricCollector` and `NEATMetricCollector`).
- **FR-012**: The `top_k_percent` parameter on `EnsembleMetricCollector` MUST be configurable at construction time and MUST default to `10.0` (representing 10% of the population).
- **FR-013**: The `elite_size` parameter controlling how many individuals constitute the "elite set" for Expert Turnover MUST be configurable at construction time and MUST default to `None` (auto-derived: `ceil(top_k_percent / 100 * population_size)`).
- **FR-014**: `EnsembleMetricCollector` MUST be instantiated in `evolve/core/engine.py` at engine `__init__` time, guarded by `"ensemble" in config.metric_categories` (matching the `frozenset[str]` dispatch pattern used by all other collectors in `EvolutionEngine`). Its `collect()` method MUST be called in the per-generation metric collection loop alongside other collectors. No new registry abstraction is introduced.
- **FR-015**: The reference guide section for each metric MUST include: metric key name, the mathematical formula, a plain-language intuition sentence, an evolutionary interpretation sentence describing what high and low values indicate, and a degenerate-case note stating what the metric returns (and why) when the formula denominator is zero or the input is otherwise undefined. For `ensemble/participation_ratio` this note MUST state: returns `float(population_size)` when total fitness is zero, because all individuals contribute equally in the limit of uniform zero fitness. For `ensemble/specialization_index` this note MUST state: returns `0.0` when total fitness variance is zero, because a converged population exhibits no species specialization by definition.

### Key Entities

- **EnsembleMetricCollector**: New collector class. Key attributes: `top_k_percent: float` (default 10.0), `elite_size: int | None` (default None, auto-derived). Implements `MetricCollector` protocol. Stateless — `reset()` is a no-op. Expert Turnover reads elite identity exclusively from `context.previous_elites`; the engine owns elite history.
- **MetricCategory.ENSEMBLE**: New enum value `"ensemble"` added to the existing `MetricCategory` enum. Controls instantiation of `EnsembleMetricCollector` in the engine. Not auto-enabled by `UnifiedConfig` — must be explicitly configured.
- **Metric Reference Guide**: Markdown document at `docs/guides/metric-collectors-reference.md`. One section per collector class (ten total) with a consistent structure: collector name, enabling mechanism (MetricCategory value / auto-enabled / manual instantiation), and a table with columns — metric key, formula, intuition, evolutionary interpretation, degenerate-case behavior (what is returned and why when the formula is undefined or the denominator is zero). `MergeMetricCollector` and `NEATMetricCollector` are included with their actual enabling mechanism noted rather than a MetricCategory value.
- **Gini Coefficient** (`ensemble/gini_coefficient`): $G = \frac{\sum_{i}\sum_{j}|f_i - f_j|}{2N\sum_i f_i}$ where $f_i$ are non-negative fitness values. Range [0, 1].
- **Participation Ratio** (`ensemble/participation_ratio`): $PR = \frac{(\sum_i f_i)^2}{\sum_i f_i^2}$. Range [1, N]. Borrowed from statistical physics; measures effective number of contributors. **Degenerate case**: returns `float(population_size)` when $\sum_i f_i = 0$ (or equivalently when all $f_i = 0$), because all N individuals contribute equally in the limit of uniform zero fitness — consistent with the uniform-fitness limit of the formula.
- **Top-k Concentration** (`ensemble/top_k_concentration`): $C_k = \frac{\sum_{i \in \text{top-}k} f_i}{\sum_i f_i}$ where top-k covers the $\lceil kN/100 \rceil$ highest-fitness individuals. Range [0, 1].
- **Expert Turnover** (`ensemble/expert_turnover`): $T = \frac{|\text{elite}_t \setminus \text{elite}_{t-1}|}{|\text{elite}_t|}$ using a stable individual identifier for set membership: `ind.id` when present (UUID preserved across `with_fitness()` calls), otherwise Python object identity via `id(ind)`. Range [0, 1]. Only present when `previous_elites` is not `None`.
- **Specialization Index** (`ensemble/specialization_index`): $\eta^2 = SS_{\text{between}} / SS_{\text{total}}$ where $SS_{\text{between}} = \sum_s n_s(\bar{f}_s - \bar{f})^2$ and $SS_{\text{total}} = \sum_i (f_i - \bar{f})^2$. Range [0, 1]. **Degenerate case**: returns `0.0` when $SS_{\text{total}} = 0$ (all individuals have identical fitness), because a fully converged population has no variance for species membership to explain — zero specialization is the correct semantic interpretation. Only present in result dict when `species_info` is not `None`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All five ensemble metrics are computable without error on any valid `CollectionContext` containing a non-empty population with scalar fitness values.
- **SC-002**: Each metric value falls within its documented range for all valid inputs — `gini_coefficient`, `top_k_concentration`, `expert_turnover`, and `specialization_index` ∈ [0, 1]; `participation_ratio` ∈ [1, N] — no metric produces `NaN`, `Inf`, or raises an exception on valid inputs including edge cases.
- **SC-003**: Enabling `MetricCategory.ENSEMBLE` in a `TrackingConfig` causes all applicable metric keys to appear in logged run data without any other configuration changes.
- **SC-004**: The reference guide covers 100% of the collector classes present in `evolve/experiment/collectors/` at the time of merge, with no collector left undocumented.
- **SC-005**: The reference guide section for each metric includes a formula, an intuition sentence, and an evolutionary interpretation sentence — fully readable without consulting source code.
- **SC-006**: Unit tests achieve branch coverage for all edge cases enumerated in the Edge Cases section (empty population, uniform fitness, monopoly fitness, zero-total fitness, missing optional context fields, negative fitness).
- **SC-007**: The Gini coefficient produces ground-truth values: uniform distribution → 0.0; one individual holds all fitness → $(N-1)/N$.
- **SC-008**: Expert Turnover correctly identifies identity changes: fully replaced elite set → 1.0; fully stable elite set → 0.0.

## Clarifications

### Session 2026-04-27

- Q: Expert Turnover implementation strategy — stateless (context-driven via `previous_elites`) vs. stateful (collector tracks elite history internally) vs. hybrid → A: Option A — stateless; collector reads elite identity purely from `context.previous_elites`; `reset()` is a no-op; engine owns elite history.
- Q: Participation Ratio return value when total fitness is zero (0/0 case) — return `float(population_size)`, return `1.0`, or omit metric → A: Option A — return `float(population_size)`; when all individuals are equally fit (including equally zero), all N individuals contribute equally, consistent with the uniform-fitness limit.
- Q: Specialization Index return value when total fitness variance is zero (0/0 case) — return `0.0`, omit metric, or return `1.0` → A: Option A — return `0.0`; a converged population has no specialization by any species; metric remains present in result dict so convergence is visible in logs.
- Q: Engine dispatch pattern for `EnsembleMetricCollector` — engine-init guard (same as existing collectors), new registry abstraction, or manual wiring by user → A: Option A — engine instantiates `EnsembleMetricCollector` at `__init__` time guarded by `tracking.has_category(MetricCategory.ENSEMBLE)` and calls `collect()` in the generation loop alongside other collectors; no new abstraction introduced.
- Q: Reference guide scope for collectors without a formal `MetricCategory` gate (`MergeMetricCollector`, `NEATMetricCollector`) — include all with actual enabling mechanism documented, exclude informal ones, or include all in a separate "Advanced / Manual" section → A: Option A — include all ten collectors (nine existing + `EnsembleMetricCollector`); document each with its actual enabling mechanism (`MetricCategory` value, auto-enabled by engine config, or manual instantiation) so the guide is complete without requiring users to read source code.

## Assumptions

- Fitness values are accessible on individuals via the same access pattern used by existing collectors (e.g., `individual.fitness.value`). Individuals with `None` fitness are skipped with a debug-level warning, consistent with existing collector behavior.
- Individual identity for Expert Turnover is determined by Python object identity (`id()`), matching the implied semantics of `previous_elites: list[Individual[Any]]`. No assumption is made about genome equality operators being defined.
- The `species_info` field follows the existing `dict[int, list[int]]` contract where values are indices into `context.population`.
- `MetricCategory.ENSEMBLE` does not auto-enable via `UnifiedConfig` derived configuration (unlike `ERP`, `MULTIOBJECTIVE`, `SYMBIOGENESIS`). It must be explicitly added by the user.
- The reference guide is a hand-maintained Markdown file in `docs/guides/`. It is not auto-generated from source. Future maintainers are responsible for keeping it in sync when adding or changing collectors.
- Negative fitness values are handled by shifting all values by `abs(min(fitness))` before computing Gini, Participation Ratio, and Top-k Concentration. A single debug-level warning is logged per generation where this shift is applied.
- The `EnsembleMetricCollector` is pure-observability: it reads population state and produces numbers. It does not write back to any population, individual, fitness, or engine state.

