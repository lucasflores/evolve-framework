# Research: Evaluator Registry & UnifiedConfig Declarative Completeness

**Feature**: 008-evaluator-registry-config  
**Date**: 2026-04-06

## R1. Registry Pattern Analysis

**Decision**: Follow the existing `OperatorRegistry` / `GenomeRegistry` singleton + lazy-init pattern exactly.

**Rationale**: Both existing registries use the same architecture:
- Private instance `_instance: ClassVar[OperatorRegistry | None] = None`
- Module-level `get_*_registry()` function creating the singleton on first call
- `_ensure_initialized()` that registers built-ins once, sets `_initialized = True`
- `register()` overwrites silently (last-write-wins)
- `get()` instantiates via `cls(**params)` or `factory(**params)`

The new `EvaluatorRegistry` and `CallbackRegistry` MUST mirror this pattern to maintain consistency. The `GenomeRegistry` pattern is the closer fit (maps names to factory callables) vs `OperatorRegistry` (maps category+name pairs to classes).

**Alternatives considered**:
- Abstract base registry class: Would reduce duplication but adds a new abstraction for only 4 classes. Rejected per Constitution VI (Extensibility Over Premature Optimization) — don't abstract until empirical need justifies it.
- Plugin discovery (entry_points): Out of scope per spec. Registrations remain explicit.

## R2. Evaluator Name → Factory Mapping

**Decision**: Register evaluator factories as callables `(**params) -> Evaluator`. Built-in registrations use wrapper factories that handle deferred imports.

**Rationale**: The codebase has these evaluator types to register:

| Name | Class | Location | Requires Deferred Import? |
|------|-------|----------|---------------------------|
| `"function"` | `FunctionEvaluator` | `evolve/evaluation/evaluator.py` | No (core) |
| `"llm_judge"` | `LLMJudgeEvaluator` | `evolve/evaluation/llm_judge.py` | Yes (transformers) |
| `"ground_truth"` | `GroundTruthEvaluator` | `evolve/evaluation/benchmark.py` | Yes (torch, SoftPromptDecoder) |
| `"scm"` | `SCMEvaluator` | `evolve/evaluation/scm_evaluator.py` | No (networkx only, already a dep) |
| `"rl"` | `RLEvaluator` | `evolve/rl/evaluator.py` | Yes (gymnasium) |
| `"meta"` | `MetaEvaluator` | `evolve/meta/evaluator.py` | No (core) |
| `"benchmark"` | Wrapper around `FunctionEvaluator` + `BENCHMARK_FUNCTIONS` | `evolve/evaluation/reference/functions.py` | No (numpy only) |

**Note**: The spec references `BenchmarkEvaluator` but no such class exists. The "benchmark" registry entry will be a factory that: (1) looks up `function_name` in `BENCHMARK_FUNCTIONS`, (2) wraps it in `FunctionEvaluator`. This provides a clean declarative path: `evaluator="benchmark", evaluator_params={"function_name": "sphere", "dimensions": 10}`.

**Deferred import pattern**: Factories for `llm_judge`, `ground_truth`, and `rl` will be thin wrapper functions that import their class at call time:
```python
def _create_llm_judge(**params):
    from evolve.evaluation.llm_judge import LLMJudgeEvaluator
    return LLMJudgeEvaluator(**params)
```
This satisfies Constitution I (Model-Agnostic) — no torch/transformers at import time.

**Alternatives considered**:
- Registering classes directly: Would require top-level imports of heavy deps. Rejected per Constitution I.
- String-based lazy class resolution: More complex, less debuggable. Wrapper factories are simpler.

## R3. `compute_hash()` Backward Compatibility Strategy

**Decision**: Only include new fields in the hash when they have non-default values. When `evaluator is None` and `custom_callbacks` is empty, the hash dict is identical to pre-feature output.

**Rationale**: Current `compute_hash()` calls `self.to_dict()`, removes metadata keys, then hashes the JSON. If `to_dict()` includes `"evaluator": None` and `"custom_callbacks": []`, the JSON string changes and hashes break.

Strategy: In `compute_hash()`, also pop `"evaluator"`, `"evaluator_params"`, and `"custom_callbacks"` when they equal their defaults (`None`, `{}`, `()`/`[]`). This makes the hash byte-identical for legacy configs.

**Alternatives considered**:
- Schema versioned hashing: Hash algorithm changes with schema version. More complex, still breaks caches on upgrade.
- Separate hash method: `compute_hash_v2()`. Confusing API surface.

## R4. `custom_callbacks` Data Structure in Frozen Dataclass

**Decision**: Store `custom_callbacks` as `tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]` — a tuple of (name, sorted-params-as-tuples) pairs. Expose a helper for constructing from dicts.

**Rationale**: `UnifiedConfig` is a frozen dataclass. Dict values aren't hashable, so `dict` fields use `field(default_factory=dict)` but still work because frozen doesn't require `__hash__` by default in Python — it means you can't reassign fields, not that the instance must be hashable.

However, looking at the existing codebase: `selection_params`, `crossover_params`, `mutation_params`, and `genome_params` are all `dict[str, Any]` with `field(default_factory=dict)`. So `dict` is already used in the frozen dataclass. `custom_callbacks` can use a similar pattern.

**Revised decision**: Use `tuple[dict[str, Any], ...]` for `custom_callbacks` — a tuple of dicts, each with `"name"` (str) and `"params"` (dict) keys. The outer tuple is immutable. This is consistent with how `tags` uses `tuple[str, ...]`.

In `to_dict()`, convert to `list[dict]`. In `from_dict()`, convert back to `tuple[dict]`.

## R5. `runtime_overrides` Structure

**Decision**: `runtime_overrides` is a flat `dict[str, Any]` passed to `create_engine`. The factory merges it with `evaluator_params` before calling the evaluator factory (runtime_overrides keys win on conflict).

**Rationale**: The most common use case is supplying non-serializable objects for a specific evaluator (e.g., `decoder` for `LLMJudgeEvaluator`). A flat dict is simplest. If callbacks also need overrides, the factory can namespace by convention (e.g., `runtime_overrides={"decoder": ..., "callback:my_cb:logger": ...}`), but for now a flat merge with evaluator_params covers the primary need.

**Alternatives considered**:
- Nested dict `{"evaluator": {...}, "callbacks": {"name": {...}}}`: More structured but over-engineered for the current use cases. Can be added later if needed.
- Separate `evaluator_overrides` and `callback_overrides` kwargs: More explicit but proliferates kwargs.

## R6. Signature Introspection for genome_params Validation

**Decision**: Use `inspect.signature(factory)` at validation time (not at registration time). Cache the result per factory.

**Rationale**: Introspecting at validation time (when `create()` is called with params) keeps registration simple. The introspection result can be cached after first call. Factories with `**kwargs` in their signature opt out of validation — `inspect.Parameter.VAR_KEYWORD` is detected and validation is skipped.

Implementation:
1. On `genome_registry.create(name, **params)`: get factory, introspect signature
2. Collect `{p.name for p in sig.parameters.values() if p.kind in (POSITIONAL_OR_KEYWORD, KEYWORD_ONLY)}`
3. Remove `"rng"` (injected by registry, not user-supplied)
4. If any param in `**kwargs` that's `VAR_KEYWORD`: skip validation
5. Otherwise: `unknown = set(params) - accepted_params` → raise if non-empty

**Alternatives considered**:
- Validation at registration time: Can't introspect dynamically generated factories. Too early.
- Explicit `accepted_params` set: More metadata burden on registrants.
