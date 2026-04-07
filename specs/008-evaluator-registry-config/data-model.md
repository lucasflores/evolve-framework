# Data Model: Evaluator Registry & UnifiedConfig Declarative Completeness

**Feature**: 008-evaluator-registry-config  
**Date**: 2026-04-06

## Entities

### EvaluatorRegistry

**Purpose**: Singleton registry mapping string names to evaluator factory callables.

**Fields**:
- `_factories: dict[str, Callable[..., Evaluator]]` — Maps evaluator name → factory callable
- `_initialized: bool` — Lazy initialization flag (default: `False`)

**Operations**:
- `register(name: str, factory: Callable[..., Evaluator]) -> None` — Register/overwrite a factory
- `get(name: str, **params: Any) -> Evaluator` — Instantiate evaluator by name
- `is_registered(name: str) -> bool` — Check if name exists
- `list_evaluators() -> list[str]` — Return all registered names

**Lifecycle**: Singleton created via `get_evaluator_registry()`. Built-in evaluators registered on first access via `_ensure_initialized()`.

**Built-in Registrations**:

| Name | Factory Target | Deferred Import? |
|------|---------------|-----------------|
| `"benchmark"` | Wrapper: looks up `BENCHMARK_FUNCTIONS[function_name]` → `FunctionEvaluator` | No |
| `"function"` | `FunctionEvaluator(**params)` | No |
| `"llm_judge"` | Deferred wrapper → `LLMJudgeEvaluator(**params)` | Yes |
| `"ground_truth"` | Deferred wrapper → `GroundTruthEvaluator(**params)` | Yes |
| `"scm"` | `SCMEvaluator(**params)` | No |
| `"rl"` | Deferred wrapper → `RLEvaluator(**params)` | Yes |
| `"meta"` | `MetaEvaluator(**params)` | No |

---

### CallbackRegistry

**Purpose**: Singleton registry mapping string names to callback factory callables.

**Fields**:
- `_factories: dict[str, Callable[..., Callback]]` — Maps callback name → factory callable
- `_initialized: bool` — Lazy initialization flag (default: `False`)

**Operations**:
- `register(name: str, factory: Callable[..., Callback]) -> None` — Register/overwrite a factory
- `get(name: str, **params: Any) -> Callback` — Instantiate callback by name
- `is_registered(name: str) -> bool` — Check if name exists
- `list_callbacks() -> list[str]` — Return all registered names

**Built-in Registrations**:

| Name | Factory Target |
|------|---------------|
| `"logging"` | `LoggingCallback(**params)` |
| `"checkpoint"` | `CheckpointCallback(**params)` |
| `"print"` | `PrintCallback(**params)` |
| `"history"` | `HistoryCallback(**params)` |

---

### UnifiedConfig (Extended Fields)

**New fields added to the existing frozen dataclass**:

| Field | Type | Default | JSON Key | Hash Participation |
|-------|------|---------|----------|-------------------|
| `evaluator` | `str \| None` | `None` | `"evaluator"` | Yes (when non-None) |
| `evaluator_params` | `dict[str, Any]` | `{}` | `"evaluator_params"` | Yes (when non-empty) |
| `custom_callbacks` | `tuple[dict[str, Any], ...]` | `()` | `"custom_callbacks"` | Yes (when non-empty) |

**Validation rules** (in `__post_init__`):
- If `evaluator` is not None, it MUST be a non-empty string
- If `evaluator_params` is non-empty, `evaluator` SHOULD be set (warning if not)
- Each entry in `custom_callbacks` MUST have a `"name"` key (str) and optional `"params"` key (dict)

**Serialization**:
- `to_dict()`: `evaluator` → string or None; `evaluator_params` → dict; `custom_callbacks` → list of dicts
- `from_dict()`: Missing keys use defaults. `custom_callbacks` list → tuple of dicts.

**Hash behavior**:
- `compute_hash()` excludes `evaluator`, `evaluator_params`, `custom_callbacks` when they equal defaults — preserving backward-compatible hashes for legacy configs.

---

### create_engine (Extended Signature)

**New signature**:
```
create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable | None = None,  # Changed: now optional
    seed: int | None = None,
    callbacks: list[Callback] | None = None,
    runtime_overrides: dict[str, Any] | None = None,  # NEW
) -> EvolutionEngine
```

**Resolution logic**:
1. If explicit `evaluator` argument is provided → use it (override)
2. Else if `config.evaluator` is set → resolve from registry with `{**config.evaluator_params, **runtime_overrides}`
3. Else → raise `ValueError("No evaluator provided...")`

**Callback merge order**:
1. `CallbackConfig`-derived callbacks (logging, checkpointing, tracking)
2. `custom_callbacks` resolved from registry (in declared order)
3. Explicitly passed `callbacks` list

---

## Relationships

```
UnifiedConfig
  ├── evaluator (str) ──resolves──► EvaluatorRegistry ──produces──► Evaluator instance
  ├── evaluator_params (dict) ──merges with──► runtime_overrides ──passed to──► evaluator factory
  ├── custom_callbacks (tuple[dict]) ──resolves each──► CallbackRegistry ──produces──► Callback instances
  └── genome_params (dict) ──validated against──► GenomeRegistry factory signature
```
