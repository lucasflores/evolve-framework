# Public API Contracts: Evaluator Registry & UnifiedConfig Declarative Completeness

**Feature**: 008-evaluator-registry-config  
**Date**: 2026-04-06

## EvaluatorRegistry API

### `get_evaluator_registry() -> EvaluatorRegistry`

Module-level singleton accessor. Creates the registry on first call.

### `reset_evaluator_registry() -> None`

Reset singleton (for testing). Clears all registrations including built-ins.

### `EvaluatorRegistry.register(name, factory)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique name (e.g., `"benchmark"`, `"llm_judge"`) |
| `factory` | `Callable[..., Evaluator]` | Factory callable accepting `**params` |

**Returns**: `None`  
**Behavior**: Overwrites silently if name already exists (last-write-wins).  
**Raises**: `TypeError` if `factory` is not callable.

### `EvaluatorRegistry.get(name, **params)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registered evaluator name |
| `**params` | `Any` | Keyword arguments passed to factory |

**Returns**: `Evaluator` instance  
**Raises**: `KeyError` with message listing available evaluators if name not found. Propagates factory exceptions with context (name, params).

### `EvaluatorRegistry.is_registered(name) -> bool`

### `EvaluatorRegistry.list_evaluators() -> list[str]`

**Returns**: Sorted list of all registered evaluator names.

---

## CallbackRegistry API

Identical to `EvaluatorRegistry` API with s/evaluator/callback/:

### `get_callback_registry() -> CallbackRegistry`
### `reset_callback_registry() -> None`
### `CallbackRegistry.register(name, factory)`
### `CallbackRegistry.get(name, **params) -> Callback`
### `CallbackRegistry.is_registered(name) -> bool`
### `CallbackRegistry.list_callbacks() -> list[str]`

---

## UnifiedConfig New Fields

### Constructor (new keyword arguments)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator` | `str \| None` | `None` | Evaluator registry name |
| `evaluator_params` | `dict[str, Any]` | `{}` | Params passed to evaluator factory |
| `custom_callbacks` | `tuple[dict[str, Any], ...]` | `()` | List of `{"name": str, "params": dict}` entries |

### `compute_hash() -> str`

**Changed behavior**: When `evaluator`, `evaluator_params`, or `custom_callbacks` have non-default values, they are included in the hash. When all are at defaults, hash is identical to pre-feature output.

### `to_dict() -> dict[str, Any]`

**Changed behavior**: Output dict now includes `"evaluator"`, `"evaluator_params"`, `"custom_callbacks"` keys.

### `from_dict(data) -> UnifiedConfig`

**Changed behavior**: Missing keys use defaults (`None`, `{}`, `()`). `"custom_callbacks"` list is converted to tuple.

---

## create_engine Updated Signature

```python
def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable[[Any], float] | None = None,
    seed: int | None = None,
    callbacks: list[Callback] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> EvolutionEngine:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `UnifiedConfig` | required | Experiment configuration |
| `evaluator` | `Evaluator \| Callable \| None` | `None` | Explicit evaluator (overrides config) |
| `seed` | `int \| None` | `None` | Random seed override |
| `callbacks` | `list[Callback] \| None` | `None` | Additional callbacks (appended last) |
| `runtime_overrides` | `dict[str, Any] \| None` | `None` | Non-serializable params merged with evaluator/callback params |

**Evaluator resolution**:
1. Explicit `evaluator` argument → used directly (wrapped in `FunctionEvaluator` if callable)
2. `config.evaluator` set → resolved from `EvaluatorRegistry` with `{**config.evaluator_params, **(runtime_overrides or {})}`
3. Neither → `ValueError`

**Callback merge order**:
1. `CallbackConfig`-derived (logging, checkpointing, tracking)
2. `custom_callbacks` from registry (each entry resolved in order)
3. Explicitly passed `callbacks` list

---

## GenomeRegistry.create() — Validation Change

### `GenomeRegistry.create(name, rng=None, **params)`

**Changed behavior**: Before instantiation, validates `params` against the factory's signature using `inspect.signature()`. Unrecognized parameter names raise `ValueError` listing the unknown keys and accepted parameters.

**Exception**: Factories accepting `**kwargs` skip strict validation.

---

## Error Messages

All error messages from the factory follow this template:

**Registry lookup failure**:
```
EvaluatorRegistry: '{name}' is not registered.
Available evaluators: {sorted_list}
```

**Factory instantiation failure**:
```
Failed to create evaluator '{name}' with params {params}: {original_error}
```

**Missing evaluator**:
```
No evaluator provided. Either set config.evaluator to a registered evaluator name
or pass an evaluator argument to create_engine().
Available evaluators: {sorted_list}
```

**Genome params validation failure**:
```
Unrecognized genome_params for '{genome_type}': {unknown_keys}.
Accepted parameters: {accepted_keys}
```
