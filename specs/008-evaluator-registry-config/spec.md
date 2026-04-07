# Feature Specification: Evaluator Registry & UnifiedConfig Declarative Completeness

**Feature Branch**: `008-evaluator-registry-config`  
**Created**: 2026-04-06  
**Status**: Draft  
**Input**: User description: "Extend UnifiedConfig and the factory system so that evaluators, callbacks, and initialization strategies are first-class declarative components — specified by name and parameters in config, resolved through registries at factory time."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Declare Evaluator in Config and Run (Priority: P1)

A researcher specifies an evaluator by name and parameters directly in a `UnifiedConfig` JSON file. When they call `create_engine(config)` without passing an evaluator argument, the factory resolves the evaluator from the registry and returns a ready-to-run engine. The researcher never writes imperative evaluator construction code.

**Why this priority**: The evaluator is the single most important experimental parameter. Without declarative evaluator support, `UnifiedConfig` cannot fulfill its role as a complete experiment specification. This is the core gap that motivated the feature.

**Independent Test**: Create a `UnifiedConfig` with `evaluator="benchmark"` and `evaluator_params={"function_name": "sphere", "dimensions": 10}`, call `create_engine(config)` with no evaluator argument, and verify the engine runs a generation successfully.

**Acceptance Scenarios**:

1. **Given** a `UnifiedConfig` with `evaluator="benchmark"` and valid `evaluator_params`, **When** the researcher calls `create_engine(config)`, **Then** the factory resolves the evaluator from the registry and returns a functional engine.
2. **Given** a `UnifiedConfig` with `evaluator="benchmark"`, **When** the researcher also passes an explicit evaluator to `create_engine(config, evaluator=my_eval)`, **Then** the explicit evaluator overrides the config-declared one.
3. **Given** a `UnifiedConfig` with `evaluator=None` (the default), **When** the researcher calls `create_engine(config)` without an evaluator argument, **Then** the factory raises a clear error indicating that an evaluator must be provided.

---

### User Story 2 — Register and Use a Custom Evaluator (Priority: P2)

A domain researcher (e.g., working in LLM evolution) registers a custom evaluator with the registry at startup, then references it by name in their config files. Multiple experiments can share the same evaluator name with different parameters, without duplicating construction code.

**Why this priority**: Extensibility via user-defined evaluators is what makes the registry pattern valuable beyond built-in types. Without runtime registration, each new domain would require framework source modifications.

**Independent Test**: Register a custom evaluator factory under the name `"my_domain_eval"`, create a config referencing that name, and verify `create_engine` resolves and instantiates it correctly.

**Acceptance Scenarios**:

1. **Given** a user-defined evaluator factory registered as `"custom_fitness"`, **When** a `UnifiedConfig` references `evaluator="custom_fitness"` with appropriate params, **Then** `create_engine(config)` creates the evaluator from the registry and produces a working engine.
2. **Given** no evaluator registered under the name `"nonexistent"`, **When** a config references `evaluator="nonexistent"`, **Then** the factory raises a clear error listing available evaluators.
3. **Given** a registered evaluator name, **When** a researcher calls the registry's listing method, **Then** the custom evaluator appears alongside built-in evaluators.

---

### User Story 3 — Declare Custom Callbacks in Config (Priority: P3)

A researcher declares domain-specific callbacks by name in their config's `custom_callbacks` field. The factory resolves them from the callback registry and wires them alongside the standard logging/checkpointing callbacks from `CallbackConfig`. The researcher can share the config file and the recipient gets identical callback behavior.

**Why this priority**: Callbacks are the second-most-common imperative wiring point. Declarative callbacks complete the "hand someone a JSON file" promise for experiment reproduction.

**Independent Test**: Register a callback under `"my_tracker"`, add it to `custom_callbacks` in a config, call `create_engine`, and verify the callback's `on_generation_end` fires during a run.

**Acceptance Scenarios**:

1. **Given** a `UnifiedConfig` with `custom_callbacks=[{"name": "espo_callback", "params": {"log_diversity": true}}]` and the callback registered, **When** `create_engine(config)` is called, **Then** the resolved callback is active during the run alongside any `CallbackConfig`-driven callbacks.
2. **Given** a `custom_callbacks` entry referencing an unregistered name, **When** the factory attempts resolution, **Then** a clear error is raised listing available callbacks.
3. **Given** both `CallbackConfig` (with logging enabled) and `custom_callbacks` entries, **When** the engine runs, **Then** all callbacks fire without interference.

---

### User Story 4 — Experiment Hash Reflects Full Specification (Priority: P2)

Two researchers create configs that are identical except for the evaluator (one uses `"sphere"`, the other uses `"rastrigin"`). When they compute the config hash, the hashes differ. Previously these would have been indistinguishable.

**Why this priority**: Hash correctness is critical for experiment tracking, caching, and meta-evolution. Without it, the registry feature creates a silent correctness bug — different experiments appear identical.

**Independent Test**: Create two configs differing only in `evaluator` or `evaluator_params`, compute their hashes, and verify they differ.

**Acceptance Scenarios**:

1. **Given** two configs identical except for `evaluator` name, **When** `compute_hash()` is called on each, **Then** the hashes differ.
2. **Given** two configs identical except for `evaluator_params`, **When** `compute_hash()` is called on each, **Then** the hashes differ.
3. **Given** two configs identical except for `custom_callbacks`, **When** `compute_hash()` is called on each, **Then** the hashes differ.
4. **Given** a config with `evaluator=None` (legacy mode), **When** `compute_hash()` is called, **Then** the hash matches what the same config produced before this feature was added (backward-compatible hash for configs that don't use the new fields).

---

### User Story 5 — Genome Params Validation (Priority: P3)

A researcher accidentally includes a typo in `genome_params` (e.g., `"dimensons"` instead of `"dimensions"`). Instead of silently ignoring the misspelled key and producing a genome with default dimensions, the factory rejects the config with a clear error identifying the unrecognized parameter.

**Why this priority**: Silent parameter ignoring undermines trust in the declarative config. Researchers must be confident that every parameter they declare actually affects the experiment.

**Independent Test**: Create a config with an unrecognized `genome_params` key for the `"vector"` genome type, call `create_engine`, and verify a validation error is raised naming the unrecognized key.

**Acceptance Scenarios**:

1. **Given** a config with `genome_type="vector"` and `genome_params={"dimensons": 10}`, **When** the factory processes the config, **Then** an error is raised identifying `"dimensons"` as an unrecognized parameter for the `"vector"` genome type.
2. **Given** a config with `genome_type="vector"` and `genome_params={"dimensions": 10, "bounds": (-5.0, 5.0)}`, **When** the factory processes the config, **Then** both parameters are accepted and applied correctly.

---

### User Story 6 — Full Serialization Roundtrip (Priority: P2)

A researcher creates a `UnifiedConfig` with an evaluator, custom callbacks, and genome params, saves it to JSON, and a colleague loads it on a different machine. The loaded config is identical, and `create_engine` produces an equivalent engine (assuming the same evaluators/callbacks are registered).

**Why this priority**: Serialization roundtrip is the mechanism for reproducibility. If new fields are lost during serialization, the feature fails its primary goal.

**Independent Test**: Create a config with all new fields populated, serialize to JSON, deserialize, and assert equality of all fields and hash.

**Acceptance Scenarios**:

1. **Given** a config with `evaluator="benchmark"`, `evaluator_params={"function_name": "sphere"}`, and `custom_callbacks=[{"name": "tracker", "params": {}}]`, **When** serialized via `to_json()` and deserialized via `from_json()`, **Then** all fields are preserved and `compute_hash()` matches.
2. **Given** a config saved via `to_file()`, **When** loaded via `from_file()`, **Then** the loaded config is identical to the original.
3. **Given** a legacy JSON file (no `evaluator` or `custom_callbacks` fields), **When** loaded via `from_json()`, **Then** the config is created with `evaluator=None` and `custom_callbacks` as an empty collection — backward compatible.

---

### Edge Cases

- What happens when a registered evaluator factory raises an exception during instantiation? The factory MUST propagate the error with context (evaluator name, params passed) rather than swallowing it.
- What happens when `evaluator_params` contains parameters the evaluator factory doesn't accept? The factory MUST raise a clear error rather than silently dropping unknown params.
- What happens when `custom_callbacks` is an empty list? It MUST be treated the same as no custom callbacks — no error, no change in behavior.
- What happens when the same callback name appears twice in `custom_callbacks`? The system MUST allow it (a researcher may want two instances with different params).
- What happens when a built-in evaluator has ML dependencies (e.g., `LLMJudgeEvaluator` depends on a decoder and model)? Registration MUST use deferred imports — the registry entry exists at import time, but dependencies are loaded only when the evaluator is actually instantiated.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide an evaluator registry that maps string names to evaluator factory callables, following the same singleton + lazy-initialization pattern as the existing operator and genome registries.
- **FR-002**: The evaluator registry MUST support `register`, `get`, `is_registered`, and `list_evaluators` operations at minimum.
- **FR-003**: The following 7 built-in evaluators MUST be registered by default during lazy initialization: `"benchmark"` (wraps `FunctionEvaluator` + `BENCHMARK_FUNCTIONS`), `"function"` (`FunctionEvaluator`), `"llm_judge"` (`LLMJudgeEvaluator`, deferred import), `"ground_truth"` (`GroundTruthEvaluator`, deferred import), `"scm"` (`SCMEvaluator`), `"rl"` (`RLEvaluator`, deferred import), `"meta"` (`MetaEvaluator`).
- **FR-004**: The evaluator registry MUST support runtime registration of user-defined evaluator factories. Re-registering an existing name MUST silently overwrite the previous entry (last-write-wins), allowing users to shadow built-in evaluators.
- **FR-005**: `UnifiedConfig` MUST include an optional `evaluator` field (string name, defaulting to `None`) and an `evaluator_params` field (parameter dictionary, defaulting to empty).
- **FR-006**: The `create_engine` factory MUST resolve the evaluator from the registry when `config.evaluator` is set and no explicit evaluator argument is passed.
- **FR-007**: The `create_engine` factory MUST continue to accept an explicit evaluator argument that overrides the config-declared evaluator.
- **FR-008**: The `create_engine` factory MUST raise a clear error when neither `config.evaluator` is set nor an explicit evaluator is passed.
- **FR-009**: The system MUST provide a callback registry that maps string names to callback factory callables, following the same pattern as the evaluator registry.
- **FR-010**: The callback registry MUST support `register`, `get`, `is_registered`, and `list_callbacks` operations.
- **FR-011**: `UnifiedConfig` MUST include an optional `custom_callbacks` field (a tuple of name+params entries, defaulting to empty) that the factory resolves from the callback registry.
- **FR-012**: The factory MUST merge callbacks resolved from `custom_callbacks` with callbacks derived from `CallbackConfig` and any explicitly passed callback list. Execution order MUST be: `CallbackConfig`-derived callbacks first, then `custom_callbacks` in their declared order, then explicitly passed callbacks last.
- **FR-013**: `compute_hash()` MUST incorporate `evaluator`, `evaluator_params`, and `custom_callbacks` into the hash when they are present.
- **FR-014**: `compute_hash()` MUST produce the same hash as before for configs that do not use the new fields (backward-compatible hashing).
- **FR-015**: Genome factories MUST validate `genome_params` and reject unrecognized parameters with a clear error message, rather than silently ignoring them. Validation MUST use signature introspection (`inspect.signature`) on the factory callable to determine accepted parameter names. Factories whose signatures include `**kwargs` opt out of strict validation.
- **FR-016**: All new fields MUST survive a full serialization roundtrip (`to_dict` → `from_dict`, `to_json` → `from_json`, `to_file` → `from_file`).
- **FR-017**: Legacy JSON configs (missing the new fields) MUST deserialize successfully with default values (`evaluator=None`, `evaluator_params={}`, `custom_callbacks=()`).
- **FR-018**: Evaluator factories with heavy ML dependencies MUST use deferred imports — the registry entry is available at import time, but dependencies are loaded only at instantiation.
- **FR-019**: Error messages from the factory MUST include actionable context: the name that failed resolution, the params that were passed, and the list of available registered names.
- **FR-020**: `create_engine` MUST accept an optional `runtime_overrides` parameter (a dictionary) that supplies non-serializable values to the evaluator factory. The factory MUST merge `runtime_overrides` with the JSON-serializable `evaluator_params` before invoking the evaluator registry factory, with `runtime_overrides` taking precedence on key conflicts. Callback factories receive only the `params` dict declared in their `custom_callbacks` entry; callback runtime overrides are out of scope for this feature.

### Key Entities

- **EvaluatorRegistry**: Singleton registry mapping string names to evaluator factory callables. Lazy-initializes 7 built-in evaluators on first access.
- **CallbackRegistry**: Singleton registry mapping string names to callback factory callables. Same lifecycle as `EvaluatorRegistry`.
- **UnifiedConfig (extended)**: Gains `evaluator`, `evaluator_params`, and `custom_callbacks` fields. Remains a frozen dataclass. The new fields participate in hashing, serialization, and validation.
- **Evaluator factory callable**: A function or class that accepts keyword parameters and returns an object satisfying the `Evaluator` protocol.
- **Callback factory callable**: A function or class that accepts keyword parameters and returns an object satisfying the `Callback` protocol.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can define a complete experiment — including fitness function and domain callbacks — in a single JSON config file and run it with `create_engine(config)` alone, with zero imperative wiring code.
- **SC-002**: Two configs that differ only in evaluator or callback configuration produce different hashes 100% of the time.
- **SC-003**: All built-in evaluators and callbacks are resolvable by name from a freshly-initialized registry without any manual registration.
- **SC-004**: A config serialized to JSON and deserialized back produces a byte-identical hash (full roundtrip fidelity).
- **SC-005**: Unrecognized `genome_params` keys cause a validation error at factory time in 100% of cases — no silent ignoring.
- **SC-006**: Existing code that calls `create_engine(config, evaluator)` continues to work without modification (zero breaking changes to the public API).
- **SC-007**: Registering a custom evaluator and using it via config requires no more than 3 lines of setup code (register call + config creation).

## Clarifications

### Session 2026-04-06

- Q: When a user registers a name that already exists in the evaluator or callback registry, what should happen? → A: Overwrite silently — last `register` wins, allowing users to shadow built-ins freely.
- Q: In what order should callbacks from the three sources (CallbackConfig, custom_callbacks, explicit argument) fire? → A: Config → Custom → Explicit — `CallbackConfig` callbacks first, then `custom_callbacks` in declared order, then explicitly passed callbacks last.
- Q: How should non-serializable evaluator dependencies (e.g., decoder, task_spec for LLMJudgeEvaluator) be supplied? → A: Via a `runtime_overrides` dict kwarg on `create_engine` — a single general-purpose mechanism that merges non-serializable params for any component (evaluator, callbacks) with the JSON-serializable config params.
- Q: How should genome factories declare which parameters they accept for validation? → A: Signature introspection — the registry inspects the factory callable's signature (`inspect.signature`) at validation time. Factories accepting `**kwargs` opt out of strict validation.

## Assumptions

- The existing `OperatorRegistry` and `GenomeRegistry` patterns are the established conventions for registries in this framework. The new registries follow the same singleton + lazy-init + `register`/`get` pattern.
- `UnifiedConfig` remains a frozen dataclass. New fields use immutable defaults (`None`, empty `dict`, empty `tuple`).
- `custom_callbacks` uses a tuple of frozen-compatible structures to preserve hashability of the frozen dataclass.
- Evaluator factories are callables that accept `**params` and return an `Evaluator`-protocol-compatible object. They receive only the declared `evaluator_params`, not the full `UnifiedConfig`.
- The `"benchmark"` evaluator registry entry is a factory that accepts `function_name` and optional params, looks up the function in `BENCHMARK_FUNCTIONS`, and wraps it in `FunctionEvaluator`. No `BenchmarkEvaluator` class exists. `LLMJudgeEvaluator` requires params like `task_spec` and `judge_model_id` — some of which are non-serializable runtime objects. These MUST be supplied via the `runtime_overrides` kwarg on `create_engine`, which merges them with the JSON-serializable `evaluator_params` before invoking the evaluator registry factory.
- `genome_params` validation is implemented at the genome registry level via signature introspection (`inspect.signature`) on the registered factory callable. Unrecognized keys are rejected unless the factory accepts `**kwargs`, in which case strict validation is skipped.
- The `FunctionEvaluator` registration accepts a `fitness_fn` name that maps to known benchmark functions; wrapping arbitrary user lambdas remains an imperative operation (callables are not JSON-serializable).

## Out of Scope

- **Initializer Registry**: The original input description mentioned "initialization strategies" as a declarative component. Constitution III mandates a registry for every behavioral component type including initializers. An initializer registry is deferred to a follow-up feature (tracked separately) to keep this feature focused on evaluators, callbacks, and config completeness. This is an acknowledged Constitution III gap that will be closed in a subsequent spec.
