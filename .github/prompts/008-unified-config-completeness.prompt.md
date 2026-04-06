---
agent: speckit.specify
---

## Constitution Amendment (apply before writing the spec)

Amend the project constitution (`.specify/memory/constitution.md`) by adding the following principle between Principle II (Separation of Concerns) and Principle III (Acceleration as Optional Execution Detail). Renumber subsequent principles accordingly.

### III. Declarative Completeness (NON-NEGOTIABLE)

`UnifiedConfig` is the single source of truth for experiment specification. Every component that affects experimental outcomes — operators, genomes, evaluators, callbacks, initialization strategies — MUST be expressible declaratively through `UnifiedConfig` and MUST be fully functional when resolved by the factory. The `create_engine` factory and associated factories MUST produce a ready-to-run system from a `UnifiedConfig` alone (plus any non-serializable runtime overrides passed explicitly).

**Rationale**: A "complete experiment specification" that omits the fitness function, requires manual callback wiring, or silently ignores declared configuration parameters is incomplete by definition. If a parameter appears in `genome_params`, `evaluator_params`, or `callback_params`, the corresponding factory MUST honor it. "Could work" is a design defect; "does work" is the standard. Incomplete declarative coverage forces users into imperative orchestration, defeats reproducibility (JSON configs can't capture the missing pieces), breaks experiment hashing (two experiments differing only in their evaluator produce identical hashes), and prevents meta-evolution from exploring the full parameter space.

**Enforcement**:
- Every behavioral component type (operators, genomes, evaluators, callbacks, initializers) MUST have a corresponding registry
- Registries MUST support runtime registration of user-defined implementations
- `UnifiedConfig` MUST carry a string name + parameter dict for each component type, resolved by the factory through the appropriate registry
- `compute_hash()` MUST incorporate all fields that affect experimental outcomes
- Adding a new behavioral component type without a registry entry and config surface is a spec violation
- Integration tests MUST verify that any component expressible in config works end-to-end through the factory without manual wiring

---

## Feature: Evaluator Registry & UnifiedConfig Declarative Completeness

### What

Extend `UnifiedConfig` and the factory system so that **evaluators** (fitness functions) are first-class declarative components — specified by name and parameters in config, resolved through a registry at factory time — exactly like operators and genomes already are. Additionally, close the same gap for **callbacks** and **population initialization strategies** so that `UnifiedConfig` truly is a complete, self-contained experiment specification.

Concretely, the feature adds:

1. **EvaluatorRegistry** — A registry (following the same singleton + `register`/`get` pattern as `OperatorRegistry` and `GenomeRegistry`) that maps string names to evaluator factories. Built-in evaluators like `GroundTruthEvaluator` and `LLMJudgeEvaluator` are registered by default. Users can register custom evaluators at runtime.

2. **`evaluator` and `evaluator_params` fields on `UnifiedConfig`** — String name + parameter dict, defaulting to `None` (preserving backward compatibility). When present, the factory resolves the evaluator from the registry. When absent, the evaluator MUST still be passed to `create_engine()` as today — the existing API is an override, not the only path.

3. **CallbackRegistry** — Same pattern. Built-in callbacks (logging, checkpointing, tracking) remain in `CallbackConfig`. Domain-specific callbacks (e.g., `ESPOCallback`) can be registered and referenced by name. `UnifiedConfig` gains a `custom_callbacks` field (list of `{name, params}` dicts) that the factory resolves alongside the existing `CallbackConfig`.

4. **Population initialization through `genome_params`** — When `genome_params` contains keys that imply initialization strategy (e.g., `seed_text` for embedding genomes), the genome factory or a dedicated initializer registry MUST handle them. The factory MUST NOT silently ignore declared parameters.

5. **`compute_hash()` updated** — The hash MUST now incorporate evaluator name/params and custom callback names/params, so experiments that differ in fitness function produce different hashes.

### Why

`UnifiedConfig` was designed as the *"complete experiment specification"* — a single JSON-serializable object that defines everything needed to reproduce a run. Today it falls short of that promise in three ways:

- **Evaluators are outside the config.** Two experiments with identical configs but different fitness functions are indistinguishable by hash, unshareable as JSON, and invisible to meta-evolution's parameter space. The evaluator is arguably the single most important parameter in an evolutionary experiment, yet it's the one thing the config can't express.

- **Domain callbacks require imperative wiring.** `ESPOCallback`, or any future domain-specific callback, must be manually constructed and passed to `create_engine(callbacks=[...])`. This breaks the declarative promise — a researcher can't hand someone a JSON file and say "run this."

- **`genome_params` is a lossy pass-through.** Parameters like `seed_text` can be placed in `genome_params` but are silently ignored by the genome factory because handling them requires a decoder. This violates the principle of least surprise — if the config accepts a parameter, the system should use it or reject it, never ignore it.

These aren't edge cases. They're the most common integration points for any domain-specific use of the framework (LLM evolution, RL, neuroevolution). Every new domain will hit the same gaps.

### Scope

**In scope:**
- `EvaluatorRegistry` with `register`, `get`, `is_compatible` (mirroring `OperatorRegistry`)
- `evaluator: str | None` and `evaluator_params: dict` fields on `UnifiedConfig`
- `CallbackRegistry` with same pattern
- `custom_callbacks` field on `UnifiedConfig`
- Factory updates to resolve evaluators and callbacks from registries
- `compute_hash()` incorporating evaluator and callback identity
- `genome_params` validation — factories MUST reject unknown parameters rather than ignoring them
- Backward compatibility: existing `create_engine(config, evaluator)` API continues to work as an override; `create_engine(config)` becomes valid when `config.evaluator` is set
- Serialization roundtrip: `to_json()` / `from_json()` / `to_file()` / `from_file()` capture full experiment state
- Registration of existing built-in evaluators (`GroundTruthEvaluator`, `LLMJudgeEvaluator`) and callbacks

**Out of scope:**
- New evaluator or callback implementations (those already exist)
- Changes to the `Evaluator` protocol itself
- GUI or web-based config editors
- Distributed or remote registry services
- Auto-discovery / plugin loading (registrations remain explicit)

### Constraints
- Backward compatibility with existing `create_engine(config, evaluator)` call sites
- Constitution principles I (Model-Agnostic) and III (Declarative Completeness) must both hold — evaluator factories with ML dependencies use deferred imports, not top-level torch/transformers imports
- `UnifiedConfig` must remain a frozen dataclass; registries are runtime singletons, not config state
- JSON serialization must not attempt to serialize callables — only string names and parameter dicts
