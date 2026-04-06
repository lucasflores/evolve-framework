# Research: Evolutionary Soft-Prompt Optimization (ESPO)

**Feature Branch**: `007-llm-soft-prompt-evolution`
**Date**: 2026-04-04

## Research Tasks

### R1: Embedding Genome Integration with Framework Protocols

**Context**: EmbeddingGenome must implement `Genome` and `SerializableGenome` protocols while storing a 2D numpy array `(n_tokens, embed_dim)`.

**Decision**: Implement `EmbeddingGenome` as a frozen dataclass in `evolve/representation/embedding.py`, following the exact pattern of `VectorGenome` in `evolve/representation/vector.py`. The genome stores:
- `embeddings: np.ndarray` — shape `(n_tokens, embed_dim)`, made immutable via `flags.writeable = False`
- `model_id: str` — target model identifier
- `seed_text: str | None` — original seed text
- `n_tokens: int` — number of virtual tokens (derived from shape)
- `embed_dim: int` — embedding dimensionality (derived from shape)

**Rationale**: The existing `VectorGenome` pattern (frozen dataclass, numpy immutability, `to_dict`/`from_dict` for serialization, `copy`, `__eq__`, `__hash__`) is proven and consistent with framework conventions. Storing the 2D array natively (not flattened) preserves token structure for token-aware operators while the `flat()` property provides a 1D view for compatibility with existing flat-vector operators (`GaussianMutation`, `BlendCrossover`, etc.).

**Alternatives considered**:
- Subclass `VectorGenome`: Rejected — VectorGenome is 1D-only and frozen, so extending to 2D would require breaking its invariants. Composition is cleaner.
- Store as 1D with reshape metadata: Rejected — forces all token-aware operators to repeatedly reshape, increasing complexity and error risk.

---

### R2: Flat-Vector Operator Compatibility

**Context**: FR-009 requires existing framework flat-vector operators (`GaussianMutation`, `BlendCrossover`, `SBX`, etc.) to work via a flat accessor.

**Decision**: `EmbeddingGenome` provides:
- `flat() -> np.ndarray` — returns a 1D view of shape `(n_tokens * embed_dim,)` (no copy, just reshape)
- `@classmethod from_flat(cls, flat: np.ndarray, n_tokens: int, embed_dim: int, model_id: str, seed_text: str | None) -> EmbeddingGenome` — reconstructs from flat array

Flat-vector operators operate on a `VectorGenome` wrapper created via `to_vector_genome()` / `from_vector_genome()` helper methods. This keeps the operator code unchanged.

**Rationale**: Existing operators expect `VectorGenome` with `.genes` attribute. Rather than modifying every operator, we provide adapters. This follows Principle V (extensibility over premature optimization).

**Alternatives considered**:
- Modify operators to accept either type: Rejected — violates separation of concerns and requires changes across many files.
- Duck-typing with `.genes` property: Rejected — fragile and obscures the 2D structure.

---

### R3: Dimensionality Strategies

**Context**: FR-002 specifies three strategies: full-space, compressed subspace, minimal tokens.

**Decision**:
1. **Full-space**: Genome shape is `(n_tokens, embed_dim)` where `embed_dim` matches the target model's embedding dimension (e.g., 4096 for LLaMA-7B). Total parameters = `n_tokens × embed_dim`.
2. **Compressed subspace**: Genome evolves in a lower-dimensional space of shape `(n_tokens, subspace_dim)`. A pre-computed projection matrix `W: (subspace_dim, embed_dim)` maps back to full space at decode time. `W` is computed once (e.g., PCA on model's embedding matrix) and fixed throughout evolution.
3. **Minimal tokens** (DEFAULT): Uses full `embed_dim` but only 4–8 virtual tokens, keeping total parameters manageable (e.g., 8 × 4096 = 32K parameters).

The strategy is configured via an enum `DimensionalityStrategy` and stored in `EmbeddingGenomeConfig`. The projection matrix (if any) is stored externally in the config, not in the genome itself.

**Rationale**: Separating the projection matrix from the genome keeps the genome lightweight and serializable. Minimal tokens as default balances expressiveness with search efficiency — 4–8 tokens provide sufficient prompt capacity while keeping the search space under 33K dimensions.

**Alternatives considered**:
- Store projection matrix in genome: Rejected — inflates genome size, makes serialization expensive, and the matrix is shared across all individuals.
- Auto-detect strategy from model: Rejected — strategy choice is a researcher decision, not automatable.

---

### R4: Coherence Radius Calibration

**Context**: FR-012 requires an L2 norm bound on mutation magnitude to prevent coherence collapse.

**Decision**: The coherence radius is calibrated relative to the model's embedding statistics:
- Compute the mean pairwise L2 distance between token embeddings in the model's vocabulary.
- Set the default coherence radius to a fraction (e.g., 0.1–0.3) of this mean distance.
- The radius is configurable and can be overridden by the researcher.
- Mutation clamping: after adding Gaussian noise, if the L2 norm of the perturbation exceeds the radius, scale the perturbation to exactly the radius magnitude.

**Rationale**: Model-relative calibration ensures the radius is meaningful across models with different embedding scales (e.g., LLaMA's embeddings have different norms than GPT-NeoX). A fraction of mean pairwise distance keeps mutations within the "plausible token" region of embedding space.

**Alternatives considered**:
- Fixed absolute radius: Rejected — meaningless across different models with different embedding scales.
- Per-dimension bounds: Rejected — overly restrictive and loses correlation structure.

---

### R5: CMA-ES and Adaptive Operator Integration

**Context**: Spec assumes CMA-ES may be available as a separate framework feature.

**Decision**: CMA-ES is NOT currently implemented in the framework (not found in operator registry or codebase). ESPO will not depend on CMA-ES for v1. The token-aware Gaussian mutation with configurable sigma and the existing framework operators provide sufficient variation. CMA-ES can be added as a future enhancement.

**Rationale**: Adding CMA-ES is a separate feature that should not block ESPO. The existing operators (Gaussian mutation, polynomial mutation, SBX crossover) combined with new token-aware operators provide a complete operator suite for initial experiments.

**Alternatives considered**:
- Implement CMA-ES as part of ESPO: Rejected — violates atomic feature scope and separation of concerns.
- Require CMA-ES as prerequisite: Rejected — would block ESPO on an unrelated feature.

---

### R6: SoftPromptDecoder Architecture

**Context**: FR-003/FR-004 require a decoder that injects embeddings into a model's embedding layer.

**Decision**: `SoftPromptDecoder` lives in `evolve/meta/soft_prompt/decoder.py` and:
1. Loads the target model and tokenizer (lazy, cached).
2. On `decode(genome, task_input)`:
   - Tokenizes `task_input` → input_ids → input embeddings via model's embedding layer.
   - Prepends `genome.embeddings` to the input embeddings.
   - Runs model.generate() with the combined embeddings.
   - Decodes output tokens to text.
3. Validates `genome.model_id == self.model_id` before decoding (FR-004).

The decoder depends on `torch` and `transformers` — these are optional deps, imported conditionally.

**Rationale**: Embedding-layer injection is the standard approach for soft-prompt tuning (Lester et al., 2021). Lazy model loading avoids importing torch at module load time. The decoder is isolated in `meta/soft_prompt/` to enforce the ML import boundary.

**Alternatives considered**:
- Token-level injection (modify input_ids): Rejected — soft prompts operate in continuous embedding space, not discrete token space.
- Adapter-based approach: Rejected — more complex, less transparent, and requires model modification.

---

### R7: Evaluator Integration Patterns

**Context**: FR-005/FR-006 require benchmark and LLM-as-judge evaluators that conform to the framework's `Evaluator` protocol.

**Decision**: Both evaluators implement the `Evaluator[EmbeddingGenome]` protocol:
- `GroundTruthEvaluator`: Takes a `TaskSpec` with ground-truth answers. Decodes each individual, compares to ground truth, returns `Fitness.scalar()` for single-metric or `Fitness(values=np.array([...]))` for multi-metric.
- `LLMJudgeEvaluator`: Takes a `TaskSpec` with a rubric. Decodes each individual, sends output to judge LLM, parses per-criterion scores, returns `Fitness(values=np.array([score1, score2, ...]))` for multi-objective selection.

Both evaluators use `SoftPromptDecoder` internally and accept explicit seeds for determinism. The `capabilities` property declares `supports_gpu=True`, `batchable=True`.

**Rationale**: The existing `Evaluator` protocol and `FunctionEvaluator` wrapper establish the pattern. ESPO evaluators follow the same protocol, enabling seamless integration with the engine and selection operators (including NSGA-II for multi-objective).

**Alternatives considered**:
- Single evaluator with mode flag: Rejected — benchmark and judge evaluation have fundamentally different interfaces and dependencies.
- Evaluation as external service: Rejected — adds network dependency; local execution is simpler and more reproducible.

---

### R8: Population Initialization Strategies

**Context**: FR-010/FR-011 require noise-based and LLM-discovered initialization.

**Decision**:
1. **Noise-based** (`NoiseInitializer`): Embeds seed text → pads/truncates to `n_tokens` → adds calibrated Gaussian noise (sigma scaled to coherence radius) → produces N distinct genomes. Pure numpy after initial embedding.
2. **LLM-discovered** (`LLMVariationInitializer`): Prompts an LLM to generate paraphrases/variants of the seed text → embeds each variant → produces N genomes. Falls back to noise-based if LLM is unavailable.

Both return `list[EmbeddingGenome]` compatible with the framework's population initialization.

**Rationale**: Two complementary strategies — noise-based is fast, deterministic, and doesn't require an LLM; LLM-discovered produces semantically diverse starting points. Noise-based is the reliable default.

**Alternatives considered**:
- Random embedding initialization (no seed): Rejected — starting from noise ignores the task prompt entirely, making convergence much harder.
- K-means clustering of vocabulary: Rejected — doesn't leverage task-specific seed text.

---

### R9: MLflow Integration for ESPO Metrics

**Context**: FR-018 requires comprehensive per-generation MLflow logging.

**Decision**: Use the existing `MLflowTracker.log_generation()` method with ESPO-specific metrics:
- `best_fitness`, `mean_fitness`, `worst_fitness` — standard fitness stats
- `diversity_mean_l2`, `diversity_std_l2` — pairwise L2 distance stats in embedding space
- `infeasibility_rate` — fraction of individuals marked infeasible by coherence defense
- `mutation_magnitude_mean` — average L2 norm of mutations applied
- `best_decoded_text` — logged as artifact (not metric) via `log_artifact()`

This integrates with the existing callback mechanism — an `ESPOCallback` extends `Callback` to collect and report these metrics.

**Rationale**: The framework already has MLflow integration (`evolve/experiment/tracking/mlflow_tracker.py`) and a callback system. ESPO adds domain-specific metrics without modifying the core tracking infrastructure.

**Alternatives considered**:
- Custom logging outside MLflow: Rejected — violates Principle VII and fragments observability.
- Log raw embeddings per individual: Rejected — too much data; aggregate statistics are more useful.

---

### R10: Registry Integration

**Context**: The framework uses `GenomeRegistry` and `OperatorRegistry` singletons for discovery and factory creation.

**Decision**: Register ESPO components in the existing registries:
- `GenomeRegistry`: Register `"embedding"` with factory that creates `EmbeddingGenome` from config.
- `OperatorRegistry`: Register `"token_gaussian"` (mutation), `"token_single_point"` and `"token_two_point"` (crossover) with `compatible_genomes={"embedding"}`.
- Registration happens in `_register_builtin_genomes()` and `_register_builtin_operators()` via lazy initialization.

**Rationale**: Follows the established registration pattern. Users can create ESPO experiments via the unified config system with `genome_type="embedding"`, `mutation="token_gaussian"`, etc.

**Alternatives considered**:
- Plugin-based registration: Rejected — ESPO is a core feature, not a third-party plugin. Built-in registration is simpler.
- Manual instantiation only: Rejected — breaks the unified config workflow.

## Summary of Resolved Items

| Item | Resolution |
|------|-----------|
| Genome protocol integration | Frozen dataclass with 2D numpy array, `Genome` + `SerializableGenome` protocols |
| Flat-vector compatibility | `to_vector_genome()` / `from_vector_genome()` adapter methods |
| Dimensionality strategies | Enum-configured, minimal tokens (4–8) as default, projection matrix external |
| Coherence radius | Model-relative calibration (fraction of mean pairwise embedding distance) |
| CMA-ES dependency | Not available, not required for v1 |
| Decoder architecture | Embedding-layer injection, lazy model loading, model ID validation |
| Evaluator patterns | Two evaluators implementing `Evaluator[EmbeddingGenome]` protocol |
| Population initialization | Noise-based (default) and LLM-discovered strategies |
| MLflow metrics | ESPO-specific callback using existing MLflowTracker |
| Registry integration | Built-in registration in genome and operator registries |
