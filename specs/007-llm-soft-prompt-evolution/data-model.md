# Data Model: Evolutionary Soft-Prompt Optimization (ESPO)

**Feature Branch**: `007-llm-soft-prompt-evolution`
**Date**: 2026-04-04

## Entity Relationship Diagram

```text
┌──────────────────────┐
│   EmbeddingGenome    │
│──────────────────────│
│ embeddings: ndarray  │──────── shape (n_tokens, embed_dim)
│ model_id: str        │
│ seed_text: str | None│
│ strategy: Dim.Strat. │
│ projection: ndarray? │──────── optional, shape (subspace_dim, embed_dim)
└──────────┬───────────┘
           │ implements
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Genome (Protocol)   │     │ SerializableGenome   │
│──────────────────────│     │   (Protocol)         │
│ copy() → Self        │     │──────────────────────│
│ __eq__() → bool      │     │ to_dict() → dict     │
│ __hash__() → int     │     │ from_dict() → Self   │
└──────────────────────┘     └──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  EmbeddingGenomeConfig│        │ DimensionalityStrategy│
│──────────────────────│         │   (Enum)             │
│ n_tokens: int        │         │──────────────────────│
│ embed_dim: int       │────────▶│ FULL_SPACE           │
│ model_id: str        │         │ COMPRESSED_SUBSPACE  │
│ strategy: Dim.Strat. │         │ MINIMAL_TOKENS       │
│ seed_text: str | None│         └──────────────────────┘
│ coherence_radius: flt│
│ subspace_dim: int?   │
│ projection_matrix: ?  │
└──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  SoftPromptDecoder   │         │    TaskSpec          │
│──────────────────────│         │──────────────────────│
│ model_id: str        │         │ task_type: str       │
│ model: PreTrainedModel│        │ inputs: list[dict]   │
│ tokenizer: Tokenizer │         │ ground_truth: list?  │
│──────────────────────│         │ rubric: list[Crit.]? │
│ decode(genome, input)│         │ metrics: list[str]   │
│   → str              │         │ max_gen_tokens: int  │
└──────────────────────┘         └──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│ GroundTruthEvaluator │         │  LLMJudgeEvaluator   │
│──────────────────────│         │──────────────────────│
│ decoder: Decoder     │         │ decoder: Decoder     │
│ task_spec: TaskSpec  │         │ task_spec: TaskSpec  │
│──────────────────────│         │ judge_model: str     │
│ evaluate(inds, seed) │         │──────────────────────│
│   → list[Fitness]    │         │ evaluate(inds, seed) │
│ capabilities         │         │   → list[Fitness]    │
└──────────────────────┘         │ capabilities         │
                                 └──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  TokenAwareMutator   │         │ TokenLevelCrossover  │
│──────────────────────│         │──────────────────────│
│ mutation_rate: float │         │ crossover_type: str  │
│ sigma: float         │         │   (single/two_point) │
│ coherence_radius: flt│         │──────────────────────│
│──────────────────────│         │ crossover(p1, p2, rng)│
│ mutate(genome, rng)  │         │   → (child1, child2) │
│   → EmbeddingGenome  │         └──────────────────────┘
└──────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│ PopulationInitializer│         │  CoherenceDefense    │
│──────────────────────│         │──────────────────────│
│ strategy: str        │         │ enable_mutation_clamp│
│   (noise / llm_var)  │         │ enable_perplexity   │
│ config: GenomeConfig │         │ enable_fitness_sel   │
│──────────────────────│         │ coherence_radius: flt│
│ initialize(seed, n)  │         │ perplexity_thresh:flt│
│   → list[Emb.Genome] │         │──────────────────────│
└──────────────────────┘         │ check(genome) → bool │
                                 │ clamp(delta) → delta │
                                 └──────────────────────┘
```

## Entities

### EmbeddingGenome

**Module**: `evolve/representation/embedding.py`
**Implements**: `Genome`, `SerializableGenome`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `embeddings` | `np.ndarray` | shape `(n_tokens, embed_dim)`, immutable | 2D continuous soft-prompt embedding array |
| `model_id` | `str` | non-empty | Target model identifier (e.g., `"meta-llama/Llama-2-7b-hf"`) |
| `seed_text` | `str \| None` | — | Original seed text used for initialization |
| `strategy` | `DimensionalityStrategy` | enum value | Which dimensionality strategy produced this genome |

**Derived properties**:
- `n_tokens: int` → `embeddings.shape[0]`
- `embed_dim: int` → `embeddings.shape[1]`

**Methods**:
- `copy() → EmbeddingGenome` — deep copy with `embeddings.copy()`
- `flat() → np.ndarray` — 1D view of shape `(n_tokens * embed_dim,)`
- `to_vector_genome() → VectorGenome` — adapter for flat-vector operators
- `from_vector_genome(vg, model_id, seed_text, strategy) → EmbeddingGenome` — reconstruct from flat
- `to_dict() → dict` — JSON-serializable dict (embeddings as nested list)
- `from_dict(data) → EmbeddingGenome` — reconstruct from dict
- `__eq__`, `__hash__` — structural equality on embeddings

**Validation rules**:
- `embeddings.ndim == 2`
- `embeddings.shape[0] >= 1` (at least 1 token)
- `embeddings.shape[1] >= 1` (at least 1 dimension)
- `model_id` is non-empty string
- `embeddings` made immutable in `__post_init__`

---

### DimensionalityStrategy

**Module**: `evolve/representation/embedding_config.py`

| Value | Description |
|-------|-------------|
| `FULL_SPACE` | Full `embed_dim` per token, full search space |
| `COMPRESSED_SUBSPACE` | Reduced subspace_dim per token, projected at decode time |
| `MINIMAL_TOKENS` | Full `embed_dim` but only 4–8 tokens (DEFAULT) |

---

### EmbeddingGenomeConfig

**Module**: `evolve/representation/embedding_config.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_tokens` | `int` | `8` | Number of virtual tokens |
| `embed_dim` | `int` | required | Model embedding dimension |
| `model_id` | `str` | required | Target model identifier |
| `strategy` | `DimensionalityStrategy` | `MINIMAL_TOKENS` | Dimensionality strategy |
| `seed_text` | `str \| None` | `None` | Seed text for initialization |
| `coherence_radius` | `float` | `0.1` | L2 norm bound (fraction of mean embedding distance) |
| `subspace_dim` | `int \| None` | `None` | Required when strategy=COMPRESSED_SUBSPACE |
| `projection_matrix` | `np.ndarray \| None` | `None` | Required when strategy=COMPRESSED_SUBSPACE, shape `(subspace_dim, embed_dim)` |

**Validation rules**:
- `n_tokens >= 1`
- `embed_dim >= 1`
- `coherence_radius > 0`
- If `strategy == COMPRESSED_SUBSPACE`: `subspace_dim` and `projection_matrix` must be provided
- If `strategy == MINIMAL_TOKENS`: `n_tokens` should be in range [4, 8] (soft constraint, warning if outside)

---

### TaskSpec

**Module**: `evolve/evaluation/task_spec.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_type` | `str` | required | `"qa"`, `"generation"`, `"classification"`, `"instruction_following"` |
| `inputs` | `list[dict[str, str]]` | required | List of task inputs, each with at least `"input"` key |
| `ground_truth` | `list[str] \| None` | `None` | Ground-truth answers (required for benchmark eval) |
| `rubric` | `list[RubricCriterion] \| None` | `None` | Scoring criteria (required for LLM-judge eval) |
| `metrics` | `list[str]` | `["accuracy"]` | Metrics to compute: `"accuracy"`, `"f1"`, `"exact_match"`, `"pass_at_k"` |
| `max_generation_tokens` | `int` | `256` | Max tokens for model generation |
| `sample_size` | `int \| None` | `None` | Subsample inputs per evaluation (None = use all) |

---

### RubricCriterion

**Module**: `evolve/evaluation/task_spec.py`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Criterion name (e.g., `"clarity"`, `"relevance"`) |
| `description` | `str` | What the judge should evaluate |
| `scale_min` | `float` | Minimum score (e.g., 0.0) |
| `scale_max` | `float` | Maximum score (e.g., 1.0) |

---

### SoftPromptDecoder

**Module**: `evolve/meta/soft_prompt/decoder.py`
**Dependencies**: `torch`, `transformers` (optional imports)

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Model identifier for loading |
| `device` | `str` | `"cpu"` or `"cuda"` |
| `_model` | `PreTrainedModel \| None` | Lazily loaded model |
| `_tokenizer` | `PreTrainedTokenizer \| None` | Lazily loaded tokenizer |

**Methods**:
- `decode(genome: EmbeddingGenome, task_input: str, max_tokens: int) → str`
- `_load_model() → None` — lazy model initialization
- `_validate_genome(genome: EmbeddingGenome) → None` — checks model_id match

---

### GroundTruthEvaluator

**Module**: `evolve/evaluation/benchmark.py`
**Implements**: `Evaluator[EmbeddingGenome]`

| Field | Type | Description |
|-------|------|-------------|
| `decoder` | `SoftPromptDecoder` | Shared decoder instance |
| `task_spec` | `TaskSpec` | Task configuration with ground truth |

**Methods**:
- `evaluate(individuals, seed) → Sequence[Fitness]`
- `capabilities → EvaluatorCapabilities`

---

### LLMJudgeEvaluator

**Module**: `evolve/evaluation/llm_judge.py`
**Implements**: `Evaluator[EmbeddingGenome]`

| Field | Type | Description |
|-------|------|-------------|
| `decoder` | `SoftPromptDecoder` | Shared decoder instance |
| `task_spec` | `TaskSpec` | Task configuration with rubric |
| `judge_model_id` | `str` | Judge model identifier |
| `temperature` | `float` | Judge temperature (0.0 for determinism) |

**Methods**:
- `evaluate(individuals, seed) → Sequence[Fitness]` — returns multi-dimensional fitness
- `capabilities → EvaluatorCapabilities`

---

### TokenAwareMutator

**Module**: `evolve/core/operators/token_mutation.py`
**Implements**: `MutationOperator[EmbeddingGenome]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mutation_rate` | `float` | `0.1` | Per-token mutation probability |
| `sigma` | `float` | `0.1` | Gaussian noise standard deviation |
| `coherence_radius` | `float \| None` | `None` | L2 norm clamp (None = no clamping) |

**Methods**:
- `mutate(genome: EmbeddingGenome, rng: Random) → EmbeddingGenome`

---

### TokenLevelCrossover

**Module**: `evolve/core/operators/token_crossover.py`
**Implements**: `CrossoverOperator[EmbeddingGenome]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `crossover_type` | `str` | `"single_point"` | `"single_point"` or `"two_point"` |

**Methods**:
- `crossover(parent1: EmbeddingGenome, parent2: EmbeddingGenome, rng: Random) → tuple[EmbeddingGenome, EmbeddingGenome]`

**Validation**: Parents must have same `n_tokens`, `embed_dim`, and `model_id`.

---

### PopulationInitializer

**Module**: `evolve/meta/soft_prompt/initializer.py`

| Field | Type | Description |
|-------|------|-------------|
| `config` | `EmbeddingGenomeConfig` | Genome configuration |
| `decoder` | `SoftPromptDecoder` | For embedding seed text |

**Methods**:
- `noise_init(seed_text: str, n: int, rng: Random) → list[EmbeddingGenome]`
- `llm_variation_init(seed_text: str, n: int, rng: Random) → list[EmbeddingGenome]`

---

### CoherenceDefense

**Module**: `evolve/meta/soft_prompt/coherence.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_mutation_clamp` | `bool` | `True` | Layer 1: L2 norm bound |
| `enable_perplexity_check` | `bool` | `True` | Layer 2: cheap feasibility check |
| `enable_fitness_selection` | `bool` | `True` | Layer 3: fitness-based selection |
| `coherence_radius` | `float` | `0.1` | L2 norm bound for mutation clamping |
| `perplexity_threshold` | `float` | `100.0` | Max output perplexity for feasibility |

**Methods**:
- `clamp_mutation(original: np.ndarray, mutated: np.ndarray) → np.ndarray` — Layer 1
- `check_feasibility(genome: EmbeddingGenome) → bool` — Layer 2, returns True if feasible
- `mark_infeasible(fitness: Fitness) → Fitness` — Adds constraint violation to Fitness

---

## State Transitions

### Genome Lifecycle

```text
[Seed Text] ──embed──▶ [Initial Genome] ──mutate/crossover──▶ [Offspring Genome]
                              │                                        │
                              ▼                                        ▼
                       [Decode → Text] ──evaluate──▶ [Fitness]  [Decode → Text]
                                                                       │
                                                                       ▼
                                                                  [Fitness]
```

### Coherence Defense Pipeline

```text
[Mutation Applied]
       │
       ▼
┌─ Layer 1: Mutation Clamp ─────┐
│ if ‖delta‖₂ > radius:        │
│   delta ← delta × radius/‖δ‖ │
└───────────────┬───────────────┘
                │
                ▼
┌─ Layer 2: Perplexity Check ───┐
│ if perplexity(decode(g)) > θ: │
│   mark infeasible             │
└───────────────┬───────────────┘
                │
                ▼
┌─ Layer 3: Fitness Selection ──┐
│ Framework handles infeasible  │
│ via Fitness.constraints       │
└───────────────────────────────┘
```

### All-Infeasible Recovery

```text
[All individuals infeasible in gen N]
       │
       ▼
[Restore gen N-1 population]
       │
       ▼
[Reduce mutation magnitude by 50%]
       │
       ▼
[Retry gen N with reduced mutations]
```
