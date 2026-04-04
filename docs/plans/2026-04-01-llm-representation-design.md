# LLM Representation Design: Evolutionary Soft-Prompt Optimization (ESPO)

**Date:** 2026-04-01
**Status:** Draft / Ideation

---

## 1. Overview

This document captures the design for evolving natural language artifacts (system prompts, agent files, skills, scripts, etc.) within the evolve-framework. The core idea: represent text as **continuous soft-prompt embeddings** and evolve them using the framework's existing evolutionary operators, fitness evaluation, and population management infrastructure.

**Working name:** Evolutionary Soft-Prompt Optimization (ESPO)

### Key Properties

- **Artifact-agnostic**: The design does not assume any specific text artifact type. The same genome representation works for system prompts, Python scripts, YAML configs, etc.
- **Open-weight models only**: Requires embedding-layer access for soft-prompt injection. API-only models are not supported (though text-mediated transfer to API models is possible).
- **Zero engine changes**: Plugs into the existing framework via new `Genome`, `Decoder`, `Evaluator`, and operator implementations.

---

## 2. Representation: Soft-Prompt Embedding Genome

### Genome Structure

The genome is a 2D continuous array of shape `(n_tokens, embed_dim)`:

- `n_tokens`: Number of virtual/soft tokens prepended to model input (typically 4–128)
- `embed_dim`: The target model's embedding dimension (e.g., 4096 for LLaMA-8B)
- Total genome dimensionality: `n_tokens × embed_dim` floating-point values

The genome implements the framework's `Genome` and `SerializableGenome` protocols. It stores:
- `embeddings: np.ndarray` — shape `(n_tokens, embed_dim)`, immutable
- `model_id: str` — identifier for the target model (e.g., `"meta-llama/Llama-3-8B"`)
- `seed_text: str | None` — the original text this genome was derived from (for lineage tracking)

### Design Rationale

- **Why continuous embeddings, not discrete tokens?** Continuous space enables standard evolutionary operators (Gaussian mutation, blend crossover, SBX) without modification. Discrete token evolution requires combinatorial operators and has a much larger, sparser search space.
- **Why 2D structure, not flat vector?** Preserving the token × dim structure allows token-aware operators (e.g., mutate whole tokens, crossover at token boundaries) while still supporting flat-vector operators via a `.flat` accessor.
- **Why not a structured JSON genome?** A JSON spec genome requires defining all evolvable fields upfront per artifact type — not artifact-agnostic. Embeddings sidestep this by letting the model's learned representation define the variation space.

### Dimensionality: A Research Question

The genome dimensionality (e.g., 32 × 4096 = 131K floats) is high for evolutionary search. Three strategies should be experimentally compared:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Full-space** | Evolve the complete `(n_tokens, embed_dim)` vector | Maximum expressiveness; dimensionality curse |
| **Compressed subspace** | PCA/random projection to reduce `embed_dim` to ~64–256 dims; project back at decode time | Tractable search; limited by projection quality |
| **Minimal tokens** | Use few virtual tokens (4–8) to keep total dims manageable | Small search space; limited prompt expressiveness |

Design the genome type to support all three. Subspace evolution means the genome stores a low-dimensional vector plus a projection matrix.

---

## 3. Decoding: Soft-Prompt Injection

### Mechanism

The `Decoder` converts an `EmbeddingGenome` into a **callable phenotype** that takes task input text and returns model output text:

1. Convert the genome's numpy embedding array to a torch tensor
2. Prepend the soft-prompt tensor to the model's input embeddings
3. Run the model's `generate()` method
4. Return the decoded output text

The phenotype is a function: `(task_input: str) → output_text: str`

### Key Properties

- **No lossy decode step**: The embedding IS the phenotype. No need to convert back to text for evaluation.
- **Requires model internals**: Must access the model's embedding layer to inject soft-prompt vectors. This rules out API-only models.
- **Model-specific**: An evolved soft prompt is tied to the model it was evolved for (see Multi-Model Transfer below).

### The NO ML IMPORTS Boundary

Per framework convention, the genome module (`embedding.py`) contains no ML framework imports — it uses only numpy. The decoder module (`embedding_decoder.py`) sits outside this boundary and imports torch/transformers as needed.

---

## 4. Fitness Evaluation

### Priority: Start with Benchmarks

Deterministic benchmark evaluation should be the starting point. It's cheaper, reproducible, and easier to debug. LLM-as-judge is added when tasks lack ground truth.

### Evaluator Types

#### 4.1 Benchmark Evaluator (Primary)

For tasks with known correct answers (QA, code generation, classification):
1. Prepend soft-prompt embedding to each test input
2. Run model forward pass → get output
3. Compare output to ground truth
4. Fitness = accuracy / F1 / exact-match / pass@k

Properties: `batchable=True`, `stochastic=False`, deterministic.

#### 4.2 LLM-as-Judge Evaluator

For open-ended quality assessment:
1. Prepend soft-prompt → get model output
2. Send output + rubric to a judge LLM
3. Judge scores on configurable criteria (helpfulness, coherence, relevance, etc.)
4. Each criterion becomes a fitness objective → multi-objective support via `Fitness(values=np.array([score1, score2, ...]))`

Properties: `batchable=True`, `stochastic=True` (judge responses vary).

#### 4.3 Hybrid Evaluator

Composes the above. Uses cheap deterministic metrics as filters (e.g., perplexity, format compliance) and expensive LLM judge as secondary objectives.

### Task Specification

The evaluation is parameterized by a `TaskSpec` configuration:
- **task_type**: generation, classification, QA, code completion, instruction following, etc.
- **input_set**: Inputs the soft prompt is evaluated on, split into train/validation/test
- **scoring_rubric**: For LLM-as-judge, the criteria and weights. For deterministic, the metric function.
- **generalization_requirement**: Whether fitness is averaged across all inputs (robust prompt) or measured per-input

The same genome type + different `TaskSpec` = different experiments. The genome is artifact-agnostic; the `TaskSpec` is where domain-specificity lives.

### Generalization

Always evaluate on the full input set (or stratified sample), report mean + variance. Evaluating on a single input leads to overfitting. Including validation-set fitness as a secondary objective or constraint handles this.

---

## 5. Evolutionary Operators

### Mutation

**Token-aware Gaussian mutation** (primary):
- For each virtual token (with probability `token_mutation_rate`): add Gaussian noise `N(0, σ)` to the token's embedding vector
- `σ` (noise scale) is a key hyperparameter, calibrated per model
- Respects token boundaries — a mutation affects one token's full embedding, not random individual dimensions across tokens

Existing framework operators that work as fallbacks (on the flattened vector):
- `GaussianMutation`, `PolynomialMutation`, `CreepMutation`

### Crossover

**Token-level crossover** (primary):
- Single-point or two-point crossover at token boundaries
- Swap whole virtual tokens between parents, not individual embedding dimensions
- Preserves per-token coherence

**Blend crossover** in embedding space:
- Interpolate between parent embeddings: `child = α × parent1 + (1-α) × parent2`
- Works at the full array level

Existing framework operators that work as fallbacks:
- `BlendCrossover`, `SimulatedBinaryCrossover`, `UniformCrossover` (on flattened vector)

### Adaptive Operators

For full-space evolution, CMA-ES or separable CMA-ES is strongly recommended. The framework should support this as an operator or engine variant. CMA-ES learns the covariance structure from selection pressure, effectively discovering which directions in embedding space matter — addressing the "what does it mean to perturb in a meaningful direction?" question through data rather than assumptions.

---

## 6. Population Initialization

The `PopulationInitializer` takes a seed text and `n_tokens` config, returns `list[EmbeddingGenome]`.

### Strategy 1: Noise-Based

1. Tokenize seed text, pass through model's embedding layer → get reference embedding `(k, embed_dim)`
2. Truncate or pad to `(n_tokens, embed_dim)`
3. For each individual in population: `seed_embedding + N(0, σI)`
4. Calibrate σ: binary search for the σ where ~80% of perturbations produce coherent model output (quick perplexity check)

**Strengths**: Minimal bias. With adaptive operators (CMA-ES), the search discovers meaningful directions from selection pressure alone.

**Weaknesses**: May waste early evaluations on incoherent individuals. Slow convergence in very high-dimensional spaces.

### Strategy 2: LLM-Discovered Variation Axes

1. Give the seed text to an LLM: *"What are the 10 most impactful dimensions along which this text could meaningfully vary?"*
2. For each discovered dimension, ask the LLM: *"Rewrite this text, pushing dimension D to [low / medium / high]."*
3. Generate ~30 text variants (10 dimensions × 3 levels)
4. Tokenize each variant, get embeddings, truncate/pad to `(n_tokens, embed_dim)`

**Strengths**: Principled coverage of meaningful variation space. Interpretable initial population (you know what each variant changed). Artifact-agnostic because the LLM discovers relevant axes per artifact.

**Weaknesses**: Requires N LLM calls at init time. Axis discovery may miss non-obvious dimensions.

### Token Count Handling

Both strategies respect the configured `n_tokens`:
- If the embedded text has fewer tokens than `n_tokens`: pad with zero vectors (or a learned initialization vector)
- If more tokens: truncate to the first `n_tokens` embeddings

---

## 7. Coherence Collapse Defense

When mutation/crossover produces embeddings far from the valid text manifold, the model may produce gibberish. Three-layer defense:

### Layer 1: Mutation Magnitude Constraint
Bound the L2 norm of any mutation to stay within a "coherence radius" of the parent. Calibrated per model: measure average L2 distance between embeddings of semantically similar texts, use as upper bound.

### Layer 2: Cheap Feasibility Check
Before expensive evaluation, do a quick forward pass with soft prompt + trivial input. If model output perplexity exceeds a threshold, mark as infeasible. Uses the framework's `Fitness.constraints` mechanism: `constraints = np.array([perplexity - threshold])`. Feasibility-based selection handles the rest.

### Layer 3: Fitness Selection
Among feasible (coherent) individuals, fitness ranking handles quality differentiation.

These layers are independently toggleable for research purposes (e.g., test whether evolution can find coherent regions from pure noise).

---

## 8. Framework-Level Enhancements Triggered by This Design

These are generic framework features that benefit all genome types, not just LLM:

### 8.1 Generic Fitness Caching

A `CachingEvaluator` wrapper that works with any `Evaluator`:
- Cache key: `hash(genome)` → fitness
- For stochastic evaluators: cache N evaluations and return average
- Configurable cache size and eviction policy
- Framework-level, not LLM-specific

### 8.2 Adaptive Population Sizing

Engine-level support for changing population size across generations:
- Start small + high mutation for exploration
- Grow population as search narrows
- Schedule defined in config (e.g., `pop_size_schedule: {0: 30, 50: 100, 80: 200}`)

---

## 9. Multi-Model Transfer

### Text-Mediated Transfer (Primary Approach)

When transferring an evolved soft prompt from model A to model B:

1. Take the best individual's embedding from model A
2. Use an LLM-conditioned decode: find k-nearest text neighbors in a reference library, generate a text interpretation
3. Use the decoded text as a standard (hard) prompt for model B
4. Optionally: use the decoded text as a seed to initialize a new ESPO run on model B

This loses soft-prompt precision but provides universal transferability.

### Same-Family Transfer (Future Work)

For models in the same family (e.g., LLaMA-8B → LLaMA-70B) that share tokenizers, a learned linear projection between embedding spaces may preserve more of the evolved structure. This requires training the projection on paired embeddings.

---

## 10. Literature Positioning

| Method | Search Space | Optimizer | Differentiable? | Multi-objective? |
|--------|-------------|-----------|-----------------|-----------------|
| Prompt Tuning (Lester 2021) | Continuous embeddings | Gradient descent | Yes (required) | No |
| EvoPrompting (Chen 2023) | Discrete text | Evolutionary | No | No |
| PromptBreeder (Fernando 2023) | Discrete text | Self-referential LLM | No | No |
| BBT/BBTv2 (Sun 2022) | Continuous (random subspace) | Gradient estimation | Sort of | No |
| **ESPO (this design)** | **Continuous embeddings** | **Evolutionary (EA/CMA-ES)** | **No** | **Yes (NSGA-II)** |

### Key Differentiators

1. **No differentiability requirement**: Works on quantized, distilled, or any model with embedding-layer access
2. **Multi-objective native**: Multiple quality criteria as separate objectives via NSGA-II
3. **Multi-modal exploration**: EA searches across multiple fitness landscape modes; gradient descent converges to nearest local optimum
4. **Evolvable Reproduction Protocols**: The framework's ERP feature could evolve *how* prompts mate, potentially discovering useful crossover strategies autonomously
5. **Framework integration**: Inherits all evolve-framework infrastructure (speciation, diversity maintenance, MLflow tracking, etc.)

---

## 11. Future Extensions (Not in Scope)

These are noted for completeness but intentionally excluded from the initial design:

- **Discrete token genome**: Evolve integer token IDs with combinatorial operators. Different genome type, different operators.
- **Structured JSON genome**: Evolve a schema-based representation. Useful for artifacts with known structure. Requires per-artifact-type schema design.
- **Surrogate-assisted evaluation**: Train a cheap model to predict fitness from embeddings. Shelved — adds overhead of maintaining/retraining the surrogate per fitness function change.
- **Variable-length soft prompts**: Allow `n_tokens` to vary per individual. Requires sequence-style operators for length changes. Keep `n_tokens` fixed per experiment initially; sweep across experiments.
- **Universal embedding space**: Evolve in a model-agnostic embedding space (e.g., sentence-transformers) and project into model-specific spaces. Decouples search from any single model.

---

## 12. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target artifact | Artifact-agnostic | Generality from day one |
| Genome type | Continuous embedding vectors `(n_tokens, embed_dim)` | Enables standard EA operators |
| Decode strategy | Soft-prompt injection | No lossy decode; embedding IS phenotype |
| Model requirement | Open-weight only | Required for embedding-layer access |
| Dimensionality handling | Research variable: full-space, compressed, minimal tokens | Core experimental question |
| Init strategy A | Noise-based + adaptive operators | Minimal bias; let CMA-ES learn directions |
| Init strategy B | LLM-discovered variation axes | Principled coverage; artifact-agnostic |
| Fitness (primary) | Benchmark datasets with deterministic metrics | Reproducible, cheap, debuggable |
| Fitness (secondary) | LLM-as-judge with configurable rubric | For open-ended quality; multi-objective |
| Coherence defense | 3-layer: mutation bound + perplexity check + fitness | Prevents wasted evaluations |
| Token count | Fixed per experiment; sweep across experiments | Simplifies operators |
| Cross-model transfer | Text-mediated decode | Universal; no new infrastructure |
| Framework enhancements | Generic fitness caching + adaptive population sizing | Benefits all genome types |
