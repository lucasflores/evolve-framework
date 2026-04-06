# Feature Specification: Evolutionary Soft-Prompt Optimization (ESPO)

**Feature Branch**: `007-llm-soft-prompt-evolution`
**Created**: 2026-04-04
**Status**: Draft
**Input**: User description: "Evolutionary Soft-Prompt Optimization - evolve natural language artifacts as continuous soft-prompt embeddings using evolutionary operators"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Evolve a Soft Prompt on a Benchmark Task (Priority: P1)

A researcher wants to optimize an LLM's behavior on a question-answering benchmark by evolving soft-prompt embeddings. They configure an experiment with a seed text, a target open-weight model, a benchmark dataset with ground-truth answers, and run the evolutionary loop. The system produces a population of soft-prompt individuals, evaluates them deterministically, and returns the best-performing soft prompt along with fitness metrics.

**Why this priority**: This is the core value proposition — demonstrating that evolutionary search over continuous embeddings can improve LLM task performance without gradient-based training. Everything else builds on this working end-to-end.

**Independent Test**: Can be fully tested by running a single ESPO experiment on a small QA dataset (e.g., 50 questions) with a local open-weight model and verifying that the best individual's accuracy exceeds the seed prompt's accuracy.

**Acceptance Scenarios**:

1. **Given** a seed text, a target open-weight model identifier, and a benchmark dataset with ground-truth answers, **When** the user configures and runs an ESPO experiment, **Then** the system initializes a population of soft-prompt embedding genomes, runs evolutionary selection for the configured number of generations, and reports the best individual's fitness score.
2. **Given** an ESPO experiment is running, **When** each generation completes, **Then** the system evaluates every individual by injecting their soft-prompt embeddings into the target model, comparing outputs to ground truth, and computing deterministic fitness (accuracy/F1/exact-match).
3. **Given** the experiment completes, **When** the user inspects results, **Then** the best individual's fitness score is equal to or better than the seed text's baseline fitness, and per-generation metrics (best/mean/worst fitness, diversity stats, infeasibility rate, mutation magnitude, best decoded text) are logged for reproducibility.

---

### User Story 2 - Create and Configure an Embedding Genome (Priority: P1)

A researcher wants to create soft-prompt embedding genomes that plug into the existing evolve-framework population management. They specify a target model, a number of virtual tokens, and a seed text. The system produces genome objects compatible with the framework's `Genome` and `SerializableGenome` protocols, enabling standard framework operations (serialization, hashing, population management).

**Why this priority**: The genome is the foundational data structure. Without it, no other ESPO component can function. It must integrate cleanly with the existing framework protocols.

**Independent Test**: Can be tested by creating an `EmbeddingGenome`, verifying its shape, serializing/deserializing it, and confirming it satisfies the framework's genome protocol requirements.

**Acceptance Scenarios**:

1. **Given** a model identifier and a token count, **When** the user creates an embedding genome from a seed text, **Then** the genome stores a 2D embedding array of shape `(n_tokens, embed_dim)`, the model identifier, and the original seed text.
2. **Given** an embedding genome, **When** the user serializes and then deserializes it, **Then** the reconstructed genome is identical to the original (embeddings, model ID, seed text).
3. **Given** an embedding genome, **When** the user requests a flat-vector view, **Then** the system returns the embeddings as a 1D array of length `n_tokens × embed_dim` suitable for flat-vector operators.

---

### User Story 3 - Apply Token-Aware Evolutionary Operators (Priority: P2)

A researcher wants to mutate and recombine soft-prompt genomes using operators that respect the token-level structure of embeddings. Token-aware Gaussian mutation perturbs whole token embeddings, and token-level crossover swaps complete tokens between parents, preserving per-token coherence.

**Why this priority**: Custom operators that respect the 2D token structure are expected to outperform naive flat-vector operators. This is a key differentiator of the ESPO approach.

**Independent Test**: Can be tested by applying token-aware mutation to a genome and verifying that only selected tokens are perturbed while others remain unchanged, and by applying token-level crossover to two parents and verifying that offspring tokens are valid whole-token copies from one parent or the other.

**Acceptance Scenarios**:

1. **Given** an embedding genome and a token mutation rate, **When** token-aware Gaussian mutation is applied, **Then** each token is independently selected for mutation with the configured probability, and mutated tokens have Gaussian noise added to their full embedding vector.
2. **Given** two parent embedding genomes with the same shape, **When** token-level crossover is applied, **Then** the offspring genome contains whole tokens from one parent or the other (no partial-token mixing), and the offspring has the same shape as the parents.
3. **Given** an embedding genome, **When** the existing framework flat-vector operators (GaussianMutation, BlendCrossover, etc.) are applied via the flat accessor, **Then** they produce valid offspring genomes with correct shape and model association.

---

### User Story 4 - Initialize a Population from Seed Text (Priority: P2)

A researcher wants to create a diverse initial population from a single seed text. Two strategies are available: noise-based initialization (perturb the seed embedding with calibrated Gaussian noise) and LLM-discovered variation axes (use an LLM to generate meaningful text variants, then embed them).

**Why this priority**: Good initialization accelerates convergence and improves final solution quality. Both strategies are needed to compare their effectiveness experimentally.

**Independent Test**: Can be tested by initializing a population from a seed text using each strategy and verifying that all individuals have valid genome shapes, are distinct from each other, and produce coherent model outputs.

**Acceptance Scenarios**:

1. **Given** a seed text and a population size, **When** noise-based initialization is used, **Then** the system embeds the seed text, pads or truncates to the configured token count, and produces the requested number of distinct genomes by adding calibrated Gaussian noise.
2. **Given** a seed text and a population size, **When** LLM-discovered variation axis initialization is used, **Then** the system identifies meaningful variation dimensions, generates text variants along those axes, embeds each variant, and returns a diverse population of genomes.
3. **Given** a seed text with fewer tokens than the configured `n_tokens`, **When** the population is initialized, **Then** the system pads the embedding to the configured token count. Conversely, if the seed text has more tokens, the system truncates to `n_tokens`.

---

### User Story 5 - Decode Soft Prompts via Model Injection (Priority: P2)

A researcher wants to see what a soft-prompt genome actually does — the system takes a genome and a task input, injects the genome's embeddings into the target model's embedding layer, runs generation, and returns the model's output text.

**Why this priority**: Decoding is essential for both fitness evaluation and for the researcher to inspect and understand what evolved soft prompts produce.

**Independent Test**: Can be tested by creating a genome, decoding it with a trivial input on a local model, and verifying that the output is a valid text string.

**Acceptance Scenarios**:

1. **Given** an embedding genome and a task input string, **When** the decoder is invoked, **Then** the system prepends the genome's embeddings to the model's input embeddings and returns the model's generated output text.
2. **Given** a genome evolved for model A and a task input, **When** decoding is attempted with a different model B, **Then** the system rejects the operation with a clear error indicating model mismatch.

---

### User Story 6 - Defend Against Coherence Collapse (Priority: P3)

A researcher running ESPO experiments wants the system to prevent wasted evaluations on genomes that produce gibberish. The system applies a configurable multi-layer defense: mutation magnitude constraints, a cheap perplexity-based feasibility check, and fitness-based selection.

**Why this priority**: Coherence collapse wastes compute and degrades search efficiency. However, the core loop can function without these defenses (just less efficiently), making this a quality-of-life enhancement.

**Independent Test**: Can be tested by creating deliberately extreme mutations and verifying that each defense layer (independently toggleable) correctly identifies and handles infeasible individuals.

**Acceptance Scenarios**:

1. **Given** a mutation operator and a configured coherence radius, **When** the mutation would produce an embedding beyond the L2 norm bound, **Then** the mutation is clamped to stay within the coherence radius.
2. **Given** a mutated genome, **When** the cheap feasibility check is enabled, **Then** the system runs a quick forward pass and marks genomes with output perplexity above the threshold as infeasible using the framework's constraint mechanism.
3. **Given** coherence defense layers are individually toggled off, **When** an experiment runs, **Then** only the enabled layers are applied, allowing researchers to study their individual effects.

---

### User Story 7 - Evaluate with LLM-as-Judge for Open-Ended Tasks (Priority: P3)

A researcher wants to optimize a soft prompt for an open-ended task (creative writing, instruction following) where no ground-truth answers exist. The system sends model outputs to a judge LLM that scores on configurable rubric criteria, with each criterion becoming a separate fitness objective for multi-objective optimization.

**Why this priority**: LLM-as-judge evaluation extends ESPO beyond benchmark tasks to real-world open-ended optimization. It depends on the deterministic evaluation pipeline (P1) being stable first.

**Independent Test**: Can be tested by configuring a rubric with two criteria, evaluating a genome's output, and verifying that the returned fitness contains a value for each criterion.

**Acceptance Scenarios**:

1. **Given** a soft-prompt genome, a task input, and a scoring rubric with multiple criteria, **When** the LLM-as-judge evaluator runs, **Then** the judge scores the model's output on each criterion and returns a multi-dimensional fitness vector.
2. **Given** a multi-objective fitness from LLM-as-judge evaluation, **When** the fitness is used in selection, **Then** the framework's multi-objective selection (e.g., NSGA-II) handles it correctly.

---

### User Story 8 - Transfer Evolved Prompts Across Models via Text (Priority: P3)

A researcher evolves a high-performing soft prompt on model A and wants to use it with model B (which may be a different architecture or an API-only model). The system decodes the best soft prompt into a text interpretation and uses that text as a standard hard prompt for the target model.

**Why this priority**: Cross-model transfer broadens the utility of ESPO results but is not required for the core evolutionary loop.

**Independent Test**: Can be tested by evolving a prompt on one model, transferring it to text, and verifying that the text can be used as a standard prompt input for a different model.

**Acceptance Scenarios**:

1. **Given** a best-performing embedding genome from model A, **When** text-mediated transfer is invoked, **Then** the system generates a text interpretation of the soft prompt suitable for use as a hard prompt.
2. **Given** a transferred text prompt, **When** it is used as a seed for a new ESPO run on model B, **Then** the system initializes a new population from that text on model B's embedding space.

---

### Edge Cases

- What happens when the target model is not available locally or lacks embedding-layer access?
- How does the system handle genomes with mismatched `embed_dim` (e.g., genome created for one model applied to another)?
- What happens when all individuals in a generation are marked infeasible by the coherence defense? The system falls back to the previous generation's population and retries with reduced mutation magnitude.
- How does the system handle seed text that tokenizes to zero tokens (empty string)?
- What happens when the LLM-as-judge evaluator is unreachable or returns malformed scores?
- How does the system handle extremely high-dimensional genomes (e.g., 128 tokens × 4096 dims) in terms of memory and serialization?

## Clarifications

### Session 2026-04-04

- Q: When all individuals in a generation are marked infeasible by the coherence defense, what should the evolutionary loop do? → A: Fall back to previous generation's population and retry with reduced mutation magnitude.
- Q: What is the acceptable wall-clock time budget per individual per generation for benchmark evaluation? → A: ≤60 seconds per individual.
- Q: For compressed subspace evolution, who provides the projection matrix and when? → A: Pre-computed once before the experiment, fixed throughout evolution.
- Q: What level of observability should the system provide during an ESPO run? → A: Comprehensive per-generation logging: best/mean/worst fitness, diversity stats, infeasibility rate, mutation magnitude, and best individual's decoded text.
- Q: What should the default dimensionality strategy be when the user does not explicitly choose one? → A: Minimal tokens (4–8 tokens at full embed_dim).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide an embedding genome type that stores a 2D continuous array of shape `(n_tokens, embed_dim)`, a model identifier, and an optional seed text, implementing the framework's `Genome` and `SerializableGenome` protocols.
- **FR-002**: System MUST support three dimensionality strategies for the genome: full-space evolution, compressed subspace evolution (with a projection matrix pre-computed once before the experiment and fixed throughout evolution), and minimal-token evolution. The default strategy when the user does not explicitly choose MUST be minimal tokens (4–8 tokens at full `embed_dim`).
- **FR-003**: System MUST provide a decoder that injects a genome's embeddings into a target model's embedding layer and returns the model's generated output text.
- **FR-004**: System MUST reject decode operations when the genome's model identifier does not match the loaded model.
- **FR-005**: System MUST provide a benchmark evaluator that computes deterministic fitness (accuracy, F1, exact-match, pass@k) by comparing model outputs to ground-truth answers.
- **FR-006**: System MUST provide an LLM-as-judge evaluator that scores model outputs against configurable rubric criteria and returns multi-dimensional fitness values.
- **FR-007**: System MUST provide token-aware Gaussian mutation that independently perturbs whole-token embedding vectors with configurable mutation rate and noise scale.
- **FR-008**: System MUST provide token-level crossover operators (single-point and two-point) that swap whole tokens between parents, preserving per-token coherence.
- **FR-009**: System MUST support existing framework flat-vector operators (GaussianMutation, BlendCrossover, SBX, etc.) via a flat accessor on the genome.
- **FR-010**: System MUST provide noise-based population initialization that embeds seed text, pads/truncates to the configured token count, and perturbs with calibrated Gaussian noise (noise scale calibrated relative to the target model's embedding norm distribution per research.md R4).
- **FR-011**: System MUST provide LLM-discovered variation axis initialization that generates diverse text variants along meaningful dimensions and embeds them.
- **FR-012**: System MUST provide a three-layer coherence defense: mutation magnitude constraint (L2 norm bound), cheap perplexity-based feasibility check (using the framework's constraint mechanism), and fitness-based selection. When all individuals in a generation are infeasible, the system MUST fall back to the previous generation's population and retry with reduced mutation magnitude.
- **FR-013**: Each coherence defense layer MUST be independently toggleable via configuration.
- **FR-014**: System MUST provide text-mediated cross-model transfer that generates a text interpretation from an evolved soft prompt for use as a hard prompt on a different model.
- **FR-015**: System MUST provide a task specification configuration that parameterizes evaluation by task type, input set, scoring rubric, and generalization requirements.
- **FR-016**: The genome module MUST contain no ML framework imports (torch, transformers, etc.) — it uses only numpy. ML imports are confined to the decoder and evaluator modules.
- **FR-017**: System MUST support using evolved text-mediated prompts as seeds for new ESPO runs on different models.
- **FR-018**: System MUST log comprehensive per-generation metrics via the framework's MLflow integration: best/mean/worst fitness, population diversity statistics (pairwise L2 distance), infeasibility rate, mutation magnitude, and the best individual's decoded text sample.

### Key Entities

- **EmbeddingGenome**: The core individual representation — a 2D continuous array of soft-prompt embeddings tied to a specific model, with optional subspace projection support.
- **SoftPromptDecoder**: Converts an embedding genome into model output by injecting embeddings into the target model's embedding layer and running generation.
- **TaskSpec**: Configuration that defines what is being optimized — task type, input dataset, scoring metrics or rubric, and generalization requirements.
- **GroundTruthEvaluator**: Computes deterministic fitness by comparing decoded outputs to ground-truth answers using standard metrics.
- **LLMJudgeEvaluator**: Computes multi-dimensional fitness by sending decoded outputs to a judge LLM that scores against a configurable rubric.
- **TokenAwareMutator**: Mutation operator that perturbs whole-token embeddings with Gaussian noise, respecting token boundaries.
- **TokenLevelCrossover**: Crossover operator that swaps whole tokens between parents at configurable crossover points.
- **PopulationInitializer**: Creates diverse initial populations from seed text using noise-based or LLM-discovered variation strategies.
- **CoherenceDefense**: Three-layer system (mutation constraint, perplexity check, fitness selection) that prevents evaluation of incoherent individuals.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An ESPO experiment on a standard QA benchmark produces a best individual whose accuracy exceeds the seed text baseline within 50 generations.
- **SC-002**: The embedding genome correctly implements framework genome protocols, passing all framework protocol compliance tests.
- **SC-003**: Token-aware operators produce offspring that maintain coherent model outputs at least 80% of the time (measured by perplexity below the coherence threshold).
- **SC-004**: Population initialization produces individuals that are measurably diverse (average pairwise L2 distance above a model-calibrated minimum) while all producing coherent model outputs.
- **SC-005**: The coherence defense reduces wasted evaluations (infeasible individuals reaching full evaluation) by at least 60% compared to no defense.
- **SC-006**: Deterministic benchmark evaluation is fully reproducible — running the same genome on the same inputs produces identical fitness scores across runs.
- **SC-007**: Multi-objective evaluation via LLM-as-judge returns per-criterion scores that integrate correctly with the framework's multi-objective selection.
- **SC-008**: Text-mediated transfer produces a usable hard prompt that achieves at least 70% of the original soft prompt's benchmark performance on the target model.
- **SC-009**: The genome module contains zero ML framework imports, verified by static analysis.
- **SC-010**: Benchmark evaluation of a single individual against the full input set completes within 60 seconds wall-clock time on a single GPU with a 7–8B parameter model.

## Assumptions

- Target users have access to open-weight LLMs with embedding-layer access (e.g., LLaMA, Mistral families) running locally or on accessible GPU infrastructure.
- The evolve-framework's existing `Genome`, `SerializableGenome`, `Evaluator`, and operator protocols are stable and will not change during this feature's development.
- `n_tokens` is fixed per experiment; variable-length soft prompts are out of scope for the initial implementation.
- CMA-ES or similar adaptive operator integration is available or will be developed as a separate framework feature if not yet present.
- LLM-as-judge evaluation uses an external judge model that may differ from the target model being optimized.
- GPU memory is sufficient to hold the target model plus a batch of soft-prompt evaluations (this constrains practical population sizes).
- The framework's MLflow tracking integration is used for experiment logging and reproducibility.
- Same-family transfer via learned projections (e.g., LLaMA-8B to LLaMA-70B) is out of scope for the initial implementation.
- Surrogate-assisted evaluation, discrete token genomes, structured JSON genomes, and variable-length soft prompts are explicitly excluded from scope per the design document.
