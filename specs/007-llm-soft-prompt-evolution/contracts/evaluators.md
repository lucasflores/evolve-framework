# Contract: Evaluators

**Modules**: `evolve.evaluation.benchmark`, `evolve.evaluation.llm_judge`

## GroundTruthEvaluator

### Interface

```python
class GroundTruthEvaluator:
    def __init__(
        self,
        decoder: SoftPromptDecoder,
        task_spec: TaskSpec,
    ) -> None: ...

    @property
    def capabilities(self) -> EvaluatorCapabilities: ...

    def evaluate(
        self,
        individuals: Sequence[Individual[EmbeddingGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]: ...
```

### Capabilities

```python
EvaluatorCapabilities(
    batchable=True,
    stochastic=False,   # Deterministic per FR-005, SC-006
    stateful=False,
    n_objectives=len(task_spec.metrics),  # e.g., 1 for accuracy-only
    n_constraints=0,     # Coherence defense adds constraints separately
    supports_diagnostics=True,
    supports_gpu=True,
    supports_jit=False,
)
```

### Evaluate Contract

**Preconditions**:
- `task_spec.ground_truth is not None` (benchmark requires ground truth)
- All individuals have genomes with matching `model_id`

**Process**:
1. For each individual, decode genome with each task input
2. Compare outputs to ground truth using configured metrics
3. Aggregate per-input scores into fitness

**Postconditions**:
- Returns one `Fitness` per individual, in same order as input
- `Fitness.values` shape: `(len(task_spec.metrics),)`
- Deterministic: same inputs + same seed → same outputs (SC-006)
- Wall-clock per individual ≤ 60s on single GPU with 7–8B model (SC-010)

### Supported Metrics

| Metric | Description | Return Range |
|--------|-------------|-------------|
| `"accuracy"` | Exact string match rate | [0.0, 1.0] |
| `"f1"` | Token-level F1 score | [0.0, 1.0] |
| `"exact_match"` | Binary exact match | {0.0, 1.0} |
| `"pass_at_k"` | Functional correctness (k samples) | [0.0, 1.0] |

---

## LLMJudgeEvaluator

### Interface

```python
class LLMJudgeEvaluator:
    def __init__(
        self,
        decoder: SoftPromptDecoder,
        task_spec: TaskSpec,
        judge_model_id: str,
        temperature: float = 0.0,
    ) -> None: ...

    @property
    def capabilities(self) -> EvaluatorCapabilities: ...

    def evaluate(
        self,
        individuals: Sequence[Individual[EmbeddingGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]: ...
```

### Capabilities

```python
EvaluatorCapabilities(
    batchable=True,
    stochastic=False,   # temperature=0.0 for determinism
    stateful=False,
    n_objectives=len(task_spec.rubric),  # One objective per criterion
    n_constraints=0,
    supports_diagnostics=True,
    supports_gpu=True,
    supports_jit=False,
)
```

### Evaluate Contract

**Preconditions**:
- `task_spec.rubric is not None` and `len(task_spec.rubric) >= 1`
- Judge model is accessible

**Process**:
1. For each individual, decode genome with each task input
2. Format judge prompt: system instructions + rubric criteria + model output
3. Send to judge model with `temperature=0.0`
4. Parse per-criterion scores from judge response
5. Average across task inputs → per-criterion fitness

**Postconditions**:
- Returns one `Fitness` per individual
- `Fitness.values` shape: `(len(task_spec.rubric),)` — one value per rubric criterion
- Multi-objective fitness integrates correctly with NSGA-II selection (SC-007)

### Judge Prompt Format

```text
You are evaluating the quality of an AI response. Score on each criterion.

Criteria:
1. {criterion.name}: {criterion.description} (scale: {scale_min}–{scale_max})
2. ...

Response to evaluate:
{model_output}

Provide scores as JSON: {"criterion_name": score, ...}
```

### Error Handling

| Condition | Behavior |
|-----------|----------|
| Judge returns malformed JSON | Retry once; if still malformed, assign minimum scores for all criteria |
| Judge is unreachable | Raise `EvaluationError` with descriptive message |
| Score out of range | Clamp to `[scale_min, scale_max]` |
