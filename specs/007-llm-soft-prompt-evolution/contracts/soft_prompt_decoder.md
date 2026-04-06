# Contract: SoftPromptDecoder

**Module**: `evolve.meta.soft_prompt.decoder`

## Interface

```python
class SoftPromptDecoder:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        max_generation_tokens: int = 256,
    ) -> None: ...

    def decode(
        self,
        genome: EmbeddingGenome,
        task_input: str,
        max_tokens: int | None = None,
    ) -> str: ...

    @property
    def embed_dim(self) -> int: ...

    def embed_text(self, text: str) -> np.ndarray: ...
```

## Decode Contract

**Preconditions**:
- `genome.model_id == self.model_id` (raises `ValueError` on mismatch — FR-004)
- `genome.embeddings.shape[1] == self.embed_dim` (raises `ValueError` on dimension mismatch)
- `task_input` is a non-empty string

**Process**:
1. Lazy-load model and tokenizer if not already loaded
2. Tokenize `task_input` → `input_ids`
3. Get input embeddings: `model.get_input_embeddings()(input_ids)`
4. Prepend `genome.embeddings` to input embeddings along sequence dimension
5. Run `model.generate()` with combined embeddings
6. Decode output tokens to text (skip input tokens)
7. Return generated text string

**Postconditions**:
- Returns a non-empty string (model output)
- Model state is unchanged (no gradients, no parameter updates)

## Embed Text Contract

```python
# Embed seed text for population initialization
embeddings = decoder.embed_text("Answer the following question:")
# shape: (n_actual_tokens, embed_dim) — actual token count from tokenizer
```

**Postconditions**:
- Returns numpy array of shape `(n_tokens, embed_dim)` where `n_tokens` is the actual tokenized length
- Array is detached from computation graph (pure numpy)

## Error Conditions

| Condition | Error |
|-----------|-------|
| `genome.model_id != self.model_id` | `ValueError("Model mismatch: genome targets '{genome.model_id}' but decoder loaded '{self.model_id}'")` |
| `genome.embed_dim != self.embed_dim` | `ValueError("Embedding dimension mismatch: genome has {genome.embed_dim}, model has {self.embed_dim}")` |
| torch/transformers not installed | `ImportError("SoftPromptDecoder requires torch and transformers. Install with: pip install evolve-framework[pytorch]")` |
| Model cannot be loaded | `RuntimeError("Failed to load model '{model_id}': {original_error}")` |

## Dependencies

This module requires `torch` and `transformers` as optional dependencies. Import is deferred to first use.
