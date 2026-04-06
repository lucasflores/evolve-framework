# Contract: EmbeddingGenome

**Module**: `evolve.representation.embedding`

## Protocol Compliance

`EmbeddingGenome` implements two framework protocols:

### Genome Protocol

```python
@runtime_checkable
class Genome(Protocol):
    def copy(self) -> Self: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
```

### SerializableGenome Protocol

```python
@runtime_checkable
class SerializableGenome(Genome, Protocol):
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...
```

## Construction

```python
from evolve.representation.embedding import EmbeddingGenome
from evolve.representation.embedding_config import DimensionalityStrategy

# Direct construction
genome = EmbeddingGenome(
    embeddings=np.random.randn(8, 4096).astype(np.float32),
    model_id="meta-llama/Llama-2-7b-hf",
    seed_text="Answer the following question:",
    strategy=DimensionalityStrategy.MINIMAL_TOKENS,
)
```

**Preconditions**:
- `embeddings` is a 2D numpy array with shape `(n_tokens, embed_dim)` where both dimensions ≥ 1
- `model_id` is a non-empty string
- `strategy` is a valid `DimensionalityStrategy` enum value

**Postconditions**:
- `embeddings` array is immutable (`flags.writeable == False`)
- `n_tokens` and `embed_dim` properties reflect array shape

## Serialization Contract

```python
# Round-trip identity
d = genome.to_dict()
restored = EmbeddingGenome.from_dict(d)
assert genome == restored

# Dict structure
{
    "embeddings": [[0.1, 0.2, ...], ...],  # nested list of floats
    "model_id": "meta-llama/Llama-2-7b-hf",
    "seed_text": "Answer the following question:",
    "strategy": "MINIMAL_TOKENS",
    "n_tokens": 8,
    "embed_dim": 4096
}
```

## Flat-Vector Adapter Contract

```python
# To VectorGenome (for existing flat operators)
vg = genome.to_vector_genome()
assert isinstance(vg, VectorGenome)
assert len(vg.genes) == genome.n_tokens * genome.embed_dim

# From VectorGenome (reconstruct)
restored = EmbeddingGenome.from_vector_genome(
    vg, model_id=genome.model_id,
    seed_text=genome.seed_text,
    strategy=genome.strategy,
)
assert genome == restored
```

## Invariants

- `genome.embeddings.ndim == 2`
- `genome.embeddings.shape == (genome.n_tokens, genome.embed_dim)`
- `genome.embeddings.flags.writeable == False`
- `genome == genome.copy()`
- `hash(genome) == hash(genome.copy())`
- `genome == EmbeddingGenome.from_dict(genome.to_dict())`

## Import Boundary

**FR-016**: This module MUST NOT import `torch`, `transformers`, or any ML framework. Only imports allowed: `numpy`, `dataclasses`, `typing`, Python stdlib.
