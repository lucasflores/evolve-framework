# Contract: Operators

**Modules**: `evolve.core.operators.token_mutation`, `evolve.core.operators.token_crossover`

## TokenAwareMutator

### Interface

```python
@dataclass
class TokenAwareMutator:
    mutation_rate: float = 0.1
    sigma: float = 0.1
    coherence_radius: float | None = None

    def mutate(
        self,
        genome: EmbeddingGenome,
        rng: Random,
    ) -> EmbeddingGenome: ...
```

### Implements

`MutationOperator[EmbeddingGenome]` protocol.

### Mutate Contract

**Preconditions**:
- `genome` is a valid `EmbeddingGenome`
- `rng` is a seeded `Random` instance

**Process**:
1. For each token index `i` in `[0, n_tokens)`:
   - With probability `mutation_rate`, select token for mutation
   - Generate noise vector: `delta = np.array([rng.gauss(0, sigma) for _ in range(embed_dim)])`
   - If `coherence_radius is not None`:
     - Compute `norm = np.linalg.norm(delta)`
     - If `norm > coherence_radius`: `delta = delta * (coherence_radius / norm)`
   - `new_embeddings[i] = genome.embeddings[i] + delta`
2. Return new `EmbeddingGenome` with mutated embeddings

**Postconditions**:
- Returns a NEW `EmbeddingGenome` (genome is immutable)
- Same `model_id`, `seed_text`, `strategy` as input
- Same shape `(n_tokens, embed_dim)` as input
- Unmutated tokens are identical to input (bitwise)
- If `coherence_radius` set, `‖mutated[i] - original[i]‖₂ ≤ coherence_radius` for all mutated tokens

### Registry Entry

```python
OperatorRegistry.register(
    "mutation", "token_gaussian", TokenAwareMutator,
    compatible_genomes={"embedding"}
)
```

---

## TokenLevelCrossover

### Interface

```python
@dataclass
class TokenLevelCrossover:
    crossover_type: str = "single_point"  # "single_point" | "two_point"

    def crossover(
        self,
        parent1: EmbeddingGenome,
        parent2: EmbeddingGenome,
        rng: Random,
    ) -> tuple[EmbeddingGenome, EmbeddingGenome]: ...
```

### Implements

`CrossoverOperator[EmbeddingGenome]` protocol.

### Crossover Contract

**Preconditions**:
- `parent1.n_tokens == parent2.n_tokens`
- `parent1.embed_dim == parent2.embed_dim`
- `parent1.model_id == parent2.model_id`
- Raises `ValueError` if any precondition violated

**Process (single_point)**:
1. Choose crossover point `k` uniformly from `[1, n_tokens - 1]`
2. `child1 = parent1[:k] + parent2[k:]` (whole-token swap)
3. `child2 = parent2[:k] + parent1[k:]`

**Process (two_point)**:
1. Choose two points `k1, k2` with `k1 < k2`, both in `[1, n_tokens - 1]`
2. `child1 = parent1[:k1] + parent2[k1:k2] + parent1[k2:]`
3. `child2 = parent2[:k1] + parent1[k1:k2] + parent2[k2:]`

**Postconditions**:
- Returns two NEW `EmbeddingGenome` instances
- Each token in each child is a COMPLETE copy from one parent (no partial-token mixing)
- Same shape, `model_id`, `strategy` as parents
- `seed_text` is `None` for children (they are recombinations, not seeded)

### Registry Entries

```python
OperatorRegistry.register(
    "crossover", "token_single_point", TokenLevelCrossover,
    compatible_genomes={"embedding"}
)
OperatorRegistry.register(
    "crossover", "token_two_point",
    lambda: TokenLevelCrossover(crossover_type="two_point"),
    compatible_genomes={"embedding"}
)
```
