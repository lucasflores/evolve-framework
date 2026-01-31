# Research: SCM Representation for Causal Discovery

**Feature**: 003-scm-representation  
**Date**: 2026-01-30  
**Phase**: 0 - Research

## Research Questions

1. Stack-based GP encoding patterns for SCM representation
2. NetworkX APIs for cycle detection and ancestor validation
3. Expression AST design for immutable equation representation
4. Heterogeneous gene handling in SequenceGenome

---

## 1. Stack-Based Genetic Programming Encoding

### Decision
Use linear postfix (RPN) encoding with typed genes: operands (variables, constants), operators (+, -, *, /), and STORE_X assignment genes.

### Rationale
- **Trivial decoding**: Simple stack machine, O(n) complexity
- **Natural junk emergence**: Unused stack values, failed operations become evolutionary neutral
- **Crossover-friendly**: One/two-point crossover on linear sequence preserves some locality
- **SequenceGenome compatible**: Directly wraps existing infrastructure

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|------------------|
| Tree GP | Complex crossover (subtree swap), hard to control bloat |
| Cartesian GP | Grid structure constraints, less flexible |
| Gene Expression Programming | Head/tail structure adds complexity, less intuitive |
| Direct graph encoding | Crossover destroys structure, hard to evolve |

### Implementation Pattern

```python
# Gene sequence example: A B + STORE_C
# Execution:
#   A     → stack: [A]
#   B     → stack: [A, B]  
#   +     → pop B, A; push (A + B); stack: [(A + B)]
#   STORE_C → pop (A + B); emit equation C = A + B; stack: []
```

### Junk Gene Scenarios

1. **Stack underflow**: Operator with insufficient operands → gene marked junk
2. **Empty STORE**: STORE_X with empty stack → gene marked junk
3. **Orphan values**: Values left on stack at end → implicitly junk
4. **Overwritten equations**: Conflict resolution may mark equations as junk

---

## 2. NetworkX Cycle Detection APIs

### Decision
Use `nx.is_directed_acyclic_graph(G)` for DAG check, `nx.simple_cycles(G)` for cycle enumeration.

### API Reference

```python
import networkx as nx

# DAG check - O(V + E)
is_dag = nx.is_directed_acyclic_graph(G)

# Enumerate all simple cycles - Johnson's algorithm
cycles = list(nx.simple_cycles(G))  # List of node lists

# Find ancestors of a node - BFS/DFS
ancestors = nx.ancestors(G, node)  # Set of ancestor nodes

# Topological generations (for evaluation order)
generations = list(nx.topological_generations(G))
```

### Rationale
- NetworkX is already a project dependency
- Well-tested, standard implementations
- Consistent with graph module patterns

### Cycle Handling Strategies

| Strategy | Implementation |
|----------|----------------|
| `reject` | Return `is_dag=False`, evaluator returns None fitness |
| `penalty_only` | Count cycles via `simple_cycles`, multiply by penalty |
| `acyclic_subgraph` | Find maximal DAG, evaluate only those nodes |
| `parse_order` | Remove later edges that create cycles during construction |

---

## 3. Expression AST Design

### Decision
Immutable dataclasses with `@dataclass(frozen=True)` forming a simple ADT: `Var`, `Const`, `BinOp`.

### Type Hierarchy

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Var:
    """Variable reference."""
    name: str

@dataclass(frozen=True)
class Const:
    """Numeric constant."""
    value: float

@dataclass(frozen=True)
class BinOp:
    """Binary operation."""
    op: str  # '+', '-', '*', '/'
    left: 'Expression'
    right: 'Expression'

Expression = Union[Var, Const, BinOp]
```

### Rationale
- **Immutable**: `frozen=True` enables hashing, prevents accidental mutation
- **Pattern matching**: Python 3.10+ structural pattern matching compatible
- **Simple**: No need for SymPy complexity; we only need structure, not algebra
- **Serializable**: Trivial recursive to_dict/from_dict

### AST Complexity Metric

```python
def complexity(expr: Expression) -> int:
    """Count total AST nodes for simplicity objective."""
    match expr:
        case Var(_) | Const(_):
            return 1
        case BinOp(_, left, right):
            return 1 + complexity(left) + complexity(right)
```

---

## 4. Heterogeneous Gene Handling

### Decision
Use `SequenceGenome[str | float]` with ERC values stored in a separate dict keyed by gene index.

### Design

```python
@dataclass(frozen=True)
class SCMGenome:
    """SCM genome wrapping SequenceGenome."""
    inner: SequenceGenome[str | float]
    config: SCMConfig
    erc_values: tuple[tuple[int, float], ...]  # (index, value) pairs
    
    @property
    def genes(self) -> tuple[str | float, ...]:
        return self.inner.genes
```

### Rationale
- **Union type**: Python 3.10+ supports `str | float` natively
- **ERC tracking**: Separate tuple enables ERC-specific mutation (perturbation)
- **Immutable**: Tuple of tuples for hashability

### ERC Mutation

```python
def mutate_erc(genome: SCMGenome, index: int, sigma: float, rng: Random) -> SCMGenome:
    """Perturb a single ERC value with Gaussian noise."""
    old_value = dict(genome.erc_values).get(index, 0.0)
    new_value = old_value + rng.gauss(0, sigma)
    # Rebuild genome with updated ERC
    ...
```

---

## 5. Default Parameter Values

### Decision
Use sensible defaults from causal discovery literature and GP practice.

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_latent_variables` | 3 | Conservative; most causal models have few latents |
| `conflict_resolution` | "first_wins" | Deterministic, simple |
| `acyclicity_mode` | "reject" | Strict DAG enforcement by default |
| `cycle_penalty_per_cycle` | 1.0 | One fitness unit per cycle |
| `incomplete_coverage_penalty` | 10.0 | High penalty for missing equations |
| `conflict_penalty` | 1.0 | Moderate penalty for conflicts |
| `div_zero_penalty` | 5.0 | Higher than conflict (evaluation failure) |
| `erc_sigma_init` | 1.0 | Standard normal for initial sampling |
| `erc_sigma_perturb` | 0.1 | Small perturbation for fine-tuning |

---

## 6. Integration with Existing Framework

### Genome Protocol Compliance

SCMGenome must implement:
- `copy() -> SCMGenome`: Deep copy via dataclass replace
- `__eq__`: Structural equality on inner genes and ERC values
- `__hash__`: Hash of (genes tuple, ERC values tuple)

### SerializableGenome Protocol Compliance

- `to_dict()`: Returns `{"genes": [...], "erc_values": [...], "config": {...}}`
- `from_dict()`: Reconstructs SCMGenome from dict

### Decoder Protocol Compliance

SCMDecoder implements `Decoder[SCMGenome, DecodedSCM]`:
- Single `decode(genome: SCMGenome) -> DecodedSCM` method
- Stateless; all config via SCMConfig in genome

### Evaluator Protocol Compliance

SCMEvaluator implements `Evaluator[SCMGenome]`:
- `capabilities`: Returns multi-objective config (n_objectives=3)
- `evaluate()`: Batch evaluation with decoder + fitness computation

---

## Summary

All technical unknowns resolved. Ready for Phase 1 design.

| Topic | Resolution |
|-------|------------|
| Encoding | Linear postfix with typed genes |
| Cycle detection | NetworkX standard APIs |
| Expression AST | Frozen dataclasses (Var, Const, BinOp) |
| Heterogeneous genes | SequenceGenome[str \| float] + ERC tuple |
| Default parameters | Documented with rationale |
| Protocol compliance | All three protocols (Genome, Serializable, Decoder, Evaluator) mapped |
