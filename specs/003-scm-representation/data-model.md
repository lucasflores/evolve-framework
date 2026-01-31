# Data Model: SCM Representation for Causal Discovery

**Feature**: 003-scm-representation  
**Date**: 2026-01-30  
**Phase**: 1 - Design

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCM Domain Model                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    composes    ┌─────────────────┐              │
│  │ SCMGenome │───────────────▶│ SequenceGenome  │              │
│  └─────┬─────┘                │   [str|float]   │              │
│        │                      └─────────────────┘              │
│        │ configured by                                          │
│        ▼                                                        │
│  ┌───────────┐    generates   ┌─────────────────┐              │
│  │ SCMConfig │───────────────▶│  SCMAlphabet    │              │
│  └───────────┘                └─────────────────┘              │
│                                                                 │
│  ┌───────────┐    decodes     ┌─────────────────┐              │
│  │SCMDecoder │───────────────▶│   DecodedSCM    │              │
│  └───────────┘                └────────┬────────┘              │
│                                        │                        │
│                               contains │                        │
│                                        ▼                        │
│                 ┌──────────────────────┴──────────────────┐    │
│                 │                      │                  │    │
│           ┌─────▼─────┐          ┌─────▼─────┐     ┌─────▼───┐│
│           │ equations │          │nx.DiGraph │     │ metadata ││
│           │dict[str,  │          │  (graph)  │     │   dict   ││
│           │Expression]│          └───────────┘     └──────────┘│
│           └───────────┘                                        │
│                                                                 │
│  ┌─────────────┐  evaluates   ┌─────────────────┐              │
│  │SCMEvaluator │─────────────▶│     Fitness     │              │
│  └─────────────┘              │ (multi-obj)     │              │
│                               └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Entities

### SCMConfig

Configuration for SCM genome creation and decoding.

```python
@dataclass(frozen=True)
class SCMConfig:
    """Configuration for SCM evolution."""
    
    # Variable configuration
    observed_variables: tuple[str, ...]  # e.g., ("A", "B", "C")
    max_latent_variables: int = 3        # Creates H1, H2, H3
    
    # Decoding behavior
    conflict_resolution: ConflictResolution = ConflictResolution.FIRST_WINS
    
    # Evaluation behavior
    acyclicity_mode: AcyclicityMode = AcyclicityMode.REJECT
    acyclicity_strategy: AcyclicityStrategy = AcyclicityStrategy.ACYCLIC_SUBGRAPH
    
    # Objectives and constraints
    objectives: tuple[str, ...] = ("data_fit", "sparsity", "simplicity")
    constraints: tuple[str, ...] = ("acyclicity",)
    
    # Penalty weights
    cycle_penalty_per_cycle: float = 1.0
    incomplete_coverage_penalty: float = 10.0
    conflict_penalty: float = 1.0
    div_zero_penalty: float = 5.0
    latent_ancestor_penalty: float = 10.0  # Penalty for latent variables without observed ancestors
    
    # ERC parameters
    erc_sigma_init: float = 1.0
    erc_sigma_perturb: float = 0.1
    erc_count: int = 5  # Number of ERC slots in alphabet
```

**Validation Rules:**
- `observed_variables` must be non-empty
- `max_latent_variables` >= 0
- All penalty values >= 0
- `erc_sigma_init` and `erc_sigma_perturb` > 0

---

### ConflictResolution (Enum)

```python
class ConflictResolution(Enum):
    """How to handle multiple STORE_X for same variable."""
    FIRST_WINS = "first_wins"   # Keep first equation, later are junk
    LAST_WINS = "last_wins"     # Keep last equation, earlier are junk
    ALL_JUNK = "all_junk"       # Discard all conflicting equations
```

---

### AcyclicityMode (Enum)

```python
class AcyclicityMode(Enum):
    """How to handle cyclic SCMs."""
    REJECT = "reject"       # Return None fitness
    PENALIZE = "penalize"   # Apply penalty, may partial-evaluate
```

---

### AcyclicityStrategy (Enum)

```python
class AcyclicityStrategy(Enum):
    """Partial evaluation strategy when mode=PENALIZE."""
    ACYCLIC_SUBGRAPH = "acyclic_subgraph"  # Evaluate maximal DAG
    PARSE_ORDER = "parse_order"            # Break cycles by parse order
    PENALTY_ONLY = "penalty_only"          # No partial eval, just penalty
    PARENT_INHERITANCE = "parent_inheritance"  # ERP-aware
    COMPOSITE = "composite"                # Subgraph + proportional penalty
```

---

### SCMAlphabet

Factory for generating genome alphabet from config.

```python
@dataclass(frozen=True)
class SCMAlphabet:
    """Symbol set for SCM genomes."""
    
    # All symbols as frozen set for SequenceGenome
    symbols: frozenset[str | float]
    
    # Categorized symbols for mutation/interpretation
    variable_refs: frozenset[str]     # A, B, C, H1, H2...
    store_genes: frozenset[str]       # STORE_A, STORE_B, STORE_H1...
    operators: frozenset[str]         # +, -, *, /
    constants: frozenset[float]       # 0, 1, 2, -1, 0.5, PI
    erc_slots: frozenset[str]         # ERC_0, ERC_1, ERC_2...
    
    @classmethod
    def from_config(cls, config: SCMConfig) -> "SCMAlphabet":
        """Generate alphabet from configuration."""
        ...
```

**Symbol Categories:**
| Category | Examples | Count |
|----------|----------|-------|
| Observed vars | A, B, C | len(observed_variables) |
| Latent vars | H1, H2, H3 | max_latent_variables |
| Store genes | STORE_A, STORE_H1 | num_vars * 2 |
| Operators | +, -, *, / | 4 |
| Constants | 0, 1, 2, -1, 0.5, π | 6 |
| ERC slots | ERC_0, ERC_1 | erc_count |

---

### SCMGenome

Wrapper around SequenceGenome with SCM semantics.

```python
@dataclass(frozen=True)
class SCMGenome:
    """
    SCM genome encoding potential causal model.
    
    Wraps SequenceGenome[str | float] with SCM-specific
    alphabet and ERC value tracking.
    """
    
    inner: SequenceGenome[str | float]
    config: SCMConfig
    erc_values: tuple[tuple[int, float], ...]  # (slot_index, value) pairs
    
    # Genome protocol
    def copy(self) -> "SCMGenome": ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    
    # SerializableGenome protocol
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCMGenome": ...
    
    # Convenience
    @property
    def genes(self) -> tuple[str | float, ...]: ...
    
    @classmethod
    def random(
        cls, 
        config: SCMConfig, 
        length: int,
        rng: Random
    ) -> "SCMGenome": ...
```

**Relationships:**
- Composes `SequenceGenome[str | float]`
- Configured by `SCMConfig`
- ERC values stored separately for targeted mutation

---

### Expression (AST)

Immutable expression tree for equations.

```python
@dataclass(frozen=True)
class Var:
    """Variable reference in expression."""
    name: str

@dataclass(frozen=True)  
class Const:
    """Numeric constant in expression."""
    value: float

@dataclass(frozen=True)
class BinOp:
    """Binary operation in expression."""
    op: str  # '+', '-', '*', '/'
    left: "Expression"
    right: "Expression"

# Type alias
Expression = Var | Const | BinOp
```

**Operations:**
- `complexity(expr) -> int`: Count AST nodes
- `variables(expr) -> set[str]`: Extract variable names
- `evaluate(expr, env: dict[str, float]) -> float`: Compute value
- `to_string(expr) -> str`: Human-readable representation

---

### DecodedSCM

Phenotype containing interpreted causal model.

```python
@dataclass(frozen=True)
class SCMMetadata:
    """Metadata from decoding process."""
    conflict_count: int
    junk_gene_indices: tuple[int, ...]
    is_cyclic: bool
    cycles: tuple[tuple[str, ...], ...]  # Each cycle as variable tuple
    latent_variables_used: frozenset[str]
    coverage: float  # Fraction of observed vars with equations

@dataclass(frozen=True)
class DecodedSCM:
    """
    Decoded Structural Causal Model.
    
    Contains equations, causal graph, and decoding metadata.
    """
    
    # Core SCM content
    equations: dict[str, Expression]  # {variable: expression}
    graph: nx.DiGraph                 # Causal graph (RHS vars → LHS var)
    
    # Decoding metadata
    metadata: SCMMetadata
    
    # Convenience properties
    @property
    def is_cyclic(self) -> bool:
        return self.metadata.is_cyclic
    
    @property
    def variables(self) -> frozenset[str]:
        """All variables (observed + latent used)."""
        ...
    
    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()
    
    @property
    def total_complexity(self) -> int:
        """Sum of AST complexity across equations."""
        ...
```

**Graph Structure:**
- Nodes: Variables with equations (endogenous) and referenced variables (may be exogenous)
- Edges: RHS variable → LHS variable (causal direction)
- Example: Equation `C = A + B` creates edges `A → C` and `B → C`

---

### SCMDecoder

Stateless decoder transforming genome to phenotype.

```python
class SCMDecoder:
    """
    Stack machine decoder for SCM genomes.
    
    Implements Decoder[SCMGenome, DecodedSCM] protocol.
    """
    
    def __init__(self, config: SCMConfig) -> None:
        self.config = config
    
    def decode(self, genome: SCMGenome) -> DecodedSCM:
        """
        Decode genome to SCM via stack machine execution.
        
        Deterministic: same genome always produces same SCM.
        """
        ...
    
    def _execute_stack_machine(
        self, 
        genes: Sequence[str | float],
        erc_values: dict[int, float]
    ) -> tuple[dict[str, Expression], list[int]]:
        """Execute genes, return equations and junk indices."""
        ...
    
    def _build_graph(
        self, 
        equations: dict[str, Expression]
    ) -> nx.DiGraph:
        """Build causal graph from equation dependencies."""
        ...
    
    def _detect_cycles(
        self, 
        graph: nx.DiGraph
    ) -> tuple[bool, list[tuple[str, ...]]]:
        """Check for cycles and enumerate them."""
        ...
```

**Stack Machine Rules:**
1. Variable/Constant → Push to stack
2. Operator → Pop operands, push result expression
3. STORE_X → Pop expression, create equation X = expr
4. Underflow → Mark gene as junk, continue
5. Empty stack on STORE → Mark as junk, continue

---

### SCMFitnessConfig

Configuration for fitness evaluation.

```python
@dataclass(frozen=True)
class SCMFitnessConfig:
    """Configuration for SCM fitness evaluation."""
    
    # Which objectives to compute
    objectives: tuple[str, ...] = ("data_fit", "sparsity", "simplicity")
    
    # Penalty configuration
    cycle_penalty_per_cycle: float = 1.0
    incomplete_coverage_penalty: float = 10.0
    conflict_penalty: float = 1.0
    div_zero_penalty: float = 5.0
    
    # Acyclicity handling
    acyclicity_mode: AcyclicityMode = AcyclicityMode.REJECT
    acyclicity_strategy: AcyclicityStrategy = AcyclicityStrategy.ACYCLIC_SUBGRAPH
```

---

### SCMEvaluator

Multi-objective fitness evaluator.

```python
class SCMEvaluator:
    """
    Multi-objective fitness evaluator for SCMs.
    
    Implements Evaluator[SCMGenome] protocol.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        variable_names: Sequence[str],
        config: SCMFitnessConfig,
        decoder: SCMDecoder | None = None,
    ) -> None:
        self.data = data
        self.variable_names = variable_names
        self.config = config
        self.decoder = decoder or SCMDecoder(...)
    
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(
            batchable=True,
            stochastic=False,
            n_objectives=len(self.config.objectives),
            n_constraints=len(self.config.constraints),
        )
    
    def evaluate(
        self,
        individuals: Sequence[Individual[SCMGenome]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """Evaluate batch of SCM individuals."""
        ...
    
    def _compute_data_fit(
        self, 
        scm: DecodedSCM
    ) -> float:
        """Compute negative MSE on observed endogenous variables."""
        ...
    
    def _compute_sparsity(
        self, 
        scm: DecodedSCM
    ) -> float:
        """Return negative edge count."""
        return -scm.edge_count
    
    def _compute_simplicity(
        self, 
        scm: DecodedSCM
    ) -> float:
        """Return negative total AST complexity."""
        return -scm.total_complexity
```

---

## State Transitions

### Genome Lifecycle

```
┌──────────────┐   random()    ┌──────────────┐
│   SCMConfig  │──────────────▶│  SCMGenome   │
└──────────────┘               └──────┬───────┘
                                      │
                               mutation/crossover
                                      │
                                      ▼
                               ┌──────────────┐
                               │  SCMGenome'  │
                               └──────┬───────┘
                                      │
                                 decode()
                                      │
                                      ▼
                               ┌──────────────┐
                               │  DecodedSCM  │
                               └──────┬───────┘
                                      │
                                evaluate()
                                      │
                                      ▼
                               ┌──────────────┐
                               │   Fitness    │
                               └──────────────┘
```

### Decoding State Machine

```
┌─────────────────────────────────────────────────────────────┐
│                    Stack Machine Execution                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  for gene in genome.genes:                                  │
│      ┌─────────────┐                                        │
│      │ Is Variable │──yes──▶ Push Var(name) to stack       │
│      │ or Constant?│                                        │
│      └──────┬──────┘                                        │
│             │ no                                            │
│             ▼                                               │
│      ┌─────────────┐                                        │
│      │ Is Operator?│──yes──▶ Pop operands, push BinOp      │
│      │             │        (underflow → mark junk)         │
│      └──────┬──────┘                                        │
│             │ no                                            │
│             ▼                                               │
│      ┌─────────────┐                                        │
│      │ Is STORE_X? │──yes──▶ Pop expr, emit equation       │
│      │             │        (empty stack → mark junk)       │
│      └──────┬──────┘                                        │
│             │ no                                            │
│             ▼                                               │
│      ┌─────────────┐                                        │
│      │ Is ERC_n?   │──yes──▶ Push Const(erc_values[n])     │
│      └─────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Validation Rules

### SCMConfig Validation

| Field | Rule |
|-------|------|
| observed_variables | len > 0, all strings non-empty |
| max_latent_variables | >= 0 |
| penalty values | all >= 0 |
| erc_sigma_* | > 0 |
| objectives | subset of valid objectives |
| constraints | subset of valid constraints |

### DecodedSCM Validation

| Property | Rule |
|----------|------|
| equations.keys() | subset of all variables |
| graph nodes | superset of equation LHS variables |
| graph edges | match equation dependencies |
| metadata.junk_gene_indices | valid indices into genome |

### Latent Variable Constraint

Latent variables must have at least one observed ancestor:
```python
for latent in metadata.latent_variables_used:
    ancestors = nx.ancestors(graph, latent)
    observed_ancestors = ancestors & set(config.observed_variables)
    if not observed_ancestors:
        # Constraint violation: latent without observed ancestor
```

---

## Serialization Format

### SCMGenome.to_dict()

```json
{
  "type": "SCMGenome",
  "version": "1.0",
  "genes": ["A", "B", "+", "STORE_C", "ERC_0", "STORE_A"],
  "erc_values": [[0, 0.5], [1, -1.2]],
  "config": {
    "observed_variables": ["A", "B", "C"],
    "max_latent_variables": 3,
    "conflict_resolution": "first_wins",
    ...
  }
}
```

### DecodedSCM (for debugging/logging, not required)

```json
{
  "equations": {
    "C": {"type": "BinOp", "op": "+", "left": {"type": "Var", "name": "A"}, "right": {"type": "Var", "name": "B"}}
  },
  "graph": {
    "nodes": ["A", "B", "C"],
    "edges": [["A", "C"], ["B", "C"]]
  },
  "metadata": {
    "conflict_count": 0,
    "junk_gene_indices": [4, 5],
    "is_cyclic": false,
    "cycles": []
  }
}
```
