# Representation Module API

The `evolve.representation` module provides genome types for different problem domains.

## Vector Genome

For continuous optimization problems.

```python
from evolve.representation import VectorGenome
import numpy as np

# Create with specific genes
genome = VectorGenome(
    genes=np.array([0.5, -0.3, 1.2]),
    bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
)

# Random initialization
genome = VectorGenome.random(
    n_dims=10,
    bounds=(-5.12, 5.12),
    rng=rng
)

# Operations
copied = genome.copy()
clipped = genome.clip_to_bounds()
```

**Attributes:**
- `genes: np.ndarray` - The gene values
- `bounds: tuple[np.ndarray, np.ndarray] | None` - (lower, upper) bounds

**Methods:**
- `copy()` - Deep copy
- `clip_to_bounds()` - Constrain to bounds
- `random(n_dims, bounds, rng)` - Factory for random genome

---

## Tree Genome

For genetic programming / symbolic regression.

```python
from evolve.representation import TreeGenome, TreeNode

# Define function and terminal sets
functions = {
    '+': lambda a, b: a + b,
    '*': lambda a, b: a * b,
    'sin': lambda a: np.sin(a),
}
terminals = ['x', 1.0, 2.0]

# Create tree: (x + 1) * sin(x)
tree = TreeGenome(
    root=TreeNode(
        op='*',
        children=[
            TreeNode('+', [TreeNode('x'), TreeNode(1.0)]),
            TreeNode('sin', [TreeNode('x')])
        ]
    )
)

# Evaluate
result = tree.evaluate({'x': 0.5})

# Random generation
tree = TreeGenome.random(
    functions=functions,
    terminals=terminals,
    max_depth=5,
    rng=rng
)
```

**Attributes:**
- `root: TreeNode` - Root node of expression tree
- `depth: int` - Maximum depth
- `size: int` - Total nodes

**Methods:**
- `evaluate(variables)` - Compute tree expression
- `copy()` - Deep copy
- `random(functions, terminals, max_depth, rng)` - Factory

---

## Graph Genome

For network/topology evolution (NEAT-style).

```python
from evolve.representation import GraphGenome, NodeGene, ConnectionGene

genome = GraphGenome(
    nodes=[
        NodeGene(id=0, type='input'),
        NodeGene(id=1, type='input'),
        NodeGene(id=2, type='hidden', bias=0.1),
        NodeGene(id=3, type='output', bias=-0.2),
    ],
    connections=[
        ConnectionGene(in_node=0, out_node=2, weight=0.5, enabled=True, innovation=1),
        ConnectionGene(in_node=1, out_node=2, weight=-0.3, enabled=True, innovation=2),
        ConnectionGene(in_node=2, out_node=3, weight=1.0, enabled=True, innovation=3),
    ]
)

# Feed-forward evaluation
output = genome.activate(inputs=[1.0, 0.5])
```

**Attributes:**
- `nodes: list[NodeGene]` - Node genes
- `connections: list[ConnectionGene]` - Connection genes

**Methods:**
- `activate(inputs)` - Feed-forward evaluation
- `add_node(connection, innovation)` - NEAT add-node mutation
- `add_connection(in_id, out_id, innovation)` - NEAT add-connection mutation
- `copy()` - Deep copy

---

## Permutation Genome

For combinatorial optimization (TSP, scheduling).

```python
from evolve.representation import PermutationGenome

# Create permutation of 0..n-1
genome = PermutationGenome(
    permutation=np.array([3, 1, 4, 0, 2])
)

# Random permutation
genome = PermutationGenome.random(n=10, rng=rng)

# Operations
inverted = genome.invert(i=1, j=4)  # 2-opt move
```

**Attributes:**
- `permutation: np.ndarray` - Integer permutation

**Methods:**
- `copy()` - Deep copy
- `invert(i, j)` - Reverse segment [i, j]
- `swap(i, j)` - Swap positions i and j
- `random(n, rng)` - Factory for random permutation

---

## Genome Protocol

All genomes implement this protocol:

```python
from typing import Protocol

class Genome(Protocol):
    """Protocol for genome types."""
    
    def copy(self) -> Self:
        """Create a deep copy."""
        ...
```

Custom genomes only need to implement `copy()` to be compatible with the framework.

---

## SCM Genome (Structural Causal Model)

For causal discovery - evolving structural causal models from observational data.

```python
from random import Random
from evolve.representation.scm import SCMConfig, SCMGenome, SCMAlphabet
from evolve.representation.scm_decoder import SCMDecoder, to_string

# Configure for your observed variables
config = SCMConfig(
    observed_variables=("A", "B", "C"),
    max_latent_variables=2,  # Allow hidden confounders
)

# Create random genome
rng = Random(42)
genome = SCMGenome.random(config, length=50, rng=rng)

# Decode to causal model
decoder = SCMDecoder(config)
scm = decoder.decode(genome)

# Inspect discovered structure
print(f"Equations: {len(scm.equations)}")
print(f"Graph edges: {scm.edge_count}")
print(f"Is acyclic: {not scm.is_cyclic}")

for var, expr in scm.equations.items():
    print(f"  {var} = {to_string(expr)}")
```

**SCMConfig Attributes:**
- `observed_variables: tuple[str, ...]` - Variable names from data
- `max_latent_variables: int` - Maximum hidden confounders (H1, H2, ...)
- `conflict_resolution: ConflictResolution` - How to handle duplicate equations
- `acyclicity_mode: AcyclicityMode` - How to handle cyclic graphs

**SCMGenome Attributes:**
- `inner: SequenceGenome` - Underlying gene sequence
- `config: SCMConfig` - Configuration used
- `erc_values: tuple[tuple[int, float], ...]` - Ephemeral random constants

**SCMGenome Methods:**
- `copy()` - Deep copy
- `random(config, length, rng)` - Factory for random genome
- `mutate_erc(rng, slot)` - Perturb ERC values
- `to_dict() / from_dict(data)` - Serialization

**DecodedSCM Attributes:**
- `equations: dict[str, Expression]` - Variable -> equation mapping
- `graph: nx.DiGraph` - Causal graph (NetworkX directed graph)
- `metadata: SCMMetadata` - Decoding info (conflicts, junk genes, cycles)
- `edge_count: int` - Number of causal edges
- `is_cyclic: bool` - Whether graph contains cycles

### Fitness Evaluation

```python
from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
import numpy as np

# Your observational data
data = np.column_stack([A, B, C])  # shape: (n_samples, n_vars)

# Configure fitness objectives
fitness_config = SCMFitnessConfig(
    objectives=("data_fit", "sparsity", "simplicity"),
)

evaluator = SCMEvaluator(
    data=data,
    variable_names=["A", "B", "C"],
    config=fitness_config,
    decoder=decoder,
)

# Evaluate individuals
fitness_results = evaluator.evaluate(individuals)
```

**Available Objectives:**
- `"data_fit"` - Negative MSE (higher = better fit)
- `"sparsity"` - Negative edge count (higher = simpler)
- `"simplicity"` - Negative AST complexity (higher = simpler expressions)
- `"coverage"` - Fraction of observed variables with equations
- `"latent_parsimony"` - Negative latent variable count

### Distance Functions for ERP Integration

```python
from evolve.representation.scm import scm_distance

# Compute distance between two SCM genomes
# Combines sequence similarity and structural similarity
distance = scm_distance(
    genome_a, genome_b, decoder,
    structural_weight=0.5  # 0=sequence only, 1=structure only
)
```
