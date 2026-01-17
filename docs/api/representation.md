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
