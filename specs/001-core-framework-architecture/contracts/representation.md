# Representation Interfaces Contract

**Module**: `evolve.representation`  
**Purpose**: Define genome, phenotype, and decoder abstractions

---

## Genome Protocol

```python
from typing import Protocol, TypeVar, Self, Any
from dataclasses import dataclass
import numpy as np

class Genome(Protocol):
    """
    Framework-neutral genetic representation.
    
    Genomes MUST:
    - Be immutable (return new instances on modification)
    - Be serializable (pickle or custom)
    - Support equality and hashing
    - NOT contain PyTorch/JAX types
    """
    
    def copy(self) -> Self:
        """Create deep copy of genome."""
        ...
    
    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        ...
    
    def __hash__(self) -> int:
        """Hash for set/dict membership."""
        ...


class SerializableGenome(Protocol):
    """Genome with portable serialization."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        ...
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct from dict."""
        ...
```

---

## Vector Genome

```python
@dataclass(frozen=True)
class VectorGenome:
    """
    Fixed-length real-valued vector genome.
    
    Common for continuous optimization, neuroevolution weights.
    """
    genes: np.ndarray  # Shape: (n_genes,)
    bounds: tuple[np.ndarray, np.ndarray] | None = None  # (lower, upper)
    
    def __post_init__(self):
        # Ensure immutable
        self.genes.flags.writeable = False
        if self.bounds is not None:
            self.bounds[0].flags.writeable = False
            self.bounds[1].flags.writeable = False
    
    def copy(self) -> 'VectorGenome':
        return VectorGenome(
            genes=self.genes.copy(),
            bounds=self.bounds
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorGenome):
            return False
        return np.array_equal(self.genes, other.genes)
    
    def __hash__(self) -> int:
        return hash(self.genes.tobytes())
    
    def __len__(self) -> int:
        return len(self.genes)
    
    def clip_to_bounds(self) -> 'VectorGenome':
        """Return genome with genes clipped to bounds."""
        if self.bounds is None:
            return self
        clipped = np.clip(self.genes, self.bounds[0], self.bounds[1])
        return VectorGenome(genes=clipped, bounds=self.bounds)
    
    def to_dict(self) -> dict:
        return {
            'genes': self.genes.tolist(),
            'bounds': [b.tolist() for b in self.bounds] if self.bounds else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VectorGenome':
        bounds = None
        if data['bounds']:
            bounds = (np.array(data['bounds'][0]), np.array(data['bounds'][1]))
        return cls(genes=np.array(data['genes']), bounds=bounds)
    
    @classmethod
    def random(
        cls,
        n_genes: int,
        bounds: tuple[np.ndarray, np.ndarray],
        rng: 'Random'
    ) -> 'VectorGenome':
        """Create random genome within bounds."""
        genes = rng.uniform(bounds[0], bounds[1])
        return cls(genes=genes, bounds=bounds)
```

---

## Sequence Genome

```python
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass(frozen=True)
class SequenceGenome(Generic[T]):
    """
    Variable-length sequence genome.
    
    Useful for genetic programming, variable-length encodings.
    """
    genes: tuple[T, ...]  # Immutable sequence
    alphabet: frozenset[T] | None = None  # Valid gene values
    min_length: int = 1
    max_length: int | None = None
    
    def copy(self) -> 'SequenceGenome[T]':
        return SequenceGenome(
            genes=self.genes,
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGenome):
            return False
        return self.genes == other.genes
    
    def __hash__(self) -> int:
        return hash(self.genes)
    
    def __len__(self) -> int:
        return len(self.genes)
    
    def __getitem__(self, idx: int) -> T:
        return self.genes[idx]
    
    def with_gene(self, idx: int, value: T) -> 'SequenceGenome[T]':
        """Return copy with gene at idx replaced."""
        new_genes = list(self.genes)
        new_genes[idx] = value
        return SequenceGenome(
            genes=tuple(new_genes),
            alphabet=self.alphabet,
            min_length=self.min_length,
            max_length=self.max_length
        )
    
    def to_dict(self) -> dict:
        return {
            'genes': list(self.genes),
            'alphabet': list(self.alphabet) if self.alphabet else None,
            'min_length': self.min_length,
            'max_length': self.max_length
        }
```

---

## Graph Genome (NEAT-style)

```python
@dataclass(frozen=True)
class NodeGene:
    """Node in a graph genome."""
    id: int
    node_type: str  # "input" | "output" | "hidden"
    activation: str = "sigmoid"  # Activation function name
    bias: float = 0.0

@dataclass(frozen=True)
class ConnectionGene:
    """Connection in a graph genome."""
    innovation: int  # Global innovation number
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True

@dataclass(frozen=True)
class GraphGenome:
    """
    NEAT-style graph genome for evolving topologies.
    
    Nodes and connections identified by innovation numbers
    for alignment during crossover.
    """
    nodes: frozenset[NodeGene]
    connections: frozenset[ConnectionGene]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    
    def copy(self) -> 'GraphGenome':
        return GraphGenome(
            nodes=self.nodes,
            connections=self.connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphGenome):
            return False
        return (self.nodes == other.nodes and 
                self.connections == other.connections)
    
    def __hash__(self) -> int:
        return hash((self.nodes, self.connections))
    
    def get_node(self, node_id: int) -> NodeGene | None:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_connection(self, innovation: int) -> ConnectionGene | None:
        """Get connection by innovation number."""
        for conn in self.connections:
            if conn.innovation == innovation:
                return conn
        return None
    
    def enabled_connections(self) -> frozenset[ConnectionGene]:
        """Get only enabled connections."""
        return frozenset(c for c in self.connections if c.enabled)
    
    def add_node(
        self,
        node: NodeGene,
        split_connection: ConnectionGene,
        new_conn1: ConnectionGene,
        new_conn2: ConnectionGene
    ) -> 'GraphGenome':
        """Add node by splitting a connection (NEAT add-node mutation)."""
        new_nodes = self.nodes | {node}
        new_connections = (
            self.connections - {split_connection} | 
            {split_connection._replace(enabled=False), new_conn1, new_conn2}
        )
        return GraphGenome(
            nodes=new_nodes,
            connections=new_connections,
            input_ids=self.input_ids,
            output_ids=self.output_ids
        )
    
    def add_connection(self, connection: ConnectionGene) -> 'GraphGenome':
        """Add new connection."""
        return GraphGenome(
            nodes=self.nodes,
            connections=self.connections | {connection},
            input_ids=self.input_ids,
            output_ids=self.output_ids
        )
    
    def to_dict(self) -> dict:
        return {
            'nodes': [
                {'id': n.id, 'type': n.node_type, 
                 'activation': n.activation, 'bias': n.bias}
                for n in self.nodes
            ],
            'connections': [
                {'innovation': c.innovation, 'from': c.from_node,
                 'to': c.to_node, 'weight': c.weight, 'enabled': c.enabled}
                for c in self.connections
            ],
            'input_ids': list(self.input_ids),
            'output_ids': list(self.output_ids)
        }
```

---

## Phenotype Protocol

```python
class Phenotype(Protocol):
    """
    Decoded form of a genome that can be evaluated.
    
    Phenotypes MAY be backend-specific (tensors, graphs).
    The Evaluation Layer handles phenotypes, not the Evolution Core.
    """
    
    def __call__(self, inputs: Any) -> Any:
        """Apply phenotype to inputs (e.g., forward pass)."""
        ...


class StatefulPhenotype(Protocol):
    """Phenotype with internal state (e.g., RNN)."""
    
    def reset(self) -> None:
        """Reset internal state."""
        ...
    
    def __call__(self, inputs: Any) -> Any:
        """Apply with state update."""
        ...
```

---

## Decoder Protocol

```python
P = TypeVar('P', bound=Phenotype)

class Decoder(Protocol[G, P]):
    """
    Maps genome to phenotype.
    
    Decoders are the bridge between framework-neutral
    genomes and potentially backend-specific phenotypes.
    """
    
    def decode(self, genome: G) -> P:
        """
        Convert genome to evaluable phenotype.
        
        Args:
            genome: Framework-neutral genome
            
        Returns:
            Phenotype (may be backend-specific)
        """
        ...


class BatchDecoder(Protocol[G, P]):
    """Decoder optimized for batch processing."""
    
    def decode_batch(self, genomes: Sequence[G]) -> Sequence[P]:
        """Decode multiple genomes efficiently."""
        ...
```

---

## Reference Decoder Implementations

```python
class VectorIdentityDecoder:
    """
    Identity decoder for vector genomes.
    
    Phenotype is just the genes array.
    """
    
    def decode(self, genome: VectorGenome) -> np.ndarray:
        return genome.genes


class GraphToNetworkDecoder:
    """
    Decode graph genome to feedforward network.
    
    Uses NumPy for CPU reference.
    """
    
    def decode(self, genome: GraphGenome) -> 'NumpyNetwork':
        """
        Create network from graph structure.
        
        Performs topological sort to determine
        evaluation order.
        """
        ...


class NumpyNetwork:
    """
    Simple feedforward network using NumPy.
    
    CPU reference implementation for neuroevolution.
    """
    
    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activations: list[Callable]
    ):
        self.weights = weights
        self.biases = biases
        self.activations = activations
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        for w, b, act in zip(self.weights, self.biases, self.activations):
            x = act(x @ w + b)
        return x
```
