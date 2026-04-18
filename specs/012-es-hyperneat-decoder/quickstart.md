# Quickstart: ES-HyperNEAT Decoder

**Feature**: 012-es-hyperneat-decoder | **Date**: 2026-04-17

## Direct Usage (Programmatic)

```python
import numpy as np
from evolve.representation.graph import GraphGenome, NodeGene, ConnectionGene
from evolve.representation.cppn_decoder import CPPNToNetworkDecoder

# 1. Define a simple CPPN genome (4 inputs: x1,y1,x2,y2 → 1 output: weight)
nodes = frozenset([
    NodeGene(0, "input"),   # x1
    NodeGene(1, "input"),   # y1
    NodeGene(2, "input"),   # x2
    NodeGene(3, "input"),   # y2
    NodeGene(4, "output", activation="tanh"),
])
connections = frozenset([
    ConnectionGene(1, 0, 4, 1.0),   # x1 → output
    ConnectionGene(2, 1, 4, 0.5),   # y1 → output
    ConnectionGene(3, 2, 4, -1.0),  # x2 → output
    ConnectionGene(4, 3, 4, -0.5),  # y2 → output
])
cppn_genome = GraphGenome(nodes, connections, (0, 1, 2, 3), (4,))

# 2. Create decoder with substrate configuration
decoder = CPPNToNetworkDecoder(
    input_positions=[(0.0, -1.0), (1.0, -1.0)],    # 2 input neurons
    output_positions=[(0.5, 1.0)],                   # 1 output neuron
    weight_threshold=0.2,
    variance_threshold=0.03,
    max_quadtree_depth=3,
)

# 3. Decode CPPN into a substrate network
network = decoder.decode(cppn_genome)

# 4. Run the network
output = network(np.array([1.0, 0.5]))
print(f"Network output: {output}")

# 5. Check decode statistics
stats = decoder.last_decode_stats
print(f"Hidden neurons discovered: {stats.neurons_discovered}")
print(f"Neurons after pruning: {stats.neurons_final}")
print(f"Connections: {stats.connections_after_pruning}")
```

## Declarative Usage (UnifiedConfig + create_engine)

```python
from evolve.config.unified import UnifiedConfig
from evolve.factory.engine import create_engine

config = UnifiedConfig(
    name="es_hyperneat_xor",
    population_size=150,
    max_generations=200,
    genome_type="graph",
    genome_params={"input_nodes": 4, "output_nodes": 1},
    selection="tournament",
    selection_params={"tournament_size": 3},
    crossover="neat",
    mutation="neat",
    mutation_params={
        "add_node_prob": 0.05,
        "add_connection_prob": 0.1,
    },
    decoder="cppn_to_network",
    decoder_params={
        "input_positions": [(0.0, -1.0), (1.0, -1.0)],
        "output_positions": [(0.5, 1.0)],
        "weight_threshold": 0.3,
        "variance_threshold": 0.03,
        "max_quadtree_depth": 4,
        "hidden_activation": "sigmoid",
    },
    evaluator="benchmark",
    evaluator_params={"function_name": "xor"},
    seed=42,
)

engine = create_engine(config)
# result = engine.run(initial_population)
```

## Using CPPN Activation Functions

```python
from evolve.representation.network import get_activation

# New CPPN-specific activations
sin_fn = get_activation("sin")       # np.sin(x)
abs_fn = get_activation("abs")       # np.abs(x)

# Already available
gauss_fn = get_activation("gaussian")  # exp(-x²)
linear_fn = get_activation("linear")   # identity
sigmoid_fn = get_activation("sigmoid")
```
