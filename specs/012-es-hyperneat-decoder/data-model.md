# Data Model: ES-HyperNEAT Decoder

**Feature**: 012-es-hyperneat-decoder | **Date**: 2026-04-17

## Entities

### CPPNToNetworkDecoder

The main decoder class. Stateless per-genome — one instance can decode many CPPNs.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `input_positions` | `list[tuple[float, float]]` | 2D coordinates for input neurons | Non-empty; validated at `__init__` |
| `output_positions` | `list[tuple[float, float]]` | 2D coordinates for output neurons | Non-empty; validated at `__init__` |
| `weight_threshold` | `float` | Minimum \|CPPN output\| to create connection | > 0; default `0.3` |
| `variance_threshold` | `float` | Minimum variance for quadtree subdivision | > 0; default `0.03` |
| `max_quadtree_depth` | `int` | Maximum recursion depth for quadtree | ≥ 1; default `4` |
| `distance_input` | `bool` | Whether CPPN receives Euclidean distance as 5th input | default `False` |
| `hidden_activation` | `str` | Activation function name for discovered hidden neurons | Must be in `ACTIVATIONS`; default `"sigmoid"` |
| `last_decode_stats` | `DecodeStats \| None` | Statistics from most recent `decode()` call | Set after each decode; initially `None` |

**Validation rules**:
- `__init__` raises `ValueError` if `input_positions` or `output_positions` is empty (FR-015).
- `hidden_activation` is validated against `get_activation()` at init time.

### DecodeStats

Immutable dataclass holding statistics from a single decode operation.

| Field | Type | Description |
|-------|------|-------------|
| `neurons_discovered` | `int` | Hidden neurons found by quadtree |
| `connections_before_pruning` | `int` | Connections after CPPN query + thresholding |
| `neurons_pruned` | `int` | Hidden neurons removed by reachability pruning |
| `connections_after_pruning` | `int` | Final connection count |
| `neurons_final` | `int` | Final total neuron count (input + output + surviving hidden) |

### QuadTreeNode

Internal data structure for the quadtree decomposition. Not part of the public API.

| Field | Type | Description |
|-------|------|-------------|
| `x` | `float` | Center x-coordinate of this region |
| `y` | `float` | Center y-coordinate of this region |
| `half_size` | `float` | Half the width/height of this region |
| `depth` | `int` | Current depth in the tree |
| `children` | `list[QuadTreeNode] \| None` | Four children if subdivided, else `None` |

### Activation Functions (additions to existing ACTIVATIONS dict)

| Name | Function | Formula |
|------|----------|---------|
| `"sin"` | `sin_activation` | `np.sin(x)` |
| `"abs"` | `abs_activation` | `np.abs(x)` |

These join the existing activations: sigmoid, tanh, relu, leaky_relu, identity, linear, softmax, step, gaussian.

## Relationships

```
GraphGenome (CPPN)
    │
    ▼  [GraphToNetworkDecoder.decode() — internal]
NEATNetwork (executable CPPN)
    │
    ▼  [CPPNToNetworkDecoder._discover_hidden_neurons() — quadtree]
list[tuple[float, float]]  (hidden neuron positions)
    │
    ▼  [CPPNToNetworkDecoder._query_connections() — CPPN evaluation + thresholding]
dict[(src, tgt), weight]  (raw connections)
    │
    ▼  [CPPNToNetworkDecoder._prune_disconnected() — reachability]
dict[(src, tgt), weight]  (pruned connections)
    │
    ▼  [Build NEATNetwork with topological sort]
NEATNetwork (decoded substrate network)
```

## State Transitions

The decoder itself is stateless per-genome. The decoding pipeline is a pure function (deterministic) with the following phases:

1. **Build CPPN** → GraphGenome → NEATNetwork (via existing GraphToNetworkDecoder)
2. **Discover neurons** → Quadtree decomposition → list of (x, y) positions
3. **Assign neuron IDs** → Input positions get IDs 0..N-1, output positions get IDs N..N+M-1, hidden neurons get IDs from N+M onward
4. **Query connections** → For every (source, target) pair, query CPPN → threshold → create connection
5. **Prune** → Bidirectional reachability from inputs and to outputs → remove orphaned neurons and their connections
6. **Build network** → Topological sort on surviving graph → NEATNetwork

## Registry Integration

The `DecoderRegistry` gains one new entry:

| Registry Name | Factory | Parameters |
|--------------|---------|------------|
| `"cppn_to_network"` | `create_cppn_to_network_decoder(**kwargs)` | `input_positions`, `output_positions`, `weight_threshold`, `variance_threshold`, `max_quadtree_depth`, `distance_input`, `hidden_activation` |

Resolution chain: `UnifiedConfig.decoder="cppn_to_network"` → `DecoderRegistry.get("cppn_to_network", **config.decoder_params)` → `CPPNToNetworkDecoder(...)`.
