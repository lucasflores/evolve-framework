# Contract: CPPNToNetworkDecoder

**Feature**: 012-es-hyperneat-decoder | **Date**: 2026-04-17

## Public API

### `CPPNToNetworkDecoder`

```python
from evolve.representation.cppn_decoder import CPPNToNetworkDecoder, DecodeStats

class CPPNToNetworkDecoder:
    """
    ES-HyperNEAT decoder: CPPN GraphGenome → NEATNetwork.

    Discovers hidden neuron positions via quadtree decomposition of
    CPPN output variance, queries pairwise connection weights, prunes
    disconnected neurons, and returns a callable NEATNetwork.

    Registry name: "cppn_to_network"
    """

    def __init__(
        self,
        input_positions: list[tuple[float, float]],
        output_positions: list[tuple[float, float]],
        weight_threshold: float = 0.3,
        variance_threshold: float = 0.03,
        max_quadtree_depth: int = 4,
        distance_input: bool = False,
        hidden_activation: str = "sigmoid",
    ) -> None:
        """
        Initialize the decoder.

        Args:
            input_positions: 2D coordinates for input neurons.
            output_positions: 2D coordinates for output neurons.
            weight_threshold: Minimum |CPPN output| to create a connection.
            variance_threshold: Minimum variance for quadtree subdivision.
            max_quadtree_depth: Maximum quadtree recursion depth.
            distance_input: If True, CPPN receives 5 inputs (x1,y1,x2,y2,d).
            hidden_activation: Activation function name for discovered hidden neurons.

        Raises:
            ValueError: If input_positions or output_positions is empty.
            ValueError: If hidden_activation is not a registered activation.
            ValueError: If weight_threshold <= 0, variance_threshold <= 0, or max_quadtree_depth < 1.
        """
        ...

    def decode(self, genome: GraphGenome) -> NEATNetwork:
        """
        Decode a CPPN GraphGenome into a callable NEATNetwork.

        Three-phase pipeline:
        1. Build executable CPPN from GraphGenome
        2. Discover hidden neuron positions via quadtree
        3. Query connections, threshold, prune, build network

        Args:
            genome: CPPN represented as a GraphGenome.

        Returns:
            Callable NEATNetwork with discovered topology.
        """
        ...

    @property
    def last_decode_stats(self) -> DecodeStats | None:
        """Statistics from the most recent decode() call."""
        ...
```

### `DecodeStats`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DecodeStats:
    """Statistics from a single ES-HyperNEAT decode operation."""
    neurons_discovered: int
    connections_before_pruning: int
    neurons_pruned: int
    connections_after_pruning: int
    neurons_final: int
```

## Activation Function Additions

```python
# Added to evolve/representation/network.py ACTIVATIONS dict

def sin_activation(x: np.ndarray) -> np.ndarray:
    """Sine activation: sin(x)."""
    return np.sin(x)

def abs_activation(x: np.ndarray) -> np.ndarray:
    """Absolute value activation: |x|."""
    return np.abs(x)

# Registry additions:
# "sin": sin_activation
# "abs": abs_activation
```

## Registry Factory

```python
# Added to evolve/registry/decoders.py :: _register_builtin_decoders()

def create_cppn_to_network_decoder(**kwargs: Any) -> CPPNToNetworkDecoder:
    """Create a CPPNToNetworkDecoder from decoder_params."""
    from evolve.representation.cppn_decoder import CPPNToNetworkDecoder
    return CPPNToNetworkDecoder(**kwargs)

registry.register("cppn_to_network", create_cppn_to_network_decoder)
```

## UnifiedConfig Usage

```python
config = UnifiedConfig(
    genome_type="graph",
    genome_params={"input_nodes": 4, "output_nodes": 1},  # CPPN: 4 or 5 inputs
    decoder="cppn_to_network",
    decoder_params={
        "input_positions": [(0.0, 0.0)],
        "output_positions": [(1.0, 1.0)],
        "weight_threshold": 0.3,
        "variance_threshold": 0.03,
        "max_quadtree_depth": 4,
        "distance_input": False,
        "hidden_activation": "sigmoid",
    },
    # ... other config fields ...
)
engine = create_engine(config, evaluator=my_evaluator)
```

## Error Conditions

| Condition | Exception | Message Pattern |
|-----------|-----------|-----------------|
| Empty `input_positions` | `ValueError` | `"input_positions must not be empty"` |
| Empty `output_positions` | `ValueError` | `"output_positions must not be empty"` |
| Unknown `hidden_activation` | `KeyError` | Propagated from `get_activation()` |
| CPPN input count mismatch | `ValueError` | `"CPPN has {n} inputs but expected {expected} (distance_input={flag})"` |
