# Research: ES-HyperNEAT Decoder

**Feature**: 012-es-hyperneat-decoder | **Date**: 2026-04-17

## R1: ES-HyperNEAT Quadtree Decomposition Algorithm

**Decision**: Implement the Evolvable-Substrate HyperNEAT (ES-HyperNEAT) quadtree algorithm as described in Risi & Stanley (2012).

**Rationale**: ES-HyperNEAT is the standard extension to HyperNEAT that automatically discovers hidden neuron placement. The algorithm uses quadtree decomposition of the CPPN's output space to identify regions of high information density (high variance), placing neurons at the centers of subdivided leaf nodes. This eliminates the need for the user to manually define substrate geometry for hidden layers.

**Algorithm outline**:
1. Start with the full 2D coordinate space (e.g., [-1,1]×[-1,1]) as the root quadtree node.
2. For each leaf node, sample the CPPN at the four quadrant centers.
3. Compute variance of CPPN outputs across the four samples.
4. If variance > `variance_threshold` and depth < `max_quadtree_depth`, subdivide into four children.
5. Repeat until convergence. Leaf nodes whose band (range of CPPN outputs) exceeds a threshold become neuron positions.
6. The spec simplifies this to: leaf nodes at sufficient depth where subdivision was warranted → place neuron at center.

**Alternatives considered**:
- **Fixed substrate (HyperNEAT)**: Requires user to pre-specify all neuron positions. Rejected — the whole point of this feature is automatic topology discovery.
- **Random sampling**: Loses the geometric regularity that CPPNs exploit. Rejected.
- **Octree (3D)**: Out of scope per spec assumption (2D only).

## R2: CPPN Execution Strategy

**Decision**: Reuse the existing `GraphToNetworkDecoder` to convert the CPPN `GraphGenome` into a `NEATNetwork`, then query it as a callable.

**Rationale**: The CPPN is itself a neural network with arbitrary topology (just like any NEAT-evolved network). The existing `GraphToNetworkDecoder.decode()` already handles topological sorting and builds a callable `NEATNetwork`. Composing on top of this avoids code duplication and leverages the battle-tested decoder.

**Alternatives considered**:
- **Custom CPPN evaluator**: Would duplicate topological sort and network evaluation logic. Rejected — unnecessary complexity.
- **NetworkX-based evaluation**: Would add a runtime dependency on NetworkX for what's already solved. Rejected.

## R3: Connection Weight Querying

**Decision**: After neuron positions are established (inputs + outputs + discovered hidden), query the CPPN at every (source, target) coordinate pair. Create a connection only where |CPPN_output| > `weight_threshold`. Use the raw CPPN output as the connection weight.

**Rationale**: This is the standard HyperNEAT/ES-HyperNEAT approach. The CPPN acts as a pattern-producing function over the connection space. Thresholding on absolute value allows both positive and negative weights while filtering weak connections.

**Alternatives considered**:
- **Normalized weights**: Apply min-max or z-score normalization after querying. Rejected per spec assumption: "Connection weights from the CPPN output are used directly."
- **Separate weight/expression outputs**: Use one CPPN output for weight and another for whether to create the connection (LEO extension). Rejected — out of scope for initial implementation.

## R4: Pruning Disconnected Neurons

**Decision**: After thresholding, perform bidirectional reachability analysis: keep only neurons that lie on at least one path from an input to an output.

**Rationale**: The quadtree may discover neurons in regions where the CPPN has high variance, but after weight thresholding, some of those neurons may have no incoming or outgoing connections that form a complete path. Pruning these dead-end neurons prevents wasted computation and ensures a clean network.

**Algorithm**:
1. Forward pass: BFS/DFS from all input neurons following outgoing connections → mark reachable nodes.
2. Backward pass: BFS/DFS from all output neurons following incoming connections → mark reachable nodes.
3. Keep only neurons in the intersection of both reachable sets.

**Alternatives considered**:
- **Prune only unreachable from inputs**: Would leave neurons with no path to outputs. Rejected — incomplete.
- **No pruning**: Would leave dead-end neurons. Rejected — violates FR-005.

## R5: Activation Functions (sin, abs)

**Decision**: Add `sin` and `abs` to the existing `ACTIVATIONS` dict in `evolve/representation/network.py`. The `gaussian` and `linear` (identity) activations already exist.

**Rationale**: The canonical CPPN activation set is {sin, gaussian, abs, linear, sigmoid}. Currently the framework has gaussian, linear (alias for identity), and sigmoid. Missing: `sin` and `abs`. These are mathematically trivial to implement with NumPy.

**Alternatives considered**:
- **Separate CPPN activation registry**: Would fragment the activation system. Rejected — the main `ACTIVATIONS` dict is the right place; these functions are generally useful beyond CPPNs.

## R6: NEATNetwork as Output Type

**Decision**: Always return `NEATNetwork` from `decode()`. Do not introduce a new network type.

**Rationale**: `NEATNetwork` already supports arbitrary topology with topological ordering. ES-HyperNEAT produces arbitrary (non-layered) topologies, so `NEATNetwork` is the natural fit. Reusing the existing type avoids new abstractions and ensures compatibility with existing evaluators and callbacks.

**Alternatives considered**:
- **New `SubstrateNetwork` type**: Would add a type that duplicates `NEATNetwork` functionality. Rejected — no additional capability needed.
- **`NumpyNetwork` (layer-based)**: Cannot represent arbitrary topology. Rejected.

## R7: Topological Sort for Decoded Network

**Decision**: After pruning, construct the decoded network's node order by running topological sort on the surviving neurons and connections. Reuse the existing `GraphToNetworkDecoder._topological_sort()` logic pattern (Kahn's algorithm).

**Rationale**: The decoded network may have arbitrary topology (not just layers). Topological sort is required for correct feedforward evaluation order. The implementation is a direct adaptation of the existing algorithm, operating on the decoded connection set rather than a `GraphGenome`.

## R8: Distance Input (5th CPPN Input)

**Decision**: When `distance_input=True`, prepend the Euclidean distance `d = sqrt((x2-x1)² + (y2-y1)²)` as a 5th input to the CPPN query. The CPPN `GraphGenome` must have been evolved with 5 input nodes (or 4 if `distance_input=False`).

**Rationale**: Distance input allows the CPPN to produce distance-dependent patterns (e.g., local connectivity, distance-based weight decay) without learning the distance function. This is a standard ES-HyperNEAT configuration option.

**Note**: It is the user's responsibility to ensure the CPPN genome's `input_ids` count matches the expected input dimensionality (4 or 5). The decoder will raise a clear error on mismatch.

## R9: Observability / Structured Events

**Decision**: After decoding, emit a structured dictionary of statistics via the framework's logging/tracking infrastructure. The decoder itself returns the stats alongside the network (or as a logged side-effect). Specifically:
- `neurons_discovered`: count of quadtree-discovered hidden neurons
- `connections_before_pruning`: count of connections after thresholding
- `neurons_pruned`: count of neurons removed by pruning
- `connections_after_pruning`: final connection count
- `neurons_final`: final neuron count

**Rationale**: Constitution Principle VIII requires observability. These metrics help researchers understand the decoder's behavior and tune thresholds.

**Implementation**: The `decode()` method returns a `NEATNetwork`. Statistics are available via a `last_decode_stats` property on the decoder, following a lightweight pattern that avoids complicating the return type.
