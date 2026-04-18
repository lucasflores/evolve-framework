# Feature Specification: ES-HyperNEAT Decoder (CPPN-to-Network Indirect Encoding)

**Feature Branch**: `012-es-hyperneat-decoder`
**Created**: 2026-04-17
**Status**: Draft
**Input**: User description: "Add an ES-HyperNEAT decoder that takes a CPPN (represented as a GraphGenome) and decodes it into an arbitrary neural network where both topology (neuron placement, connectivity) and weights are determined by the CPPN's output."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Decode a CPPN into a Neural Network with Discovered Topology (Priority: P1)

A researcher has evolved a CPPN (GraphGenome) and wants to decode it into a functional neural network whose hidden neuron positions and connection weights are automatically determined by querying the CPPN. They configure the decoder with input/output neuron positions and thresholds, invoke `decode()`, and receive an executable neural network.

**Why this priority**: This is the core value proposition — without the ability to decode a CPPN into a network with automatically discovered topology, the feature has no utility.

**Independent Test**: Can be fully tested by constructing a simple CPPN GraphGenome with known outputs, decoding it, and verifying the resulting network has neurons placed in information-dense regions and connections where the CPPN output exceeds the weight threshold.

**Acceptance Scenarios**:

1. **Given** a CPPN GraphGenome, input positions [(0,0)], output positions [(1,1)], and a weight threshold, **When** the user calls `decoder.decode(genome)`, **Then** the decoder returns a callable neural network whose hidden neuron positions were discovered via quadtree decomposition and whose connections reflect CPPN-queried weights above the threshold.
2. **Given** a CPPN whose output is uniform (low variance everywhere), **When** decoded, **Then** the resulting network contains only the specified input and output neurons with no hidden neurons discovered.
3. **Given** a CPPN whose output has high variance in a specific spatial region, **When** decoded, **Then** hidden neurons are placed in that region and the resulting network produces correct forward-pass outputs when called with input data.

---

### User Story 2 - Configure and Run ES-HyperNEAT via UnifiedConfig and create_engine() (Priority: P2)

A user wants to set up an ES-HyperNEAT experiment entirely through configuration — specifying `decoder="cppn_to_network"` and providing `decoder_params` (input positions, output positions, thresholds, quadtree depth) in their UnifiedConfig. They call `create_engine()` and the decoder is wired in automatically, ready for an evolutionary run.

**Why this priority**: Declarative configuration is how users interact with the framework. Without registry integration the decoder would require manual wiring, breaking the framework's convention.

**Independent Test**: Can be tested by creating a UnifiedConfig with `decoder="cppn_to_network"` and appropriate `decoder_params`, calling `create_engine()`, and verifying the engine contains a correctly configured CPPNToNetworkDecoder.

**Acceptance Scenarios**:

1. **Given** a UnifiedConfig with `decoder="cppn_to_network"` and `decoder_params` including input/output positions, **When** `create_engine()` is called, **Then** the engine is created with a CPPNToNetworkDecoder using those parameters.
2. **Given** a UnifiedConfig with `decoder="cppn_to_network"` but missing required `decoder_params`, **When** `create_engine()` is called, **Then** the factory raises a clear error indicating which parameters are missing.
3. **Given** a UnifiedConfig with `decoder="cppn_to_network"`, **When** the user lists available decoders via DecoderRegistry, **Then** `"cppn_to_network"` appears in the list.

---

### User Story 3 - CPPN Activation Functions Available for GraphGenome Evolution (Priority: P3)

A user evolving CPPNs needs the canonical CPPN activation functions (sin, Gaussian, abs, linear) to be available in the framework's activation function set so that NEAT mutation can assign them to CPPN nodes. These functions produce the geometric regularities (repetition, symmetry, gradients, reflection) that make indirect encoding powerful.

**Why this priority**: The decoder functions correctly with any activation set, but the quality of evolved CPPNs depends critically on having these specific activation functions. This is an enabler for good results, not a functional prerequisite.

**Independent Test**: Can be tested by verifying that `get_activation("sin")`, `get_activation("gaussian")`, `get_activation("abs")`, and `get_activation("linear")` all return valid callable functions, and that NEATMutation can assign these activations to new nodes.

**Acceptance Scenarios**:

1. **Given** the framework's activation function registry, **When** a user requests activation functions "sin", "gaussian", "abs", and "linear", **Then** each returns a valid callable that computes the expected mathematical function.
2. **Given** a NEATMutation operator configured with the CPPN activation set, **When** an add-node mutation occurs, **Then** the new node's activation is drawn from the set that includes sin, Gaussian, abs, linear, and sigmoid.

---

### User Story 4 - Pruning Disconnected Neurons from Decoded Network (Priority: P3)

After the CPPN is queried for all pairwise connections and low-weight connections are removed, some discovered hidden neurons may become disconnected — they have no path from any input to any output. These dead-end neurons must be pruned so the resulting network is clean and efficient.

**Why this priority**: Pruning is essential for correctness (disconnected neurons waste computation and can cause evaluation errors) but is an internal step of the decoding process, not a standalone user-facing feature.

**Independent Test**: Can be tested by constructing a scenario where the CPPN produces connections that leave some discovered neurons isolated, decoding, and verifying the returned network contains only neurons reachable from inputs to outputs.

**Acceptance Scenarios**:

1. **Given** a CPPN that produces connections leaving hidden neuron H3 with no path to any output, **When** the genome is decoded, **Then** H3 does not appear in the resulting network.
2. **Given** a CPPN that produces connections leaving hidden neuron H5 with no path from any input, **When** the genome is decoded, **Then** H5 does not appear in the resulting network.

---

### User Story 5 - Optional Euclidean Distance Input to CPPN (Priority: P4)

A user wants to control whether the CPPN receives a 5th input representing the Euclidean distance between source and target coordinates. This enables the CPPN to produce distance-dependent connection patterns (e.g., local connectivity) without having to learn the distance function from raw coordinates.

**Why this priority**: This is a configuration option that enhances expressiveness but is not required for basic functionality.

**Independent Test**: Can be tested by decoding the same CPPN genome with `distance_input=True` and `distance_input=False` and verifying the CPPN receives 5 vs. 4 inputs respectively.

**Acceptance Scenarios**:

1. **Given** `distance_input=True`, **When** the CPPN is queried for a connection between (0,0) and (1,1), **Then** the CPPN receives 5 inputs: x1=0, y1=0, x2=1, y2=1, d=√2.
2. **Given** `distance_input=False`, **When** the CPPN is queried, **Then** the CPPN receives 4 inputs: x1, y1, x2, y2.

---

### Edge Cases

- What happens when the CPPN output is constant (zero variance everywhere)? The decoder produces a network with only input and output neurons and direct connections (if above threshold) — no hidden neurons are discovered.
- What happens when all CPPN-queried weights fall below the weight threshold? The decoder produces a network with neurons but no connections, which will output zeros (or biases only) on forward pass.
- What happens when max_quadtree_depth is reached before variance falls below threshold? Subdivision stops at the maximum depth, placing neurons at the finest granularity allowed.
- What happens when input_positions and output_positions contain duplicate coordinates? The decoder treats each position entry as a distinct neuron regardless of coordinate overlap.
- What happens when the CPPN GraphGenome has no hidden nodes (minimal CPPN)? The decoder still functions — the CPPN is a simple input-to-output mapping that produces a valid (if simple) network.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `CPPNToNetworkDecoder` class that implements the `Decoder[GraphGenome, NEATNetwork]` protocol, accepting a GraphGenome (representing a CPPN) and returning a callable `NEATNetwork`.
- **FR-002**: The decoder MUST execute a three-phase decoding pipeline: (1) build an executable CPPN from the GraphGenome, (2) discover hidden neuron positions via quadtree decomposition of CPPN output variance, (3) query the CPPN for all pairwise connection weights and apply thresholding.
- **FR-003**: The quadtree decomposition MUST subdivide spatial regions where the CPPN's output variance exceeds a configurable variance threshold, recursively up to a configurable maximum depth, placing neurons at the centers of information-dense leaf nodes.
- **FR-004**: The decoder MUST query the CPPN at every (source_neuron, target_neuron) coordinate pair and create a connection only when the absolute CPPN output exceeds the configurable weight threshold.
- **FR-005**: The decoder MUST prune any discovered hidden neurons that have no path from any input neuron to any output neuron in the resulting network.
- **FR-006**: The decoder MUST accept configurable parameters: `input_positions` (list of 2D coordinates for input neurons, required), `output_positions` (list of 2D coordinates for output neurons, required), `weight_threshold` (minimum absolute weight for connection creation, default `0.3`, must be > 0), `variance_threshold` (minimum variance for quadtree subdivision, default `0.03`, must be > 0), `max_quadtree_depth` (maximum recursion depth, default `4`, must be ≥ 1), `distance_input` (whether to include Euclidean distance as a 5th CPPN input, default `False`), and `hidden_activation` (activation function name for discovered hidden neurons, default `\"sigmoid\"`, must be a valid name in the framework's activation registry).
- **FR-007**: The decoder MUST be registered in the DecoderRegistry under the name `"cppn_to_network"` with a factory function that accepts `decoder_params` and returns a configured `CPPNToNetworkDecoder`.
- **FR-008**: The decoder MUST be resolvable via UnifiedConfig's `decoder` field so that `create_engine()` can wire it in automatically without manual construction.
- **FR-009**: The framework's activation function set MUST include `sin`, `gaussian`, `abs`, and `linear` so that CPPN node genes can use these activations for producing geometric regularities.
- **FR-010**: When `distance_input` is `True`, the CPPN MUST receive 5 inputs (x1, y1, x2, y2, d) where d is the Euclidean distance between (x1, y1) and (x2, y2). When `False`, the CPPN MUST receive 4 inputs (x1, y1, x2, y2).
- **FR-011**: The decoded neural network MUST be callable, accepting input values and producing output values via forward propagation through the discovered topology.
- **FR-012**: *(Consolidated into FR-006)* Hidden neuron activation is configurable via `hidden_activation` parameter — see FR-006 for details.", "oldString": "- **FR-012**: Discovered hidden neurons MUST use a configurable activation function specified via `decoder_params` (e.g., `hidden_activation=\"sigmoid\"`). The default MUST be `\"sigmoid\"`. The activation name MUST resolve via the framework's `get_activation()` registry.
- **FR-013**: The `decode()` method MUST always return a `NEATNetwork` instance, which natively supports the arbitrary topologies produced by ES-HyperNEAT.
- **FR-014**: The decoder MUST emit structured events via the framework's callback/tracking system reporting discovery and pruning statistics (neurons discovered, connections created, neurons pruned) to support experiment observability.
- **FR-015**: The `CPPNToNetworkDecoder.__init__()` MUST raise `ValueError` with a clear message if `input_positions` or `output_positions` is empty.
- **FR-016**: The `CPPNToNetworkDecoder.__init__()` MUST raise `ValueError` if the `hidden_activation` name is not found in the framework's activation function registry.

### Key Entities

- **CPPNToNetworkDecoder**: The decoder object. Holds configuration (positions, thresholds, depth settings) and implements the `decode(genome) -> network` method. Stateless with respect to individual genomes — the same decoder instance can decode multiple CPPNs.
- **CPPN (Compositional Pattern-Producing Network)**: An executable neural network built from a GraphGenome. Each node may have a distinct activation function. Takes spatial coordinates as input and produces weight values as output. Represented internally as a NEATNetwork obtained via the existing GraphToNetworkDecoder.
- **Quadtree**: A spatial decomposition structure used during hidden neuron discovery. Each node represents a region of 2D space; leaf nodes with high CPPN output variance are subdivided until the variance threshold or depth limit is reached. Neuron positions are extracted from the centers of selected leaf nodes.
- **Decoded Network**: The final output neural network containing input neurons (at user-specified positions), output neurons (at user-specified positions), discovered hidden neurons (at quadtree-determined positions), and connections with CPPN-determined weights. Callable as a standard network.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can decode any valid CPPN GraphGenome into a callable neural network in a single `decode()` call, with no manual topology specification required.
- **SC-002**: The decoder correctly discovers hidden neuron positions — networks decoded from CPPNs with spatially varying output have more hidden neurons than networks from CPPNs with uniform output.
- **SC-003**: A user can configure and run an ES-HyperNEAT experiment entirely through UnifiedConfig and `create_engine()` with no manual decoder wiring.
- **SC-004**: The decoded network produces deterministic outputs — decoding the same CPPN with the same parameters always yields an identical network.
- **SC-005**: Networks with thousands of connections (generated from CPPNs with high quadtree depth) can be decoded and executed without errors.
- **SC-006**: All existing tests continue to pass — the addition does not break any existing decoder, registry, or configuration functionality.

## Assumptions

- The CPPN genome is a standard GraphGenome with node and connection genes, evolved by existing NEAT operators (NEATCrossover, NEATMutation, InnovationTracker). No new reproduction operators are needed.
- Neuron positions are 2D coordinates. Extension to higher dimensions is out of scope for this feature.
- The existing GraphToNetworkDecoder is used internally to build the executable CPPN from the GraphGenome. The CPPNToNetworkDecoder composes on top of it.
- The decoded output network type is NEATNetwork from the existing representation module (always, per clarification). No new network type is needed.
- Performance optimization (GPU-accelerated CPPN queries, batched evaluation) is out of scope for the initial implementation. The decoder operates on CPU with NumPy.
- The quadtree operates over a normalized 2D coordinate space (e.g., [-1, 1] × [-1, 1]). The user is responsible for specifying input/output positions within this space.
- Connection weights from the CPPN output are used directly (after thresholding). No additional weight normalization or scaling is applied beyond what the CPPN produces.
- Hidden neurons in the decoded substrate network use a single configurable activation function (default sigmoid), specified via `hidden_activation` in `decoder_params`.

## Clarification Log

| # | Question | Answer | Rationale |
|---|----------|--------|-----------|
| 1 | Hidden neuron activation function | Configurable via `decoder_params` (`hidden_activation`, default `"sigmoid"`) | Allows experimentation with different substrate activations |
| 2 | Return type of `decode()` | Always `NEATNetwork` | Arbitrary topology; no layer structure guarantee |
| 3 | Observability | Structured events via callback/tracking system | Aligns with Constitution Principle VIII |
| 4 | CPPN output interpretation | Raw output = weight; threshold on absolute value | Standard ES-HyperNEAT approach |
| 5 | Empty positions validation | `ValueError` at `__init__` — fail fast | Clear error at construction time |
