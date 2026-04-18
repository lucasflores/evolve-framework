# Implementation Plan: ES-HyperNEAT Decoder (CPPN-to-Network Indirect Encoding)

**Branch**: `012-es-hyperneat-decoder` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/012-es-hyperneat-decoder/spec.md`

## Summary

Add a `CPPNToNetworkDecoder` that implements the ES-HyperNEAT algorithm: given a CPPN represented as a `GraphGenome`, discover hidden neuron positions via quadtree decomposition of CPPN output variance, query the CPPN for pairwise connection weights, prune disconnected neurons, and return a callable `NEATNetwork`. The decoder integrates into the `DecoderRegistry` as `"cppn_to_network"` and is resolvable via `UnifiedConfig` + `create_engine()`. Two missing CPPN activation functions (`sin`, `abs`) are added to the existing activation registry.

## Technical Context

**Language/Version**: Python ≥3.10
**Primary Dependencies**: NumPy ≥1.24.0 (CPU-only, no ML frameworks)
**Storage**: N/A
**Testing**: pytest
**Target Platform**: Cross-platform (Linux, macOS, Windows)
**Project Type**: Library
**Performance Goals**: Decode CPPNs with up to `max_quadtree_depth=8` (≤~65k leaf nodes) in reasonable time. Performance optimization (GPU, batching) is explicitly out of scope per spec assumptions.
**Constraints**: CPU-only NumPy; no ML framework imports in core modules. Must maintain determinism from seed.
**Scale/Scope**: Single new module (`evolve/representation/cppn_decoder.py`) + activation additions + registry wiring.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | ✅ PASS | Decoder is a plugin in `evolve/representation/`. Core modules unchanged. GraphGenome and NEATNetwork are existing abstractions — no hard dependency on neural networks introduced in core. |
| II. Separation of Concerns | ✅ PASS | Decoder is an independent, composable component. Decoding logic is separate from evolution logic, evaluation, and execution backends. |
| III. Declarative Completeness | ✅ PASS | Decoder is registered in DecoderRegistry, resolvable via `UnifiedConfig.decoder="cppn_to_network"` + `decoder_params`. `create_engine()` wires it automatically. |
| IV. Acceleration as Optional | ✅ PASS | Implementation is CPU-only NumPy. No GPU/JIT dependencies. |
| V. Determinism and Reproducibility | ✅ PASS | `decode()` is deterministic — same CPPN + same parameters = same network. No RNG used in decoding (topology is deterministic from CPPN outputs). |
| VI. Extensibility Over Premature Optimization | ✅ PASS | Clean interface, no premature optimization. Quadtree is a straightforward recursive structure. |
| VII. Multi-Domain Algorithm Support | ✅ PASS | Adds neuroevolution capability via typed extension. Core abstractions unchanged. |
| VIII. Observability and Experiment Tracking | ✅ PASS | Decoder emits structured events (neurons discovered, connections created, neurons pruned) via callback/tracking system (FR-014). |

**Gate result: PASS** — no violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/012-es-hyperneat-decoder/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── decoder.md       # CPPNToNetworkDecoder public API contract
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
evolve/
├── representation/
│   ├── network.py            # MODIFY: add sin, abs activations to ACTIVATIONS dict
│   ├── decoder.py            # EXISTING: GraphToNetworkDecoder (composed internally)
│   └── cppn_decoder.py       # NEW: CPPNToNetworkDecoder, QuadTree, pruning logic
├── registry/
│   └── decoders.py           # MODIFY: register "cppn_to_network" factory

tests/
├── unit/
│   └── representation/
│       ├── test_cppn_decoder.py      # NEW: core decoder unit tests
│       └── test_activations.py       # NEW: sin, abs, gaussian activation tests
│   └── registry/
│       └── test_decoders.py          # MODIFY: add cppn_to_network registry tests
└── integration/
    └── test_cppn_engine.py           # NEW: UnifiedConfig → create_engine() integration
```

**Structure Decision**: Follows existing project layout. New decoder lives in `evolve/representation/` alongside `decoder.py` (GraphToNetworkDecoder). No new top-level packages needed.

## Complexity Tracking

No constitution violations — table not needed.
