# Implementation Plan: Symbiogenetic Merge Operator

**Branch**: `013-symbiogenetic-merge` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/013-symbiogenetic-merge/spec.md`

## Summary

Add a representation-agnostic symbiogenetic merge operator that permanently absorbs one individual (symbiont) into another (host), producing a single offspring of greater compositional complexity. The operator integrates into the evolutionary engine as a post-mutation phase with its own configurable rate, supports multiple symbiont sourcing strategies, and provides tracking/observability via a new metric category. The GraphGenome implementation handles NEAT-specific concerns (node ID remapping, innovation number remapping, interface connections), while simpler implementations cover SequenceGenome, VectorGenome, and EmbeddingGenome.

## Technical Context

**Language/Version**: Python 3.10+, full type hints (mypy strict)
**Primary Dependencies**: NumPy (numeric ops); no ML framework deps in core
**Storage**: N/A
**Testing**: pytest; TDD (Red-Green-Refactor); tests in `tests/` mirroring `evolve/` structure
**Target Platform**: Cross-platform (Python library)
**Project Type**: Library
**Performance Goals**: Merge operator must not dominate generation time — O(N+M) where N,M are genome sizes
**Constraints**: All genomes immutable (frozen dataclasses with frozensets/immutable arrays); no global mutable state; explicit RNG threading
**Scale/Scope**: ~19 functional requirements, 4 genome types, 2 symbiont sourcing strategies, 1 new config section, 1 new metric category

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | **PASS** | Merge protocol is generic over `G`; no neural network or ML framework dependencies. GraphGenome-specific logic isolated in a plugin implementation. |
| II. Separation of Concerns | **PASS** | Merge operator is a standalone component; does not access evaluation functions. Symbiont sourcing strategies are separate from the operator. Engine integration is via a clean phase insertion. |
| III. Declarative Completeness | **PASS** | `MergeConfig` section added to `UnifiedConfig`; merge operator registered in operator registry; `create_engine` factory resolves merge from config. All merge parameters are declarative. |
| IV. Acceleration as Optional | **PASS** | No GPU/JIT/vectorization dependencies. Pure Python + NumPy. |
| V. Determinism and Reproducibility | **PASS** | Explicit `rng: Random` parameter on all merge operations. Node ID/innovation remapping uses deterministic counters. |
| VI. Extensibility | **PASS** | SymbiogeneticMerge protocol allows user-defined merge strategies. Registry pattern for discovery. |
| VII. Multi-Domain Support | **PASS** | Implementations for 4 genome types spanning neuroevolution (GraphGenome), sequence optimization (SequenceGenome), continuous optimization (VectorGenome), and soft-prompt evolution (EmbeddingGenome). |
| VIII. Observability | **PASS** | New `SYMBIOGENESIS` metric category; MergeMetricCollector tracks merge count, genome complexity, complexity delta. Merged offspring carry lineage metadata. |

**Gate result**: All 8 principles PASS. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/013-symbiogenetic-merge/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
evolve/
├── core/
│   ├── operators/
│   │   └── merge.py              # SymbiogeneticMerge protocol + GraphGenome impl
│   └── engine.py                 # Merge phase insertion in _step()
├── config/
│   ├── merge.py                  # MergeConfig frozen dataclass
│   ├── unified.py                # Add merge: MergeConfig | None field
│   └── tracking.py               # Add SYMBIOGENESIS MetricCategory
├── registry/
│   └── operators.py              # Add "merge" category + builtin registrations
├── factory/
│   └── engine.py                 # Resolve merge operator from config
├── experiment/
│   └── collectors/
│       └── merge.py              # MergeMetricCollector
└── representation/
    ├── graph.py                  # InnovationTracker.reserve_range() helper
    ├── sequence.py               # (no changes — concatenation via SequenceGenome constructor)
    ├── vector.py                 # (no changes — concatenation via np.concatenate)
    └── embedding.py              # (no changes — stacking via np.vstack)

tests/
├── unit/
│   ├── core/
│   │   └── operators/
│   │       └── test_merge.py     # Protocol, GraphGenome merge, non-graph merges
│   ├── config/
│   │   └── test_merge_config.py  # MergeConfig validation
│   └── experiment/
│       └── collectors/
│           └── test_merge_collector.py
└── integration/
    └── test_engine_merge.py      # End-to-end engine with merge enabled
```

**Structure Decision**: Follows existing project layout. New merge operator lives in `evolve/core/operators/merge.py` alongside existing crossover/mutation/selection. Config in `evolve/config/merge.py` following the ERPSettings/MetaEvolutionConfig pattern. Collector in `evolve/experiment/collectors/merge.py` following NEATMetricCollector pattern.
