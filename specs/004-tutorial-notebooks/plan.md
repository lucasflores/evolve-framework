# Implementation Plan: Tutorial Notebooks for Evolve Framework

**Branch**: `004-tutorial-notebooks` | **Date**: 2026-01-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-tutorial-notebooks/spec.md`

## Summary

Create 5 comprehensive Jupyter tutorial notebooks plus a shared utility module (`tutorial_utils.py`) teaching the Evolve framework's representation types to ML/DS practitioners with no prior EA experience. Each notebook demonstrates end-to-end evolution with synthetic data, ML-to-EA terminology bridging, island model parallelism, GPU acceleration benchmarks, and advanced features (speciation for NEAT, NSGA-II for SCM). Visualizations use beautiful-mermaid (github-light theme) and plotly for interactive Pareto exploration.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: evolve (core framework), numpy, matplotlib, plotly, beautiful-mermaid, gymnasium (RL notebook)  
**Storage**: N/A (notebooks are self-contained, synthetic data generated at runtime)  
**Testing**: pytest for tutorial_utils.py module; notebook execution validation via papermill  
**Target Platform**: Jupyter Notebook/Lab, VS Code notebooks, Google Colab compatible  
**Project Type**: Documentation/tutorials (notebooks + utility module)  
**Performance Goals**: Each notebook completes in <90 minutes including all code execution  
**Constraints**: CPU-only execution must work; GPU optional; island models require 4+ cores for speedup demo  
**Scale/Scope**: 5 notebooks (~500-800 cells total), 1 shared module (~1000 LOC)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: Tutorials demonstrate framework's model-agnostic design; no hard PyTorch/JAX deps in tutorial_utils
- [x] **Separation of Concerns**: Each notebook showcases decoupled components (genome, evaluator, operators, backend)
- [x] **Optional Acceleration**: GPU sections are clearly marked optional; CPU reference implementations shown first
- [x] **Determinism**: All examples use explicit seeds; users learn reproducibility as core concept
- [x] **Extensibility**: Tutorials demonstrate extension points; no premature optimization in utility module
- [x] **Multi-Domain Support**: 5 notebooks span classical EA, neuroevolution, multi-objective, causal discovery, RL
- [x] **Observability**: Callbacks and metrics logging demonstrated in every notebook
- [x] **Clear Abstractions**: Type annotations in tutorial_utils; explicit interfaces documented
- [x] **Composability**: No global state; components independently imported and combined
- [x] **Test-First**: tutorial_utils.py will have comprehensive unit tests

**Violations requiring justification**: None - tutorials are documentation that demonstrate constitution compliance

## Project Structure

### Documentation (this feature)

```text
specs/004-tutorial-notebooks/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts for tutorial_utils)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
docs/tutorials/
├── utils/
│   ├── __init__.py
│   └── tutorial_utils.py       # Shared data gen, viz, terminology (FR-001 to FR-010)
├── 01_vector_genome.ipynb      # Continuous optimization (FR-027 to FR-030)
├── 02_sequence_genome.ipynb    # Genetic programming (FR-031 to FR-034)
├── 03_graph_genome_neat.ipynb  # Neuroevolution + speciation (FR-035 to FR-040)
├── 04_rl_neuroevolution.ipynb  # Policy evolution (FR-041 to FR-046)
├── 05_scm_multiobjective.ipynb # Causal discovery + NSGA-II (FR-047 to FR-054)
└── README.md                   # Tutorial index and learning path

tests/tutorials/
├── test_tutorial_utils.py      # Unit tests for shared module
└── test_notebook_execution.py  # Papermill-based notebook validation
```

**Structure Decision**: Tutorial/documentation project. Notebooks live in `docs/tutorials/` alongside existing documentation. Shared utilities in a submodule `utils/` to avoid polluting the main package namespace. Tests in `tests/tutorials/` following existing test organization.

## Complexity Tracking

No constitution violations - this feature is documentation that demonstrates constitution compliance.

## Phase 0 Outputs

- [x] [research.md](research.md) - Technology decisions and implementation patterns
- [x] All NEEDS CLARIFICATION markers resolved

## Phase 1 Outputs

- [x] [data-model.md](data-model.md) - Data structures for tutorial utilities
- [x] [contracts/tutorial_utils_api.md](contracts/tutorial_utils_api.md) - Public API contract
- [x] [quickstart.md](quickstart.md) - Development setup and patterns
- [x] Agent context updated

## Phase 2: Next Step

Run `/speckit.tasks` to generate implementation task breakdown.

## Implementation Phases

### Phase A: Foundation (Priority P1-P2)
1. Create `tutorial_utils.py` with data generators and visualization functions
2. Implement VectorGenome notebook (simplest, validates all shared code)
3. Write tests for tutorial_utils module

### Phase B: Core Notebooks (Priority P2-P3)
4. SequenceGenome notebook (genetic programming)
5. GraphGenome/NEAT notebook with full speciation

### Phase C: Advanced Topics (Priority P4-P5)
6. RL/Neuroevolution notebook (CartPole + optional LunarLander)
7. SCMGenome notebook with full NSGA-II multi-objective

### Phase D: Polish
8. README.md with learning path
9. Final review and runtime validation
10. Documentation integration

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| beautiful-mermaid breaking changes | Pin version, provide text fallback |
| Gymnasium API changes | Pin gymnasium>=0.29, test in CI |
| Notebook execution >90 min | Add checkpoints, early stopping demos |
| GPU not available to most users | Make GPU sections clearly optional |
| Island model overhead masks speedup | Use ring topology, tune migration interval |

