# Implementation Plan: Comprehensive Documentation Refactor Centered on UnifiedConfig

**Branch**: `011-docs-unifiedconfig-refactor` | **Date**: 2026-04-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/011-docs-unifiedconfig-refactor/spec.md`

## Summary

Refactor all user-facing documentation (README, tutorials, examples, guides, docstrings, API docs) so that `UnifiedConfig` + `create_engine()` is the primary, consistent entry point shown across every artifact. Tutorials are renumbered (UnifiedConfig becomes 01), hand-rolled operator code is replaced with framework API calls, examples are converted to declarative patterns, a single Advanced Configuration Guide is created, and docstrings are updated with registry names and UnifiedConfig usage examples.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: evolve-framework (this project), Jupyter notebooks, Sphinx (docs build)
**Storage**: N/A (documentation-only changes)
**Testing**: pytest (existing tests), notebook execution validation, sphinx-build
**Target Platform**: Cross-platform (documentation consumed on any OS)
**Project Type**: Library documentation
**Performance Goals**: N/A
**Constraints**: All code snippets must execute correctly; no breaking changes to framework source
**Scale/Scope**: ~7 tutorial notebooks, 4 examples, 1 guide (updated) + 1 guide (new), README, ~15 docstring updates, Sphinx API docs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Model-Agnostic Architecture | ✅ PASS | Documentation changes only; no new hard dependencies |
| II. Separation of Concerns | ✅ PASS | No architectural changes |
| III. Declarative Completeness | ✅ PASS | This spec directly reinforces this principle |
| IV. Acceleration as Optional | ✅ PASS | Tutorials will handle optional deps gracefully |
| V. Determinism and Reproducibility | ✅ PASS | All examples/tutorials will include seed values |
| VI. Extensibility Over Premature Optimization | ✅ PASS | Documentation shows extension points (registries) |
| VII. Multi-Domain Algorithm Support | ✅ PASS | Tutorials cover vector, sequence, graph, SCM, RL, ERP |
| VIII. Observability and Experiment Tracking | ✅ PASS | Tracking guide included in Advanced Configuration Guide |

All gates pass. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/011-docs-unifiedconfig-refactor/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
# Files modified by this feature (documentation-only):

README.md                                    # Quickstart rewrite, representations section

docs/
├── guides/
│   ├── erp-best-practices.md                # Rewrite to UnifiedConfig + appendix
│   └── advanced-configuration.md            # NEW: consolidated advanced features guide
├── tutorials/
│   ├── README.md                            # Rewrite learning path, renumbered
│   ├── 01_unified_config.ipynb              # RENAMED from 07 → 01
│   ├── 02_vector_genome.ipynb               # RENAMED from 01 → 02, rewritten
│   ├── 03_sequence_genome.ipynb             # RENAMED from 02 → 03, rewritten
│   ├── 04_graph_genome_neat.ipynb           # RENAMED from 03 → 04, rewritten
│   ├── 05_rl_neuroevolution.ipynb           # RENAMED from 04 → 05, rewritten
│   ├── 06_scm_multiobjective.ipynb          # RENAMED from 05 → 06, rewritten
│   ├── 07_evolvable_reproduction.ipynb      # RENAMED from 06 → 07, rewritten
│
examples/
├── mlflow_tracking_demo.py                  # Already uses UnifiedConfig (minor updates)
├── sexual_selection.py                      # Rewrite to UnifiedConfig + manual override
├── protocol_evolution.py                    # Rewrite to UnifiedConfig + manual override
└── speciation_demo.py                       # Rewrite to UnifiedConfig + manual override

evolve/                                      # Docstring updates only
├── core/engine.py
├── core/population.py
├── core/operators/selection.py
├── core/operators/crossover.py
├── core/operators/mutation.py
├── config/unified.py
├── factory/engine.py
├── representation/vector.py
├── representation/sequence.py
├── representation/graph.py
├── representation/scm.py
├── registry/operators.py
├── registry/evaluators.py
├── registry/callbacks.py
└── registry/genomes.py
```

**Structure Decision**: This is a documentation-only feature modifying existing files and creating two new guide files. No new source code directories.

## Implementation Strategy

### Work Stream 1: Tutorial Renumbering & Rewriting (P1)
1. Rename tutorial files to new numbering (07→01, 01→02, 02→03, etc.)
2. Rewrite each tutorial to use `UnifiedConfig` + `create_engine()` as primary pattern
3. Replace hand-rolled operator code with framework API calls
4. Keep concept explanations in markdown cells, framework-only code in code cells
5. Add prerequisites/dependency cells to each notebook
6. Update `docs/tutorials/README.md` with new numbering and descriptions

### Work Stream 2: README Overhaul (P1)
1. Rewrite quickstart section with `UnifiedConfig` + `create_engine()` example
2. Add representations overview section (genome_type/genome_params)
3. Add registry reference section (built-in operators, evaluators, callbacks, genomes)

### Work Stream 3: Examples Conversion (P2)
1. Convert sexual_selection.py to UnifiedConfig + "Advanced: Manual Override"
2. Convert protocol_evolution.py to UnifiedConfig + "Advanced: Manual Override"
3. Convert speciation_demo.py to UnifiedConfig + "Advanced: Manual Override"
4. Minor updates to mlflow_tracking_demo.py (already compliant)

### Work Stream 4: Guides (P2)
1. Rewrite erp-best-practices.md primary examples to UnifiedConfig; add appendix
2. Create advanced-configuration.md consolidating: custom evaluators, custom callbacks, multi-objective, meta-evolution, MLflow tracking

### Work Stream 5: Docstrings (P3)
1. Update engine/factory docstrings with UnifiedConfig examples
2. Update genome class docstrings with genome_type + genome_params
3. Update operator class docstrings with registry names
4. Update registry class docstrings with usage patterns

### Work Stream 6: API Docs & Validation (P3)
1. Verify Sphinx build passes with zero errors/warnings
2. Ensure all public modules appear in generated output
3. Run all tutorial notebooks and examples to confirm execution

## Complexity Tracking

No constitution violations. Table not applicable.
