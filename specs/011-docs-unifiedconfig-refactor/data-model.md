# Data Model: Documentation Refactor

**Branch**: `011-docs-unifiedconfig-refactor` | **Date**: 2026-04-13

## Entities

This feature modifies documentation artifacts, not data models. The key entities are the documentation artifacts themselves:

### Documentation Artifacts

| Artifact Type | Location | Count | Format |
|--------------|----------|-------|--------|
| README | `README.md` | 1 | Markdown |
| Tutorial Notebooks | `docs/tutorials/*.ipynb` | 7→8 | Jupyter |
| Tutorial README | `docs/tutorials/README.md` | 1 | Markdown |
| Examples | `examples/*.py` | 4 | Python |
| Guides | `docs/guides/*.md` | 1 existing + 1 new | Markdown |
| Docstrings | `evolve/**/*.py` | ~15 files | Python |
| API Docs | `docs/api/*.rst` | Existing | reStructuredText |

### Relationships

- README quickstart → references UnifiedConfig, create_engine(), genome types
- Tutorials README → indexes all tutorial notebooks by number and description
- Tutorial notebooks → import from `evolve.*`, use `UnifiedConfig` + `create_engine()`
- Examples → import from `evolve.*`, use `UnifiedConfig` + `create_engine()`
- Guides → reference registry entries, UnifiedConfig fields, sub-configs
- Docstrings → embedded in source files, consumed by Sphinx API docs
- API Docs (.rst) → auto-generate from docstrings via Sphinx autodoc

### State Transitions

Tutorial notebook renumbering:
```
07_unified_config.ipynb    → 01_unified_config.ipynb
01_vector_genome.ipynb     → 02_vector_genome.ipynb
02_sequence_genome.ipynb   → 03_sequence_genome.ipynb
03_graph_genome_neat.ipynb → 04_graph_genome_neat.ipynb
04_rl_neuroevolution.ipynb → 05_rl_neuroevolution.ipynb
05_scm_multiobjective.ipynb → 06_scm_multiobjective.ipynb
06_evolvable_reproduction_protocols.ipynb → 07_evolvable_reproduction.ipynb
```
