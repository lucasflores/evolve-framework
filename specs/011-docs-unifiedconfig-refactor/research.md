# Research: Documentation Refactor Centered on UnifiedConfig

**Branch**: `011-docs-unifiedconfig-refactor` | **Date**: 2026-04-13

## Research Summary

No NEEDS CLARIFICATION items existed in Technical Context. All decisions were resolved during clarification. This document records the key technical findings from codebase analysis.

## Key Findings

### 1. Current UnifiedConfig Declarative Coverage

**Decision**: UnifiedConfig is comprehensive enough to serve as the sole configuration entry point for all standard experiment types.

**Rationale**: Analysis of `evolve/config/unified.py` shows 40+ parameters covering: population/evolution params, operator selection (string name + params dict), genome type (string + params dict), evaluator (string + params dict), callbacks (custom + built-in), stopping criteria, ERP, multi-objective, meta-evolution, tracking, and dataset config. The `create_engine()` factory fully resolves all of these.

**Alternatives considered**: Keeping separate config patterns per experiment type — rejected because it contradicts Constitution Principle III (Declarative Completeness).

### 2. ERP Declarative Limitations

**Decision**: Some ERP features (individual-level protocol assignment, custom matchability functions) currently require imperative code after `create_engine()`.

**Rationale**: The `_create_erp_engine()` factory path handles `ERPSettings` integration, but protocols with custom matchability functions (e.g., cosine similarity) cannot be expressed as a string name in config. The 3 ERP examples all use manual `ERPEngine` construction for this reason.

**Alternatives considered**: Extending the registry to support matchability functions — deferred (out of scope for documentation refactor). Documentation will use "Advanced: Manual Override" pattern to address this.

### 3. Tutorial "Reinventing the Wheel" Pattern

**Decision**: All hand-rolled operator code in tutorials will be replaced with framework API calls via registries.

**Rationale**: Tutorials 01-06 construct operators manually (e.g., `TournamentSelection(tournament_size=3)`) and pass them directly to `EvolutionEngine`. While technically using framework classes, this bypasses the declarative config → factory pipeline. The correct pattern is `config = UnifiedConfig(selection="tournament", selection_params={"tournament_size": 3})` → `create_engine(config, evaluator)`.

**Alternatives considered**: Keeping manual construction as a "how it works" teaching tool — rejected for primary code paths per clarification Q3. Concept explanations stay in markdown cells.

### 4. Registry Built-in Entries

**Decision**: All registry entries will be documented in the Advanced Configuration Guide.

**Findings**:
- **Operator Registry**: selection (tournament, roulette, rank, crowded_tournament), crossover (single_point, two_point, uniform, sbx, blend), mutation (gaussian, uniform, polynomial, creep)
- **Evaluator Registry**: benchmark, function, llm_judge, ground_truth, scm, rl
- **Callback Registry**: logging, checkpoint, print, history
- **Genome Registry**: vector, sequence, graph, scm, network

### 5. Tutorial Renumbering Impact

**Decision**: Renumber 07→01, 01→02, 02→03, 03→04, 04→05, 05→06, 06→07. No 08 needed unless a tracking tutorial is added.

**Rationale**: Git handles renames well. The tutorials README is the primary index — updating it ensures discoverability. No external links are known to reference specific tutorial numbers.
