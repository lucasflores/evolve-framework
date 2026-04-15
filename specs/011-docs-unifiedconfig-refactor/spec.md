# Feature Specification: Comprehensive Documentation Refactor Centered on UnifiedConfig

**Feature Branch**: `011-docs-unifiedconfig-refactor`  
**Created**: 2026-04-13  
**Status**: Draft  
**Input**: User description: "Comprehensive update/refactor of all documentation, tutorials, readmes, examples, guides, and docstrings to be in sync with current working functionality, centered on UnifiedConfig as the primary declarative entry point for all experiment types."

## Clarifications

### Session 2026-04-13

- Q: Should tutorials be renumbered to place UnifiedConfig earlier? → A: Yes, move UnifiedConfig tutorial to position 01 and renumber all others (02–07).
- Q: For ERP features that can't be fully declarative, what pattern? → A: Show UnifiedConfig for everything possible, then a clearly-labeled "Advanced: Manual Override" section for imperative parts.
- Q: Should tutorials retain EA/GA concept explanations? → A: Yes, keep concept explanations in markdown cells, but all code cells must use framework API only (no custom implementations).
- Q: What should happen to ERP best-practices guide manual patterns? → A: Rewrite primary examples to UnifiedConfig; keep one "Advanced: Direct ERPEngine Usage" appendix section for power users.
- Q: Separate guides per advanced feature or consolidated? → A: Single "Advanced Configuration Guide" covering all features (custom evaluators, custom callbacks, multi-objective, meta-evolution, MLflow tracking) in sections.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First-Time User Runs an Experiment via UnifiedConfig (Priority: P1)

A researcher new to the evolve-framework reads the README, follows the getting-started guide, and is able to define and run a complete evolutionary experiment using only a `UnifiedConfig` and `create_engine()` — without ever constructing engine components manually.

**Why this priority**: If the primary learning path does not lead users to UnifiedConfig as the canonical entry point, the entire documentation effort fails. This is the foundational user journey.

**Independent Test**: Can be fully tested by verifying a new user can go from `pip install` to a working experiment using only README and getting-started content, with all code using `UnifiedConfig` + `create_engine()`.

**Acceptance Scenarios**:

1. **Given** a user reads the README, **When** they follow the quickstart section, **Then** they can define a `UnifiedConfig`, call `create_engine()`, and get evolution results — all within 5 minutes of reading.
2. **Given** the README quickstart code, **When** a user copies and runs it, **Then** it executes without errors and produces meaningful output.
3. **Given** a user wants to understand what configuration options are available, **When** they read the README or getting-started guide, **Then** they find a clear reference to `UnifiedConfig` fields and the registry system.

---

### User Story 2 - Notebook Tutorials Teach Concepts via Framework Methods (Priority: P1)

A researcher working through representation-specific tutorials (vector, sequence, graph, SCM, RL, ERP) uses `UnifiedConfig` and built-in framework methods throughout. Tutorials explain EA/GA concepts (selection, crossover, mutation) while demonstrating how the framework implements them — never rolling custom implementations of functionality the framework already provides.

**Why this priority**: Tutorials that bypass framework methods teach users the wrong patterns and undermine the entire framework purpose. This is tied with P1 because incorrect tutorials actively harm adoption.

**Independent Test**: Each tutorial notebook can be executed end-to-end and every fitness evaluation, operator invocation, and engine construction uses evolve-framework public API (registries, factories, `create_engine()`).

**Acceptance Scenarios**:

1. **Given** any tutorial notebook (01-07, renumbered with UnifiedConfig as 01), **When** a user runs all cells, **Then** every evolutionary operation uses framework-provided operators resolved through registries or `create_engine()` — no hand-rolled selection, crossover, or mutation logic appears.
2. **Given** a tutorial that teaches crossover concepts, **When** the user reads the explanation, **Then** markdown cells explain the concept while all code cells demonstrate it using a registered crossover operator (e.g., `crossover="single_point"` in config) rather than implementing crossover from scratch.
3. **Given** a tutorial that previously used manual `EvolutionEngine(...)` construction, **When** it is updated, **Then** it uses `UnifiedConfig` + `create_engine()` as the primary pattern, with the old manual pattern removed or relegated to an "under the hood" sidebar.

---

### User Story 3 - Examples All Use UnifiedConfig (Priority: P2)

A developer looking at `examples/` can copy any example and adapt it, always using the `UnifiedConfig` + `create_engine()` pattern — including ERP, multi-objective, and tracking examples.

**Why this priority**: Examples serve as copy-paste templates. If they use inconsistent patterns, users learn inconsistent patterns.

**Independent Test**: Each `.py` file in `examples/` runs successfully and uses `UnifiedConfig` as its configuration mechanism.

**Acceptance Scenarios**:

1. **Given** any example in `examples/`, **When** a user reads it, **Then** the experiment is configured via `UnifiedConfig` (with appropriate sub-configs like `ERPSettings`, `MultiObjectiveConfig`, `TrackingConfig`).
2. **Given** the ERP examples (sexual_selection.py, protocol_evolution.py, speciation_demo.py), **When** they are updated, **Then** they use `UnifiedConfig.with_erp()` and `create_engine()` for all declaratively-expressible configuration, followed by a clearly-labeled "Advanced: Manual Override" section for any imperative parts that cannot be expressed declaratively.
3. **Given** any example, **When** a user runs it (possibly with `--help`), **Then** it produces meaningful output without requiring external dependencies beyond the framework's optional extras.

---

### User Story 4 - Guides Cover Advanced Declarative Patterns (Priority: P2)

A researcher wanting to use advanced features (ERP, multi-objective, meta-evolution, MLflow tracking, custom evaluators, custom callbacks) finds a guide that explains how to configure each entirely through `UnifiedConfig` and the registry system.

**Why this priority**: Advanced users who can't find declarative patterns for advanced features fall back to manual wiring, defeating the purpose of the config system.

**Independent Test**: Each guide contains runnable code snippets that compile/run correctly and use declarative configuration.

**Acceptance Scenarios**:

1. **Given** the ERP best-practices guide, **When** a user reads it, **Then** primary configuration examples use `UnifiedConfig` with `ERPSettings`, with one "Advanced: Direct ERPEngine Usage" appendix section for power users needing manual patterns.
2. **Given** a need for custom evaluator/callback, **When** a user reads the single Advanced Configuration Guide, **Then** they learn how to register it and reference it by name in `UnifiedConfig`.
3. **Given** the tracking section of the Advanced Configuration Guide, **When** a user follows it, **Then** MLflow tracking is enabled purely through `TrackingConfig` in `UnifiedConfig`.

---

### User Story 5 - Docstrings Are Accurate and Reference UnifiedConfig (Priority: P3)

A developer using IDE autocompletion reads module/class/method docstrings that accurately reflect current behavior, include working examples, and reference `UnifiedConfig` as the recommended usage pattern where relevant.

**Why this priority**: Docstrings are the most accessible form of documentation. Inaccurate docstrings cause subtle misuse.

**Independent Test**: Docstring code examples can be extracted and run as doctests. Key classes reference `UnifiedConfig` in their usage examples.

**Acceptance Scenarios**:

1. **Given** the `EvolutionEngine` class docstring, **When** a developer reads it, **Then** the primary usage example shows `create_engine(config)` rather than manual construction.
2. **Given** any genome class docstring (VectorGenome, SequenceGenome, etc.), **When** a developer reads it, **Then** it shows how to configure that genome type via `genome_type` and `genome_params` in `UnifiedConfig`.
3. **Given** any operator class docstring, **When** a developer reads it, **Then** it mentions the registry name (e.g., `"tournament"`) that can be used in `UnifiedConfig`.

---

### User Story 6 - API Reference Docs Build and Are Current (Priority: P3)

A developer browsing the Sphinx-generated API docs finds accurate, complete documentation that matches the current codebase, with working cross-references.

**Why this priority**: API reference is a long-tail resource; it doesn't block initial adoption but is critical for sustained usage.

**Independent Test**: `make html` in `docs/` succeeds without warnings, and all public modules appear in the generated output.

**Acceptance Scenarios**:

1. **Given** the Sphinx docs source, **When** `make html` runs, **Then** it completes with zero errors and zero warnings about missing references.
2. **Given** any public module in `evolve/`, **When** a user searches the API docs, **Then** that module appears with complete class/method documentation.
3. **Given** the API docs index, **When** a user navigates to UnifiedConfig, **Then** all fields, methods, and nested config types are documented.

---

### User Story 7 - Tutorials README Provides Clear Learning Path (Priority: P2)

A user reading `docs/tutorials/README.md` finds a clear, accurate learning path that starts with UnifiedConfig fundamentals and progresses through representation-specific tutorials — with accurate descriptions matching updated notebook content.

**Why this priority**: The tutorials README is the navigation hub for the tutorial series. If it's out of sync, users get confused.

**Independent Test**: Every notebook referenced in the README exists, and the description matches the notebook's actual content.

**Acceptance Scenarios**:

1. **Given** `docs/tutorials/README.md`, **When** a user reads the learning path, **Then** UnifiedConfig is listed as tutorial 01 (the starting point), with all other tutorials renumbered sequentially (02–07) after it.
2. **Given** the README descriptions, **When** a user opens a referenced notebook, **Then** the actual content matches the README description and the file numbering matches the README ordering.

---

### Edge Cases

- What happens when a tutorial references a feature that requires optional dependencies (JAX, torch, RL libraries)? Each tutorial must clearly state required extras and gracefully skip if unavailable.
- What happens when an ERP feature genuinely cannot be expressed declaratively? The documentation must explicitly note the limitation in a clearly-labeled "Advanced: Manual Override" section showing the minimal imperative code needed.
- What happens when code examples in docstrings reference deprecated patterns? All deprecated patterns must be removed or marked with deprecation warnings pointing to UnifiedConfig equivalents.
- What happens when the Sphinx build encounters a module with deferred imports? The API docs must handle optional-dependency modules without failing the build.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The README MUST contain a quickstart section where the primary code example uses `UnifiedConfig` + `create_engine()` to define and run a complete experiment.
- **FR-002**: Tutorial notebooks MUST be renumbered with UnifiedConfig as tutorial 01, and all others renumbered sequentially (02–07). Every tutorial MUST use `UnifiedConfig` + `create_engine()` as the primary pattern.
- **FR-003**: Tutorial notebooks MUST NOT implement operator logic (selection, crossover, or mutation algorithms) from scratch in code cells when the framework already provides equivalent registered implementations. Using framework operator classes directly (e.g., `TournamentSelection()`) is permitted only in clearly-labeled "under the hood" educational sidebars, not as the primary usage pattern. Concept explanations in markdown cells are retained.
- **FR-004**: Every `.py` example in `examples/` MUST configure experiments via `UnifiedConfig` (with appropriate sub-configs) rather than manual engine construction.
- **FR-005**: The ERP examples (sexual_selection.py, protocol_evolution.py, speciation_demo.py) MUST use `UnifiedConfig.with_erp()` and `create_engine()` for all declaratively-expressible configuration, with any imperative overrides in a clearly-labeled "Advanced: Manual Override" section.
- **FR-006**: All guides in `docs/guides/` MUST present configuration examples using `UnifiedConfig` and the registry system as the primary pattern.
- **FR-007**: The ERP best-practices guide MUST be updated so primary code examples use declarative `UnifiedConfig` + `ERPSettings` configuration, with one "Advanced: Direct ERPEngine Usage" appendix section preserved for power users.
- **FR-008**: A single "Advanced Configuration Guide" MUST be created covering custom evaluators (registry), custom callbacks (registry), multi-objective configuration, meta-evolution configuration, and MLflow tracking configuration — all showing the declarative `UnifiedConfig` approach in organized sections.
- **FR-009**: Class and method docstrings for all public API entry points (`EvolutionEngine`, `create_engine`, genome classes, operator classes) MUST include a usage example that references `UnifiedConfig` as the recommended pattern.
- **FR-010**: Operator class docstrings MUST include the registry name string (e.g., `"tournament"`, `"gaussian"`) that users pass in `UnifiedConfig`.
- **FR-011**: Genome class docstrings MUST include the `genome_type` string (e.g., `"vector"`, `"sequence"`) and a representative `genome_params` dict for `UnifiedConfig`.
- **FR-012**: The `docs/tutorials/README.md` MUST present an accurate learning path where UnifiedConfig is listed as tutorial 01 (starting point) with all others renumbered as 02–07.
- **FR-013**: All tutorial and guide code snippets MUST execute without errors when the required dependencies are installed.
- **FR-014**: The Sphinx API documentation MUST build without errors or missing-reference warnings, and all public modules in `evolve/` MUST appear in the generated output.
- **FR-015**: Every tutorial notebook MUST include a dependencies/prerequisites cell that lists required extras and gracefully handles missing optional dependencies.
- **FR-016**: The project README MUST include a section describing the available representations (vector, sequence, graph, SCM) and how each is configured via `genome_type`/`genome_params` in `UnifiedConfig`.
- **FR-017**: Deprecated or legacy patterns (manual engine construction, raw operator instantiation outside of educational "under the hood" sidebars) MUST be removed from primary documentation paths or clearly marked as advanced/internal.
- **FR-018**: Every registry (operator, evaluator, callback, genome) MUST have its built-in entries documented in a discoverable location (guide, README section, or dedicated reference page).

### Key Entities

- **UnifiedConfig**: The single declarative entry point for defining experiments. All documentation must treat it as the canonical configuration mechanism.
- **create_engine()**: The factory function that resolves a UnifiedConfig into a runnable EvolutionEngine. All documentation must pair it with UnifiedConfig.
- **Registries** (Operator, Evaluator, Callback, Genome): The lookup systems that resolve string names in UnifiedConfig to concrete implementations. Documentation must teach users how to discover and extend registered entries.
- **Tutorial Notebooks**: The 7 Jupyter notebooks in `docs/tutorials/` that serve as the primary learning resource for representation-specific concepts.
- **Examples**: The `.py` files in `examples/` that serve as copy-paste starter templates for common experiment types.
- **Guides**: The prose documents in `docs/guides/` that explain advanced features and best practices.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of tutorial notebooks (01–07) execute end-to-end without errors with appropriate dependencies installed.
- **SC-002**: 100% of `.py` examples in `examples/` run without errors.
- **SC-003**: 100% of tutorial notebooks and examples use `UnifiedConfig` + `create_engine()` as the primary experiment configuration pattern.
- **SC-004**: 0 instances of hand-rolled selection/crossover/mutation logic in tutorials where an equivalent registered operator exists.
- **SC-005**: Sphinx documentation builds with zero errors and zero missing-reference warnings.
- **SC-006**: Every public class in `evolve/` that is part of the user-facing API has a docstring containing a `UnifiedConfig`-based usage example or registry name reference.
- **SC-007**: A new user following the README quickstart can define and run an experiment using only `UnifiedConfig` + `create_engine()` in under 5 minutes of reading.

## Assumptions

- The current `UnifiedConfig`, `create_engine()` factory, and registry system are functionally complete for the representations and operators that exist today. This spec does not add new framework features — it updates documentation to reflect what already works.
- ERP features that can be configured via `UnifiedConfig.with_erp()` + `ERPSettings` + `create_engine()` will use that path; any features that genuinely require imperative construction will be documented as known limitations with the minimal imperative code needed.
- Optional dependencies (JAX, torch, RL environments) are not assumed to be installed; tutorials and examples requiring them must handle their absence gracefully.
- The Sphinx documentation build infrastructure (`docs/conf.py`, Makefile) already exists and is functional; this spec updates content, not build tooling.
- Tutorial notebooks will be renumbered: UnifiedConfig becomes 01, all others shift to 02–07. This is a confirmed decision, not optional.
