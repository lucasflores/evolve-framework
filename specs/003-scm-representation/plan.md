# Implementation Plan: SCM Representation for Causal Discovery

**Branch**: `003-scm-representation` | **Date**: 2026-01-30 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-scm-representation/spec.md`

## Summary

Implement evolutionary representation for Structural Causal Models (SCMs) enabling causal discovery through evolution. The implementation provides `SCMGenome` (wrapping `SequenceGenome` with stack-based postfix encoding), `SCMDecoder` (stack machine producing `DecodedSCM` with NetworkX graph), and `SCMEvaluator` (multi-objective fitness with configurable penalties for cycles, conflicts, and coverage).

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: NumPy (array ops), NetworkX (graph representation, cycle detection)  
**Storage**: N/A (in-memory, checkpoint via existing infrastructure)  
**Testing**: pytest with property-based tests (hypothesis)  
**Target Platform**: Cross-platform (Linux, macOS, Windows)  
**Project Type**: Single project - extends existing `evolve/` package  
**Performance Goals**: Decode 500-gene genome in <10ms, support 1000+ population  
**Constraints**: No ML framework dependencies, deterministic decoding from seed  
**Scale/Scope**: 3 new modules (~1500 LOC), ~500 lines of tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with Evolve Framework Constitution principles:

- [x] **Model-Agnostic Architecture**: SCM representation is domain-specific but uses no PyTorch/JAX/RL dependencies
- [x] **Separation of Concerns**: Genome (representation), Decoder (transformation), Evaluator (fitness) are separate modules
- [x] **Optional Acceleration**: CPU-only initial implementation; GPU acceleration explicitly out of scope
- [x] **Determinism**: Stack machine decoding is deterministic; ERC sampling uses explicit Random instance with seed
- [x] **Extensibility**: SCMConfig provides extension points for objectives, constraints, strategies; Decoder protocol enables alternatives
- [x] **Multi-Domain Support**: Adds causal discovery domain to existing classical EA, neuroevolution, multi-objective support
- [x] **Observability**: Decoder exposes junk genes, conflict metadata; Evaluator returns structured fitness with penalty breakdown
- [x] **Clear Abstractions**: Full type annotations, dataclasses with frozen=True, explicit Decoder[G, P] protocol
- [x] **Composability**: No global state; SCMGenome wraps SequenceGenome via composition; all components constructor-injected
- [x] **Test-First**: Test strategy defined with unit, property, and integration tests before implementation

**Violations requiring justification**: None

## Project Structure

### Documentation (this feature)

```text
specs/003-scm-representation/
├── plan.md              # This file
├── research.md          # Phase 0: Stack encoding patterns, cycle detection algorithms
├── data-model.md        # Phase 1: Entity definitions, type hierarchy
├── quickstart.md        # Phase 1: Usage examples
├── contracts/           # Phase 1: API contracts
│   ├── scm_genome.py    # SCMGenome, SCMConfig, SCMAlphabet interfaces
│   ├── scm_decoder.py   # SCMDecoder, DecodedSCM, Expression interfaces
│   └── scm_evaluator.py # SCMEvaluator, SCMFitnessConfig interfaces
└── tasks.md             # Phase 2: Implementation tasks (via /speckit.tasks)
```

### Source Code (repository root)

```text
evolve/
├── representation/
│   ├── scm.py           # SCMGenome, SCMConfig, SCMAlphabet, ConflictResolution
│   └── scm_decoder.py   # SCMDecoder, DecodedSCM, Expression AST nodes
└── evaluation/
    └── scm_evaluator.py # SCMEvaluator, SCMFitnessConfig, partial eval strategies

tests/
├── unit/
│   ├── test_scm_genome.py        # Genome protocol, alphabet, ERC, serialization
│   ├── test_scm_decoder.py       # Stack machine, junk detection, conflict resolution
│   └── test_scm_evaluator.py     # Objectives, constraints, penalties, cycle handling
├── property/
│   └── test_scm_properties.py    # Determinism, round-trip serialization
└── integration/
    └── test_scm_discovery.py     # End-to-end synthetic SCM discovery
```

**Structure Decision**: Follows existing `evolve/representation/` pattern with paired decoder. Evaluator in `evolve/evaluation/` consistent with existing evaluators. Tests mirror source structure.

## Complexity Tracking

> No Constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

---

## Phase 0: Outline & Research

### Technical Context Unknowns

All technical context items resolved via user input:
- ✅ Language/Version: Python 3.11+
- ✅ Dependencies: NumPy, NetworkX
- ✅ Testing: pytest
- ✅ Architecture: Follows existing representation patterns

### Research Tasks

1. **Stack-based GP encoding patterns**: Best practices for RPN/postfix genetic programming
2. **NetworkX cycle detection**: API for `is_directed_acyclic_graph`, `simple_cycles`, `ancestors`
3. **Expression AST design**: Immutable AST nodes for equation representation
4. **ERC handling in sequence genomes**: Heterogeneous typing for str | float genes

### Research Findings

**Decision 1: Stack Encoding Pattern**
- Decision: Use linear postfix encoding with typed genes (operators, operands, STORE)
- Rationale: Simpler than tree-based GP, natural junk emergence, compatible with SequenceGenome
- Alternatives: Tree GP (complex crossover), CGP (grid constraints), GEP (gene-specific mutation)

**Decision 2: Heterogeneous SequenceGenome**
- Decision: `SequenceGenome[str | float]` with runtime type checking in alphabet
- Rationale: Existing SequenceGenome is generic; union type handles ERC floats alongside symbol strings
- Alternatives: Separate ERC array (complicates crossover), tagged union class (over-engineering)

**Decision 3: Expression AST**
- Decision: Immutable dataclasses with `@dataclass(frozen=True)`: `Var`, `Const`, `BinOp`, `UnaryOp`
- Rationale: Matches framework pattern, enables hashing for deduplication, simple pattern matching
- Alternatives: SymPy expressions (heavy dependency), string expressions (no structure)

**Decision 4: Cycle Detection**
- Decision: Use `nx.is_directed_acyclic_graph(G)` for check, `nx.simple_cycles(G)` for enumeration
- Rationale: NetworkX standard, well-tested, O(V+E) for DAG check
- Alternatives: Custom Tarjan (unnecessary), DFS coloring (reinventing)

**Decision 5: Ancestor Validation**
- Decision: Use `nx.ancestors(G, node)` intersected with observed variable set
- Rationale: Direct NetworkX API, clear semantics
- Alternatives: Custom BFS (unnecessary complexity)

---

## Phase 1: Design & Contracts

### Data Model

See [data-model.md](data-model.md) for complete entity definitions.

**Core Entities:**

| Entity | Description | Key Attributes |
|--------|-------------|----------------|
| `SCMConfig` | Configuration dataclass | observed_variables, max_latent_variables, penalties |
| `SCMAlphabet` | Symbol set factory | variables, operators, constants, ERC slots |
| `SCMGenome` | Genome wrapper | inner: SequenceGenome, config: SCMConfig, erc_values: dict |
| `Expression` | AST base | Abstract, subclassed by Var, Const, BinOp |
| `DecodedSCM` | Phenotype | equations: dict, graph: nx.DiGraph, metadata |
| `SCMDecoder` | Decoder | config: SCMConfig, deterministic stack machine |
| `SCMEvaluator` | Evaluator | data: np.ndarray, config: SCMFitnessConfig |

### API Contracts

See [contracts/](contracts/) for interface definitions.

**Key Interfaces:**

```python
# SCMGenome implements Genome and SerializableGenome protocols
class SCMGenome:
    def copy(self) -> SCMGenome: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SCMGenome: ...

# SCMDecoder implements Decoder[SCMGenome, DecodedSCM] protocol  
class SCMDecoder:
    def decode(self, genome: SCMGenome) -> DecodedSCM: ...

# SCMEvaluator implements Evaluator[SCMGenome] protocol
class SCMEvaluator:
    @property
    def capabilities(self) -> EvaluatorCapabilities: ...
    def evaluate(self, individuals: Sequence[Individual[SCMGenome]], seed: int | None = None) -> Sequence[Fitness]: ...
```

### Quickstart

See [quickstart.md](quickstart.md) for usage examples.

---

## Phase 2: Implementation Tasks

> Generated by `/speckit.tasks` command - not created by `/speckit.plan`

Task breakdown will be generated in `tasks.md` covering:
1. Core types (SCMConfig, SCMAlphabet, Expression AST)
2. SCMGenome with SequenceGenome composition
3. SCMDecoder stack machine implementation
4. DecodedSCM with NetworkX graph construction
5. SCMEvaluator multi-objective fitness
6. Unit tests for each component
7. Property tests for determinism
8. Integration test for synthetic discovery

---

## Testing Strategy

### Unit Tests

| Component | Test Focus | Key Scenarios |
|-----------|-----------|---------------|
| SCMAlphabet | Symbol generation | All symbol types present, ERC slots |
| SCMGenome | Protocol compliance | copy(), __eq__, __hash__, to_dict/from_dict |
| SCMDecoder | Stack machine | Valid equations, junk emergence, underflow |
| SCMDecoder | Conflict resolution | first_wins, last_wins, all_junk |
| SCMDecoder | Cycle detection | Cyclic graphs detected, cycles enumerated |
| SCMEvaluator | Objectives | data_fit, sparsity, simplicity computation |
| SCMEvaluator | Constraints | acyclicity reject/penalize modes |
| SCMEvaluator | Penalties | div_zero, cycle, conflict penalties |

### Property Tests

| Property | Description |
|----------|-------------|
| Determinism | Same genome → same DecodedSCM (no RNG in decode) |
| Serialization round-trip | genome == from_dict(to_dict(genome)) |
| Junk gene accounting | len(junk_indices) + effective_genes == genome_length |

### Integration Tests

| Test | Description |
|------|-------------|
| Synthetic discovery | Evolve population, verify true SCM structure discovered |
| Checkpoint round-trip | Save/restore population, verify continued evolution |

---

## Dependencies

### External Dependencies (already in project)

- `numpy`: Array operations, MSE computation
- `networkx`: Graph representation, cycle detection

### New Dependencies

None required. NetworkX already used in existing graph representation.

### Optional Dependencies

- `dowhy`: Causal inference metrics (not required for core implementation)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance: large genome decoding | Medium | Medium | Profile early, optimize hot paths |
| Complexity: partial eval strategies | Low | Medium | Implement penalty_only first, others as needed |
| Integration: ERP matchability | Low | Low | P3 priority, can defer if complex |

---

## Notes

- SCMGenome composes SequenceGenome rather than inheriting to maintain clear separation
- Expression AST is frozen dataclass for immutability and hashing
- Decoder is stateless; all configuration passed via SCMConfig
- Evaluator requires data at construction time (stateful w.r.t. observed data)
