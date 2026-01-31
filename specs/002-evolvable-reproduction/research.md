# Research: Evolvable Reproduction Protocols (ERP)

**Feature**: 002-evolvable-reproduction  
**Date**: January 28, 2026  
**Phase**: 0 - Research

## Research Tasks

### 1. Protocol Execution Sandboxing

**Question**: How should protocol execution be sandboxed to prevent runaway computation and state corruption?

**Research**:
- Python's `sys.settrace()` can count execution steps but has significant overhead (10-100x slowdown)
- RestrictedPython provides AST-level sandboxing but is complex and may be overkill
- Simple approach: Protocol functions are pure Python callables with timeout via threading
- Step counting can be implemented via a context manager that wraps operations

**Decision**: Use a lightweight step-counting context manager that increments on each matchability/crossover call. Protocol functions receive a `StepCounter` that raises `StepLimitExceeded` when budget exhausted.

**Rationale**: 
- Simpler than full AST sandboxing
- No external dependencies
- Predictable performance overhead
- Easy to test and debug

**Alternatives rejected**:
- RestrictedPython: Too heavyweight for numeric/boolean operations
- Threading timeout: Non-deterministic, harder to reproduce
- sys.settrace: Unacceptable performance overhead

---

### 2. Protocol Genome Representation

**Question**: How should the Reproduction Protocol Genome (RPG) be structured to support evolvability and junk code?

**Research**:
- Genetic Programming (GP) trees allow variable-length expressions but are complex
- Cartesian Genetic Programming (CGP) uses fixed-size arrays with inactive nodes
- Simple parameterized approach: Fixed structure with evolvable parameters + activation flags

**Decision**: Use a hybrid approach:
- **Matchability**: Parameterized threshold functions (distance threshold, fitness ratio, etc.) with activation flags
- **Intent**: Parameterized policies (fitness threshold, budget counter) with activation flags  
- **Crossover**: Enum selector + parameters (crossover point, mixing rate)
- **Junk regions**: Additional inactive parameter slots that can be activated by mutation

**Rationale**:
- CGP-style inactive regions provide neutral drift capability
- Fixed structure is easier to inherit and mutate
- Parameters are continuous (easy gradient-free optimization)
- Activation flags enable dormant logic

**Alternatives rejected**:
- Full GP trees: Too complex for initial implementation; can add later
- Pure enum-based: No continuous parameters to evolve
- Neural network policies: Violates model-agnostic principle

---

### 3. Matchability Function Design

**Question**: What inputs should matchability functions receive, and what built-in functions should be provided?

**Research**:
- NEAT speciation uses genetic distance (compatibility distance)
- Assortative mating in biology uses phenotypic similarity
- Novelty search uses behavioral distance

**Decision**: Provide a `MateContext` dataclass with:
- `partner_distance`: Genetic distance to potential mate
- `partner_fitness_rank`: Rank in population (0 = best)
- `partner_fitness_ratio`: Partner fitness / self fitness
- `partner_niche_id`: Species/niche label (optional)
- `population_diversity`: Current population diversity metric
- `crowding_distance`: Multi-objective crowding (optional)

Built-in matchability functions:
1. `AcceptAll`: Always returns True
2. `DistanceThreshold`: Accept if distance > min_distance
3. `SimilarityThreshold`: Accept if distance < max_distance  
4. `FitnessRatioThreshold`: Accept if partner fitness ratio in range
5. `DifferentNiche`: Accept if different niche_id
6. `Probabilistic`: Accept with probability based on distance

**Rationale**: Covers speciation (distance), assortative mating (similarity), and diversity preservation (niche). All can be parameterized and combined.

---

### 4. Crossover Protocol Inheritance

**Question**: How should offspring inherit crossover protocols when parents differ?

**Research**:
- Biological: Single parent contributes mitochondria (asymmetric)
- GP: Subtree from one parent, structure from other
- Self-adaptation: Parameters inherited separately from strategy

**Decision**: Random single-parent inheritance (50/50) as default, configurable:
- Offspring gets complete protocol from randomly selected parent
- Protocol then undergoes mutation with configurable rate
- Future extension: Protocol recombination for complex protocols

**Rationale**: 
- Simplest approach that maintains protocol integrity
- Allows protocol selection pressure to operate
- Matches clarification decision from spec

---

### 5. Recovery Mechanism Design

**Question**: How should the system recover when all individuals reject each other?

**Research**:
- Island models use migration to restore connectivity
- Speciation collapse in NEAT is handled by species extinction thresholds
- Population re-initialization is a common fallback

**Decision**: Implement `ImmigrationRecovery`:
- Triggered when successful_matings == 0 for a generation
- Inject immigrants equal to 5-10% of population size (configurable)
- Immigrants have default "accept-all" matchability
- Immigrants' primary genomes initialized via configurable strategy (random, clone elite with mutation)

**Rationale**:
- Matches clarification decision from spec
- Accept-all protocols restore mating connectivity
- Preserves existing population (doesn't reset evolution)
- Configurable immigrant genome source allows research flexibility

---

### 6. Multi-Objective Integration

**Question**: How should ERP interact with NSGA-II style selection?

**Research**:
- NSGA-II uses binary tournament on rank + crowding distance
- Mating restriction in NSGA-II is not standard (random within selected pool)
- Reference-point based methods (NSGA-III) also use random mating

**Decision**: ERP operates as a mating filter AFTER selection:
1. Selection operator selects mating pool (as usual)
2. ERP filters mating pool by intent (who wants to reproduce)
3. For each reproduction attempt, both parties must pass matchability
4. If no compatible pairs found, use recovery mechanism

Crowding distance exposed via `MateContext.crowding_distance`.

**Rationale**:
- Selection remains authoritative (FR-023, FR-024)
- ERP adds individual-level mating preferences on top
- Clean separation of concerns
- Existing multi-objective infrastructure unchanged

---

### 7. Performance Considerations

**Question**: How can protocol evaluation overhead be kept under 20%?

**Research**:
- Current reproduction phase: ~10% of generation time (selection + crossover + mutation)
- Protocol evaluation adds: intent check, matchability check (2 per pair), crossover selection
- With 100 individuals and ~50 matings: ~200 protocol evaluations per generation

**Decision**: Optimizations:
1. Cache intent evaluations per individual per generation (evaluate once, reuse)
2. Batch matchability evaluations where possible
3. Use NumPy operations in built-in matchability functions
4. Profile and optimize hot paths after initial implementation

Target budget: 200 protocol evaluations * 50 microseconds each = 10ms overhead (acceptable for typical generation times of 50-500ms).

**Rationale**: Initial implementation prioritizes correctness; optimize based on profiling data per constitution.

---

## Summary of Decisions

| Topic | Decision |
|-------|----------|
| Sandboxing | Step-counting context manager with StepLimitExceeded exception |
| RPG Structure | Parameterized functions with activation flags (CGP-style) |
| Matchability Inputs | MateContext dataclass with distance, fitness, niche, crowding |
| Protocol Inheritance | Random single-parent (50/50) with subsequent mutation |
| Recovery | Immigration at 5-10% population with accept-all protocols |
| Multi-Objective | ERP as post-selection mating filter; crowding in MateContext |
| Performance | Cache intent, batch where possible, profile-driven optimization |
