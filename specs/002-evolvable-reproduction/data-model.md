# Data Model: Evolvable Reproduction Protocols (ERP)

**Feature**: 002-evolvable-reproduction  
**Date**: January 28, 2026  
**Phase**: 1 - Design

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Individual[G]                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ id: UUID                                                           │ │
│  │ genome: G (primary problem genome)                                 │ │
│  │ fitness: Fitness | None                                            │ │
│  │ metadata: IndividualMetadata                                       │ │
│  │ protocol: ReproductionProtocol | None  ←── NEW (optional RPG)      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ has-a (optional)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ReproductionProtocol                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ matchability: MatchabilityFunction                                 │ │
│  │ intent: ReproductionIntentPolicy                                   │ │
│  │ crossover: CrossoverProtocolSpec                                   │ │
│  │ junk_data: dict[str, Any]  ←── inactive/dormant parameters         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│MatchabilityFunc  │  │IntentPolicy      │  │CrossoverProtocolSpec     │
├──────────────────┤  ├──────────────────┤  ├──────────────────────────┤
│ type: str        │  │ type: str        │  │ type: CrossoverType      │
│ params: dict     │  │ params: dict     │  │ params: dict             │
│ active: bool     │  │ active: bool     │  │ active: bool             │
└──────────────────┘  └──────────────────┘  └──────────────────────────┘
```

## Core Entities

### ReproductionProtocol

The evolvable genome component that governs reproduction behavior.

```python
@dataclass(frozen=True)
class ReproductionProtocol:
    """
    Evolvable reproduction protocol attached to an individual.
    
    This is the Reproduction Protocol Genome (RPG) that encodes:
    - Matchability: Who is an acceptable mate
    - Intent: When to attempt reproduction
    - Crossover: How to combine genetic material
    
    Attributes:
        matchability: Function determining mate acceptability
        intent: Policy determining reproduction willingness
        crossover: Specification for offspring genome construction
        junk_data: Inactive parameters for neutral drift
    """
    matchability: MatchabilityFunction
    intent: ReproductionIntentPolicy
    crossover: CrossoverProtocolSpec
    junk_data: dict[str, Any] = field(default_factory=dict)
```

**Validation Rules**:
- All components must be serializable (for checkpointing)
- junk_data keys must be strings; values must be JSON-serializable
- Protocol is immutable (frozen dataclass)

**State Transitions**: None (immutable)

---

### MatchabilityFunction

Determines whether another individual is an acceptable mating partner.

```python
@dataclass(frozen=True)
class MatchabilityFunction:
    """
    Evolvable function determining mate acceptability.
    
    Attributes:
        type: Function type identifier (e.g., "accept_all", "distance_threshold")
        params: Type-specific parameters
        active: Whether this function is active (inactive = always reject)
    """
    type: str
    params: dict[str, float]
    active: bool = True
```

**Supported Types**:

| Type | Parameters | Behavior |
|------|------------|----------|
| `accept_all` | (none) | Always accept |
| `reject_all` | (none) | Always reject |
| `distance_threshold` | `min_distance: float` | Accept if distance > min |
| `similarity_threshold` | `max_distance: float` | Accept if distance < max |
| `fitness_ratio` | `min_ratio: float, max_ratio: float` | Accept if ratio in range |
| `different_niche` | (none) | Accept if different niche_id |
| `probabilistic` | `base_prob: float, distance_weight: float` | Probability based on distance |

**Validation Rules**:
- type must be a registered matchability function
- params must contain all required parameters for the type
- Probability parameters must be in [0, 1]
- Distance parameters must be non-negative

---

### ReproductionIntentPolicy

Determines when an individual attempts to reproduce.

```python
@dataclass(frozen=True)
class ReproductionIntentPolicy:
    """
    Evolvable policy determining reproduction willingness.
    
    Attributes:
        type: Policy type identifier
        params: Type-specific parameters
        active: Whether policy is active (inactive = always willing)
    """
    type: str
    params: dict[str, float]
    active: bool = True
```

**Supported Types**:

| Type | Parameters | Behavior |
|------|------------|----------|
| `always_willing` | (none) | Always attempt reproduction |
| `never_willing` | (none) | Never attempt reproduction |
| `fitness_threshold` | `threshold: float` | Willing if fitness >= threshold |
| `fitness_rank_threshold` | `max_rank: int` | Willing if rank <= max_rank |
| `resource_budget` | `max_offspring: int` | Willing until budget exhausted |
| `age_dependent` | `min_age: int, max_age: int` | Willing if age in range |
| `probabilistic` | `probability: float` | Willing with fixed probability |

**State**: `resource_budget` requires per-generation state tracking (offspring count).

---

### CrossoverProtocolSpec

Specifies how offspring genomes are constructed.

```python
class CrossoverType(Enum):
    """Supported crossover strategies."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLEND = "blend"
    MODULE_EXCHANGE = "module_exchange"
    CLONE = "clone"  # No-op fallback

@dataclass(frozen=True)
class CrossoverProtocolSpec:
    """
    Evolvable specification for offspring genome construction.
    
    Attributes:
        type: Crossover strategy to use
        params: Type-specific parameters
        active: Whether protocol is active (inactive = clone)
    """
    type: CrossoverType
    params: dict[str, float]
    active: bool = True
```

**Type-Specific Parameters**:

| Type | Parameters |
|------|------------|
| `SINGLE_POINT` | `point_ratio: float` (0-1, relative position) |
| `TWO_POINT` | `point1_ratio: float, point2_ratio: float` |
| `UNIFORM` | `swap_prob: float` |
| `BLEND` | `alpha: float` (BLX-alpha parameter) |
| `MODULE_EXCHANGE` | `module_prob: float` (for graph genomes) |
| `CLONE` | (none) |

---

### MateContext

Context provided to matchability functions during evaluation.

```python
@dataclass(frozen=True)
class MateContext:
    """
    Context for matchability evaluation.
    
    Provides information about potential mate and population state.
    All fields are read-only to prevent side effects.
    
    Attributes:
        partner_distance: Genetic distance to potential mate
        partner_fitness_rank: Partner's rank (0 = best)
        partner_fitness_ratio: Partner fitness / self fitness
        partner_niche_id: Partner's species/niche label (None if N/A)
        population_diversity: Current population diversity (0-1)
        crowding_distance: Multi-objective crowding (None if single-obj)
        self_fitness: Own fitness value(s)
        partner_fitness: Partner's fitness value(s)
    """
    partner_distance: float
    partner_fitness_rank: int
    partner_fitness_ratio: float
    partner_niche_id: int | None
    population_diversity: float
    crowding_distance: float | None
    self_fitness: np.ndarray
    partner_fitness: np.ndarray
```

---

### IntentContext

Context provided to intent policies during evaluation.

```python
@dataclass(frozen=True)
class IntentContext:
    """
    Context for intent policy evaluation.
    
    Attributes:
        fitness: Own fitness value(s)
        fitness_rank: Own rank in population (0 = best)
        age: Generations survived
        offspring_count: Offspring produced this generation
        generation: Current generation number
        population_size: Current population size
    """
    fitness: np.ndarray
    fitness_rank: int
    age: int
    offspring_count: int
    generation: int
    population_size: int
```

---

### StepCounter

Resource limiter for sandboxed protocol execution.

```python
@dataclass
class StepCounter:
    """
    Counts execution steps and enforces limits.
    
    Attributes:
        limit: Maximum steps allowed
        count: Current step count
    """
    limit: int = 1000
    count: int = 0
    
    def step(self, n: int = 1) -> None:
        """Increment counter; raise if limit exceeded."""
        self.count += n
        if self.count > self.limit:
            raise StepLimitExceeded(self.count, self.limit)
    
    def reset(self) -> None:
        """Reset counter for reuse."""
        self.count = 0
```

---

### ReproductionEvent

Record of a reproduction attempt (for observability).

```python
@dataclass(frozen=True)
class ReproductionEvent:
    """
    Record of a reproduction attempt.
    
    Emitted for logging and metrics collection.
    
    Attributes:
        generation: When the event occurred
        parent1_id: First parent UUID
        parent2_id: Second parent UUID
        success: Whether reproduction produced offspring
        failure_reason: Why reproduction failed (if applicable)
        offspring_ids: UUIDs of produced offspring (if successful)
        matchability_result: Result of matchability evaluation
        intent_result: Result of intent evaluation
    """
    generation: int
    parent1_id: UUID
    parent2_id: UUID
    success: bool
    failure_reason: str | None
    offspring_ids: tuple[UUID, ...] | None
    matchability_result: tuple[bool, bool]  # (parent1_accepts, parent2_accepts)
    intent_result: tuple[bool, bool]  # (parent1_willing, parent2_willing)
```

---

## Relationships

```
Individual ─────────────── 0..1 ──────────────→ ReproductionProtocol
     │                                                  │
     │                                                  ├── MatchabilityFunction
     │                                                  ├── ReproductionIntentPolicy
     │                                                  └── CrossoverProtocolSpec
     │
     └── fitness ─────────────→ Fitness (existing)

Population ────────────── 1..* ──────────────→ Individual

ReproductionEvent ─────── 2 ──────────────────→ Individual (parents)
ReproductionEvent ─────── 0..* ───────────────→ Individual (offspring)
```

## Default Protocol

When an individual has no explicit protocol (`protocol=None`), the system uses:

```python
DEFAULT_PROTOCOL = ReproductionProtocol(
    matchability=MatchabilityFunction(type="accept_all", params={}, active=True),
    intent=ReproductionIntentPolicy(type="always_willing", params={}, active=True),
    crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT, params={"point_ratio": 0.5}, active=True),
    junk_data={},
)
```

This ensures backward compatibility with existing evolution runs.
