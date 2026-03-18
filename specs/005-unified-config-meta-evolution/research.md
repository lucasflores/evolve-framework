# Research: Unified Configuration & Meta-Evolution Framework

**Date**: March 12, 2026  
**Branch**: `005-unified-config-meta-evolution`

## Research Tasks

This document resolves all "NEEDS CLARIFICATION" items and documents best practices research for implementing the unified configuration and meta-evolution system.

---

## 1. Registry Pattern Best Practices

### Decision
Use a lazy-initialized singleton registry with explicit accessor functions (e.g., `get_operator_registry()`).

### Rationale
- **Lazy initialization**: Avoids import-time side effects; built-in operators registered on first access
- **Singleton via module**: Use module-level instance with accessor function rather than class-level singleton (more Pythonic)
- **Explicit access**: `get_operator_registry()` is explicit and testable vs. importing a global instance

### Alternatives Considered
1. **Class-level singleton**: Rejected because it complicates testing (hard to reset state)
2. **Import-time registration**: Rejected because it creates import order dependencies
3. **Dependency injection only**: Rejected because it requires passing registries everywhere, increasing boilerplate

### Implementation Pattern
```python
# In evolve/registry/operators.py
_operator_registry: OperatorRegistry | None = None

def get_operator_registry() -> OperatorRegistry:
    """Get or create the global operator registry."""
    global _operator_registry
    if _operator_registry is None:
        _operator_registry = OperatorRegistry()
        _register_builtin_operators(_operator_registry)
    return _operator_registry
```

---

## 2. Configuration Schema Design

### Decision
Use a frozen dataclass with nested dataclasses for subsections, supporting JSON serialization via `to_dict()`/`from_dict()` methods.

### Rationale
- **Frozen dataclass**: Ensures immutability after creation; hashable for caching
- **Nested structure**: Groups related config (e.g., `stopping: StoppingConfig`) for clarity
- **Dict-based serialization**: More flexible than JSON Schema; allows incremental validation

### Schema Versioning Strategy
```python
@dataclass(frozen=True)
class UnifiedConfig:
    schema_version: str = "1.0.0"  # Semantic versioning
    # ... other fields
    
    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedConfig":
        version = data.get("schema_version", "1.0.0")
        if not is_compatible_version(version, CURRENT_SCHEMA_VERSION):
            raise IncompatibleSchemaError(version, CURRENT_SCHEMA_VERSION)
        return cls(**migrate_if_needed(data, version))
```

### Alternatives Considered
1. **JSON Schema validation**: Rejected because it adds external dependency and is verbose
2. **Pydantic models**: Rejected to avoid new runtime dependency
3. **YAML primary format**: Rejected because JSON is more portable and safer

---

## 3. Operator Compatibility Tracking

### Decision
Store compatibility as metadata on registration; validate at factory build time.

### Rationale
- **Registration metadata**: Simple dictionary mapping `{operator_name: set[genome_type]}`
- **Build-time validation**: Catches errors early without over-constraining registration
- **Wildcard support**: Some operators work with any genome; mark with `{"*"}`

### Implementation Pattern
```python
class OperatorRegistry:
    def __init__(self):
        self._operators: dict[tuple[str, str], type] = {}
        self._compatibility: dict[str, set[str]] = {}
    
    def register(
        self,
        category: str,
        name: str,
        cls: type,
        compatible_genomes: set[str] | None = None,
    ):
        """Register operator with optional genome compatibility."""
        self._operators[(category, name)] = cls
        if compatible_genomes is not None:
            self._compatibility[name] = compatible_genomes
```

---

## 4. Stopping Criteria Composition

### Decision
Stopping criteria are specified as a list; any condition triggers stop (OR semantics).

### Rationale
- **Simple OR composition**: Covers typical use cases (stop at generation limit OR fitness target)
- **Flat list structure**: Easier to serialize than nested AND/OR trees
- **Extensible**: Can add AND semantics later via nested "all_of" condition type

### Configuration Structure
```python
@dataclass(frozen=True)
class StoppingConfig:
    max_generations: int | None = None
    fitness_threshold: float | None = None
    stagnation_generations: int | None = None  # No improvement for N gens
    time_limit_seconds: float | None = None
    
    # Any non-None condition triggers stop when met
```

---

## 5. Multi-Objective Configuration

### Decision
Objectives specified as list of `ObjectiveSpec` with name, direction, and optional weight for scalarization.

### Rationale
- **Explicit direction per objective**: Clearer than global `minimize` flag
- **Reference point in config**: Required for hypervolume calculation
- **Constraint as objective**: Constraint violations add implicit minimization objective

### Configuration Structure
```python
@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: Literal["minimize", "maximize"]
    weight: float = 1.0  # For weighted sum scalarization
    
@dataclass(frozen=True)
class MultiObjectiveConfig:
    objectives: tuple[ObjectiveSpec, ...]
    reference_point: tuple[float, ...] | None = None  # For hypervolume
    constraint_handling: Literal["dominance", "penalty"] = "dominance"
```

---

## 6. Meta-Evolution Configuration Codec

### Decision
Use a `ParameterSpec` list defining evolvable parameters, their bounds, and encoding strategy.

### Rationale
- **Explicit bounds**: Required for continuous parameters; prevents invalid configs
- **Categorical as index**: Map categorical choices to integer indices, then to floats [0, N-1]
- **Base config inheritance**: Fixed parameters come from base config; codec only encodes evolvables

### Configuration Structure
```python
@dataclass(frozen=True)
class ParameterSpec:
    path: str  # Dot-notation path, e.g., "mutation_params.sigma"
    param_type: Literal["continuous", "categorical", "integer"]
    bounds: tuple[float, float] | None = None  # For continuous/integer
    choices: tuple[Any, ...] | None = None  # For categorical
    
@dataclass(frozen=True)
class MetaEvolutionConfig:
    base_config: UnifiedConfig
    evolvable_params: tuple[ParameterSpec, ...]
    outer_population_size: int = 20
    outer_generations: int = 10
    trials_per_config: int = 1
    aggregation: Literal["mean", "median", "best"] = "mean"
```

---

## 7. Callback Configuration

### Decision
Built-in callbacks configurable via flags and parameters; custom callbacks passed separately to factory.

### Rationale
- **Separation of declarative/imperative**: Config captures what can be JSON-serialized
- **Custom callbacks need code**: Can't serialize arbitrary Python callables safely
- **Built-in set is limited**: Logging, checkpointing, early stopping cover most needs

### Configuration Structure
```python
@dataclass(frozen=True)
class CallbackConfig:
    enable_logging: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    
    enable_checkpointing: bool = False
    checkpoint_dir: str | None = None
    checkpoint_frequency: int = 10  # Every N generations
    
    # Early stopping is handled via StoppingConfig, not callbacks
```

---

## 8. Factory Function Design

### Decision
Single `create_engine()` function that inspects config and builds appropriate engine type.

### Rationale
- **Single entry point**: Researcher uses one function regardless of experiment type
- **Config-driven dispatch**: `erp_enabled` flag determines engine type
- **Composable with defaults**: Registries provide defaults; config can override

### Function Signature
```python
def create_engine(
    config: UnifiedConfig,
    evaluator: Evaluator | Callable,  # Accept either
    seed: int | None = None,  # Override config seed if provided
    callbacks: Sequence[Callback] | None = None,  # Custom callbacks
) -> EvolutionEngine | ERPEngine:
    """Create engine from unified configuration."""
```

---

## 9. Existing Framework Patterns Reviewed

### DEAP (Distributed Evolutionary Algorithms in Python)
- Uses `toolbox` pattern for operator registration
- Registration is explicit: `toolbox.register("select", tools.selTournament, tournsize=3)`
- Insight: String-based registration works well; parameters at registration time

### inspyred
- Uses decorator-based candidate generation
- Configuration via constructor parameters
- Insight: Separating "what" (operators) from "how" (parameters) is useful

### PyGMO (Pagmo)
- Problem-centric design; algorithms separate
- JSON serialization via `to_json()` methods
- Insight: Immutable problem/algorithm specs aid reproducibility

---

## 10. Inner Loop Seeding for Meta-Evolution

### Decision
Seed inner runs using hash of (config_hash, trial_index).

### Rationale
- **Deterministic**: Same config+trial always produces same result
- **Independent**: Different configs get different seeds even with same trial number
- **Reproducible**: Re-running meta-evolution produces identical inner runs

### Implementation Pattern
```python
def compute_inner_seed(config: UnifiedConfig, trial: int) -> int:
    """Compute deterministic seed for inner evolution run."""
    config_hash = hash(config)  # Uses config's __hash__
    combined = f"{config_hash}:{trial}"
    return int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
```

---

## Summary of Decisions

| Topic | Decision |
|-------|----------|
| Registry pattern | Lazy singleton with accessor function |
| Config schema | Frozen dataclass with `to_dict()`/`from_dict()` |
| Operator compatibility | Metadata on registration; build-time validation |
| Stopping criteria | OR composition via flat list |
| Multi-objective | `ObjectiveSpec` list with per-objective direction |
| Meta-evolution codec | `ParameterSpec` list with explicit bounds |
| Callbacks | Built-ins configurable; custom via factory param |
| Factory design | Single `create_engine()` with config-driven dispatch |
| Inner seeding | Hash of (config_hash, trial_index) |

All "NEEDS CLARIFICATION" items from Technical Context have been resolved.
