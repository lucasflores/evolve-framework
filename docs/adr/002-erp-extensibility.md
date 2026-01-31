# ADR-002: ERP Extensibility Design

**Status**: Accepted  
**Date**: 2026-01-29  
**Context**: Evolvable Reproduction Protocols (ERP) Feature

## Summary

This Architecture Decision Record documents how the ERP implementation satisfies extensibility requirements FR-026, FR-027, and FR-028, enabling future integration with RL-trained policies, memetic local search, and island model migration without requiring core refactoring.

## Motivation

The ERP system must be designed for extensibility to support advanced evolutionary computation techniques:

1. **RL-trained policies (FR-026)**: Matchability decisions could benefit from learned policies trained via reinforcement learning
2. **Memetic local search (FR-027)**: Offspring could benefit from local optimization after crossover
3. **Protocol-aware migration (FR-028)**: Island models should be able to migrate individuals while preserving their reproductive protocols

## Decision

### FR-026: RL-Trained Matchability Policies

**Requirement**: Design MUST allow future addition of RL-trained policies without core refactoring

**Verification**: ✅ SATISFIED

The `MatchabilityEvaluator` protocol interface is designed to support arbitrary evaluation logic:

```python
@runtime_checkable
class MatchabilityEvaluator(Protocol):
    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> bool | float:
        ...
```

**Extension Point**: An RL-trained policy can be implemented by:

```python
class RLTrainedMatchability:
    """Matchability evaluator using RL-trained policy network."""
    
    def __init__(self, model_path: str):
        self.policy_network = load_policy(model_path)
    
    def evaluate(
        self,
        context: MateContext,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> float:
        counter.step()
        
        # Extract features from MateContext
        features = np.array([
            context.partner_distance,
            context.partner_fitness_ratio,
            context.population_diversity,
            float(context.partner_fitness_rank) / 100.0,
        ])
        
        # Policy network returns acceptance probability
        counter.step(10)  # Account for inference steps
        return float(self.policy_network(features))

# Register the RL evaluator
MatchabilityRegistry.register("rl_trained", RLTrainedMatchability("path/to/model"))
```

**Why It Works**:
- The `MateContext` provides all necessary state information (fitness, distance, diversity, ranks)
- The `params` dictionary can store model configuration or hyperparameters
- The `StepCounter` ensures bounded execution even with neural inference
- The registry pattern allows runtime addition without code changes
- Probabilistic output (float) is natively supported

### FR-027: Memetic Local Search Post-Crossover

**Requirement**: Design MUST allow future addition of memetic local search post-crossover

**Verification**: ✅ SATISFIED

The `CrossoverExecutor` protocol interface can be extended to include local search:

```python
@runtime_checkable
class CrossoverExecutor(Protocol[G]):
    def execute(
        self,
        parent1: G,
        parent2: G,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[G, G]:
        ...
```

**Extension Point**: A memetic crossover executor can apply local search after genetic crossover:

```python
class MemeticCrossoverExecutor:
    """Crossover followed by local search optimization."""
    
    def __init__(
        self,
        base_crossover: CrossoverExecutor,
        local_search: Callable[[np.ndarray, int], np.ndarray],
        search_steps: int = 10,
    ):
        self.base_crossover = base_crossover
        self.local_search = local_search
        self.search_steps = search_steps
    
    def execute(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        params: dict[str, float],
        rng: Random,
        counter: StepCounter,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Step 1: Standard genetic crossover
        child1, child2 = self.base_crossover.execute(
            parent1, parent2, params, rng, counter
        )
        
        # Step 2: Local search refinement
        steps = int(params.get("search_steps", self.search_steps))
        counter.step(steps * 2)  # Account for local search
        
        child1 = self.local_search(child1, steps)
        child2 = self.local_search(child2, steps)
        
        return child1, child2

# Register the memetic executor
CrossoverRegistry.register(
    CrossoverType.SINGLE_POINT,  # Override or add new type
    MemeticCrossoverExecutor(
        base_crossover=SinglePointCrossoverExecutor(),
        local_search=gradient_descent_step,
        search_steps=10,
    )
)
```

**Why It Works**:
- The `params` dictionary can control local search intensity (e.g., `search_steps`, `learning_rate`)
- The `StepCounter` bounds total computation including local search
- Generic type `G` supports any genome representation
- Composition pattern allows wrapping any base crossover
- New `CrossoverType` enum values can be added for memetic variants

**Alternative Extension via Engine Hook**:

The `ERPEngine` can also be subclassed to add post-crossover processing:

```python
class MemeticERPEngine(ERPEngine[G]):
    def __init__(self, *args, local_optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_optimizer = local_optimizer
    
    def _attempt_mating(self, parent1, parent2, population):
        offspring = super()._attempt_mating(parent1, parent2, population)
        
        if self.local_optimizer and offspring:
            for child in offspring:
                child.genome = self.local_optimizer(child.genome)
        
        return offspring
```

### FR-028: Protocol-Aware Migration in Island Models

**Requirement**: Design MUST allow protocol-aware migration in island models

**Verification**: ✅ SATISFIED

The `ReproductionProtocol` dataclass supports full serialization:

```python
@dataclass(frozen=True)
class ReproductionProtocol:
    matchability: MatchabilityFunction
    intent: ReproductionIntentPolicy
    crossover: CrossoverProtocolSpec
    junk_data: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "matchability": self.matchability.to_dict(),
            "intent": self.intent.to_dict(),
            "crossover": self.crossover.to_dict(),
            "junk_data": dict(self.junk_data),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionProtocol:
        """Deserialize from dictionary."""
        ...
```

**Extension Point**: Protocol-aware migration can be implemented:

```python
from evolve.diversity.islands import MigrationPolicy
from evolve.reproduction.protocol import ReproductionProtocol

class ProtocolAwareMigration(MigrationPolicy):
    """Migration policy that considers reproductive compatibility."""
    
    def __init__(
        self,
        compatibility_threshold: float = 0.5,
        prefer_compatible: bool = True,
    ):
        self.compatibility_threshold = compatibility_threshold
        self.prefer_compatible = prefer_compatible
    
    def select_migrants(
        self,
        source_island: Island,
        target_island: Island,
        n_migrants: int,
    ) -> list[Individual]:
        """Select migrants considering protocol compatibility."""
        candidates = source_island.population.best(n_migrants * 3)
        
        # Score candidates by compatibility with target population
        scored = []
        for candidate in candidates:
            compatibility = self._estimate_compatibility(
                candidate.protocol,
                target_island.population
            )
            scored.append((compatibility, candidate))
        
        # Select based on preference
        scored.sort(reverse=self.prefer_compatible)
        return [ind for _, ind in scored[:n_migrants]]
    
    def _estimate_compatibility(
        self,
        protocol: ReproductionProtocol,
        population: Population,
    ) -> float:
        """Estimate how compatible a protocol is with a population."""
        # Sample population and check matchability
        sample = population.sample(min(10, len(population)))
        
        compatible = 0
        for ind in sample:
            # Would the migrant accept this individual?
            context = build_mate_context(protocol, ind)
            result, _ = safe_evaluate_matchability(
                protocol.matchability, context, Random()
            )
            if result:
                compatible += 1
        
        return compatible / len(sample) if sample else 0.0
    
    def serialize_migrant(self, individual: Individual) -> dict:
        """Serialize individual including protocol for transfer."""
        return {
            "genome": individual.genome.tolist(),
            "fitness": individual.fitness.to_dict() if individual.fitness else None,
            "protocol": individual.protocol.to_dict() if individual.protocol else None,
            "metadata": {
                "origin_island": individual.metadata.origin,
                "created_at": individual.created_at,
            }
        }
    
    def deserialize_migrant(self, data: dict) -> Individual:
        """Deserialize individual including protocol from transfer."""
        protocol = None
        if data.get("protocol"):
            protocol = ReproductionProtocol.from_dict(data["protocol"])
        
        return Individual(
            id=uuid4(),
            genome=np.array(data["genome"]),
            protocol=protocol,
            fitness=Fitness.from_dict(data["fitness"]) if data.get("fitness") else None,
            metadata=IndividualMetadata(origin="migration"),
        )
```

**Why It Works**:
- `to_dict()` / `from_dict()` provide complete serialization
- Protocol components (matchability, intent, crossover) are independently serializable
- `junk_data` preserves dormant genetic material during migration
- The `Individual` dataclass includes optional `protocol` field
- Migration policies can access protocol information for selection decisions

## Extension Registry Pattern

All three extension points use a consistent registry pattern:

```python
# Matchability
MatchabilityRegistry.register("custom_type", CustomMatchabilityEvaluator())
evaluator = MatchabilityRegistry.get("custom_type")

# Intent
IntentRegistry.register("custom_intent", CustomIntentEvaluator())
evaluator = IntentRegistry.get("custom_intent")

# Crossover
CrossoverRegistry.register(CrossoverType.CUSTOM, CustomCrossoverExecutor())
executor = CrossoverRegistry.get(CrossoverType.CUSTOM)
```

This pattern enables:
- Runtime registration without code modification
- Plugin architectures for user-defined components
- Testing with mock implementations
- Gradual migration to new implementations

## Consequences

### Positive

1. **No core refactoring needed**: All three requirements can be implemented via the existing protocol interfaces
2. **Backward compatible**: Existing code continues to work unchanged
3. **Composable**: Components can be combined (e.g., RL matchability + memetic crossover)
4. **Testable**: Each extension can be unit tested in isolation
5. **Serialization-safe**: All components support checkpointing and migration

### Negative

1. **Step counting overhead**: Complex extensions must carefully track step usage
2. **Registry coordination**: Distributed systems must ensure consistent registrations
3. **Type safety**: Generic `params` dict loses some type safety (mitigated by documentation)

### Neutral

1. **Performance**: Extension complexity affects evaluation time (bounded by step limits)
2. **Memory**: RL models may require additional memory (user responsibility)

## Related Decisions

- ADR-001: Core Framework Architecture (provides base `Individual` and `Population` types)
- Spec 002: ERP Feature Specification (defines requirements FR-026, FR-027, FR-028)

## Verification Summary

| Requirement | Status | Extension Point |
|-------------|--------|-----------------|
| FR-026 (RL policies) | ✅ Satisfied | `MatchabilityEvaluator` protocol + registry |
| FR-027 (Memetic search) | ✅ Satisfied | `CrossoverExecutor` protocol + composition |
| FR-028 (Protocol migration) | ✅ Satisfied | `to_dict()` / `from_dict()` serialization |

All extensibility requirements are satisfied by the current interface design.
