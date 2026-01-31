# Quickstart: Evolvable Reproduction Protocols (ERP)

**Feature**: 002-evolvable-reproduction  
**Date**: January 28, 2026  
**Phase**: 1 - Design

## Overview

Evolvable Reproduction Protocols (ERP) allows individuals to encode their own mating strategies. Instead of a global crossover operator, each individual carries a **Reproduction Protocol Genome (RPG)** that determines:

1. **Who** they will mate with (Matchability)
2. **When** they attempt reproduction (Intent)
3. **How** offspring are created (Crossover)

## Basic Usage

### 1. Create Individuals with Protocols

```python
from evolve.core.types import Individual
from evolve.reproduction import ReproductionProtocol, MatchabilityFunction, CrossoverType

# Create an individual with a custom protocol
protocol = ReproductionProtocol(
    matchability=MatchabilityFunction(
        type="distance_threshold",
        params={"min_distance": 0.1},  # Only mate with sufficiently different partners
    ),
    intent=ReproductionIntentPolicy(
        type="always_willing",
        params={},
    ),
    crossover=CrossoverProtocolSpec(
        type=CrossoverType.UNIFORM,
        params={"swap_prob": 0.5},
    ),
)

individual = Individual(
    genome=my_genome,
    protocol=protocol,
)
```

### 2. Use ERPEngine Instead of EvolutionEngine

```python
from evolve.reproduction import ERPEngine, ERPConfig

config = ERPConfig(
    population_size=100,
    max_generations=500,
    step_limit=1000,  # Max steps per protocol evaluation
    protocol_mutation_rate=0.1,  # Rate of protocol mutations
    immigration_rate=0.05,  # 5% immigration on reproduction failure
)

engine = ERPEngine(
    config=config,
    evaluator=my_evaluator,
    selection=TournamentSelection(),
    protocol_mutator=DefaultProtocolMutator(),  # Evolves protocols
    seed=42,
)

result = engine.run(initial_population)
```

### 3. Initialize Population with Diverse Protocols

```python
from evolve.reproduction import ProtocolInitializer

# Random protocols for diversity
initializer = ProtocolInitializer(
    matchability_types=["accept_all", "distance_threshold", "similarity_threshold"],
    intent_types=["always_willing", "fitness_threshold"],
    crossover_types=[CrossoverType.SINGLE_POINT, CrossoverType.UNIFORM],
)

population = Population([
    Individual(
        genome=genome_initializer.create(rng),
        protocol=initializer.create(rng),
    )
    for _ in range(100)
])
```

## Key Concepts

### Matchability Functions

Matchability determines whether an individual accepts another as a mate. It's **asymmetric**: A may accept B while B rejects A.

| Type | Description | Parameters |
|------|-------------|------------|
| `accept_all` | Always accept | (none) |
| `distance_threshold` | Accept if genetically different | `min_distance` |
| `similarity_threshold` | Accept if genetically similar | `max_distance` |
| `fitness_ratio` | Accept based on relative fitness | `min_ratio`, `max_ratio` |
| `different_niche` | Accept if different species | (none) |
| `probabilistic` | Random acceptance | `base_prob`, `distance_weight` |

### Intent Policies

Intent determines when an individual attempts reproduction at all (before considering partners).

| Type | Description | Parameters |
|------|-------------|------------|
| `always_willing` | Always attempt | (none) |
| `fitness_threshold` | Only if fit enough | `threshold` |
| `resource_budget` | Limited offspring | `max_offspring` |
| `age_dependent` | Only at certain ages | `min_age`, `max_age` |

### Crossover Protocols

How offspring genomes are constructed when mating succeeds.

| Type | Description | Parameters |
|------|-------------|------------|
| `SINGLE_POINT` | Split at one point | `point_ratio` |
| `TWO_POINT` | Exchange middle segment | `point1_ratio`, `point2_ratio` |
| `UNIFORM` | Gene-by-gene mixing | `swap_prob` |
| `CLONE` | Copy single parent | (none) |

## Protocol Evolution

Protocols evolve alongside primary genomes:

1. **Inheritance**: Offspring inherit protocol from one parent (50/50 random)
2. **Mutation**: Protocol parameters mutate with configurable rate
3. **Selection**: Protocols that lead to fitter offspring spread

```python
# Custom protocol mutator
class MyProtocolMutator:
    def mutate(self, protocol: ReproductionProtocol, rng: Random) -> ReproductionProtocol:
        # Mutate matchability threshold
        new_params = dict(protocol.matchability.params)
        if "min_distance" in new_params:
            new_params["min_distance"] += rng.gauss(0, 0.1)
            new_params["min_distance"] = max(0, new_params["min_distance"])
        
        return ReproductionProtocol(
            matchability=MatchabilityFunction(
                type=protocol.matchability.type,
                params=new_params,
            ),
            intent=protocol.intent,  # unchanged
            crossover=protocol.crossover,  # unchanged
        )
```

## Safety Features

### Step Limits

All protocol execution is bounded:

```python
# Protocol evaluation automatically uses step counter
# Exceeding 1000 steps (default) causes safe failure

# Configure step limit in ERPConfig
config = ERPConfig(step_limit=500)  # Stricter limit
```

### Recovery on Failure

When no individuals can mate (all reject each other):

```python
# Automatic immigration with accept-all protocols
# Configurable via immigration_rate (default: 5-10% of population)

config = ERPConfig(
    immigration_rate=0.10,  # 10% immigration
    immigrant_protocol="accept_all",  # Immigrants accept everyone
)
```

## Multi-Objective Integration

ERP works with NSGA-II and other MOEAs:

```python
from evolve.multiobjective import NSGA2Selection

engine = ERPEngine(
    config=config,
    evaluator=multi_objective_evaluator,
    selection=NSGA2Selection(),  # Pareto-based selection
    # ERP adds mating preferences on top of selection
)

# Matchability can use crowding distance
protocol = ReproductionProtocol(
    matchability=MatchabilityFunction(
        type="diversity_seeking",  # Prefer partners with different crowding
        params={"crowding_weight": 0.5},
    ),
    ...
)
```

## Metrics and Observability

Track protocol evolution:

```python
from evolve.reproduction import ERPMetricsCallback

callback = ERPMetricsCallback()
result = engine.run(population, callbacks=[callback])

# Access protocol metrics
print(callback.acceptance_rate_history)  # Avg matchability acceptance per gen
print(callback.reproduction_success_rate)  # Successful matings / attempts
print(callback.protocol_diversity)  # Diversity of protocols in population
```

## Migration Path

### From Standard EvolutionEngine

```python
# Before: Standard evolution
engine = EvolutionEngine(
    config=config,
    evaluator=evaluator,
    selection=selection,
    crossover=UniformCrossover(),  # Global crossover
    mutation=mutation,
)

# After: ERP evolution
erp_config = ERPConfig.from_evolution_config(config)
erp_engine = ERPEngine(
    config=erp_config,
    evaluator=evaluator,
    selection=selection,
    # Crossover is now per-individual
)

# Existing populations work (default accept-all protocols assigned)
result = erp_engine.run(existing_population)
```
