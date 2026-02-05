# Evolvable Reproduction Protocols (ERP) - Best Practices Guide

This guide provides practical advice for using ERP effectively in your evolutionary algorithms.

## Table of Contents

1. [When to Use ERP](#when-to-use-erp)
2. [Getting Started](#getting-started)
3. [Configuration Guidelines](#configuration-guidelines)
4. [Common Pitfalls](#common-pitfalls)
5. [Performance Optimization](#performance-optimization)
6. [Debugging and Monitoring](#debugging-and-monitoring)
7. [Advanced Techniques](#advanced-techniques)

---

## When to Use ERP

### ✅ Good Use Cases

**ERP is ideal when:**
- Modeling biological phenomena (sexual selection, speciation, assortative mating)
- Exploring emergent mating strategies
- Co-evolutionary scenarios where reproductive strategies matter
- Research on self-organizing populations
- Problems where mate selection could provide useful diversity control

**Example domains:**
- Evolutionary biology research
- Multi-agent systems with reproduction
- Co-evolutionary game theory
- Artificial life simulations

### ❌ Not Recommended When

**ERP may be overkill if:**
- You're optimizing standard benchmark functions (Rastrigin, Rosenbrock, etc.)
- Speed is critical and you can't afford 10-20% overhead
- Population size is very small (<20 individuals)
- You need simple, explainable algorithms for production systems
- You're new to evolutionary algorithms (start with Tutorial 1 first!)

### Decision Matrix

| Your Goal | Use Standard EA | Use ERP |
|-----------|----------------|---------|
| Minimize sphere function | ✅ | ❌ |
| Model sexual selection dynamics | ❌ | ✅ |
| Hyperparameter optimization | ✅ | ❌ |
| Study speciation mechanisms | ❌ | ✅ |
| Fast convergence needed | ✅ | ❌ |
| Co-evolution with mating strategies | ❌ | ✅ |

---

## Getting Started

### Basic Setup

```python
from evolve.reproduction.engine import ERPEngine, ERPConfig
from evolve.reproduction.protocol import ReproductionProtocol
from evolve.reproduction.mutation import ProtocolMutator, MutationConfig

# 1. Configure ERP
erp_config = ERPConfig(
    population_size=50,
    max_generations=100,
    enable_recovery=False,  # Start simple
    protocol_mutation_rate=0.15
)

# 2. Configure protocol mutation
mutation_config = MutationConfig(
    param_mutation_rate=0.15,
    param_mutation_strength=0.1,
    type_mutation_rate=0.05
)
protocol_mutator = ProtocolMutator(config=mutation_config)

# 3. Create ERP engine
erp_engine = ERPEngine(
    config=erp_config,
    evaluator=your_evaluator,
    selection=TournamentSelection(tournament_size=3),
    crossover=UniformCrossover(swap_prob=0.5),
    mutation=GaussianMutation(mutation_rate=0.1, sigma=0.5),
    protocol_mutator=protocol_mutator,
    seed=42
)
```

### Initial Population

Individuals must have a `protocol` field:

```python
from evolve.reproduction.protocol import (
    ReproductionProtocol,
    ReproductionIntentPolicy,
    MatchabilityFunction,
    CrossoverProtocolSpec,
    CrossoverType,
)

# Create individual with protocol
protocol = ReproductionProtocol(
    intent=ReproductionIntentPolicy(type="always", params={}),
    matchability=MatchabilityFunction(
        type="fitness_threshold",
        params={"min_fitness": 0.5}
    ),
    crossover=CrossoverProtocolSpec(
        type=CrossoverType.UNIFORM,
        params={"swap_probability": 0.5}
    )
)

individual = Individual(
    genome=your_genome,
    fitness=None,
    protocol=protocol
)
```

---

## Configuration Guidelines

### Protocol Mutation Rate

**Recommended range: 0.1 - 0.2**

```python
# Conservative (stable protocols)
protocol_mutation_rate=0.05

# Balanced (default)
protocol_mutation_rate=0.15

# Exploratory (rapid protocol evolution)
protocol_mutation_rate=0.30
```

**Effects:**
- **Too low (<0.05)**: Protocols stagnate, diversity collapses
- **Too high (>0.3)**: Protocols unstable, good strategies disrupted
- **Sweet spot (0.1-0.2)**: Balance between exploration and exploitation

### Parameter Mutation Strength

**Recommended range: 0.05 - 0.15**

```python
mutation_config = MutationConfig(
    param_mutation_rate=0.15,
    param_mutation_strength=0.1,  # Controls magnitude of changes
    type_mutation_rate=0.05
)
```

**Interpretation:**
- `param_mutation_strength=0.1` means parameters change by ±10% on average
- Lower values = gradual protocol refinement
- Higher values = larger jumps in parameter space

### Matchability Thresholds

**For fitness_threshold:**
- Start with 0.3 - 0.5 for moderate selectivity
- Values > 0.7 risk population collapse without recovery
- Values < 0.2 provide minimal selection pressure

**For cosine_similarity (speciation):**
- 0.7 - 0.9 for strong assortative mating
- 0.5 - 0.7 for moderate clustering
- < 0.5 has weak effect

---

## Common Pitfalls

### 1. Population Collapse from Over-Selectivity

**Problem:** High matchability thresholds with no recovery → population dies

```python
# ❌ DANGEROUS: Will likely crash
matchability=MatchabilityFunction(
    type="fitness_threshold",
    params={"min_fitness": 0.9}  # Too high!
)
enable_recovery=False
```

**Solution:**
```python
# ✅ SAFE: Enable recovery or lower threshold
matchability=MatchabilityFunction(
    type="fitness_threshold",
    params={"min_fitness": 0.5}
)
enable_recovery=True  # Or use recovery
```

### 2. Ignoring Protocol Diversity

**Problem:** All individuals evolve identical protocols → no mating strategy diversity

```python
# Check protocol diversity
def check_protocol_diversity(population):
    intent_types = [ind.protocol.intent.type for ind in population.individuals]
    unique = len(set(intent_types))
    diversity = unique / len(intent_types)
    
    if diversity < 0.2:
        print("⚠️ WARNING: Low protocol diversity!")
```

**Solution:** Increase `protocol_mutation_rate` or use callbacks to maintain diversity.

### 3. Forgetting Mutual Consent

**Problem:** Assuming only one partner checks compatibility

**Important:** ERPEngine automatically enforces mutual consent:
- Both `A.matchability(B)` AND `B.matchability(A)` must return `True`
- You don't need to implement this yourself
- This is a core feature of ERP

### 4. Invalid Protocol Parameters After Mutation

**Problem:** Mutation creates threshold > 1.0 or swap_probability > 1.0

```python
# ❌ Risk: High mutation strength can create invalid values
param_mutation_strength=0.5  # Too high
```

**Solution:**
```python
# ✅ Use moderate mutation strength
param_mutation_strength=0.1  # Safe range

# Or implement parameter clamping in custom evaluators
```

### 5. Not Monitoring Population Health

**Problem:** Recovery keeps adding clones without notice → genetic diversity lost

```python
# ✅ Monitor population size
def check_population_health(population, initial_size):
    current_size = len(population.individuals)
    if current_size < 0.5 * initial_size:
        print(f"⚠️ Population crashed from {initial_size} to {current_size}")
        print("Consider adjusting matchability thresholds")
```

---

## Performance Optimization

### Computational Overhead

**Typical overhead: 10-20% compared to standard EA**

Factors affecting performance:
1. **Protocol complexity**: Simple protocols evaluate faster
2. **Population size**: Overhead scales with O(n²) for mate finding
3. **Matchability functions**: Fitness threshold is faster than cosine similarity
4. **Recovery mechanisms**: Immigration adds minimal cost

### Optimization Tips

```python
# 1. Use simpler matchability functions
matchability=MatchabilityFunction(
    type="fitness_threshold",  # Fast
    params={"min_fitness": 0.5}
)

# vs
matchability=MatchabilityFunction(
    type="cosine_similarity",  # Slower (computes similarity)
    params={"threshold": 0.7}
)

# 2. Lower protocol mutation rate if evolution is too slow
protocol_mutation_rate=0.1  # vs 0.2

# 3. Disable recovery if not needed
enable_recovery=False

# 4. Use smaller populations during prototyping
population_size=30  # vs 100
```

### Benchmarking

```python
import time

# Compare standard vs ERP
start = time.time()
standard_result = standard_engine.run(pop.copy())
standard_time = time.time() - start

start = time.time()
erp_result = erp_engine.run(pop.copy())
erp_time = time.time() - start

overhead_pct = ((erp_time / standard_time) - 1) * 100
print(f"ERP overhead: {overhead_pct:.1f}%")
```

---

## Debugging and Monitoring

### Health Check Function

```python
def erp_health_check(population, generation):
    """Comprehensive ERP health check."""
    print(f"\n=== ERP Health Report (Gen {generation}) ===")
    
    # 1. Population size
    pop_size = len(population.individuals)
    print(f"Population size: {pop_size}")
    if pop_size < 30:
        print("  ⚠️ WARNING: Low population")
    
    # 2. Protocol diversity
    protocols = [ind.protocol for ind in population.individuals]
    intent_types = [p.intent.type for p in protocols]
    unique_intents = len(set(intent_types))
    diversity = unique_intents / len(intent_types)
    print(f"Protocol diversity: {diversity:.2f}")
    if diversity < 0.2:
        print("  ⚠️ WARNING: Low diversity")
    
    # 3. Fitness distribution
    fitnesses = [ind.fitness for ind in population.individuals if ind.fitness]
    if fitnesses:
        print(f"Fitness: best={max(fitnesses):.2f}, mean={np.mean(fitnesses):.2f}")
    
    print("=" * 40)
```

### Using Callbacks

```python
from evolve.core.callbacks import HistoryCallback

class ERPMonitorCallback:
    def __init__(self):
        self.protocol_history = []
    
    def on_generation_end(self, engine, population, generation):
        # Track protocol distributions
        protocols = [ind.protocol for ind in population.individuals]
        intent_dist = {}
        for p in protocols:
            intent_type = p.intent.type
            intent_dist[intent_type] = intent_dist.get(intent_type, 0) + 1
        
        self.protocol_history.append({
            'generation': generation,
            'intent_distribution': intent_dist
        })
        
        # Alert on issues
        if generation % 10 == 0:
            erp_health_check(population, generation)

# Use callback
monitor = ERPMonitorCallback()
result = erp_engine.run(population, callbacks=[monitor])
```

---

## Advanced Techniques

### Custom Matchability Evaluators

```python
from evolve.reproduction.matchability import MatchabilityEvaluator
from evolve.reproduction.protocol import MateContext
from random import Random
from typing import Protocol, runtime_checkable

@runtime_checkable
class DiversityMatchability(MatchabilityEvaluator, Protocol):
    """Prefer mates with different genotypes."""
    
    min_distance: float = 2.0
    
    def evaluate(self, context: MateContext, rng: Random) -> bool:
        self_genes = context.self_individual.genome.genes
        partner_genes = context.partner_individual.genome.genes
        
        distance = np.linalg.norm(self_genes - partner_genes)
        return distance > self.min_distance
```

### Combining ERP with Multi-Objective

```python
from evolve.multiobjective.selection import NSGA2Selection

# Use Pareto selection with ERP
erp_engine = ERPEngine(
    config=erp_config,
    evaluator=multi_objective_evaluator,
    selection=NSGA2Selection(),  # Pareto-based
    crossover=UniformCrossover(swap_prob=0.5),
    mutation=GaussianMutation(mutation_rate=0.1, sigma=0.5),
    protocol_mutator=protocol_mutator,
    seed=42
)
```

**Key insight:** NSGA-II selects *which* individuals reproduce based on Pareto optimality, while ERP controls *how* they mate (protocol-driven consent).

### Analyzing Reproductive Skew

```python
def compute_reproductive_skew(mating_events):
    """Compute Gini coefficient for mate count distribution."""
    mate_counts = {}
    for event in mating_events:
        mate_counts[event.parent1_id] = mate_counts.get(event.parent1_id, 0) + 1
        mate_counts[event.parent2_id] = mate_counts.get(event.parent2_id, 0) + 1
    
    counts = np.array(list(mate_counts.values()))
    n = len(counts)
    
    # Gini coefficient
    numerator = np.sum(np.abs(counts[:, None] - counts[None, :]))
    denominator = 2 * n * np.sum(counts)
    
    return numerator / denominator

# Interpret:
# 0.0 = perfect equality (all mate equally)
# 1.0 = perfect inequality (one individual monopolizes mating)
```

---

## Related Resources

- **Tutorial 6**: [Evolvable Reproduction Protocols](../tutorials/06_evolvable_reproduction_protocols.ipynb)
- **ADR 002**: [ERP Extensibility Design](../adr/002-erp-extensibility.md)
- **API Reference**: [evolve.reproduction](../api/reproduction.md)
- **Examples**:
  - [examples/sexual_selection.py](../../examples/sexual_selection.py)
  - [examples/speciation_demo.py](../../examples/speciation_demo.py)
  - [examples/protocol_evolution.py](../../examples/protocol_evolution.py)

---

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share experiences
- **Contributing**: See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines

---

*Last updated: February 2026*
