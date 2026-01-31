# Quickstart: SCM Representation for Causal Discovery

**Feature**: 003-scm-representation  
**Date**: 2026-01-30  
**Phase**: 1 - Design

## Overview

The SCM representation enables evolutionary discovery of Structural Causal Models from observational data. This guide demonstrates basic usage patterns.

---

## 1. Basic Genome Creation

```python
from random import Random
from evolve.representation.scm import SCMConfig, SCMGenome, SCMAlphabet

# Define observed variables from your dataset
config = SCMConfig(
    observed_variables=("A", "B", "C", "D"),
    max_latent_variables=2,
)

# View the generated alphabet
alphabet = SCMAlphabet.from_config(config)
print(f"Variables: {alphabet.variable_refs}")
# Variables: frozenset({'A', 'B', 'C', 'D', 'H1', 'H2'})

print(f"Store genes: {alphabet.store_genes}")  
# Store genes: frozenset({'STORE_A', 'STORE_B', 'STORE_C', 'STORE_D', 'STORE_H1', 'STORE_H2'})

# Create a random genome
rng = Random(42)
genome = SCMGenome.random(config, length=50, rng=rng)
print(f"Genome length: {len(genome.genes)}")
# Genome length: 50
```

---

## 2. Decoding Genomes to Causal Models

```python
from evolve.representation.scm_decoder import SCMDecoder

# Create decoder
decoder = SCMDecoder(config)

# Decode genome to SCM
scm = decoder.decode(genome)

# Inspect the decoded model
print(f"Equations: {len(scm.equations)}")
print(f"Graph edges: {scm.edge_count}")
print(f"Is cyclic: {scm.is_cyclic}")
print(f"Junk genes: {len(scm.metadata.junk_gene_indices)}")

# View equations
for var, expr in scm.equations.items():
    print(f"  {var} = {expr}")
```

---

## 3. Manual Genome Construction

```python
from evolve.representation.sequence import SequenceGenome

# Create specific genome for: C = A + B
genes = ("A", "B", "+", "STORE_C")
inner = SequenceGenome(
    genes=genes,
    alphabet=alphabet.symbols,
)
genome = SCMGenome(
    inner=inner,
    config=config,
    erc_values=(),
)

# Decode
scm = decoder.decode(genome)
print(scm.equations)
# {'C': BinOp(op='+', left=Var(name='A'), right=Var(name='B'))}

# Verify graph structure
print(list(scm.graph.edges()))
# [('A', 'C'), ('B', 'C')]
```

---

## 4. Using ERCs (Ephemeral Random Constants)

```python
# Genome with ERC: C = A * 0.5
genes = ("A", "ERC_0", "*", "STORE_C")
inner = SequenceGenome(genes=genes, alphabet=alphabet.symbols)
genome = SCMGenome(
    inner=inner,
    config=config,
    erc_values=((0, 0.5),),  # ERC_0 = 0.5
)

scm = decoder.decode(genome)
print(scm.equations)
# {'C': BinOp(op='*', left=Var(name='A'), right=Const(value=0.5))}

# Mutate ERC value
new_erc_values = ((0, 0.75),)  # Perturbed value
mutated = genome.with_erc_values(new_erc_values)
```

---

## 5. Fitness Evaluation

```python
import numpy as np
from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
from evolve.core.types import Individual

# Create synthetic data
np.random.seed(42)
n_samples = 100
A = np.random.randn(n_samples)
B = np.random.randn(n_samples)
C = A + B + 0.1 * np.random.randn(n_samples)  # True model: C = A + B
D = 2 * C + np.random.randn(n_samples)        # True model: D = 2*C

data = np.column_stack([A, B, C, D])

# Create evaluator
fitness_config = SCMFitnessConfig(
    objectives=("data_fit", "sparsity", "simplicity"),
)
evaluator = SCMEvaluator(
    data=data,
    variable_names=["A", "B", "C", "D"],
    config=fitness_config,
    decoder=decoder,
)

# Evaluate individuals
individual = Individual(genome=genome, fitness=None)
fitness_results = evaluator.evaluate([individual])

print(f"Fitness: {fitness_results[0]}")
# Fitness: (-0.01, -2, -3)  # (neg MSE, neg edges, neg complexity)
```

---

## 6. Handling Conflicts and Cycles

```python
# Genome with conflict: Two equations for C
genes = ("A", "STORE_C", "B", "STORE_C")
inner = SequenceGenome(genes=genes, alphabet=alphabet.symbols)
genome = SCMGenome(inner=inner, config=config, erc_values=())

# With first_wins (default)
decoder_first = SCMDecoder(SCMConfig(
    observed_variables=("A", "B", "C"),
    conflict_resolution=ConflictResolution.FIRST_WINS,
))
scm = decoder_first.decode(genome)
print(f"Equation (first_wins): C = {scm.equations['C']}")
# Equation (first_wins): C = Var(name='A')

print(f"Conflicts: {scm.metadata.conflict_count}")
# Conflicts: 1

# Cyclic genome: A = B, B = A
genes = ("B", "STORE_A", "A", "STORE_B")
inner = SequenceGenome(genes=genes, alphabet=alphabet.symbols)
genome = SCMGenome(inner=inner, config=config, erc_values=())

scm = decoder.decode(genome)
print(f"Is cyclic: {scm.is_cyclic}")
# Is cyclic: True

print(f"Cycles: {scm.metadata.cycles}")
# Cycles: (('A', 'B'),)
```

---

## 7. Integration with Evolution Engine

```python
from evolve.core.engine import EvolutionEngine
from evolve.core.population import Population

# Create initial population
population = Population.random(
    size=100,
    genome_factory=lambda rng: SCMGenome.random(config, length=50, rng=rng),
    rng=Random(42),
)

# Standard evolution with SCMEvaluator
# (Uses existing mutation/crossover on inner SequenceGenome)
engine = EvolutionEngine(
    evaluator=evaluator,
    # ... other configuration
)
```

---

## 8. Serialization

```python
# Save genome
genome_dict = genome.to_dict()
import json
with open("genome.json", "w") as f:
    json.dump(genome_dict, f)

# Load genome
with open("genome.json") as f:
    loaded_dict = json.load(f)
restored = SCMGenome.from_dict(loaded_dict)

assert genome == restored  # Round-trip equality
```

---

## 9. End-to-End SCM Discovery Example (T110a)

Complete example discovering causal structure from synthetic data:

```python
"""
End-to-end SCM discovery on synthetic data.

True causal model: A -> B -> C
We generate data from this model and use evolution to rediscover it.
"""
from random import Random
from uuid import uuid4
import numpy as np

from evolve.representation.scm import SCMConfig, SCMGenome
from evolve.representation.scm_decoder import SCMDecoder, to_string
from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
from evolve.core.types import Individual, IndividualMetadata
from evolve.reproduction.protocol import ReproductionProtocol

# --- 1. Generate synthetic data from true model ---
np.random.seed(42)
n_samples = 200

# True model: A is exogenous, B = 2*A + noise, C = 0.5*B + noise
A = np.random.randn(n_samples)
B = 2.0 * A + 0.1 * np.random.randn(n_samples)
C = 0.5 * B + 0.1 * np.random.randn(n_samples)

data = np.column_stack([A, B, C])
var_names = ("A", "B", "C")

# --- 2. Configure evolution ---
config = SCMConfig(
    observed_variables=var_names,
    max_latent_variables=0,  # No hidden confounders
)
decoder = SCMDecoder(config)
fitness_config = SCMFitnessConfig(
    objectives=("data_fit", "sparsity"),  # Maximize fit, minimize edges
)
evaluator = SCMEvaluator(
    data=data,
    variable_names=var_names,
    config=fitness_config,
    decoder=decoder,
)

# --- 3. Simple evolutionary loop ---
population_size = 100
n_generations = 50
rng = Random(42)

# Initialize population
genomes = [
    SCMGenome.random(config, length=30, rng=Random(i))
    for i in range(population_size)
]

best_genome = None
best_fitness = float("-inf")

for gen in range(n_generations):
    # Create individuals
    individuals = [
        Individual(
            id=uuid4(),
            genome=g,
            protocol=ReproductionProtocol.default(),
            fitness=None,
            metadata=IndividualMetadata(),
            created_at=gen,
        )
        for g in genomes
    ]
    
    # Evaluate
    fitnesses = evaluator.evaluate(individuals)
    
    # Track best (handle None for rejected cyclic genomes)
    for g, f in zip(genomes, fitnesses):
        if f is not None and f.values[0] > best_fitness:
            best_fitness = f.values[0]
            best_genome = g
    
    # Selection: keep top 50%
    valid_pairs = [
        (g, f.values[0] if f is not None else float("-inf"))
        for g, f in zip(genomes, fitnesses)
    ]
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    survivors = [g for g, _ in valid_pairs[:population_size // 2]]
    
    # Reproduction: random new genomes (simplified - real impl uses crossover/mutation)
    new_genomes = survivors + [
        SCMGenome.random(config, length=30, rng=Random(gen * 1000 + i))
        for i in range(population_size - len(survivors))
    ]
    genomes = new_genomes

# --- 4. Analyze best solution ---
if best_genome:
    best_scm = decoder.decode(best_genome)
    print("\\n=== Best Discovered SCM ===")
    print(f"Data fit (neg MSE): {best_fitness:.4f}")
    print(f"Number of edges: {best_scm.edge_count}")
    print(f"Is acyclic: {not best_scm.is_cyclic}")
    print("\\nDiscovered equations:")
    for var, expr in best_scm.equations.items():
        print(f"  {var} = {to_string(expr)}")
    print("\\nCausal graph edges:")
    for src, tgt in best_scm.graph.edges():
        print(f"  {src} -> {tgt}")
```

Expected output (discovered structure should approximate true model):
```
=== Best Discovered SCM ===
Data fit (neg MSE): -0.0123
Number of edges: 2
Is acyclic: True

Discovered equations:
  B = (A * 2.0)
  C = (B * 0.5)

Causal graph edges:
  A -> B
  B -> C
```

---

## Common Patterns

### Pattern: Custom Objective Function

```python
def custom_objective(scm: DecodedSCM) -> float:
    """Penalize use of latent variables."""
    return -len(scm.metadata.latent_variables_used)

# Extend evaluator with custom objective
# (Implementation detail: subclass or compose)
```

### Pattern: Checking Latent Ancestor Constraint

```python
import networkx as nx

def check_latent_ancestors(scm: DecodedSCM, observed: set[str]) -> bool:
    """Verify all latents have observed ancestors."""
    for latent in scm.metadata.latent_variables_used:
        ancestors = nx.ancestors(scm.graph, latent)
        if not (ancestors & observed):
            return False
    return True
```

### Pattern: Extracting NetworkX Graph for DoWhy

```python
import dowhy

# Get causal graph
graph = scm.graph

# DoWhy expects specific format
# (Nodes as variables, edges as causal relationships)
model = dowhy.CausalModel(
    data=data_df,
    graph=graph,  # NetworkX DiGraph
    treatment="A",
    outcome="D",
)
```

---

## Next Steps

- See [data-model.md](data-model.md) for complete entity definitions
- See [contracts/](contracts/) for full API contracts
- Run `/speckit.tasks` to generate implementation tasks
