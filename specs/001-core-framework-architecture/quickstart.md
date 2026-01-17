# Evolve Framework Quickstart Guide

This guide demonstrates the core usage patterns for the Evolve Framework.

---

## Installation

```bash
# Core installation (NumPy only)
pip install evolve

# With optional GPU acceleration
pip install evolve[pytorch]  # or evolve[jax]

# With experiment tracking
pip install evolve[mlflow]   # or evolve[wandb]

# Full installation
pip install evolve[all]
```

---

## Example 1: Simple Function Optimization

Minimize the Rastrigin function using a standard genetic algorithm.

```python
import numpy as np
from evolve.core import EvolutionEngine, Individual
from evolve.representation import VectorGenome
from evolve.operators import TournamentSelection, UniformCrossover, GaussianMutation
from evolve.evaluation import FunctionEvaluator

# Define the problem
def rastrigin(genome: VectorGenome) -> float:
    """Rastrigin function (minimize). Negate for maximization."""
    x = genome.genes
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

# Problem bounds
n_dims = 10
bounds = (np.full(n_dims, -5.12), np.full(n_dims, 5.12))

# Create genome factory
def genome_factory(rng):
    return VectorGenome.random(n_dims, bounds, rng)

# Setup evolution
engine = EvolutionEngine(
    population_size=100,
    genome_factory=genome_factory,
    evaluator=FunctionEvaluator(rastrigin),
    selection=TournamentSelection(tournament_size=3),
    crossover=UniformCrossover(),
    mutation=GaussianMutation(sigma=0.1),
    crossover_rate=0.9,
    mutation_rate=0.1,
    seed=42  # Reproducibility
)

# Run evolution
for generation in range(100):
    engine.step()
    
    if generation % 10 == 0:
        print(f"Gen {generation}: Best = {engine.best.fitness.value:.4f}")

# Result
print(f"\nBest solution: {engine.best.genome.genes}")
print(f"Best fitness: {engine.best.fitness.value:.4f}")
```

---

## Example 2: Multi-Objective Optimization

Find Pareto front for a bi-objective problem using NSGA-II.

```python
import numpy as np
from evolve.core import Individual
from evolve.representation import VectorGenome
from evolve.multiobjective import (
    MultiObjectiveFitness,
    NSGA2Selector,
    fast_non_dominated_sort,
    pareto_front
)

# ZDT1 bi-objective benchmark
def zdt1(genome: VectorGenome) -> MultiObjectiveFitness:
    x = genome.genes
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    # Negate for maximization convention
    return MultiObjectiveFitness(objectives=np.array([-f1, -f2]))

# Setup with NSGA-II selector
selector = NSGA2Selector()

# ... rest of evolution loop uses selector for parent selection
# NSGA-II automatically handles:
# 1. Non-dominated sorting
# 2. Crowding distance calculation  
# 3. Selection based on rank and diversity

# After evolution, extract Pareto front:
fitnesses = [ind.fitness for ind in population]
front_indices = pareto_front(fitnesses)
pareto_solutions = [population[i] for i in front_indices]
```

---

## Example 3: Neuroevolution for RL

Evolve neural network controllers for a Gym environment.

```python
import numpy as np
from evolve.rl import (
    GymAdapter, 
    MLPPolicy, 
    RLEvaluator,
    evaluate_policy
)
from evolve.representation import VectorGenome

# Environment setup
def make_env():
    import gymnasium as gym
    return GymAdapter(gym.make("CartPole-v1"))

# Define network architecture
obs_dim = 4
action_dim = 2
hidden_sizes = [32, 32]

# Calculate total parameters
def count_params():
    sizes = [obs_dim] + hidden_sizes + [action_dim]
    return sum(sizes[i] * sizes[i+1] + sizes[i+1] 
               for i in range(len(sizes)-1))

n_params = count_params()

# Genome to policy decoder
class MLPDecoder:
    def decode(self, genome: VectorGenome) -> MLPPolicy:
        params = genome.genes
        weights, biases = [], []
        
        idx = 0
        sizes = [obs_dim] + hidden_sizes + [action_dim]
        
        for i in range(len(sizes) - 1):
            w_size = sizes[i] * sizes[i+1]
            b_size = sizes[i+1]
            
            w = params[idx:idx+w_size].reshape(sizes[i], sizes[i+1])
            idx += w_size
            b = params[idx:idx+b_size]
            idx += b_size
            
            weights.append(w)
            biases.append(b)
        
        return MLPPolicy(weights, biases)

# Create evaluator
evaluator = RLEvaluator(
    decoder=MLPDecoder(),
    env_factory=make_env,
    n_episodes=5,  # Average over 5 episodes
    aggregate="mean"
)

# Evolution uses evaluator to compute fitness
# fitness = mean reward over n_episodes
```

---

## Example 4: Island Model Parallelism

Run multiple populations with periodic migration.

```python
from evolve.diversity import (
    Island, 
    MigrationController,
    BestMigration,
    ring_topology
)
import numpy as np

# Create islands
n_islands = 4
topology = ring_topology(n_islands)

islands = [
    Island(
        id=i,
        population=create_population(pop_size=25),  # 100 total
        topology=topology[i],
        migration_rate=0.1
    )
    for i in range(n_islands)
]

# Migration controller
migration = MigrationController(
    policy=BestMigration(),
    migration_interval=10
)

# Evolution with migration
for generation in range(100):
    # Evolve each island independently
    for island in islands:
        island.population = evolve_one_generation(island.population)
    
    # Migrate between islands
    if migration.should_migrate(generation):
        rng = np.random.default_rng(generation)  # Deterministic
        migration.migrate(islands, rng)

# Collect best from all islands
all_individuals = [ind for island in islands for ind in island.population]
best = max(all_individuals, key=lambda i: i.fitness.value)
```

---

## Example 5: Speciated Evolution (NEAT-style)

Maintain diverse species using compatibility distance.

```python
from evolve.diversity import (
    ThresholdSpeciator,
    neat_distance,
    explicit_fitness_sharing
)
from evolve.representation import GraphGenome

# Distance function for NEAT
def compatibility(a: GraphGenome, b: GraphGenome) -> float:
    return neat_distance(a, b, c_disjoint=1.0, c_excess=1.0, c_weight=0.4)

# Speciator
speciator = ThresholdSpeciator(
    distance_fn=compatibility,
    threshold=3.0  # Compatibility threshold
)

# Initial species assignment
species = speciator.speciate(population, existing_species=[])

# Evolution with speciation
for generation in range(100):
    # Assign fitness sharing within species
    for sp in species:
        shared = explicit_fitness_sharing(
            sp.members,
            distance_fn=compatibility,
            sigma_share=3.0
        )
        for ind, fit in zip(sp.members, shared):
            ind.adjusted_fitness = fit
    
    # Reproduce within species (offspring proportional to avg fitness)
    offspring = []
    for sp in species:
        n_offspring = calculate_offspring_quota(sp, total=len(population))
        offspring.extend(reproduce_species(sp, n_offspring))
    
    population = offspring
    
    # Re-speciate
    species = speciator.speciate(population, species)
    
    # Update stagnation and potentially remove stagnant species
    for sp in species:
        sp.update_stagnation()
    species = [sp for sp in species if not sp.is_stagnant(threshold=15)]
```

---

## Example 6: Experiment Tracking

Full experiment with configuration and checkpointing.

```python
from evolve.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    LocalTracker,
    CheckpointManager
)

# Define experiment configuration
config = ExperimentConfig(
    name="rastrigin_optimization_v1",
    description="Optimize 10D Rastrigin function",
    seed=42,
    
    # Population
    population_size=100,
    n_generations=200,
    
    # Operators
    selection_method="tournament",
    selection_params={"tournament_size": 3},
    crossover_method="uniform",
    crossover_rate=0.9,
    mutation_method="gaussian",
    mutation_rate=0.1,
    mutation_params={"sigma": 0.1},
    
    # Representation
    genome_type="vector",
    genome_params={"n_dims": 10, "bounds": (-5.12, 5.12)},
    
    # Evaluation
    evaluator_type="function",
    evaluator_params={"function": "rastrigin"},
    
    # Output
    output_dir="./experiments",
    checkpoint_interval=20,
    
    # Stopping
    target_fitness=0.0,  # Global optimum
    stagnation_limit=50
)

# Validate
errors = config.validate()
if errors:
    print(f"Config errors: {errors}")
else:
    # Run experiment
    runner = ExperimentRunner(config)
    best = runner.run()
    
    print(f"Best fitness: {best.fitness.value}")
    print(f"Config hash: {config.hash()}")
```

---

## Example 7: Resuming from Checkpoint

```python
from evolve.experiment import ExperimentRunner, ExperimentConfig

# Load existing config
config = ExperimentConfig.from_json("./experiments/my_exp/config.json")

# Resume from last checkpoint
runner = ExperimentRunner(config)
best = runner.run(resume=True)  # Automatically loads latest checkpoint
```

---

## Common Patterns

### Custom Fitness Function

```python
from evolve.evaluation import Evaluator, EvaluatorCapabilities, Fitness
from evolve.core import Individual

class MyEvaluator:
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(batched=True, stochastic=False)
    
    def evaluate(self, individuals, seed=None):
        return [
            Fitness(
                value=self._compute(ind.genome),
                metadata={"custom": "data"}
            )
            for ind in individuals
        ]
    
    def _compute(self, genome):
        # Your fitness logic
        return ...
```

### Custom Mutation Operator

```python
from evolve.operators import MutationOperator
from evolve.representation import VectorGenome
import numpy as np

class AdaptiveMutation:
    """Mutation with adaptive step size."""
    
    def __init__(self, initial_sigma: float = 0.1):
        self.sigma = initial_sigma
    
    def mutate(self, genome: VectorGenome, rng) -> VectorGenome:
        noise = rng.normal(0, self.sigma, size=len(genome.genes))
        new_genes = genome.genes + noise
        return VectorGenome(genes=new_genes, bounds=genome.bounds).clip_to_bounds()
    
    def adapt(self, success_rate: float):
        """Adapt sigma based on success rate (1/5 rule)."""
        if success_rate > 0.2:
            self.sigma *= 1.22  # Increase exploration
        elif success_rate < 0.2:
            self.sigma *= 0.82  # Decrease exploration
```

### Callbacks for Custom Logging

```python
from evolve.core import Callback

class PrintProgressCallback:
    def on_generation_end(self, engine, generation):
        if generation % 10 == 0:
            print(f"Gen {generation}: Best={engine.best.fitness.value:.4f}")

class EarlyStoppingCallback:
    def __init__(self, target: float):
        self.target = target
    
    def on_generation_end(self, engine, generation):
        if engine.best.fitness.value >= self.target:
            engine.stop()  # Signal to stop evolution

# Use callbacks
engine.add_callback(PrintProgressCallback())
engine.add_callback(EarlyStoppingCallback(target=0.99))
```

---

## Reproducibility Checklist

1. **Always set seed**: `EvolutionEngine(..., seed=42)`
2. **Use framework-neutral genomes**: No PyTorch/JAX in genome
3. **Fixed evaluation order**: Process individuals deterministically
4. **Seed-derived parallelism**: Each worker gets derived seed
5. **Checkpoint regularly**: Enable resumption without drift

```python
# Verify reproducibility
results = []
for _ in range(3):
    engine = EvolutionEngine(..., seed=42)
    for _ in range(100):
        engine.step()
    results.append(engine.best.fitness.value)

assert all(r == results[0] for r in results), "Not reproducible!"
```

---

## Next Steps

- **API Reference**: See `contracts/` for detailed interface documentation
- **Examples**: Check `examples/` for complete working scripts
- **Benchmarks**: Run `python -m evolve.benchmarks` for performance tests
- **Contributing**: Read `CONTRIBUTING.md` for development guidelines
