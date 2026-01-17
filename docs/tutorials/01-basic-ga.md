# Tutorial 1: Your First Genetic Algorithm

This tutorial walks through creating a simple genetic algorithm to optimize a mathematical function.

## Goal

Minimize the **Sphere function**: $f(x) = \sum_{i=1}^{n} x_i^2$

The global minimum is at $x = (0, 0, ..., 0)$ with $f(x) = 0$.

## Step 1: Import Dependencies

```python
import numpy as np
from random import Random

from evolve.core.engine import EvolutionEngine, EvolutionConfig, create_initial_population
from evolve.core.operators import TournamentSelection, UniformCrossover, GaussianMutation
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome
```

## Step 2: Define the Fitness Function

```python
def sphere(genome: VectorGenome) -> float:
    """
    Sphere function (minimize).
    
    Global optimum: f(0, 0, ..., 0) = 0
    """
    return float(np.sum(genome.genes ** 2))
```

## Step 3: Set Up the Evolution

```python
# Configuration
n_dims = 10
bounds = (-5.12, 5.12)
seed = 42

# Evolution parameters
config = EvolutionConfig(
    population_size=100,
    max_generations=100,
    elitism=1,              # Keep best individual
    crossover_rate=0.9,     # 90% chance of crossover
    mutation_rate=0.1,      # 10% mutation per individual
    minimize=True           # Lower fitness is better
)

# Create operators
selection = TournamentSelection(tournament_size=3)
crossover = UniformCrossover()
mutation = GaussianMutation(sigma=0.1)

# Create evaluator
evaluator = FunctionEvaluator(sphere)

# Create engine
engine = EvolutionEngine(
    config=config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    seed=seed
)
```

## Step 4: Create Initial Population

```python
# Genome factory
def genome_factory(rng: Random) -> VectorGenome:
    genes = np.array([rng.uniform(bounds[0], bounds[1]) for _ in range(n_dims)])
    return VectorGenome(genes=genes)

# Create population
rng = Random(seed)
initial_population = create_initial_population(
    genome_factory=genome_factory,
    population_size=config.population_size,
    rng=rng
)
```

## Step 5: Run Evolution

```python
# Progress callback
class ProgressCallback:
    def on_generation_end(self, generation, population, metrics):
        if generation % 10 == 0:
            best_fit = metrics.get("best_fitness", "N/A")
            print(f"Generation {generation}: Best fitness = {best_fit:.6f}")

# Run
result = engine.run(initial_population, callbacks=[ProgressCallback()])

# Results
print(f"\n=== Results ===")
print(f"Best fitness: {result.best.fitness.values[0]:.6f}")
print(f"Best solution: {result.best.genome.genes}")
print(f"Generations: {result.generations}")
print(f"Stop reason: {result.stop_reason}")
```

## Expected Output

```
Generation 0: Best fitness = 23.456789
Generation 10: Best fitness = 5.123456
Generation 20: Best fitness = 1.234567
Generation 30: Best fitness = 0.345678
...
Generation 90: Best fitness = 0.001234

=== Results ===
Best fitness: 0.000456
Best solution: [-0.01  0.02 -0.01 ...]
Generations: 100
Stop reason: Generation limit reached
```

## Complete Code

```python
import numpy as np
from random import Random

from evolve.core.engine import EvolutionEngine, EvolutionConfig, create_initial_population
from evolve.core.operators import TournamentSelection, UniformCrossover, GaussianMutation
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome


def sphere(genome: VectorGenome) -> float:
    return float(np.sum(genome.genes ** 2))


def main():
    # Setup
    n_dims = 10
    bounds = (-5.12, 5.12)
    seed = 42
    
    config = EvolutionConfig(
        population_size=100,
        max_generations=100,
        elitism=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
        minimize=True
    )
    
    engine = EvolutionEngine(
        config=config,
        evaluator=FunctionEvaluator(sphere),
        selection=TournamentSelection(tournament_size=3),
        crossover=UniformCrossover(),
        mutation=GaussianMutation(sigma=0.1),
        seed=seed
    )
    
    # Initial population
    rng = Random(seed)
    population = create_initial_population(
        genome_factory=lambda r: VectorGenome(
            genes=np.array([r.uniform(bounds[0], bounds[1]) for _ in range(n_dims)])
        ),
        population_size=config.population_size,
        rng=rng
    )
    
    # Run
    result = engine.run(population)
    
    print(f"Best fitness: {result.best.fitness.values[0]:.6f}")
    print(f"Solution: {result.best.genome.genes}")


if __name__ == "__main__":
    main()
```

## Next Steps

- **[Tutorial 2](02-function-optimization.md)**: Try different benchmark functions
- **[Tutorial 3](03-multi-objective.md)**: Multi-objective optimization with NSGA-II
- **[Tutorial 4](04-custom-operators.md)**: Create your own operators
