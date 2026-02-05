# Evolve Framework

A research-grade evolutionary algorithms experimentation framework.

## Features

- **Model-Agnostic Architecture**: No hard dependencies on ML frameworks in core modules
- **Deterministic Reproducibility**: All randomness uses explicit seeding
- **Optional GPU Acceleration**: PyTorch/JAX backends available as optional extras
- **Multi-Objective Optimization**: NSGA-II with Pareto ranking and crowding distance
- **Neuroevolution Support**: NEAT-style topology evolution
- **Evolvable Reproduction Protocols (ERP)**: Individual-level mating strategies with sexual selection
- **Experiment Tracking**: MLflow and Weights & Biases integrations

## Installation

```bash
# Basic installation (NumPy only)
pip install evolve-framework

# With PyTorch GPU support
pip install evolve-framework[pytorch]

# With all optional dependencies
pip install evolve-framework[all]

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from evolve.core.engine import EvolutionEngine, EvolutionConfig, create_initial_population
from evolve.core.operators import TournamentSelection, BlendCrossover, GaussianMutation
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.evaluation.reference.functions import sphere
from evolve.representation.vector import VectorGenome
from evolve.utils.random import create_rng

# Configuration
n_dims = 10
bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))

config = EvolutionConfig(
    population_size=50,
    max_generations=100,
    elitism=2,
)

# Create engine
engine = EvolutionEngine(
    config=config,
    evaluator=FunctionEvaluator(sphere),
    selection=TournamentSelection(tournament_size=3),
    crossover=BlendCrossover(alpha=0.5),
    mutation=GaussianMutation(mutation_rate=0.1, sigma=0.1, adaptive=True),
    seed=42,
)

# Create initial population
rng = create_rng(42)
initial_pop = create_initial_population(
    genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
    population_size=config.population_size,
    rng=rng,
)

# Run evolution
result = engine.run(initial_pop)

print(f"Best fitness: {result.best.fitness.values[0]:.6f}")
print(f"Generations: {result.generations}")
```

## Architecture

The framework is organized into independent layers:

1. **Evolution Core** (`evolve.core`): Backend-agnostic population dynamics
2. **Representation** (`evolve.representation`): Genome/phenotype abstractions
3. **Evaluation** (`evolve.evaluation`): Fitness computation interfaces
4. **Backends** (`evolve.backends`): Parallel/GPU/JIT execution
5. **Reproduction** (`evolve.reproduction`): Evolvable reproduction protocols (ERP) for sexual selection
6. **Observability** (`evolve.experiment`): Tracking and reproducibility

## Tutorials

See [docs/tutorials/](docs/tutorials/) for interactive Jupyter notebooks, including:
- Tutorial 6: [Evolvable Reproduction Protocols](docs/tutorials/06_evolvable_reproduction_protocols.ipynb) - Sexual selection and speciation

## Testing

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/ -m integration

# Run property-based tests
pytest tests/property/ -m property
```

## License

MIT License
