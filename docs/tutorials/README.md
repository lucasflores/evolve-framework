# Evolve Framework Tutorials

Welcome to the Evolve Framework tutorial series! These notebooks teach evolutionary algorithms to ML practitioners by bridging familiar concepts from machine learning to evolutionary computation.

Every tutorial uses `UnifiedConfig` + `create_engine()` as the canonical pattern for defining and running experiments.

## Prerequisites

- **Audience**: ML/DS practitioners with strong statistics/linear algebra background but NO evolutionary algorithms experience
- **Python**: 3.10+
- **Dependencies**: numpy, matplotlib, plotly (optional)

## Interactive Tutorial Notebooks

| # | Tutorial | Genome Type | Problem Domain | Est. Time |
|---|----------|-------------|----------------|-----------|
| 1 | [**UnifiedConfig** (Start Here)](01_unified_config.ipynb) | Config | Declarative experiment setup | 30-45 min |
| 2 | [VectorGenome](02_vector_genome.ipynb) | Fixed-length vectors | Continuous optimization | 45-60 min |
| 3 | [SequenceGenome](03_sequence_genome.ipynb) | Variable-length sequences | Symbolic regression (GP) | 45-60 min |
| 4 | [GraphGenome/NEAT](04_graph_genome_neat.ipynb) | Network topology | Neural architecture search | 60-75 min |
| 5 | [RL/Neuroevolution](05_rl_neuroevolution.ipynb) | Policy weights | Reinforcement learning | 45-60 min |
| 6 | [SCM/Multi-Objective](06_scm_multiobjective.ipynb) | Causal graphs | Causal discovery (NSGA-II) | 60-75 min |
| 7 | [Evolvable Reproduction](07_evolvable_reproduction.ipynb) | Individual protocols | Sexual selection, speciation | 75-90 min |

## Learning Path

### Recommended Order (Complete Beginner)
```
Tutorial 01 (UnifiedConfig) → Tutorial 02 → Tutorial 03 → Tutorial 04 → Tutorial 05
```

**Tutorial 01 is required** — it introduces `UnifiedConfig`, `create_engine()`, and `create_initial_population()` which all subsequent tutorials build on.

**Prerequisites for Tutorial 07**: Complete Tutorial 01 and Tutorial 02 first to understand basic EA concepts.

### By Interest

| If you're interested in... | Start with |
|---------------------------|------------|
| **Getting started** | Tutorial 01 (UnifiedConfig) |
| **Hyperparameter optimization** | Tutorial 02 (VectorGenome) |
| **Symbolic AI / interpretable models** | Tutorial 03 (SequenceGenome) |
| **Neural architecture search** | Tutorial 04 (GraphGenome/NEAT) |
| **RL without gradients** | Tutorial 05 (RL/Neuroevolution) |
| **Causal inference** | Tutorial 06 (SCM/Multi-Objective) |
| **Sexual selection / speciation** | Tutorial 07 (Evolvable Reproduction) |

## Key Concepts Covered

### Tutorial 01: UnifiedConfig
- Declarative experiment configuration via `UnifiedConfig`
- `create_engine()` factory for one-line engine creation
- `create_initial_population()` from config
- Config serialization, hashing, and reproducibility
- Sub-configs: stopping, tracking, ERP, multi-objective, meta-evolution

### Tutorial 02: VectorGenome
- EA fundamentals: selection, crossover, mutation
- Population-based optimization vs gradient descent
- Adaptive mutation rates
- Benchmark functions (Rastrigin, Rosenbrock)

### Tutorial 03: SequenceGenome
- Variable-length representations
- Expression tree manipulation
- Subtree crossover
- Bloat control (parsimony pressure)

### Tutorial 04: GraphGenome/NEAT
- Topology evolution
- Speciation and genomic distance
- Fitness sharing
- Innovation numbers for crossover alignment

### Tutorial 05: RL/Neuroevolution
- Evolution as derivative-free optimization
- Policy networks from genomes
- Episode rollout as fitness
- Fitness aggregation strategies

### Tutorial 06: SCM/Multi-Objective
- Pareto dominance
- NSGA-II algorithm
- Crowding distance
- Multi-objective trade-offs

### Tutorial 07: Evolvable Reproduction
- Individual-level mating strategies
- Reproduction intent policies (willingness to mate)
- Matchability functions (mate selection criteria)
- Evolvable crossover protocols
- Sexual selection modeling
- Assortative mating and speciation
- Population recovery mechanisms

## Common Features

All tutorials include:
- ✅ **UnifiedConfig** declarative experiment setup
- ✅ **EA Primer sections** with ML analogies
- ✅ **Mermaid diagrams** for visual concepts
- ✅ **Interactive visualizations** (matplotlib + plotly)
- ✅ **Callbacks** for logging and early stopping
- ✅ **Checkpointing** for save/resume
- ✅ **Island model** parallelism overview
- ✅ **GPU acceleration** guidance (Tutorial 02)
- ✅ **Config serialization** and reproducibility (Tutorial 01)

## ML-to-EA Terminology

| ML Term | EA Equivalent |
|---------|---------------|
| Hyperparameters | Genome / Genotype |
| Model instance | Individual / Phenotype |
| Training batch | Population |
| Epoch | Generation |
| Loss function | Fitness function (negated) |
| Learning rate | Mutation rate |
| Training | Evolution |
| Ensemble | Population diversity |

## Setup

```bash
# From project root
pip install -e ".[tutorials]"
```

### Optional Dependencies

```bash
# For interactive 3D plots
pip install plotly

# For RL environments (Tutorial 4)
pip install gymnasium

# For notebook execution tests
pip install papermill
```

## Running Notebooks

### Jupyter Lab/Notebook
```bash
cd docs/tutorials
jupyter lab
```

### VS Code
Open any `.ipynb` file directly in VS Code with the Jupyter extension.

## Troubleshooting

### Import Errors
Ensure you're running from the project root or have installed the package:
```bash
pip install -e .
```

### Missing GPU
All tutorials gracefully degrade to CPU. GPU sections will print informative messages about expected speedups.

### Missing Gymnasium
Tutorial 4 includes a `MockCartPole` class for execution without gymnasium installed.

### Memory Issues
Reduce population sizes in the configuration cells if you encounter memory errors.

---

## Markdown Tutorials (Legacy Reference)

> **Note**: These markdown tutorials use the older `EvolutionConfig` + manual construction pattern.
> For new experiments, use the `UnifiedConfig` + `create_engine()` approach shown in the notebooks above.

These supplementary tutorials provide additional documentation in markdown format:

1. **[Basic GA Tutorial](01-basic-ga.md)** - Your first genetic algorithm
2. **[Function Optimization](02-function-optimization.md)** - Optimize benchmark functions
3. **[Multi-Objective Optimization](03-multi-objective.md)** - NSGA-II and Pareto fronts
4. **[Custom Operators](04-custom-operators.md)** - Create custom selection, crossover, mutation
5. **[Experiment Tracking](05-experiment-tracking.md)** - Logging and reproducibility
6. **[Neuroevolution](06-neuroevolution.md)** - Evolve neural network controllers
7. **[Island Model](07-island-model.md)** - Parallel populations with migration
8. **[Speciated Evolution](08-speciation.md)** - NEAT-style speciation
9. **[Novelty Search](09-novelty-search.md)** - Quality-diversity optimization
10. **[GPU Acceleration](10-gpu-acceleration.md)** - PyTorch and JAX backends
11. **[Checkpointing](11-checkpointing.md)** - Save and resume experiments

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on improving tutorials.

## License

Same as the main Evolve Framework - see project root for license information.
