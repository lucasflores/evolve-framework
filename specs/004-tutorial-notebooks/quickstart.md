# Quickstart: Tutorial Notebooks Development

**Date**: 2026-01-31  
**Branch**: `004-tutorial-notebooks`

## Prerequisites

```bash
# Clone and navigate to repo
cd evolve-framework

# Create development environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install evolve with tutorial dependencies
pip install -e ".[tutorials]"
```

## Project Structure

```
docs/tutorials/
├── utils/
│   ├── __init__.py
│   └── tutorial_utils.py       # Implement first
├── 01_vector_genome.ipynb
├── 02_sequence_genome.ipynb
├── 03_graph_genome_neat.ipynb
├── 04_rl_neuroevolution.ipynb
├── 05_scm_multiobjective.ipynb
└── README.md
```

## Development Order

### Phase 1: Shared Module (Week 1)

```bash
# 1. Create the utils module
mkdir -p docs/tutorials/utils
touch docs/tutorials/utils/__init__.py

# 2. Implement tutorial_utils.py following contracts/tutorial_utils_api.md

# 3. Write tests
pytest tests/tutorials/test_tutorial_utils.py -v
```

### Phase 2: VectorGenome Notebook (Week 2)

```bash
# Start Jupyter and create notebook
jupyter lab docs/tutorials/

# Key sections to implement:
# - EA Primer with terminology glossary
# - Rastrigin optimization demo
# - Convergence visualization
# - Island model parallel benchmark
# - GPU acceleration benchmark
```

### Phase 3: Remaining Notebooks (Weeks 3-5)

Complete in priority order:
1. `02_sequence_genome.ipynb` - Symbolic regression
2. `03_graph_genome_neat.ipynb` - XOR with speciation
3. `04_rl_neuroevolution.ipynb` - CartPole policy evolution
4. `05_scm_multiobjective.ipynb` - Causal discovery with NSGA-II

## Quick Validation

```bash
# Run all notebooks headlessly
papermill docs/tutorials/01_vector_genome.ipynb /dev/null

# Run tutorial_utils tests
pytest tests/tutorials/ -v

# Check notebook execution time
time papermill docs/tutorials/01_vector_genome.ipynb /dev/null
# Should complete in <10 minutes
```

## Key Implementation Patterns

### Mermaid Diagram

```python
from docs.tutorials.utils import render_mermaid

render_mermaid('''
graph LR
    A[VectorGenome] --> B[Identity Decoder]
    B --> C[Float Array]
    C --> D[Fitness Function]
    D --> E[Scalar Fitness]
''')
```

### Evolution with Callbacks

```python
from evolve import EvolutionEngine, EvolutionConfig
from docs.tutorials.utils import EvolutionHistory

history = EvolutionHistory()

engine = EvolutionEngine(
    config=EvolutionConfig(population_size=100, max_generations=50),
    callbacks=[history.callback()]
)
result = engine.run(initial_population)

plot_fitness_history(history)
```

### Island Model

```python
from docs.tutorials.utils import create_island_config, run_island_benchmark

config = create_island_config(
    num_islands=4,
    population_per_island=50,
    topology="ring"
)

result = run_island_benchmark(config, rastrigin_function, VectorGenome.random)
print(f"Speedup: {result.speedup_vs(baseline):.2f}x")
```

## Success Criteria Checklist

- [ ] Each notebook runs end-to-end without errors
- [ ] All notebooks complete in <90 minutes total execution
- [ ] 10+ terminology mappings per notebook
- [ ] 3+ mermaid diagrams per notebook
- [ ] Island model shows >20% speedup on 4 cores
- [ ] GPU section shows speedup for pop_size > 1000

## Common Issues

### beautiful-mermaid not rendering
```python
# Fallback to text display
try:
    render_mermaid(diagram)
except ImportError:
    print(diagram)  # Show raw mermaid code
```

### Gymnasium not installed
```python
try:
    import gymnasium as gym
except ImportError:
    print("Install with: pip install gymnasium")
    # Skip RL-specific cells
```

### GPU not available
```python
import torch
if torch.cuda.is_available():
    # Run GPU benchmark
else:
    print("GPU not available - showing CPU-only results")
```
