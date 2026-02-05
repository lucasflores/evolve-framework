# Research: Tutorial Notebooks for Evolve Framework

**Phase**: 0 - Research & Discovery  
**Date**: 2026-01-31  
**Status**: Complete

## Research Tasks

### 1. Mermaid Rendering in Jupyter

**Question**: How to render Mermaid diagrams reliably in Jupyter notebooks?

**Decision**: Use `beautiful-mermaid` library with github-light theme

**Rationale**:
- Pure Python implementation, no Node.js subprocess required
- Native Jupyter/IPython integration via `display()` magic
- Supports all standard Mermaid diagram types (flowchart, sequence, class)
- Theme customization via constructor parameter
- Fallback to static SVG if rendering fails

**Alternatives Considered**:
- `mermaid-py`: Requires Node.js subprocess, adds complexity
- Pre-rendered SVGs: Breaks sync with code, maintenance burden
- IPython.display.IFrame: External service dependency, offline issues

**Implementation Pattern**:
```python
from beautiful_mermaid import Mermaid

def render_mermaid(diagram_spec: str, theme: str = "github-light") -> None:
    """Render Mermaid diagram in notebook cell."""
    m = Mermaid(diagram_spec, theme=theme)
    display(m)
```

---

### 2. Pareto Front Visualization Strategy

**Question**: How to visualize 3-objective Pareto fronts effectively for learners?

**Decision**: 2D pairwise projections (primary) + interactive 3D plotly scatter3d (optional)

**Rationale**:
- 2D projections are easier to interpret, familiar from ML loss curves
- Three projections cover all pairwise trade-offs: fit-vs-sparsity, fit-vs-simplicity, sparsity-vs-simplicity
- Interactive 3D enables exploration of true dominance relationships
- plotly provides rotate/zoom without additional dependencies
- Progressive complexity: start with 2D, graduate to 3D

**Alternatives Considered**:
- 3D only: Steeper learning curve, harder to print/export
- Parallel coordinates: Good for high-D but less intuitive for beginners
- Static 3D matplotlib: No interactivity, awkward rotation

**Implementation Pattern**:
```python
import plotly.express as px
import plotly.graph_objects as go

def plot_pareto_2d_projections(front: np.ndarray, labels: list[str]) -> Figure:
    """Create 3 subplot 2D projections of Pareto front."""
    # Returns matplotlib figure with 1x3 subplots
    
def plot_pareto_3d_interactive(front: np.ndarray, labels: list[str]) -> go.Figure:
    """Create interactive 3D Pareto surface with plotly."""
    return px.scatter_3d(...)
```

---

### 3. Island Model Parallelism Parameters

**Question**: What island count and population sizes demonstrate meaningful speedup?

**Decision**: 4 islands × 50 individuals (200 total), ring topology default

**Rationale**:
- 4 islands matches typical consumer hardware (4-8 cores)
- 50 individuals per island provides statistical significance
- Ring topology is simple to explain and visualize
- Demonstrates ~2-3x speedup on 4 cores (Amdahl's law applies)
- Avoids oversubscription that would mask parallelism benefits

**Benchmark Expectations**:
| Configuration | Cores | Expected Speedup | Notes |
|--------------|-------|------------------|-------|
| Single pop (200) | 1 | 1.0x (baseline) | Sequential reference |
| 4 islands × 50 | 4 | 2.0-3.0x | Typical laptop |
| 4 islands × 50 | 8 | 2.5-3.5x | Migration overhead limits |

**Alternatives Considered**:
- 2 islands: Too few to show meaningful parallelism
- 8 islands: Requires 8+ cores, uncommon in tutorial environments
- Larger populations: Increases runtime beyond 90-minute target

---

### 4. RL Environment Selection

**Question**: Which Gymnasium environment best demonstrates policy evolution?

**Decision**: CartPole-v1 (primary), LunarLander-v2 (optional advanced)

**Rationale**:
- CartPole solves in ~30 seconds with small networks
- 4-dimensional state space is easy to visualize
- Clear success criterion (475+ average return over 100 episodes)
- LunarLander shows evolution handles harder problems (8D state, continuous control)
- Both environments are well-documented in RL literature

**Network Architecture for CartPole**:
- Input: 4 neurons (observation space)
- Hidden: 8 neurons (sufficient for linear separability)
- Output: 2 neurons (action probabilities)
- Total parameters: ~50 weights (small genome)

**Alternatives Considered**:
- MountainCar: Sparse reward makes evolution slow
- Pendulum: Continuous action requires different policy encoding
- Atari: Too computationally expensive for tutorial

---

### 5. Speciation Visualization

**Question**: How to visualize species dynamics in NEAT for learners?

**Decision**: Stacked area chart (primary), phylogenetic tree (optional)

**Rationale**:
- Stacked area shows population composition over time at a glance
- Familiar format from training curves and population demographics
- Easy to see species birth, growth, decline, extinction
- Colors distinguish species; legend maps to species ID
- Phylogenetic tree as optional view for users interested in ancestry

**Implementation Pattern**:
```python
def plot_species_over_generations(species_history: dict[int, list[int]]) -> Figure:
    """Stacked area chart of species population over generations."""
    # species_history: {generation: [species_id for each individual]}
    # Returns matplotlib figure
    
def plot_species_phylogeny(lineage_data: LineageTracker) -> Figure:
    """Optional: Phylogenetic tree showing species ancestry."""
    # Uses networkx for tree layout
```

---

### 6. beautiful-mermaid Theme Selection

**Question**: Which theme provides best readability across environments?

**Decision**: github-light as default

**Rationale**:
- Maximum contrast for accessibility
- Print-friendly for exported notebooks
- Matches default Jupyter light theme
- Works well in documentation/README embedding
- Users can override to dark themes if preferred

**Configuration Pattern**:
```python
# In tutorial_utils.py
DEFAULT_MERMAID_THEME = "github-light"

def render_mermaid(diagram: str, theme: str = DEFAULT_MERMAID_THEME) -> None:
    ...
```

---

## Technology Stack Summary

| Component | Choice | Version | Notes |
|-----------|--------|---------|-------|
| Core Framework | evolve | 0.1.0+ | Local dependency |
| Mermaid Rendering | beautiful-mermaid | latest | Dynamic rendering |
| Interactive Plots | plotly | 5.x | 3D Pareto, policy viz |
| Static Plots | matplotlib | 3.x | Convergence curves |
| RL Environments | gymnasium | 0.29+ | CartPole, LunarLander |
| Parallelism | multiprocessing | stdlib | Island models |
| Notebook Testing | papermill | 2.x | Execution validation |

## Dependencies to Add

```toml
# pyproject.toml [project.optional-dependencies]
tutorials = [
    "beautiful-mermaid>=0.2",
    "plotly>=5.0",
    "gymnasium>=0.29",
    "papermill>=2.0",
    "ipywidgets>=8.0",  # For interactive controls
]
```

## Open Questions (Resolved)

All research questions have been resolved. No blockers for Phase 1.
