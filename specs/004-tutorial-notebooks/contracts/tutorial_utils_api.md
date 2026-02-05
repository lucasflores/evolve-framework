# API Contract: tutorial_utils Module

**Version**: 1.0.0  
**Date**: 2026-01-31  
**Module**: `docs.tutorials.utils.tutorial_utils`

## Overview

This contract defines the public API for the shared tutorial utilities module. All functions are designed for use in Jupyter notebook environments and prioritize clarity over performance.

---

## Data Generation Functions

### Benchmark Functions (FR-001)

```python
def sphere_function(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = sum(x_i^2)
    
    Args:
        x: Input vector of any dimension
        
    Returns:
        Scalar fitness value (lower is better)
        
    Example:
        >>> sphere_function(np.array([0.0, 0.0]))
        0.0
        >>> sphere_function(np.array([1.0, 1.0]))
        2.0
    """

def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin function: highly multimodal benchmark.
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Global optimum: f(0, ..., 0) = 0
    Recommended bounds: [-5.12, 5.12]
    """

def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function: banana-shaped valley.
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Global optimum: f(1, ..., 1) = 0
    Recommended bounds: [-5, 10]
    """

def ackley_function(x: np.ndarray) -> float:
    """
    Ackley function: many local minima with global minimum at origin.
    
    Global optimum: f(0, ..., 0) = 0
    Recommended bounds: [-32.768, 32.768]
    """

def get_benchmark(name: str) -> BenchmarkFunction:
    """
    Get a benchmark function configuration by name.
    
    Args:
        name: One of "sphere", "rastrigin", "rosenbrock", "ackley"
        
    Returns:
        BenchmarkFunction dataclass with metadata and evaluate method
        
    Raises:
        ValueError: If name is not recognized
    """
```

### Symbolic Regression Data (FR-002)

```python
def generate_polynomial_data(
    degree: int = 2,
    n_samples: int = 100,
    noise_level: float = 0.1,
    seed: int = 42,
    test_fraction: float = 0.2
) -> SymbolicRegressionData:
    """
    Generate synthetic data from a polynomial function.
    
    Args:
        degree: Polynomial degree (1-5 recommended)
        n_samples: Total number of samples
        noise_level: Standard deviation of Gaussian noise (0.0-1.0)
        seed: Random seed for reproducibility
        test_fraction: Fraction of data reserved for testing
        
    Returns:
        SymbolicRegressionData with train/test split and true expression
    """

def generate_trigonometric_data(
    frequency: float = 1.0,
    n_samples: int = 100,
    noise_level: float = 0.1,
    seed: int = 42
) -> SymbolicRegressionData:
    """
    Generate synthetic data from sin/cos functions.
    
    True expression: a*sin(frequency*x) + b*cos(frequency*x) + c
    """

def generate_composite_data(
    complexity: Literal["simple", "medium", "complex"] = "medium",
    n_features: int = 2,
    n_samples: int = 200,
    noise_level: float = 0.1,
    seed: int = 42
) -> SymbolicRegressionData:
    """
    Generate data from composite mathematical expressions.
    
    Complexity levels:
    - simple: x + y, x*y
    - medium: x^2 + sin(y), x*y + exp(-x)
    - complex: nested functions, multiple interactions
    """
```

### Causal Data Generation (FR-003)

```python
def generate_causal_dag_data(
    n_variables: int = 5,
    n_samples: int = 1000,
    edge_probability: float = 0.3,
    noise_level: float = 0.1,
    hidden_fraction: float = 0.0,
    seed: int = 42
) -> CausalDAGData:
    """
    Generate synthetic observational data from a random causal DAG.
    
    Args:
        n_variables: Number of variables in the DAG
        n_samples: Number of observation samples
        edge_probability: Probability of edge between any two nodes
        noise_level: Noise in causal mechanisms (0.0-1.0)
        hidden_fraction: Fraction of variables to hide (latent variables)
        seed: Random seed
        
    Returns:
        CausalDAGData with observations, ground truth adjacency, and hidden info
        
    Example:
        >>> data = generate_causal_dag_data(n_variables=5, hidden_fraction=0.2)
        >>> data.observations.shape
        (1000, 4)  # One variable hidden
        >>> data.adjacency_matrix.shape
        (5, 5)     # Full ground truth
    """

def generate_chain_dag_data(
    n_variables: int = 5,
    n_samples: int = 1000,
    noise_level: float = 0.1,
    seed: int = 42
) -> CausalDAGData:
    """
    Generate data from a simple chain DAG: X1 -> X2 -> ... -> Xn
    
    Useful for demonstrating basic causal discovery before complex DAGs.
    """
```

---

## Visualization Functions

### Mermaid Diagrams (FR-006)

```python
def render_mermaid(
    diagram: str,
    theme: str = "github-light"
) -> None:
    """
    Render a Mermaid diagram in a Jupyter notebook cell.
    
    Args:
        diagram: Mermaid diagram specification string
        theme: Color theme ("github-light", "tokyo-night", "default")
        
    Example:
        >>> render_mermaid('''
        ... graph LR
        ...     A[Genome] --> B[Decoder]
        ...     B --> C[Phenotype]
        ...     C --> D[Evaluator]
        ...     D --> E[Fitness]
        ... ''')
    
    Note:
        Requires beautiful-mermaid package. Falls back to plain text if unavailable.
    """

# Pre-defined diagram templates
EVOLUTIONARY_LOOP_DIAGRAM: str  # Standard EA loop diagram
GENOME_PHENOTYPE_DIAGRAM: str   # Encoding/decoding pipeline
ISLAND_MODEL_DIAGRAM: str       # Island topology visualization
```

### Fitness Plots (FR-007)

```python
def plot_fitness_history(
    history: EvolutionHistory,
    show_std: bool = True,
    title: str = "Evolution Progress",
    figsize: tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot fitness over generations with best/mean/worst lines.
    
    Args:
        history: EvolutionHistory from evolution run
        show_std: Whether to show standard deviation band
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure (also displayed in notebook)
    """

def plot_fitness_comparison(
    histories: dict[str, EvolutionHistory],
    metric: Literal["best", "mean"] = "best",
    title: str = "Configuration Comparison"
) -> plt.Figure:
    """
    Compare multiple evolution runs on same axes.
    
    Args:
        histories: Dict mapping configuration names to histories
        metric: Which fitness metric to plot
        title: Plot title
    """
```

### Diversity Visualization (FR-008)

```python
def plot_population_diversity(
    population_genomes: np.ndarray,
    method: Literal["pca", "tsne"] = "pca",
    fitness_values: np.ndarray | None = None,
    title: str = "Population Diversity"
) -> plt.Figure:
    """
    Visualize population distribution in 2D projection.
    
    Args:
        population_genomes: (n_individuals, genome_length) array
        method: Dimensionality reduction method
        fitness_values: Optional fitness for color coding
        title: Plot title
        
    Returns:
        Scatter plot of projected population
    """

def plot_diversity_over_generations(
    history: EvolutionHistory,
    title: str = "Diversity Over Time"
) -> plt.Figure:
    """
    Plot population diversity metric across generations.
    """
```

### Pareto Visualization (FR-048)

```python
def plot_pareto_2d_projections(
    front: ParetoFront,
    highlight_indices: list[int] | None = None,
    figsize: tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create 3 subplot 2D projections of a 3-objective Pareto front.
    
    Subplots: obj0 vs obj1, obj0 vs obj2, obj1 vs obj2
    
    Args:
        front: ParetoFront with 3 objectives
        highlight_indices: Indices of solutions to highlight
        figsize: Figure dimensions
    """

def plot_pareto_3d_interactive(
    front: ParetoFront,
    color_by: Literal["crowding", "generation", "custom"] = "crowding",
    custom_colors: np.ndarray | None = None
) -> go.Figure:
    """
    Create interactive 3D Pareto front visualization with plotly.
    
    Args:
        front: ParetoFront with 3 objectives
        color_by: Coloring scheme
        custom_colors: Custom color values if color_by="custom"
        
    Returns:
        Plotly Figure with rotate/zoom/hover capabilities
    """

def plot_pareto_evolution(
    fronts: list[ParetoFront],
    generations: list[int],
    objective_pair: tuple[int, int] = (0, 1)
) -> plt.Figure:
    """
    Animate Pareto front evolution across generations.
    
    Args:
        fronts: List of Pareto fronts from different generations
        generations: Generation numbers corresponding to fronts
        objective_pair: Which two objectives to plot (for 2D view)
    """
```

### Speciation Visualization (FR-038)

```python
def plot_species_stacked_area(
    history: SpeciesHistory,
    max_species_shown: int = 10,
    title: str = "Species Composition Over Generations"
) -> plt.Figure:
    """
    Stacked area chart of species population composition.
    
    Args:
        history: SpeciesHistory from NEAT run
        max_species_shown: Limit species (others grouped as "Other")
        title: Plot title
    """

def plot_species_phylogeny(
    history: SpeciesHistory,
    lineage_tracker: Any | None = None
) -> plt.Figure:
    """
    Optional: Phylogenetic tree view of species relationships.
    
    Requires lineage tracking during evolution.
    """
```

---

## Terminology & Glossary (FR-009)

```python
def get_glossary() -> dict[str, TerminologyEntry]:
    """
    Get the full EA-to-ML terminology glossary.
    
    Returns:
        Dict mapping EA terms to TerminologyEntry objects
    """

def explain_term(ea_term: str) -> str:
    """
    Get a formatted explanation of an EA term.
    
    Args:
        ea_term: Term like "genome", "fitness", "crossover"
        
    Returns:
        Formatted string with ML analogy and explanation
        
    Example:
        >>> print(explain_term("fitness"))
        **Fitness** ≈ **-Loss** (ML)
        Biology: Survival/reproductive success
        In EA: The quality measure we maximize. Higher fitness = better solution.
        Example: For minimization problems, fitness = -objective_value
    """

def print_glossary_table() -> None:
    """
    Display the full glossary as a formatted table in notebook.
    """
```

---

## Statistical Utilities (FR-010)

```python
def population_stats(
    fitness_values: np.ndarray
) -> dict[str, float]:
    """
    Compute statistical summary of population fitness.
    
    Returns:
        Dict with keys: mean, std, min, max, median, q25, q75
    """

def convergence_test(
    history: EvolutionHistory,
    window: int = 10,
    threshold: float = 0.001
) -> tuple[bool, int | None]:
    """
    Test if evolution has converged.
    
    Args:
        history: Evolution history
        window: Number of generations to check
        threshold: Minimum improvement to not be considered converged
        
    Returns:
        (converged: bool, generation: int | None where convergence detected)
    """

def compare_runs_statistical(
    runs_a: list[float],
    runs_b: list[float],
    test: Literal["ttest", "mannwhitney"] = "mannwhitney"
) -> dict[str, float]:
    """
    Statistical comparison of two sets of evolution runs.
    
    Args:
        runs_a: Final fitness values from configuration A
        runs_b: Final fitness values from configuration B
        test: Statistical test to use
        
    Returns:
        Dict with p_value, effect_size, significant (at p<0.05)
    """
```

---

## Island Model Utilities (FR-020-023)

```python
def create_island_config(
    num_islands: int = 4,
    population_per_island: int = 50,
    topology: Literal["ring", "star", "fully_connected"] = "ring",
    migration_interval: int = 10,
    migration_rate: float = 0.1
) -> IslandConfig:
    """
    Create an island model configuration.
    """

def visualize_topology(
    config: IslandConfig
) -> None:
    """
    Render island topology as Mermaid diagram.
    """

def run_island_benchmark(
    config: IslandConfig,
    fitness_function: Callable,
    genome_factory: Callable,
    generations: int = 100,
    runs: int = 5
) -> BenchmarkResult:
    """
    Run island model benchmark and return timing results.
    """
```

---

## Backward Compatibility

This is version 1.0.0 - no backward compatibility constraints yet.

## Error Handling

All functions raise:
- `ValueError`: For invalid parameter values
- `ImportError`: For missing optional dependencies (with helpful message)
- `TypeError`: For incorrect argument types

No silent failures - all errors are explicit.
