"""
Tutorial utilities for the Evolve Framework.

This module provides synthetic data generators, visualization functions,
and terminology mapping for the tutorial notebooks.

Version: 1.0.0
Date: 2026-01-31
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.graph_objects as go


# =============================================================================
# DATA STRUCTURES (from data-model.md)
# =============================================================================


@dataclass
class BenchmarkFunction:
    """Configuration for a benchmark optimization function.
    
    Used in VectorGenome tutorials to provide continuous optimization
    benchmarks with known global optima for validation.
    
    Attributes:
        name: Human-readable function name (e.g., "Rastrigin")
        dimensions: Number of input dimensions
        bounds: Search space bounds (min, max) for each dimension
        global_optimum: Known optimal fitness value
        optimal_position: Position of global optimum (if unique)
        evaluate: Function to compute fitness at a position
    """
    name: str
    dimensions: int
    bounds: tuple[float, float]
    global_optimum: float
    optimal_position: np.ndarray | None
    evaluate: Callable[[np.ndarray], float] = field(repr=False)
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate fitness at position x."""
        return self.evaluate(x)


@dataclass
class SymbolicRegressionData:
    """Dataset for symbolic regression experiments.
    
    Used in SequenceGenome tutorials to provide synthetic data
    with known ground truth formulas for validation.
    
    Attributes:
        X_train: Training inputs (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_test: Test inputs
        y_test: Test targets
        true_expression: Ground truth formula string
        noise_level: Applied noise std (0.0-1.0 scale)
        seed: Random seed used for reproducibility
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    true_expression: str
    noise_level: float
    seed: int


@dataclass
class CausalDAGData:
    """Dataset with known causal structure for discovery experiments.
    
    Used in SCMGenome tutorials to provide synthetic observational
    data with known ground truth DAG for validation.
    
    Attributes:
        observations: Observed data (n_samples, n_variables) as DataFrame
        adjacency_matrix: Ground truth DAG (n_vars, n_vars)
        variable_names: Column names for variables
        hidden_variables: Variables removed from observations (latent)
        noise_level: Noise applied to causal mechanisms
        seed: Random seed used
    """
    observations: "pd.DataFrame"
    adjacency_matrix: np.ndarray
    variable_names: list[str]
    hidden_variables: list[str]
    noise_level: float
    seed: int
    
    def edge_accuracy(self, predicted: np.ndarray) -> dict[str, float]:
        """Compute precision, recall, F1 for edge recovery.
        
        Args:
            predicted: Predicted adjacency matrix (same shape as ground truth)
            
        Returns:
            Dict with keys: precision, recall, f1, accuracy
        """
        true_flat = self.adjacency_matrix.flatten() > 0
        pred_flat = predicted.flatten() > 0
        
        tp = np.sum(true_flat & pred_flat)
        fp = np.sum(~true_flat & pred_flat)
        fn = np.sum(true_flat & ~pred_flat)
        tn = np.sum(~true_flat & ~pred_flat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }


@dataclass
class EvolutionHistory:
    """Aggregated metrics from an evolutionary run.
    
    Used for convergence visualization in all tutorials.
    
    Attributes:
        generations: Generation indices [0, 1, 2, ...]
        best_fitness: Best fitness per generation
        mean_fitness: Population mean fitness
        worst_fitness: Worst fitness per generation
        std_fitness: Fitness standard deviation
        diversity: Population diversity metric (genotypic + phenotypic)
    """
    generations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    worst_fitness: list[float] = field(default_factory=list)
    std_fitness: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    
    @classmethod
    def from_callback_logs(cls, logs: list[dict[str, Any]]) -> "EvolutionHistory":
        """Construct from evolution callback logs.
        
        Args:
            logs: List of dicts with keys 'generation', 'best', 'mean', 'worst', 'std', 'diversity'
            
        Returns:
            Populated EvolutionHistory instance
        """
        history = cls()
        for log in logs:
            history.generations.append(log.get("generation", len(history.generations)))
            history.best_fitness.append(log.get("best", 0.0))
            history.mean_fitness.append(log.get("mean", 0.0))
            history.worst_fitness.append(log.get("worst", 0.0))
            history.std_fitness.append(log.get("std", 0.0))
            history.diversity.append(log.get("diversity", 0.0))
        return history
    
    def callback(self) -> Callable[[dict[str, Any]], None]:
        """Return a callback function for use with EvolutionEngine.
        
        Returns:
            Callback that appends metrics to this history
        """
        def _callback(metrics: dict[str, Any]) -> None:
            self.generations.append(metrics.get("generation", len(self.generations)))
            self.best_fitness.append(metrics.get("best", 0.0))
            self.mean_fitness.append(metrics.get("mean", 0.0))
            self.worst_fitness.append(metrics.get("worst", 0.0))
            self.std_fitness.append(metrics.get("std", 0.0))
            self.diversity.append(metrics.get("diversity", 0.0))
        return _callback


@dataclass
class SpeciesHistory:
    """Species population counts over generations.
    
    Used in NEAT/GraphGenome tutorials for speciation visualization.
    
    Attributes:
        generations: Generation indices
        species_counts: {species_id: [count_gen0, count_gen1, ...]}
        species_births: {species_id: birth_generation}
        species_extinctions: {species_id: extinction_generation}
    """
    generations: list[int] = field(default_factory=list)
    species_counts: dict[int, list[int]] = field(default_factory=dict)
    species_births: dict[int, int] = field(default_factory=dict)
    species_extinctions: dict[int, int] = field(default_factory=dict)
    
    def to_stacked_area_data(self) -> tuple[np.ndarray, list[str]]:
        """Convert to format for matplotlib stackplot.
        
        Returns:
            Tuple of (data array, species labels)
        """
        if not self.species_counts:
            return np.array([[]]), []
        
        species_ids = sorted(self.species_counts.keys())
        n_gens = len(self.generations)
        
        data = np.zeros((len(species_ids), n_gens))
        labels = []
        
        for i, sid in enumerate(species_ids):
            counts = self.species_counts[sid]
            # Pad with zeros if needed
            data[i, :len(counts)] = counts[:n_gens]
            labels.append(f"Species {sid}")
        
        return data, labels


@dataclass
class ParetoFront:
    """Non-dominated solutions from multi-objective optimization.
    
    Used in SCMGenome tutorials for Pareto visualization.
    
    Attributes:
        objectives: (n_solutions, n_objectives) fitness values
        objective_names: Names like ["data_fit", "sparsity", "simplicity"]
        solutions: Corresponding genomes/phenotypes
        generation: Generation when front was captured
    """
    objectives: np.ndarray
    objective_names: list[str]
    solutions: list[Any] = field(default_factory=list)
    generation: int = 0
    
    def dominates(self, a_idx: int, b_idx: int) -> bool:
        """Check if solution a dominates solution b.
        
        Solution a dominates b if a is at least as good in all objectives
        and strictly better in at least one.
        
        Args:
            a_idx: Index of first solution
            b_idx: Index of second solution
            
        Returns:
            True if a dominates b (assuming minimization)
        """
        a = self.objectives[a_idx]
        b = self.objectives[b_idx]
        return bool(np.all(a <= b) and np.any(a < b))
    
    def crowding_distances(self) -> np.ndarray:
        """Compute crowding distance for each solution.
        
        Crowding distance measures how spread out solutions are
        on the Pareto front - higher means more isolated.
        
        Returns:
            Array of crowding distances for each solution
        """
        n_solutions, n_objectives = self.objectives.shape
        distances = np.zeros(n_solutions)
        
        for obj_idx in range(n_objectives):
            # Sort by this objective
            sorted_indices = np.argsort(self.objectives[:, obj_idx])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Normalize by objective range
            obj_range = (
                self.objectives[sorted_indices[-1], obj_idx]
                - self.objectives[sorted_indices[0], obj_idx]
            )
            
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distances[sorted_indices[i]] += (
                        self.objectives[sorted_indices[i + 1], obj_idx]
                        - self.objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
        
        return distances


@dataclass
class TerminologyEntry:
    """Single term in the EA-to-ML glossary.
    
    Maps evolutionary algorithm terms to machine learning concepts.
    
    Attributes:
        ea_term: Evolutionary algorithm term (e.g., "Genome")
        ml_analogy: Machine learning equivalent (e.g., "Model weights")
        biology_origin: Biological inspiration (e.g., "DNA")
        explanation: Detailed explanation for learners
        example: Optional code/usage example
    """
    ea_term: str
    ml_analogy: str
    biology_origin: str
    explanation: str
    example: str | None = None


@dataclass
class IslandConfig:
    """Configuration for island model parallelism.
    
    Attributes:
        num_islands: Number of parallel populations (default: 4)
        population_per_island: Individuals per island (default: 50)
        migration_interval: Generations between migrations (default: 10)
        migration_rate: Fraction of population migrating (default: 0.1)
        topology: Connection topology ("ring", "star", "fully_connected")
    """
    num_islands: int = 4
    population_per_island: int = 50
    migration_interval: int = 10
    migration_rate: float = 0.1
    topology: Literal["ring", "star", "fully_connected"] = "ring"
    
    @property
    def total_population(self) -> int:
        """Total population across all islands."""
        return self.num_islands * self.population_per_island


@dataclass
class MigrationEvent:
    """Record of a migration between islands.
    
    Attributes:
        generation: Generation when migration occurred
        source_island: Index of source island
        destination_island: Index of destination island
        num_migrants: Number of individuals migrated
        migrant_fitness: Fitness values of migrants
    """
    generation: int
    source_island: int
    destination_island: int
    num_migrants: int
    migrant_fitness: list[float]


@dataclass
class BenchmarkResult:
    """Timing results from a benchmark comparison.
    
    Used for CPU/GPU and single/island performance comparisons.
    
    Attributes:
        configuration: Description (e.g., "CPU-single", "GPU-island")
        total_time_seconds: Total wall-clock time
        generations: Number of generations run
        population_size: Total population size
        final_best_fitness: Best fitness achieved
        generations_per_second: Throughput metric
    """
    configuration: str
    total_time_seconds: float
    generations: int
    population_size: int
    final_best_fitness: float
    generations_per_second: float = 0.0
    
    def __post_init__(self) -> None:
        if self.generations_per_second == 0.0 and self.total_time_seconds > 0:
            self.generations_per_second = self.generations / self.total_time_seconds
    
    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        """Compute speedup factor relative to baseline.
        
        Args:
            baseline: Reference benchmark result
            
        Returns:
            Speedup factor (>1 means faster than baseline)
        """
        if self.total_time_seconds <= 0:
            return float("inf")
        return baseline.total_time_seconds / self.total_time_seconds


# =============================================================================
# TERMINOLOGY GLOSSARY (FR-009)
# =============================================================================

TERMINOLOGY_GLOSSARY: dict[str, TerminologyEntry] = {
    "genome": TerminologyEntry(
        ea_term="Genome",
        ml_analogy="Model weights",
        biology_origin="DNA",
        explanation="Complete parameter encoding of a solution. Like neural network weights, "
                    "the genome contains all the information needed to produce behavior.",
        example="genome = np.random.randn(10)  # 10-dimensional parameter vector",
    ),
    "phenotype": TerminologyEntry(
        ea_term="Phenotype",
        ml_analogy="Model behavior/predictions",
        biology_origin="Organism",
        explanation="The decoded/evaluated form of a genome. The phenotype is what gets "
                    "evaluated for fitness, similar to how model predictions are what get scored.",
        example="phenotype = decoder.decode(genome)  # genome -> observable behavior",
    ),
    "fitness": TerminologyEntry(
        ea_term="Fitness",
        ml_analogy="-Loss (negative loss)",
        biology_origin="Survival/reproductive success",
        explanation="Quality measure we maximize. Higher fitness = better solution. "
                    "For minimization problems, fitness = -objective_value.",
        example="fitness = -mse_loss(predictions, targets)  # higher is better",
    ),
    "population": TerminologyEntry(
        ea_term="Population",
        ml_analogy="Ensemble of models",
        biology_origin="Species",
        explanation="Collection of candidate solutions maintained simultaneously. "
                    "Like maintaining multiple model instances to explore solution space.",
        example="population = [Individual() for _ in range(100)]",
    ),
    "generation": TerminologyEntry(
        ea_term="Generation",
        ml_analogy="Epoch/iteration",
        biology_origin="Lifespan",
        explanation="One full cycle of selection and variation. Each generation produces "
                    "a new population from the previous one.",
        example="for gen in range(100):  # run 100 generations",
    ),
    "selection": TerminologyEntry(
        ea_term="Selection",
        ml_analogy="Sampling by quality",
        biology_origin="Natural selection",
        explanation="Choosing individuals to be parents based on fitness. Higher fitness "
                    "= higher probability of being selected. Like importance sampling.",
        example="parents = tournament_select(population, k=3)  # select via tournament",
    ),
    "crossover": TerminologyEntry(
        ea_term="Crossover",
        ml_analogy="Weight interpolation",
        biology_origin="Sexual reproduction",
        explanation="Combining information from two parents to create offspring. "
                    "Similar to model interpolation or mixing weight matrices.",
        example="child = crossover(parent1, parent2, rate=0.5)",
    ),
    "mutation": TerminologyEntry(
        ea_term="Mutation",
        ml_analogy="Gradient noise / perturbation",
        biology_origin="Random mutation",
        explanation="Small random changes to a genome. Enables exploration of nearby "
                    "solutions, similar to adding noise to gradients or weights.",
        example="mutated = genome + np.random.randn(*genome.shape) * 0.1",
    ),
    "elitism": TerminologyEntry(
        ea_term="Elitism",
        ml_analogy="Best checkpoint preservation",
        biology_origin="N/A (artificial)",
        explanation="Preserving the top performers unchanged into the next generation. "
                    "Guarantees best fitness never decreases, like saving best model checkpoint.",
        example="next_gen[:n_elite] = sorted(population, key=fitness)[-n_elite:]",
    ),
    "diversity": TerminologyEntry(
        ea_term="Diversity",
        ml_analogy="Ensemble variance",
        biology_origin="Biodiversity",
        explanation="Spread of population in solution space. High diversity = exploring "
                    "widely; low diversity = converging. Prevents premature convergence.",
        example="diversity = np.mean(pairwise_distances(genomes))",
    ),
}


def get_glossary() -> dict[str, TerminologyEntry]:
    """Get the full EA-to-ML terminology glossary.
    
    Returns:
        Dict mapping EA terms (lowercase) to TerminologyEntry objects
    """
    return TERMINOLOGY_GLOSSARY.copy()


def print_glossary_table() -> None:
    """Display the full glossary as a formatted table in notebook.
    
    Renders using IPython display for rich formatting, falls back
    to plain text if not in notebook environment.
    """
    try:
        from IPython.display import HTML, display
        
        rows = []
        for entry in TERMINOLOGY_GLOSSARY.values():
            rows.append(
                f"<tr><td><b>{entry.ea_term}</b></td>"
                f"<td>{entry.ml_analogy}</td>"
                f"<td><i>{entry.biology_origin}</i></td>"
                f"<td>{entry.explanation}</td></tr>"
            )
        
        html = f"""
        <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #f0f0f0;">
                <th style="padding: 8px; border: 1px solid #ddd;">EA Term</th>
                <th style="padding: 8px; border: 1px solid #ddd;">ML Analogy</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Biology Origin</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Explanation</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
        </table>
        """
        display(HTML(html))
    except ImportError:
        # Fallback to plain text
        print("=" * 80)
        print(f"{'EA Term':<15} {'ML Analogy':<25} {'Biology':<15}")
        print("=" * 80)
        for entry in TERMINOLOGY_GLOSSARY.values():
            print(f"{entry.ea_term:<15} {entry.ml_analogy:<25} {entry.biology_origin:<15}")
        print("=" * 80)


# =============================================================================
# BENCHMARK FUNCTIONS (FR-001)
# =============================================================================


def sphere_function(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2)
    
    Simplest benchmark - unimodal, symmetric, separable.
    Global optimum: f(0, ..., 0) = 0
    Recommended bounds: [-5.12, 5.12]
    
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
    return float(np.sum(x ** 2))


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function: highly multimodal benchmark.
    
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Has approximately 10^n local minima, making it challenging.
    Global optimum: f(0, ..., 0) = 0
    Recommended bounds: [-5.12, 5.12]
    
    Args:
        x: Input vector of any dimension
        
    Returns:
        Scalar fitness value (lower is better)
    """
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function: banana-shaped valley.
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Classic benchmark with narrow curved valley.
    Global optimum: f(1, ..., 1) = 0
    Recommended bounds: [-5, 10]
    
    Args:
        x: Input vector of dimension >= 2
        
    Returns:
        Scalar fitness value (lower is better)
    """
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    )


def ackley_function(x: np.ndarray) -> float:
    """Ackley function: many local minima with global minimum at origin.
    
    Complex multimodal function with exponential and cosine terms.
    Global optimum: f(0, ..., 0) = 0
    Recommended bounds: [-32.768, 32.768]
    
    Args:
        x: Input vector of any dimension
        
    Returns:
        Scalar fitness value (lower is better)
    """
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    return float(
        -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20
        + np.e
    )


# Pre-configured benchmark instances
_BENCHMARKS: dict[str, Callable[[], BenchmarkFunction]] = {
    "sphere": lambda: BenchmarkFunction(
        name="Sphere",
        dimensions=10,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        optimal_position=np.zeros(10),
        evaluate=sphere_function,
    ),
    "rastrigin": lambda: BenchmarkFunction(
        name="Rastrigin",
        dimensions=10,
        bounds=(-5.12, 5.12),
        global_optimum=0.0,
        optimal_position=np.zeros(10),
        evaluate=rastrigin_function,
    ),
    "rosenbrock": lambda: BenchmarkFunction(
        name="Rosenbrock",
        dimensions=10,
        bounds=(-5.0, 10.0),
        global_optimum=0.0,
        optimal_position=np.ones(10),
        evaluate=rosenbrock_function,
    ),
    "ackley": lambda: BenchmarkFunction(
        name="Ackley",
        dimensions=10,
        bounds=(-32.768, 32.768),
        global_optimum=0.0,
        optimal_position=np.zeros(10),
        evaluate=ackley_function,
    ),
}


def get_benchmark(name: str, dimensions: int = 10) -> BenchmarkFunction:
    """Get a benchmark function configuration by name.
    
    Args:
        name: One of "sphere", "rastrigin", "rosenbrock", "ackley"
        dimensions: Number of input dimensions (default: 10)
        
    Returns:
        BenchmarkFunction dataclass with metadata and evaluate method
        
    Raises:
        ValueError: If name is not recognized
        
    Example:
        >>> benchmark = get_benchmark("rastrigin", dimensions=5)
        >>> benchmark.evaluate(np.zeros(5))
        0.0
    """
    name_lower = name.lower()
    if name_lower not in _BENCHMARKS:
        available = ", ".join(_BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    
    benchmark = _BENCHMARKS[name_lower]()
    
    # Update dimensions if different from default
    if dimensions != benchmark.dimensions:
        optimal_pos = (
            np.zeros(dimensions) if name_lower in ("sphere", "rastrigin", "ackley")
            else np.ones(dimensions)
        )
        benchmark = BenchmarkFunction(
            name=benchmark.name,
            dimensions=dimensions,
            bounds=benchmark.bounds,
            global_optimum=benchmark.global_optimum,
            optimal_position=optimal_pos,
            evaluate=benchmark.evaluate,
        )
    
    return benchmark


# =============================================================================
# SYMBOLIC REGRESSION DATA (FR-002)
# =============================================================================


def generate_polynomial_data(
    degree: int = 2,
    n_samples: int = 100,
    noise_level: float = 0.1,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> SymbolicRegressionData:
    """Generate synthetic data from a polynomial function.
    
    Creates data from a polynomial with random coefficients, split into
    train/test sets.
    
    Args:
        degree: Polynomial degree (1-5 recommended)
        n_samples: Total number of samples
        noise_level: Standard deviation of Gaussian noise (0.0-1.0)
        seed: Random seed for reproducibility
        test_fraction: Fraction of data reserved for testing
        
    Returns:
        SymbolicRegressionData with train/test split and true expression
        
    Example:
        >>> data = generate_polynomial_data(degree=2, seed=42)
        >>> data.true_expression
        '0.37*x^2 + -0.23*x + 0.13'
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0.0, 1.0]")
    
    rng = np.random.default_rng(seed)
    
    # Generate coefficients
    coefficients = rng.uniform(-1, 1, size=degree + 1)
    
    # Generate X values
    X = rng.uniform(-2, 2, size=(n_samples, 1))
    
    # Compute y = sum(coef_i * x^i)
    y = np.zeros(n_samples)
    expr_parts = []
    for i, coef in enumerate(coefficients):
        y += coef * (X[:, 0] ** i)
        if i == 0:
            expr_parts.append(f"{coef:.2f}")
        elif i == 1:
            expr_parts.append(f"{coef:.2f}*x")
        else:
            expr_parts.append(f"{coef:.2f}*x^{i}")
    
    # Add noise
    if noise_level > 0:
        noise_std = noise_level * np.std(y)
        y += rng.normal(0, noise_std, size=n_samples)
    
    # Train/test split
    n_test = int(n_samples * test_fraction)
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    true_expression = " + ".join(reversed(expr_parts))
    
    return SymbolicRegressionData(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        true_expression=true_expression,
        noise_level=noise_level,
        seed=seed,
    )


def generate_trigonometric_data(
    frequency: float = 1.0,
    n_samples: int = 100,
    noise_level: float = 0.1,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> SymbolicRegressionData:
    """Generate synthetic data from sin/cos functions.
    
    True expression: a*sin(frequency*x) + b*cos(frequency*x) + c
    
    Args:
        frequency: Frequency of oscillation
        n_samples: Total number of samples
        noise_level: Standard deviation of Gaussian noise (0.0-1.0)
        seed: Random seed for reproducibility
        test_fraction: Fraction for testing
        
    Returns:
        SymbolicRegressionData with train/test split
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0.0, 1.0]")
    
    rng = np.random.default_rng(seed)
    
    # Random coefficients
    a = rng.uniform(0.5, 2.0)
    b = rng.uniform(0.5, 2.0)
    c = rng.uniform(-1, 1)
    
    # Generate X values
    X = rng.uniform(-np.pi, np.pi, size=(n_samples, 1))
    
    # Compute y
    y = a * np.sin(frequency * X[:, 0]) + b * np.cos(frequency * X[:, 0]) + c
    
    # Add noise
    if noise_level > 0:
        noise_std = noise_level * np.std(y)
        y += rng.normal(0, noise_std, size=n_samples)
    
    # Train/test split
    n_test = int(n_samples * test_fraction)
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    true_expression = f"{a:.2f}*sin({frequency:.1f}*x) + {b:.2f}*cos({frequency:.1f}*x) + {c:.2f}"
    
    return SymbolicRegressionData(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        true_expression=true_expression,
        noise_level=noise_level,
        seed=seed,
    )


def generate_composite_data(
    complexity: Literal["simple", "medium", "complex"] = "medium",
    n_features: int = 2,
    n_samples: int = 200,
    noise_level: float = 0.1,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> SymbolicRegressionData:
    """Generate data from composite mathematical expressions.
    
    Complexity levels:
    - simple: x + y, x*y
    - medium: x^2 + sin(y), x*y + exp(-x)
    - complex: nested functions, multiple interactions
    
    Args:
        complexity: Expression complexity level
        n_features: Number of input features (1-3)
        n_samples: Total number of samples
        noise_level: Gaussian noise std (0.0-1.0)
        seed: Random seed
        test_fraction: Fraction for testing
        
    Returns:
        SymbolicRegressionData with composite expression
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0.0, 1.0]")
    if n_features < 1 or n_features > 3:
        raise ValueError("n_features must be 1, 2, or 3")
    
    rng = np.random.default_rng(seed)
    
    # Generate X values
    X = rng.uniform(-2, 2, size=(n_samples, n_features))
    
    # Generate y based on complexity
    if complexity == "simple":
        if n_features == 1:
            y = X[:, 0] + 1
            expr = "x + 1"
        else:
            y = X[:, 0] + X[:, 1]
            expr = "x + y"
    elif complexity == "medium":
        if n_features == 1:
            y = X[:, 0] ** 2 + np.sin(X[:, 0])
            expr = "x^2 + sin(x)"
        else:
            y = X[:, 0] ** 2 + np.sin(X[:, 1])
            expr = "x^2 + sin(y)"
    else:  # complex
        if n_features == 1:
            y = np.sin(X[:, 0] ** 2) * np.exp(-np.abs(X[:, 0]))
            expr = "sin(x^2) * exp(-|x|)"
        else:
            y = np.sin(X[:, 0] * X[:, 1]) + np.exp(-X[:, 0] ** 2)
            expr = "sin(x*y) + exp(-x^2)"
    
    # Add noise
    if noise_level > 0:
        noise_std = noise_level * np.std(y)
        y += rng.normal(0, noise_std, size=n_samples)
    
    # Train/test split
    n_test = int(n_samples * test_fraction)
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return SymbolicRegressionData(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        true_expression=expr,
        noise_level=noise_level,
        seed=seed,
    )


# =============================================================================
# CAUSAL DATA GENERATION (FR-003, FR-004, FR-005)
# =============================================================================


def generate_causal_dag_data(
    n_variables: int = 5,
    n_samples: int = 1000,
    edge_probability: float = 0.3,
    noise_level: float = 0.1,
    hidden_fraction: float = 0.0,
    seed: int = 42,
) -> CausalDAGData:
    """Generate synthetic observational data from a random causal DAG.
    
    Creates a random DAG and simulates linear Gaussian causal mechanisms.
    
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
    import pandas as pd
    
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0.0, 1.0]")
    if not 0.0 <= hidden_fraction < 1.0:
        raise ValueError("hidden_fraction must be in [0.0, 1.0)")
    
    rng = np.random.default_rng(seed)
    
    # Generate random DAG (lower triangular to ensure acyclicity)
    adjacency = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        for j in range(i):
            if rng.random() < edge_probability:
                adjacency[j, i] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    
    # Variable names
    var_names = [f"X{i}" for i in range(n_variables)]
    
    # Simulate data
    data = np.zeros((n_samples, n_variables))
    noise_std = 0.5 + noise_level * 0.5  # Scale noise
    
    for i in range(n_variables):
        # Data = sum(parent effects) + noise
        parent_effect = data @ adjacency[:, i]
        data[:, i] = parent_effect + rng.normal(0, noise_std, n_samples)
    
    # Determine hidden variables
    n_hidden = int(n_variables * hidden_fraction)
    hidden_indices = rng.choice(n_variables, size=n_hidden, replace=False).tolist()
    hidden_vars = [var_names[i] for i in hidden_indices]
    
    # Create observed DataFrame
    observed_indices = [i for i in range(n_variables) if i not in hidden_indices]
    observed_data = data[:, observed_indices]
    observed_names = [var_names[i] for i in observed_indices]
    
    observations = pd.DataFrame(observed_data, columns=observed_names)
    
    return CausalDAGData(
        observations=observations,
        adjacency_matrix=adjacency,
        variable_names=var_names,
        hidden_variables=hidden_vars,
        noise_level=noise_level,
        seed=seed,
    )


def generate_chain_dag_data(
    n_variables: int = 5,
    n_samples: int = 1000,
    noise_level: float = 0.1,
    seed: int = 42,
) -> CausalDAGData:
    """Generate data from a simple chain DAG: X1 -> X2 -> ... -> Xn
    
    Useful for demonstrating basic causal discovery before complex DAGs.
    
    Args:
        n_variables: Number of variables in the chain
        n_samples: Number of samples
        noise_level: Noise in mechanisms (0.0-1.0)
        seed: Random seed
        
    Returns:
        CausalDAGData with chain structure
    """
    import pandas as pd
    
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be in [0.0, 1.0]")
    
    rng = np.random.default_rng(seed)
    
    # Chain adjacency: each variable depends on previous
    adjacency = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        adjacency[i - 1, i] = rng.uniform(0.5, 1.5)
    
    var_names = [f"X{i}" for i in range(n_variables)]
    
    # Simulate data
    data = np.zeros((n_samples, n_variables))
    noise_std = 0.5 + noise_level * 0.5
    
    data[:, 0] = rng.normal(0, 1, n_samples)  # Root node
    for i in range(1, n_variables):
        data[:, i] = adjacency[i - 1, i] * data[:, i - 1] + rng.normal(0, noise_std, n_samples)
    
    observations = pd.DataFrame(data, columns=var_names)
    
    return CausalDAGData(
        observations=observations,
        adjacency_matrix=adjacency,
        variable_names=var_names,
        hidden_variables=[],
        noise_level=noise_level,
        seed=seed,
    )


# =============================================================================
# MERMAID RENDERING (FR-006)
# =============================================================================

# Pre-defined diagram templates
EVOLUTIONARY_LOOP_DIAGRAM = """
graph TD
    A[Initialize Population] --> B[Evaluate Fitness]
    B --> C{Termination?}
    C -->|No| D[Selection]
    D --> E[Crossover]
    E --> F[Mutation]
    F --> B
    C -->|Yes| G[Return Best]
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
"""

GENOME_PHENOTYPE_DIAGRAM = """
graph LR
    A[Genome<br/>Parameter Encoding] --> B[Decoder]
    B --> C[Phenotype<br/>Expressed Form]
    C --> D[Evaluator]
    D --> E[Fitness Score]
    
    style A fill:#fff3e0
    style C fill:#e8f5e9
    style E fill:#fce4ec
"""

ISLAND_MODEL_DIAGRAM = """
graph TD
    subgraph Island 1
        A1[Population 1]
    end
    subgraph Island 2
        A2[Population 2]
    end
    subgraph Island 3
        A3[Population 3]
    end
    subgraph Island 4
        A4[Population 4]
    end
    
    A1 <-->|Migration| A2
    A2 <-->|Migration| A3
    A3 <-->|Migration| A4
    A4 <-->|Migration| A1
    
    style A1 fill:#e3f2fd
    style A2 fill:#e8f5e9
    style A3 fill:#fff3e0
    style A4 fill:#fce4ec
"""


def render_mermaid(
    diagram: str,
    theme: str = "default",
    filename: str = None,
) -> None:
    """Render a Mermaid diagram in a Jupyter notebook cell.
    
    Uses mermaid-cli (mmdc) for high-quality PNG rendering. Falls back to 
    displaying the raw diagram text if mermaid-cli is unavailable.
    
    Args:
        diagram: Mermaid diagram specification string
        theme: Mermaid theme ("default", "forest", "dark", "neutral")
        filename: Output filename (auto-generated if None)
        
    Example:
        >>> render_mermaid('''
        ... graph LR
        ...     A[Genome] --> B[Decoder]
        ...     B --> C[Phenotype]
        ... ''', filename="genome_pipeline")
    
    Note:
        Requires mermaid-cli: npm install -g @mermaid-js/mermaid-cli puppeteer
    """
    try:
        from docs.tutorials.utils.mermaid_renderer import show_mermaid
        import hashlib
        
        # Auto-generate filename from diagram content if not provided
        if filename is None:
            # Create a short hash of the diagram content
            content_hash = hashlib.md5(diagram.encode()).hexdigest()[:8]
            filename = f"diagram_{content_hash}"
        
        show_mermaid(diagram, filename, title="", theme=theme)
    except ImportError:
        # Fallback: display as code block
        try:
            from IPython.display import Markdown, display
            display(Markdown(f"```mermaid\n{diagram}\n```"))
        except ImportError:
            print("Mermaid diagram (install mermaid-cli for rendering):")
            print(diagram)


# =============================================================================
# FITNESS PLOTS (FR-007)
# =============================================================================


def plot_fitness_history(
    history: EvolutionHistory,
    show_std: bool = True,
    title: str = "Evolution Progress",
    figsize: tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Plot fitness over generations with best/mean/worst lines.
    
    Args:
        history: EvolutionHistory from evolution run
        show_std: Whether to show standard deviation band
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure (also displayed in notebook)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gens = history.generations
    
    ax.plot(gens, history.best_fitness, label="Best", color="green", linewidth=2)
    ax.plot(gens, history.mean_fitness, label="Mean", color="blue", linewidth=1.5)
    ax.plot(gens, history.worst_fitness, label="Worst", color="red", linewidth=1, alpha=0.7)
    
    if show_std and history.std_fitness:
        mean = np.array(history.mean_fitness)
        std = np.array(history.std_fitness)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2, color="blue", label="±1 std")
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_fitness_comparison(
    histories: dict[str, EvolutionHistory],
    metric: Literal["best", "mean"] = "best",
    title: str = "Configuration Comparison",
    figsize: tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Compare multiple evolution runs on same axes.
    
    Args:
        histories: Dict mapping configuration names to histories
        metric: Which fitness metric to plot ("best" or "mean")
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, history in histories.items():
        gens = history.generations
        if metric == "best":
            values = history.best_fitness
        else:
            values = history.mean_fitness
        ax.plot(gens, values, label=name, linewidth=2)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"{metric.capitalize()} Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# DIVERSITY VISUALIZATION (FR-008)
# =============================================================================


def plot_population_diversity(
    population_genomes: np.ndarray,
    method: Literal["pca", "tsne"] = "pca",
    fitness_values: np.ndarray | None = None,
    title: str = "Population Diversity",
    figsize: tuple[int, int] = (8, 6),
) -> "plt.Figure":
    """Visualize population distribution in 2D projection.
    
    Args:
        population_genomes: (n_individuals, genome_length) array
        method: Dimensionality reduction method ("pca" or "tsne")
        fitness_values: Optional fitness for color coding
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Scatter plot of projected population
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reduce to 2D
    if method == "pca":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(population_genomes)
    else:  # tsne
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(population_genomes) - 1))
        coords = reducer.fit_transform(population_genomes)
    
    if fitness_values is not None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=fitness_values, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Fitness")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_diversity_over_generations(
    history: EvolutionHistory,
    title: str = "Diversity Over Time",
    figsize: tuple[int, int] = (10, 4),
) -> "plt.Figure":
    """Plot population diversity metric across generations.
    
    Args:
        history: EvolutionHistory with diversity data
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(history.generations, history.diversity, color="purple", linewidth=2)
    ax.fill_between(history.generations, 0, history.diversity, alpha=0.3, color="purple")
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# PARETO VISUALIZATION (FR-048, FR-049, FR-050)
# =============================================================================


def plot_pareto_2d_projections(
    front: ParetoFront,
    highlight_indices: list[int] | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> "plt.Figure":
    """Create 2D pairwise projections of a Pareto front.
    
    For 3-objective problems, creates 3 subplots showing all objective pairs.
    For 2-objective, creates single plot.
    
    Args:
        front: ParetoFront with objectives data
        highlight_indices: Indices of solutions to highlight
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure with projection subplots
    """
    import matplotlib.pyplot as plt
    
    n_obj = front.objectives.shape[1]
    
    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]
        pairs = [(0, 1)]
    else:
        # Create subplots for each pair
        n_pairs = n_obj * (n_obj - 1) // 2
        fig, axes = plt.subplots(1, min(n_pairs, 3), figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        pairs = [(i, j) for i in range(n_obj) for j in range(i + 1, n_obj)][:3]
    
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(
            front.objectives[:, i],
            front.objectives[:, j],
            alpha=0.7,
            c="steelblue",
        )
        
        if highlight_indices:
            ax.scatter(
                front.objectives[highlight_indices, i],
                front.objectives[highlight_indices, j],
                c="red",
                s=100,
                marker="*",
                label="Selected",
            )
        
        ax.set_xlabel(front.objective_names[i] if i < len(front.objective_names) else f"Obj {i}")
        ax.set_ylabel(front.objective_names[j] if j < len(front.objective_names) else f"Obj {j}")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Pareto Front (Generation {front.generation})")
    plt.tight_layout()
    return fig


def plot_pareto_3d_interactive(
    front: ParetoFront,
    color_by: Literal["crowding", "generation", "custom"] = "crowding",
    custom_colors: np.ndarray | None = None,
) -> "go.Figure":
    """Create interactive 3D Pareto front visualization with plotly.
    
    Args:
        front: ParetoFront with 3 objectives
        color_by: Coloring scheme ("crowding", "generation", "custom")
        custom_colors: Custom color values if color_by="custom"
        
    Returns:
        Plotly Figure with rotate/zoom/hover capabilities
    """
    import plotly.graph_objects as go
    
    if front.objectives.shape[1] < 3:
        raise ValueError("3D visualization requires at least 3 objectives")
    
    # Determine colors
    if color_by == "crowding":
        colors = front.crowding_distances()
        colors = np.where(np.isinf(colors), np.nanmax(colors[~np.isinf(colors)]) * 1.5, colors)
        colorbar_title = "Crowding Distance"
    elif color_by == "generation":
        colors = np.full(len(front.objectives), front.generation)
        colorbar_title = "Generation"
    else:
        colors = custom_colors if custom_colors is not None else np.arange(len(front.objectives))
        colorbar_title = "Custom"
    
    obj_names = front.objective_names if len(front.objective_names) >= 3 else ["Obj 0", "Obj 1", "Obj 2"]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=front.objectives[:, 0],
        y=front.objectives[:, 1],
        z=front.objectives[:, 2],
        mode="markers",
        marker=dict(
            size=6,
            color=colors,
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
            opacity=0.8,
        ),
        hovertemplate=(
            f"{obj_names[0]}: %{{x:.3f}}<br>"
            f"{obj_names[1]}: %{{y:.3f}}<br>"
            f"{obj_names[2]}: %{{z:.3f}}<br>"
            "<extra></extra>"
        ),
    )])
    
    fig.update_layout(
        title=f"Pareto Front (Generation {front.generation})",
        scene=dict(
            xaxis_title=obj_names[0],
            yaxis_title=obj_names[1],
            zaxis_title=obj_names[2],
        ),
        width=800,
        height=600,
    )
    
    return fig


def plot_pareto_evolution(
    fronts: list[ParetoFront],
    generations: list[int],
    objective_pair: tuple[int, int] = (0, 1),
    figsize: tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Visualize Pareto front evolution across generations.
    
    Args:
        fronts: List of Pareto fronts from different generations
        generations: Generation numbers corresponding to fronts
        objective_pair: Which two objectives to plot (for 2D view)
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure showing front progression
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(fronts)))
    
    i, j = objective_pair
    
    for front, gen, color in zip(fronts, generations, colors):
        ax.scatter(
            front.objectives[:, i],
            front.objectives[:, j],
            c=[color],
            alpha=0.6,
            label=f"Gen {gen}",
        )
    
    ax.set_xlabel(f"Objective {i}")
    ax.set_ylabel(f"Objective {j}")
    ax.set_title("Pareto Front Evolution")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_crowding_distance_visual(
    front: ParetoFront,
    objective_pair: tuple[int, int] = (0, 1),
    figsize: tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Visualize crowding distance concept on Pareto front.
    
    Shows how crowding distance is computed for each solution,
    with larger markers for more isolated (higher crowding) solutions.
    
    Args:
        front: ParetoFront to visualize
        objective_pair: Which objectives to show
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure with crowding visualization
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    i, j = objective_pair
    crowding = front.crowding_distances()
    
    # Normalize crowding for marker sizing
    finite_mask = ~np.isinf(crowding)
    if np.any(finite_mask):
        max_crowding = np.max(crowding[finite_mask])
        sizes = np.where(np.isinf(crowding), 300, 50 + 200 * crowding / max_crowding)
    else:
        sizes = np.full(len(crowding), 100)
    
    scatter = ax.scatter(
        front.objectives[:, i],
        front.objectives[:, j],
        s=sizes,
        c=crowding,
        cmap="plasma",
        alpha=0.7,
        edgecolors="black",
    )
    
    plt.colorbar(scatter, ax=ax, label="Crowding Distance")
    
    ax.set_xlabel(front.objective_names[i] if i < len(front.objective_names) else f"Obj {i}")
    ax.set_ylabel(front.objective_names[j] if j < len(front.objective_names) else f"Obj {j}")
    ax.set_title("Crowding Distance Visualization\n(Larger markers = more isolated)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SPECIATION VISUALIZATION (FR-038)
# =============================================================================


def plot_species_stacked_area(
    history: SpeciesHistory,
    max_species_shown: int = 10,
    title: str = "Species Composition Over Generations",
    figsize: tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Stacked area chart of species population composition.
    
    Args:
        history: SpeciesHistory from NEAT run
        max_species_shown: Limit species shown (others grouped as "Other")
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure with stacked area chart
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data, labels = history.to_stacked_area_data()
    
    if len(labels) == 0:
        ax.text(0.5, 0.5, "No species data", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    # Limit to top species
    if len(labels) > max_species_shown:
        # Sum counts across generations to find top species
        total_counts = data.sum(axis=1)
        top_indices = np.argsort(total_counts)[-max_species_shown:]
        other = data[~np.isin(np.arange(len(labels)), top_indices)].sum(axis=0)
        
        data = np.vstack([data[top_indices], other])
        labels = [labels[i] for i in top_indices] + ["Other"]
    
    ax.stackplot(history.generations, data, labels=labels, alpha=0.8)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Count")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_species_phylogeny(
    history: SpeciesHistory,
    lineage_tracker: Any | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> "plt.Figure":
    """Optional: Phylogenetic tree view of species relationships.
    
    Requires lineage tracking during evolution.
    
    Args:
        history: SpeciesHistory with birth/extinction data
        lineage_tracker: Optional lineage tracking object
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure with phylogenetic tree
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if not history.species_births:
        ax.text(0.5, 0.5, "No phylogeny data available", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    # Simple timeline visualization
    species_ids = sorted(history.species_births.keys())
    
    for i, sid in enumerate(species_ids):
        birth = history.species_births[sid]
        death = history.species_extinctions.get(sid, max(history.generations) if history.generations else 0)
        
        ax.plot([birth, death], [i, i], linewidth=3, solid_capstyle="round")
        ax.scatter([birth], [i], marker="o", s=50, zorder=5)
        if sid in history.species_extinctions:
            ax.scatter([death], [i], marker="x", s=50, color="red", zorder=5)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Species")
    ax.set_yticks(range(len(species_ids)))
    ax.set_yticklabels([f"Species {sid}" for sid in species_ids])
    ax.set_title("Species Lifespans")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    return fig


# =============================================================================
# CAUSAL GRAPH VISUALIZATION (FR-054)
# =============================================================================


def plot_causal_graph_comparison(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    variable_names: list[str] | None = None,
    title: str = "Predicted vs Ground Truth Causal Graph",
    figsize: tuple[int, int] = (14, 6),
) -> "plt.Figure":
    """Visualize causal graphs with ground truth comparison.
    
    Shows predicted graph, ground truth, and highlights differences.
    
    Args:
        predicted: Predicted adjacency matrix
        ground_truth: True adjacency matrix
        variable_names: Names for variables
        title: Plot title
        figsize: Figure dimensions
        
    Returns:
        Matplotlib Figure with side-by-side comparison
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    n_vars = predicted.shape[0]
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(n_vars)]
    
    def draw_graph(ax: plt.Axes, adj: np.ndarray, graph_title: str, edge_colors: dict | None = None) -> None:
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] != 0:
                    G.add_edge(variable_names[i], variable_names[j])
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=700)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        # Draw edges
        if edge_colors:
            for edge in G.edges():
                color = edge_colors.get(edge, "gray")
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], edge_color=color, arrows=True)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color="gray")
        
        ax.set_title(graph_title)
        ax.axis("off")
    
    # Predicted
    draw_graph(axes[0], predicted, "Predicted")
    
    # Ground Truth
    draw_graph(axes[1], ground_truth, "Ground Truth")
    
    # Difference (True Positives=green, False Positives=red, False Negatives=orange)
    edge_colors = {}
    for i, vi in enumerate(variable_names):
        for j, vj in enumerate(variable_names):
            pred_edge = predicted[i, j] != 0
            true_edge = ground_truth[i, j] != 0
            
            if pred_edge and true_edge:
                edge_colors[(vi, vj)] = "green"  # True positive
            elif pred_edge and not true_edge:
                edge_colors[(vi, vj)] = "red"  # False positive
    
    # Draw with all edges (pred | true)
    combined = np.logical_or(predicted != 0, ground_truth != 0).astype(float)
    draw_graph(axes[2], combined, "Comparison\n(Green=TP, Red=FP)", edge_colors)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


# =============================================================================
# STATISTICAL UTILITIES (FR-010)
# =============================================================================


def compute_population_stats(fitness_values: np.ndarray) -> dict[str, float]:
    """Compute statistical summary of population fitness.
    
    Args:
        fitness_values: Array of fitness values
        
    Returns:
        Dict with keys: mean, std, min, max, median, q25, q75
    """
    return {
        "mean": float(np.mean(fitness_values)),
        "std": float(np.std(fitness_values)),
        "min": float(np.min(fitness_values)),
        "max": float(np.max(fitness_values)),
        "median": float(np.median(fitness_values)),
        "q25": float(np.percentile(fitness_values, 25)),
        "q75": float(np.percentile(fitness_values, 75)),
    }


def convergence_test(
    history: EvolutionHistory,
    window: int = 10,
    threshold: float = 0.001,
) -> tuple[bool, int | None]:
    """Test if evolution has converged.
    
    Checks if improvement over recent generations is below threshold.
    
    Args:
        history: Evolution history
        window: Number of generations to check
        threshold: Minimum improvement to not be considered converged
        
    Returns:
        Tuple of (converged: bool, generation: int | None where convergence detected)
    """
    if len(history.best_fitness) < window:
        return False, None
    
    for i in range(window, len(history.best_fitness)):
        recent = history.best_fitness[i - window:i]
        improvement = abs(max(recent) - min(recent))
        
        if improvement < threshold:
            return True, history.generations[i]
    
    return False, None


def compare_runs_statistical(
    runs_a: list[float],
    runs_b: list[float],
    test: Literal["ttest", "mannwhitney"] = "mannwhitney",
) -> dict[str, float]:
    """Statistical comparison of two sets of evolution runs.
    
    Args:
        runs_a: Final fitness values from configuration A
        runs_b: Final fitness values from configuration B
        test: Statistical test to use
        
    Returns:
        Dict with p_value, effect_size, significant (at p<0.05)
    """
    from scipy import stats
    
    a = np.array(runs_a)
    b = np.array(runs_b)
    
    if test == "ttest":
        statistic, p_value = stats.ttest_ind(a, b)
    else:
        statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    effect_size = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
    
    return {
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "significant": bool(p_value < 0.05),
        "statistic": float(statistic),
    }


# =============================================================================
# ISLAND MODEL UTILITIES (FR-020-023)
# =============================================================================


def create_island_config(
    num_islands: int = 4,
    population_per_island: int = 50,
    topology: Literal["ring", "star", "fully_connected"] = "ring",
    migration_interval: int = 10,
    migration_rate: float = 0.1,
) -> IslandConfig:
    """Create an island model configuration.
    
    Args:
        num_islands: Number of parallel populations
        population_per_island: Individuals per island
        topology: Connection topology
        migration_interval: Generations between migrations
        migration_rate: Fraction of population migrating
        
    Returns:
        Configured IslandConfig instance
    """
    return IslandConfig(
        num_islands=num_islands,
        population_per_island=population_per_island,
        topology=topology,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
    )


def visualize_topology(config: IslandConfig) -> None:
    """Render island topology as Mermaid diagram.
    
    Args:
        config: Island configuration to visualize
    """
    if config.topology == "ring":
        edges = []
        for i in range(config.num_islands):
            next_i = (i + 1) % config.num_islands
            edges.append(f"    I{i} <--> I{next_i}")
    elif config.topology == "star":
        edges = [f"    I0 <--> I{i}" for i in range(1, config.num_islands)]
    else:  # fully_connected
        edges = []
        for i in range(config.num_islands):
            for j in range(i + 1, config.num_islands):
                edges.append(f"    I{i} <--> I{j}")
    
    diagram = f"""
graph TD
{chr(10).join(edges)}
"""
    render_mermaid(diagram)


# =============================================================================
# GPU UTILITIES (T147a - Graceful Degradation)
# =============================================================================


def check_gpu_available() -> dict[str, Any]:
    """Check GPU availability with graceful degradation.
    
    Detects PyTorch CUDA, JAX GPU, or reports CPU-only mode.
    
    Returns:
        Dict with keys:
        - available: bool - whether GPU is available
        - backend: str - detected backend ("pytorch", "jax", "cpu")
        - device_name: str - GPU name if available
        - message: str - informative message for users
    """
    result = {
        "available": False,
        "backend": "cpu",
        "device_name": "CPU",
        "message": "",
    }
    
    # Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            result["available"] = True
            result["backend"] = "pytorch"
            result["device_name"] = torch.cuda.get_device_name(0)
            result["message"] = f"GPU available: {result['device_name']} (PyTorch CUDA)"
            return result
    except ImportError:
        pass
    
    # Try JAX
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            result["available"] = True
            result["backend"] = "jax"
            result["device_name"] = str(gpu_devices[0])
            result["message"] = f"GPU available: {result['device_name']} (JAX)"
            return result
    except ImportError:
        pass
    
    # CPU fallback
    result["message"] = (
        "No GPU detected. Running in CPU mode.\n"
        "GPU sections will show placeholder benchmarks.\n"
        "Install PyTorch CUDA or JAX with GPU support for acceleration."
    )
    return result


# =============================================================================
# ERP UTILITIES (Tutorial 06 - Evolvable Reproduction Protocols)
# =============================================================================


@dataclass
class ERPHistory:
    """Evolution history with ERP-specific metrics.
    
    Extends EvolutionHistory with protocol-specific tracking.
    
    Attributes:
        generations: Generation indices
        best_fitness: Best fitness per generation
        mean_fitness: Mean fitness per generation
        worst_fitness: Worst fitness per generation
        std_fitness: Fitness standard deviation
        diversity: Genotypic diversity
        mating_success_rate: Fraction of individuals that successfully mated
        protocol_diversity: Diversity of protocol parameters
        mean_matchability_threshold: Average distance threshold (if using distance-based)
        mean_intent_threshold: Average fitness threshold (if using fitness threshold intent)
        recovery_events: List of generations where recovery was triggered
    """
    generations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    worst_fitness: list[float] = field(default_factory=list)
    std_fitness: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    mating_success_rate: list[float] = field(default_factory=list)
    protocol_diversity: list[float] = field(default_factory=list)
    mean_matchability_threshold: list[float] = field(default_factory=list)
    mean_intent_threshold: list[float] = field(default_factory=list)
    recovery_events: list[int] = field(default_factory=list)


def plot_protocol_evolution(
    history: ERPHistory,
    param_name: str,
    ax: "plt.Axes | None" = None,
    show_std: bool = True
) -> "plt.Figure":
    """Plot evolution of protocol parameters over time.
    
    Args:
        history: ERPHistory with parameter tracking
        param_name: Parameter to plot ("max_distance", "intent_threshold", etc.)
        ax: Matplotlib axes (creates new figure if None)
        show_std: Whether to show standard deviation bands
        
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    
    # Map parameter name to history attribute
    param_map = {
        "max_distance": "mean_matchability_threshold",
        "matchability_threshold": "mean_matchability_threshold",
        "intent_threshold": "mean_intent_threshold",
        "protocol_diversity": "protocol_diversity",
        "mating_success": "mating_success_rate"
    }
    
    attr_name = param_map.get(param_name, param_name)
    
    if not hasattr(history, attr_name):
        raise ValueError(f"Unknown parameter: {param_name}")
    
    values = getattr(history, attr_name)
    
    if len(values) == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    ax.plot(history.generations, values, linewidth=2, label=f"Mean {param_name}")
    
    if show_std and hasattr(history, f"{attr_name}_std"):
        std_values = getattr(history, f"{attr_name}_std")
        ax.fill_between(
            history.generations,
            np.array(values) - np.array(std_values),
            np.array(values) + np.array(std_values),
            alpha=0.3,
            label="±1 std"
        )
    
    ax.set_xlabel("Generation")
    ax.set_ylabel(param_name.replace("_", " ").title())
    ax.set_title(f"Evolution of {param_name.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_mating_network(
    population: "Population",
    mating_events: list[tuple[int, int]] | None = None,
    generation: int = 0,
    layout: str = "spring",
    ax: "plt.Axes | None" = None
) -> "plt.Figure":
    """Visualize mating network as a graph.
    
    Args:
        population: Population with individuals
        mating_events: List of (parent1_idx, parent2_idx) tuples
        generation: Generation number for title
        layout: NetworkX layout ("spring", "circular", "kamada_kawai")
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        matplotlib Figure
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("NetworkX required for mating network visualization")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    # Create graph
    G = nx.Graph()
    n_individuals = len(population.individuals)
    
    # Add nodes
    for i in range(n_individuals):
        fitness = population.individuals[i].fitness.values[0] if population.individuals[i].fitness else 0.0
        G.add_node(i, fitness=fitness)
    
    # Add edges from mating events
    if mating_events:
        for parent1_idx, parent2_idx in mating_events:
            if G.has_edge(parent1_idx, parent2_idx):
                G[parent1_idx][parent2_idx]['weight'] += 1
            else:
                G.add_edge(parent1_idx, parent2_idx, weight=1)
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Node colors by fitness
    fitness_values = [G.nodes[i]['fitness'] for i in G.nodes()]
    
    # Node sizes by degree
    degrees = [G.degree(i) * 50 + 100 for i in G.nodes()]
    
    # Draw
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=fitness_values,
        node_size=degrees,
        cmap='viridis',
        alpha=0.7,
        ax=ax
    )
    
    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges] if edges else []
    
    if edges:
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 0.5 for w in weights],
            alpha=0.5,
            ax=ax
        )
    
    ax.set_title(f"Mating Network (Generation {generation})")
    ax.axis('off')
    
    # Add colorbar for fitness
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(fitness_values), vmax=max(fitness_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Fitness')
    
    return fig


def compute_reproductive_skew(mating_counts: list[int]) -> float:
    """Compute Gini coefficient for reproductive skew.
    
    Higher values indicate more unequal reproduction (few individuals
    monopolize mating opportunities).
    
    Args:
        mating_counts: Number of offspring per individual
        
    Returns:
        Gini coefficient (0.0 = perfect equality, 1.0 = complete monopoly)
    """
    if len(mating_counts) == 0:
        return 0.0
    
    # Sort counts
    sorted_counts = np.sort(mating_counts)
    n = len(sorted_counts)
    
    # Gini coefficient formula
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    
    return float(gini)


def visualize_acceptance_matrix(
    population: "Population",
    matchability_function: "MatchabilityFunction",
    ax: "plt.Axes | None" = None
) -> "plt.Figure":
    """Visualize pairwise acceptance matrix as heatmap.
    
    Shows which individuals accept which potential mates.
    
    Args:
        population: Population to analyze
        matchability_function: Matchability function to evaluate
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        matplotlib Figure with heatmap
    """
    import matplotlib.pyplot as plt
    from random import Random
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    n = len(population.individuals)
    acceptance_matrix = np.zeros((n, n))
    
    # Evaluate all pairs
    rng = Random(42)
    for i in range(n):
        for j in range(n):
            if i == j:
                acceptance_matrix[i, j] = -1  # Self-mating marked as -1
                continue
            
            # This is simplified - actual implementation would use MateContext
            # For tutorial purposes, we'll use a distance-based heuristic
            try:
                ind_i = population.individuals[i]
                ind_j = population.individuals[j]
                
                # Calculate genetic distance if possible
                if hasattr(ind_i.genome, 'genes') and hasattr(ind_j.genome, 'genes'):
                    distance = np.linalg.norm(ind_i.genome.genes - ind_j.genome.genes)
                    
                    # Check against threshold if available
                    if hasattr(matchability_function, 'parameters'):
                        threshold = matchability_function.parameters.get('max_distance', float('inf'))
                        acceptance_matrix[i, j] = 1.0 if distance <= threshold else 0.0
                    else:
                        acceptance_matrix[i, j] = 1.0  # Accept all
                else:
                    acceptance_matrix[i, j] = 1.0  # Default accept
            except:
                acceptance_matrix[i, j] = 0.5  # Unknown
    
    # Plot heatmap
    im = ax.imshow(acceptance_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xlabel("Potential Mate Index")
    ax.set_ylabel("Evaluator Index")
    ax.set_title("Pairwise Mate Acceptance Matrix")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accept (1) / Reject (0) / Self (-1)")
    
    return fig


def plot_recovery_events(
    history: ERPHistory,
    recovery_log: list[dict],
    ax: "plt.Axes | None" = None
) -> "plt.Figure":
    """Plot population size over time with recovery events marked.
    
    Args:
        history: ERPHistory with population size tracking
        recovery_log: List of dicts with keys 'generation', 'type', 'count'
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure
    
    # Plot population size (approximated from mating success rate)
    if hasattr(history, 'population_size'):
        ax.plot(history.generations, history.population_size, linewidth=2, label="Population Size")
    else:
        # Estimate from mating success rate
        estimated_size = [100 * rate for rate in history.mating_success_rate]  # Placeholder
        ax.plot(history.generations, estimated_size, linewidth=2, label="Effective Population", alpha=0.7)
    
    # Mark recovery events
    for event in recovery_log:
        gen = event['generation']
        recovery_type = event.get('type', 'unknown')
        count = event.get('count', 0)
        
        ax.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
        ax.annotate(
            f"{recovery_type}\n(+{count})",
            xy=(gen, ax.get_ylim()[1] * 0.9),
            ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5)
        )
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Size")
    ax.set_title("Population Size with Recovery Events")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def protocol_diversity_metric(population: "Population") -> float:
    """Compute Shannon entropy over protocol types.
    
    Args:
        population: Population with reproduction protocols
        
    Returns:
        Normalized entropy (0.0 = all same, 1.0 = maximum diversity)
    """
    from collections import Counter
    
    # Extract protocol types
    protocol_types = []
    for ind in population.individuals:
        if hasattr(ind, 'reproduction_protocol') and ind.reproduction_protocol:
            protocol = ind.reproduction_protocol
            # Create signature from protocol components
            sig = (
                protocol.intent_policy.policy_type if hasattr(protocol, 'intent_policy') else 'unknown',
                protocol.matchability.function_type if hasattr(protocol, 'matchability') else 'unknown',
                protocol.crossover_spec.crossover_type if hasattr(protocol, 'crossover_spec') else 'unknown'
            )
            protocol_types.append(str(sig))
        else:
            protocol_types.append('no_protocol')
    
    # Count frequencies
    counts = Counter(protocol_types)
    n = len(protocol_types)
    
    if n == 0 or len(counts) == 1:
        return 0.0
    
    # Shannon entropy
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(counts))
    
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def create_erp_glossary() -> dict[str, dict[str, str]]:
    """Create ERP terminology mapping for tutorials.
    
    Returns:
        Dict mapping terms to definitions across multiple domains
    """
    return {
        "Intent Policy": {
            "definition": "Determines when an individual is willing to reproduce",
            "biology": "Fertility cycle, estrus, sexual maturity",
            "game_theory": "Decision to enter game or tournament",
            "ml": "Action probability, policy activation"
        },
        "Matchability Function": {
            "definition": "Determines which partners an individual will accept",
            "biology": "Mate choice, sexual selection, assortative mating",
            "game_theory": "Opponent selection strategy",
            "ml": "Partner value function, compatibility score"
        },
        "Crossover Protocol": {
            "definition": "Specifies how offspring genomes are constructed",
            "biology": "Recombination, meiosis, genetic inheritance",
            "game_theory": "Payoff distribution, credit assignment",
            "ml": "Model averaging, weight mixing strategy"
        },
        "Recovery Strategy": {
            "definition": "Mechanism to prevent population collapse",
            "biology": "Immigration, founder effect, rescue colonization",
            "game_theory": "Safety net, minimum viable population",
            "ml": "Exploration bonus, diversity maintenance"
        },
        "Reproduction Protocol": {
            "definition": "Complete specification of individual's reproductive behavior",
            "biology": "Mating strategy, life history strategy",
            "game_theory": "Player strategy profile",
            "ml": "Meta-policy, reproductive controller"
        },
        "Mutual Consent": {
            "definition": "Both partners must accept each other to mate",
            "biology": "Bilateral mate choice, courtship acceptance",
            "game_theory": "Mutual agreement, bilateral trade",
            "ml": "Bidirectional validation"
        },
        "Protocol Inheritance": {
            "definition": "How offspring acquire reproduction protocols from parents",
            "biology": "Cultural transmission, learned behavior",
            "game_theory": "Strategy copying, social learning",
            "ml": "Policy transfer, knowledge distillation"
        },
        "Reproductive Skew": {
            "definition": "Inequality in reproductive success across population",
            "biology": "Sexual selection intensity, variance in mating success",
            "game_theory": "Winner-take-all dynamics, competition outcome",
            "ml": "Reward distribution inequality"
        }
    }


def print_erp_glossary_table():
    """Print ERP terminology mapping as formatted table."""
    glossary = create_erp_glossary()
    
    print("=" * 100)
    print("ERP TERMINOLOGY MAPPING".center(100))
    print("=" * 100)
    print(f"{'Term':<25} {'Biology':<25} {'Game Theory':<25} {'ML/RL':<25}")
    print("-" * 100)
    
    for term, mappings in glossary.items():
        print(f"{term:<25} {mappings['biology']:<25} {mappings['game_theory']:<25} {mappings['ml']:<25}")
    
    print("=" * 100)


# ERP Diagram Constants
ERP_EVALUATION_DIAGRAM = """
graph TD
    A[Population] --> B{Intent Check}
    B -->|Willing| C[Find Partners]
    B -->|Not Willing| A
    C --> D{Matchability Check}
    D -->|Mutual Accept| E[Mate Pair]
    D -->|Reject| C
    E --> F[Crossover Protocol]
    F --> G[Offspring Creation]
    G --> H{Protocol Inheritance}
    H -->|50/50 single-parent| I[Offspring with Protocol]
    I -->|Mutation| J[Next Generation]
"""

PROTOCOL_INHERITANCE_DIAGRAM = """
graph LR
    P1[Parent 1 Protocol] --> C{Coin Flip}
    P2[Parent 2 Protocol] --> C
    C -->|Heads| O1[Offspring gets P1]
    C -->|Tails| O2[Offspring gets P2]
    O1 --> M1{Mutation?}
    O2 --> M2{Mutation?}
    M1 -->|Yes| F1[Mutated Protocol]
    M1 -->|No| F2[Unchanged Protocol]
    M2 -->|Yes| F3[Mutated Protocol]
    M2 -->|No| F4[Unchanged Protocol]
"""

SEXUAL_SELECTION_DIAGRAM = """
graph TD
    M[Males: Accept All] --> C[Competition]
    F[Females: Fitness-Based Choice] --> C
    C --> H[High-Fitness Males]
    C --> L[Low-Fitness Males]
    H -->|Many mates| O1[Many Offspring]
    L -->|Few/No mates| O2[Few/No Offspring]
    O1 --> S[Reproductive Skew]
    O2 --> S
"""

STANDARD_VS_ERP_DIAGRAM = """
graph LR
    subgraph Standard EA
        A1[Random Pairing] --> B1[Fixed Crossover]
        B1 --> C1[Fixed Mutation]
        C1 --> D1[Offspring]
    end
    
    subgraph ERP Evolution
        A2[Intent Policy] --> B2{Willing?}
        B2 -->|Yes| C2[Matchability Check]
        C2 -->|Accept| D2[Protocol-Specified Crossover]
        D2 --> E2[Offspring with Inherited Protocol]
    end
"""


# =============================================================================
# ERP-SPECIFIC UTILITY FUNCTIONS
# =============================================================================


def protocol_diversity_metric(population) -> float:
    """
    Compute Shannon entropy of protocol types in population.
    
    Args:
        population: Population object with individuals having protocol field
        
    Returns:
        Normalized diversity metric (0-1), where:
        - 0 = all identical protocols
        - 1 = maximum diversity
    """
    from collections import Counter
    import math
    
    # Extract protocol intent types
    intent_types = []
    for ind in population.individuals:
        if ind.protocol is not None:
            intent_types.append(ind.protocol.intent.type)
    
    if not intent_types:
        return 0.0
    
    # Compute Shannon entropy
    counts = Counter(intent_types)
    n = len(intent_types)
    entropy = -sum((count / n) * math.log2(count / n) for count in counts.values())
    
    # Normalize by maximum entropy (log2 of number of unique types)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_reproductive_skew_gini(mating_events: list) -> float:
    """
    Compute Gini coefficient for reproductive skew.
    
    Measures inequality in mating success across individuals.
    
    Args:
        mating_events: List of mating events with parent1_id and parent2_id
        
    Returns:
        Gini coefficient (0-1), where:
        - 0 = perfect equality (all mate equally)
        - 1 = perfect inequality (one individual monopolizes mating)
    """
    from collections import Counter
    
    # Count matings per individual
    mate_counts = Counter()
    for event in mating_events:
        mate_counts[event.parent1_id] += 1
        mate_counts[event.parent2_id] += 1
    
    if not mate_counts:
        return 0.0
    
    # Convert to array of counts
    counts = np.array(list(mate_counts.values()), dtype=float)
    n = len(counts)
    
    if n <= 1:
        return 0.0
    
    # Compute Gini coefficient
    # Formula: G = sum(|x_i - x_j|) / (2 * n * sum(x_i))
    numerator = np.sum(np.abs(counts[:, None] - counts[None, :]))
    denominator = 2 * n * np.sum(counts)
    
    return numerator / denominator if denominator > 0 else 0.0


def plot_mating_network_v2(population, generation: int, layout: str = "spring"):
    """
    Visualize mating network as a graph.
    
    Args:
        population: Population with individuals
        generation: Current generation number
        layout: Layout algorithm ("spring", "circular", "kamada_kawai")
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("NetworkX required for mating network visualization")
        return None
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (individuals)
    for i, ind in enumerate(population.individuals):
        fitness = ind.fitness.values[0] if ind.fitness else 0.0
        G.add_node(i, fitness=fitness)
    
    # For now, add edges based on matchability acceptance
    # (In a real scenario, you'd track actual mating events)
    rng = np.random.RandomState(generation)
    for i in range(len(population.individuals)):
        for j in range(i + 1, min(i + 5, len(population.individuals))):
            if rng.random() > 0.5:  # Simplified acceptance
                G.add_edge(i, j)
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Node sizes based on fitness
    node_sizes = [G.nodes[i].get('fitness', 0) * 100 + 100 for i in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    ax.set_title(f"Mating Network - Generation {generation}")
    ax.axis('off')
    
    return fig


def _visualize_acceptance_matrix(population):
    """
    Create heatmap showing which individuals accept which mates.
    
    Args:
        population: Population with protocol-enabled individuals
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required for acceptance matrix visualization")
        return None
    
    n = len(population.individuals)
    acceptance_matrix = np.zeros((n, n))
    
    # Simplified: check if fitnesses would be acceptable
    # (Real implementation would use matchability evaluators)
    for i, ind_i in enumerate(population.individuals):
        for j, ind_j in enumerate(population.individuals):
            if i != j and ind_i.protocol and ind_j.protocol:
                # Simplified acceptance based on fitness threshold
                if (ind_i.fitness and ind_j.fitness and 
                    ind_i.protocol.matchability.type == "fitness_threshold"):
                    threshold = ind_i.protocol.matchability.params.get("min_fitness", 0.5)
                    fitness_j = ind_j.fitness.values[0] if ind_j.fitness else 0.0
                    acceptance_matrix[i, j] = 1.0 if fitness_j >= threshold else 0.0
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(acceptance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xlabel("Potential Mate")
    ax.set_ylabel("Evaluator Individual")
    ax.set_title("Acceptance Matrix (Green = Accept, Red = Reject)")
    
    plt.colorbar(im, ax=ax, label="Acceptance")
    
    return fig


# NOTE:
# The `plot_recovery_events` function is defined earlier in this module.
# This duplicate implementation has been removed to avoid shadowing the
# original definition and to keep a single, consolidated implementation.


def create_erp_glossary() -> dict:
    """
    Create glossary of ERP terms and definitions.
    
    Returns:
        Dictionary mapping terms to definitions
    """
    return {
        "Intent Policy": "Defines when an individual is willing to reproduce (e.g., always, fitness-based, probabilistic)",
        "Matchability Function": "Criteria for accepting/rejecting potential mates (e.g., fitness threshold, similarity)",
        "Crossover Protocol": "Specifies which crossover operator to use (uniform, single-point, etc.)",
        "Reproduction Protocol": "Complete mating strategy combining intent, matchability, and crossover",
        "Protocol Inheritance": "How offspring receive protocols from parents (typically 50/50 single-parent)",
        "Protocol Mutation": "Random changes to protocol parameters or types during evolution",
        "Recovery Mechanism": "Strategies to prevent population collapse (immigration, cloning, constraint relaxation)",
        "Sexual Selection": "Reproductive competition leading to trait evolution (choosy vs. eager)",
        "Assortative Mating": "Preference for similar mates, can lead to speciation",
        "Reproductive Skew": "Inequality in mating success (measured by Gini coefficient)",
        "Mutual Consent": "Both individuals must accept each other for mating to occur",
        "Protocol Diversity": "Variety of mating strategies in the population (Shannon entropy)",
    }


def print_erp_glossary_table():
    """Print ERP glossary as formatted table."""
    glossary = create_erp_glossary()
    
    print("=" * 80)
    print("ERP TERMINOLOGY GLOSSARY")
    print("=" * 80)
    print()
    
    max_term_len = max(len(term) for term in glossary.keys())
    
    for term, definition in sorted(glossary.items()):
        print(f"{term:<{max_term_len}}  |  {definition}")
    
    print()
    print("=" * 80)
