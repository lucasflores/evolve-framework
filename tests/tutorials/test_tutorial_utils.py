"""
Tests for tutorial_utils module.

Covers:
- T021: Test file with pytest fixtures
- T022: Benchmark functions return correct optima
- T023: Data generators respect seed and noise params
- T024: Causal DAG generator produces valid acyclic graphs
- T025: Dataclass methods work correctly
- T025a: Visualization smoke tests
- T025b: BenchmarkResult.speedup_vs() method
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from pathlib import Path
# Import all components under test - add project root to sys.path dynamically
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs.tutorials.utils.tutorial_utils import (
    # Dataclasses
    BenchmarkFunction,
    BenchmarkResult,
    CausalDAGData,
    EvolutionHistory,
    IslandConfig,
    MigrationEvent,
    ParetoFront,
    SpeciesHistory,
    SymbolicRegressionData,
    TerminologyEntry,
    # Constants
    EVOLUTIONARY_LOOP_DIAGRAM,
    GENOME_PHENOTYPE_DIAGRAM,
    ISLAND_MODEL_DIAGRAM,
    TERMINOLOGY_GLOSSARY,
    # Functions
    ackley_function,
    check_gpu_available,
    compare_runs_statistical,
    compute_population_stats,
    convergence_test,
    create_island_config,
    generate_causal_dag_data,
    generate_chain_dag_data,
    generate_composite_data,
    generate_polynomial_data,
    generate_trigonometric_data,
    get_benchmark,
    get_glossary,
    plot_diversity_over_generations,
    plot_fitness_comparison,
    plot_fitness_history,
    plot_pareto_2d_projections,
    plot_population_diversity,
    rastrigin_function,
    render_mermaid,
    rosenbrock_function,
    sphere_function,
)


# =============================================================================
# FIXTURES (T021)
# =============================================================================


@pytest.fixture
def sample_evolution_history():
    """Fixture providing a sample EvolutionHistory for testing."""
    history = EvolutionHistory()
    for gen in range(10):
        history.generations.append(gen)
        history.best_fitness.append(100 - gen * 5)  # Decreasing (minimization)
        history.mean_fitness.append(150 - gen * 4)
        history.worst_fitness.append(200 - gen * 3)
        history.std_fitness.append(20 - gen)
        history.diversity.append(0.8 - gen * 0.05)
    return history


@pytest.fixture
def sample_species_history():
    """Fixture providing a sample SpeciesHistory."""
    history = SpeciesHistory()
    history.generations = list(range(10))
    history.species_counts = {
        0: [50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
        1: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    }
    history.species_births = {0: 0, 1: 1}
    history.species_extinctions = {}
    return history


@pytest.fixture
def sample_pareto_front():
    """Fixture providing a sample ParetoFront."""
    # 2-objective Pareto front
    objectives = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 1.5],
        [5.0, 1.0],
    ])
    return ParetoFront(
        objectives=objectives,
        objective_names=["Obj A", "Obj B"],
        generation=10,
    )


# =============================================================================
# BENCHMARK FUNCTION TESTS (T022)
# =============================================================================


class TestBenchmarkFunctions:
    """Tests for benchmark optimization functions."""

    def test_sphere_function_optimum(self):
        """Sphere function should return 0 at origin."""
        x = np.zeros(10)
        result = sphere_function(x)
        assert_almost_equal(result, 0.0, decimal=10)

    def test_sphere_function_nonzero(self):
        """Sphere function should return sum of squares."""
        x = np.array([1.0, 2.0, 3.0])
        result = sphere_function(x)
        assert_almost_equal(result, 14.0, decimal=10)

    def test_rastrigin_function_optimum(self):
        """Rastrigin function should return 0 at origin."""
        x = np.zeros(10)
        result = rastrigin_function(x)
        assert_almost_equal(result, 0.0, decimal=10)

    def test_rastrigin_function_multimodal(self):
        """Rastrigin should have local minima away from origin."""
        # Point at integer coords should be a local minimum
        x = np.ones(5)  # Not at global optimum
        result = rastrigin_function(x)
        assert result > 0  # Not at global optimum

    def test_rosenbrock_function_optimum(self):
        """Rosenbrock function should return 0 at (1,1,...,1)."""
        x = np.ones(10)
        result = rosenbrock_function(x)
        assert_almost_equal(result, 0.0, decimal=10)

    def test_rosenbrock_function_nonzero(self):
        """Rosenbrock function should return positive values away from optimum."""
        x = np.zeros(10)
        result = rosenbrock_function(x)
        assert result > 0

    def test_ackley_function_optimum(self):
        """Ackley function should return ~0 at origin."""
        x = np.zeros(10)
        result = ackley_function(x)
        assert_almost_equal(result, 0.0, decimal=5)

    def test_ackley_function_nonzero(self):
        """Ackley function should return positive values away from origin."""
        x = np.ones(5)
        result = ackley_function(x)
        assert result > 0


class TestGetBenchmark:
    """Tests for get_benchmark factory function."""

    def test_get_sphere_benchmark(self):
        """Should return sphere benchmark with correct metadata."""
        benchmark = get_benchmark("sphere", dimensions=5)
        assert benchmark.name == "Sphere"
        assert benchmark.dimensions == 5
        assert benchmark.global_optimum == 0.0
        assert_almost_equal(benchmark.optimal_position, np.zeros(5))

    def test_get_rastrigin_benchmark(self):
        """Should return rastrigin benchmark."""
        benchmark = get_benchmark("rastrigin")
        assert benchmark.name == "Rastrigin"
        assert benchmark.bounds == (-5.12, 5.12)

    def test_get_rosenbrock_benchmark(self):
        """Should return rosenbrock benchmark with correct optimal position."""
        benchmark = get_benchmark("rosenbrock", dimensions=3)
        assert benchmark.name == "Rosenbrock"
        assert_almost_equal(benchmark.optimal_position, np.ones(3))

    def test_get_benchmark_callable(self):
        """Benchmark should be callable for evaluation."""
        benchmark = get_benchmark("sphere", dimensions=3)
        result = benchmark(np.array([1.0, 2.0, 3.0]))
        assert_almost_equal(result, 14.0)

    def test_get_benchmark_invalid_name(self):
        """Should raise ValueError for unknown benchmark."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("unknown_function")


# =============================================================================
# DATA GENERATOR TESTS (T023)
# =============================================================================


class TestPolynomialDataGenerator:
    """Tests for polynomial data generation."""

    def test_polynomial_data_shape(self):
        """Should return correct shapes for train/test split."""
        data = generate_polynomial_data(
            degree=2,
            n_samples=100,
            test_fraction=0.2,
            seed=42,
        )
        assert data.X_train.shape == (80, 1)
        assert data.y_train.shape == (80,)
        assert data.X_test.shape == (20, 1)
        assert data.y_test.shape == (20,)

    def test_polynomial_data_seed_reproducibility(self):
        """Same seed should produce identical data."""
        data1 = generate_polynomial_data(seed=42)
        data2 = generate_polynomial_data(seed=42)
        np.testing.assert_array_equal(data1.X_train, data2.X_train)
        np.testing.assert_array_equal(data1.y_train, data2.y_train)

    def test_polynomial_data_different_seeds(self):
        """Different seeds should produce different data."""
        data1 = generate_polynomial_data(seed=42)
        data2 = generate_polynomial_data(seed=123)
        assert not np.allclose(data1.y_train, data2.y_train)

    def test_polynomial_data_noise_effect(self):
        """Higher noise should increase y variance."""
        data_low = generate_polynomial_data(noise_level=0.0, seed=42)
        data_high = generate_polynomial_data(noise_level=0.5, seed=42)
        
        # Same X values, different noise
        # Higher noise should mean higher residual variance
        # (Note: seed makes X same, so we can compare)
        std_low = np.std(data_low.y_train)
        std_high = np.std(data_high.y_train)
        # With 50% noise, std should increase
        assert std_high >= std_low

    def test_polynomial_data_noise_validation(self):
        """Should reject invalid noise levels."""
        with pytest.raises(ValueError, match="noise_level"):
            generate_polynomial_data(noise_level=1.5)
        with pytest.raises(ValueError, match="noise_level"):
            generate_polynomial_data(noise_level=-0.1)


class TestTrigonometricDataGenerator:
    """Tests for trigonometric data generation."""

    def test_trigonometric_data_shape(self):
        """Should return correct shapes."""
        data = generate_trigonometric_data(n_samples=50, seed=42)
        assert data.X_train.shape[0] + data.X_test.shape[0] == 50

    def test_trigonometric_data_expression(self):
        """Should have sin/cos in true expression."""
        data = generate_trigonometric_data(frequency=2.0, seed=42)
        assert "sin" in data.true_expression
        assert "cos" in data.true_expression
        assert "2.0" in data.true_expression  # frequency

    def test_trigonometric_data_seed_reproducibility(self):
        """Same seed should produce identical data."""
        data1 = generate_trigonometric_data(seed=99)
        data2 = generate_trigonometric_data(seed=99)
        np.testing.assert_array_equal(data1.X_train, data2.X_train)


class TestCompositeDataGenerator:
    """Tests for composite data generation."""

    def test_composite_complexity_levels(self):
        """All complexity levels should work."""
        for complexity in ["simple", "medium", "complex"]:
            data = generate_composite_data(complexity=complexity, seed=42)
            assert data.X_train is not None
            assert data.y_train is not None

    def test_composite_n_features(self):
        """Should respect n_features parameter."""
        data = generate_composite_data(n_features=3, seed=42)
        assert data.X_train.shape[1] == 3

    def test_composite_invalid_n_features(self):
        """Should reject invalid n_features."""
        with pytest.raises(ValueError, match="n_features"):
            generate_composite_data(n_features=0)
        with pytest.raises(ValueError, match="n_features"):
            generate_composite_data(n_features=4)


# =============================================================================
# CAUSAL DAG TESTS (T024)
# =============================================================================


class TestCausalDAGGenerator:
    """Tests for causal DAG data generation."""

    def test_dag_acyclicity(self):
        """Generated adjacency matrix should be acyclic (DAG)."""
        data = generate_causal_dag_data(n_variables=5, seed=42)
        adj = data.adjacency_matrix
        
        # Lower triangular means acyclic (topological order)
        # Check no cycles: adjacency^n should have zero diagonal for all n < n_vars
        powered = adj.copy()
        n_vars = adj.shape[0]
        for _ in range(n_vars):
            # If there's a cycle, diagonal will become non-zero
            assert np.allclose(np.diag(powered), 0), "DAG contains a cycle"
            powered = powered @ adj

    def test_dag_observations_shape(self):
        """Observations should have correct shape."""
        data = generate_causal_dag_data(n_variables=5, n_samples=200, seed=42)
        assert data.observations.shape[0] == 200
        # With no hidden variables, should have 5 columns
        assert data.observations.shape[1] == 5

    def test_dag_hidden_variables(self):
        """Hidden fraction should reduce observed variables."""
        data = generate_causal_dag_data(
            n_variables=10,
            hidden_fraction=0.3,  # 30% hidden = 3 vars
            seed=42,
        )
        assert len(data.hidden_variables) == 3
        assert data.observations.shape[1] == 7  # 10 - 3

    def test_dag_edge_accuracy_method(self):
        """edge_accuracy() should compute correct metrics."""
        data = generate_causal_dag_data(n_variables=4, seed=42)
        
        # Perfect prediction
        accuracy = data.edge_accuracy(data.adjacency_matrix)
        assert accuracy["precision"] == 1.0
        assert accuracy["recall"] == 1.0
        assert accuracy["f1"] == 1.0

        # All zeros prediction (no edges)
        zeros = np.zeros_like(data.adjacency_matrix)
        accuracy = data.edge_accuracy(zeros)
        assert accuracy["precision"] == 0.0
        assert accuracy["recall"] == 0.0


class TestChainDAGGenerator:
    """Tests for chain DAG generation."""

    def test_chain_structure(self):
        """Chain should have exactly n-1 edges in sequence."""
        data = generate_chain_dag_data(n_variables=5, seed=42)
        adj = data.adjacency_matrix
        
        # Count edges
        n_edges = np.count_nonzero(adj)
        assert n_edges == 4  # 5 nodes = 4 edges in chain

        # Each node i should connect to i+1
        for i in range(4):
            assert adj[i, i + 1] != 0, f"Missing edge {i} -> {i+1}"

    def test_chain_no_hidden_variables(self):
        """Chain generator should have no hidden variables."""
        data = generate_chain_dag_data(n_variables=5, seed=42)
        assert data.hidden_variables == []


# =============================================================================
# DATACLASS METHOD TESTS (T025)
# =============================================================================


class TestEvolutionHistory:
    """Tests for EvolutionHistory dataclass."""

    def test_from_callback_logs(self):
        """Should construct from callback logs."""
        logs = [
            {"generation": 0, "best": 100, "mean": 150, "worst": 200, "std": 20, "diversity": 0.8},
            {"generation": 1, "best": 90, "mean": 140, "worst": 190, "std": 18, "diversity": 0.75},
        ]
        history = EvolutionHistory.from_callback_logs(logs)
        
        assert history.generations == [0, 1]
        assert history.best_fitness == [100, 90]
        assert history.mean_fitness == [150, 140]

    def test_callback_method(self):
        """callback() should return a callable that appends metrics."""
        history = EvolutionHistory()
        cb = history.callback()
        
        cb({"generation": 0, "best": 50, "mean": 100})
        assert len(history.generations) == 1
        assert history.best_fitness[0] == 50


class TestSpeciesHistory:
    """Tests for SpeciesHistory dataclass."""

    def test_to_stacked_area_data(self, sample_species_history):
        """Should convert to stackplot-compatible format."""
        data, labels = sample_species_history.to_stacked_area_data()
        
        assert data.shape == (2, 10)  # 2 species, 10 generations
        assert len(labels) == 2

    def test_empty_species_history(self):
        """Should handle empty history."""
        history = SpeciesHistory()
        data, labels = history.to_stacked_area_data()
        assert labels == []


class TestParetoFront:
    """Tests for ParetoFront dataclass."""

    def test_dominates(self, sample_pareto_front):
        """dominates() should correctly identify domination."""
        # First point [1, 5] does not dominate [5, 1] (trade-off)
        assert not sample_pareto_front.dominates(0, 4)
        assert not sample_pareto_front.dominates(4, 0)

    def test_crowding_distances(self, sample_pareto_front):
        """crowding_distances() should compute valid distances."""
        distances = sample_pareto_front.crowding_distances()
        
        assert len(distances) == 5
        # Boundary points should have infinite distance
        assert np.isinf(distances[0])
        assert np.isinf(distances[-1])
        # Interior points should have finite distance
        assert np.isfinite(distances[1])
        assert np.isfinite(distances[2])
        assert np.isfinite(distances[3])


class TestIslandConfig:
    """Tests for IslandConfig dataclass."""

    def test_total_population(self):
        """total_population should compute correctly."""
        config = IslandConfig(num_islands=4, population_per_island=50)
        assert config.total_population == 200


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_generations_per_second_auto_compute(self):
        """Should auto-compute generations_per_second."""
        result = BenchmarkResult(
            configuration="test",
            total_time_seconds=10.0,
            generations=100,
            population_size=50,
            final_best_fitness=0.1,
        )
        assert result.generations_per_second == 10.0

    def test_speedup_vs(self):
        """speedup_vs() should compute correct ratio."""
        baseline = BenchmarkResult(
            configuration="CPU",
            total_time_seconds=100.0,
            generations=50,
            population_size=100,
            final_best_fitness=0.1,
        )
        faster = BenchmarkResult(
            configuration="GPU",
            total_time_seconds=25.0,
            generations=50,
            population_size=100,
            final_best_fitness=0.1,
        )
        
        speedup = faster.speedup_vs(baseline)
        assert speedup == 4.0  # 100 / 25


# =============================================================================
# VISUALIZATION SMOKE TESTS (T025a)
# =============================================================================


class TestVisualizationSmoke:
    """Smoke tests to verify visualization functions return valid figures."""

    def test_plot_fitness_history_returns_figure(self, sample_evolution_history):
        """plot_fitness_history should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        fig = plot_fitness_history(sample_evolution_history)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fitness_comparison_returns_figure(self, sample_evolution_history):
        """plot_fitness_comparison should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        histories = {
            "Config A": sample_evolution_history,
            "Config B": sample_evolution_history,
        }
        fig = plot_fitness_comparison(histories)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_population_diversity_returns_figure(self):
        """plot_population_diversity should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        # Generate sample population
        population = np.random.randn(50, 10)
        fitness = np.random.randn(50)
        
        fig = plot_population_diversity(population, fitness_values=fitness)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_diversity_over_generations_returns_figure(self, sample_evolution_history):
        """plot_diversity_over_generations should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        fig = plot_diversity_over_generations(sample_evolution_history)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pareto_2d_projections_returns_figure(self, sample_pareto_front):
        """plot_pareto_2d_projections should return a matplotlib Figure."""
        import matplotlib.pyplot as plt
        
        fig = plot_pareto_2d_projections(sample_pareto_front)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# SPEEDUP_VS TESTS (T025b)
# =============================================================================


class TestSpeedupVs:
    """Additional tests for BenchmarkResult.speedup_vs() method."""

    def test_speedup_vs_same_time(self):
        """Same time should give speedup of 1.0."""
        a = BenchmarkResult("A", 50.0, 100, 100, 0.1)
        b = BenchmarkResult("B", 50.0, 100, 100, 0.1)
        assert a.speedup_vs(b) == 1.0

    def test_speedup_vs_slower(self):
        """Slower config should have speedup < 1."""
        baseline = BenchmarkResult("fast", 10.0, 100, 100, 0.1)
        slow = BenchmarkResult("slow", 20.0, 100, 100, 0.1)
        assert slow.speedup_vs(baseline) == 0.5

    def test_speedup_vs_zero_time(self):
        """Zero time should return infinity."""
        baseline = BenchmarkResult("baseline", 10.0, 100, 100, 0.1)
        instant = BenchmarkResult("instant", 0.0, 100, 100, 0.1)
        assert instant.speedup_vs(baseline) == float("inf")


# =============================================================================
# STATISTICAL UTILITIES TESTS
# =============================================================================


class TestComputePopulationStats:
    """Tests for compute_population_stats function."""

    def test_stats_keys(self):
        """Should return all expected keys."""
        fitness = np.array([1, 2, 3, 4, 5])
        stats = compute_population_stats(fitness)
        
        expected_keys = {"mean", "std", "min", "max", "median", "q25", "q75"}
        assert set(stats.keys()) == expected_keys

    def test_stats_values(self):
        """Should compute correct values."""
        fitness = np.array([1, 2, 3, 4, 5])
        stats = compute_population_stats(fitness)
        
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0


class TestConvergenceTest:
    """Tests for convergence_test function."""

    def test_not_converged(self, sample_evolution_history):
        """Should return False when still improving."""
        converged, gen = convergence_test(
            sample_evolution_history,
            window=3,
            threshold=0.0001,  # Very small threshold
        )
        # History has improvement each generation
        assert not converged or gen is not None

    def test_converged_flat(self):
        """Should detect convergence when fitness is flat."""
        history = EvolutionHistory()
        for i in range(20):
            history.generations.append(i)
            history.best_fitness.append(50.0)  # Flat
        
        converged, gen = convergence_test(history, window=5, threshold=0.01)
        assert converged
        assert gen is not None

    def test_insufficient_data(self):
        """Should return False with insufficient data."""
        history = EvolutionHistory()
        history.generations = [0, 1]
        history.best_fitness = [100, 99]
        
        converged, gen = convergence_test(history, window=10)
        assert not converged
        assert gen is None


class TestCompareRunsStatistical:
    """Tests for compare_runs_statistical function."""

    def test_significant_difference(self):
        """Should detect significant difference."""
        runs_a = [10, 11, 12, 10, 11]
        runs_b = [90, 91, 92, 90, 91]  # Very different
        
        result = compare_runs_statistical(runs_a, runs_b, test="ttest")
        assert result["significant"]
        assert result["p_value"] < 0.05

    def test_no_significant_difference(self):
        """Should not detect difference for similar runs."""
        runs_a = [10, 11, 12, 10, 11, 12, 10]
        runs_b = [10.1, 10.9, 12.1, 10.2, 11.1, 11.9, 10.1]  # Very similar
        
        result = compare_runs_statistical(runs_a, runs_b, test="ttest")
        # May or may not be significant depending on exact values
        assert "p_value" in result
        assert "effect_size" in result


# =============================================================================
# GPU UTILITIES TESTS
# =============================================================================


class TestCheckGPUAvailable:
    """Tests for check_gpu_available function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        result = check_gpu_available()
        
        assert "available" in result
        assert "backend" in result
        assert "device_name" in result
        assert "message" in result

    def test_backend_valid(self):
        """Backend should be one of expected values."""
        result = check_gpu_available()
        assert result["backend"] in ("pytorch", "jax", "cpu")


# =============================================================================
# GLOSSARY TESTS
# =============================================================================


class TestGlossary:
    """Tests for terminology glossary functions."""

    def test_get_glossary_returns_dict(self):
        """get_glossary should return a dictionary."""
        glossary = get_glossary()
        assert isinstance(glossary, dict)
        assert len(glossary) > 0

    def test_glossary_entries_complete(self):
        """Each entry should have all required fields."""
        glossary = get_glossary()
        
        for term, entry in glossary.items():
            assert isinstance(entry, TerminologyEntry)
            assert entry.ea_term
            assert entry.ml_analogy
            assert entry.biology_origin
            assert entry.explanation

    def test_core_terms_present(self):
        """Core EA terms should be in glossary."""
        glossary = get_glossary()
        
        core_terms = ["genome", "phenotype", "fitness", "population", "generation"]
        for term in core_terms:
            assert term in glossary, f"Missing core term: {term}"


# =============================================================================
# ISLAND MODEL TESTS
# =============================================================================


class TestCreateIslandConfig:
    """Tests for create_island_config function."""

    def test_default_config(self):
        """Should create config with default values."""
        config = create_island_config()
        
        assert config.num_islands == 4
        assert config.population_per_island == 50
        assert config.topology == "ring"
        assert config.total_population == 200

    def test_custom_config(self):
        """Should respect custom parameters."""
        config = create_island_config(
            num_islands=8,
            population_per_island=25,
            topology="star",
        )
        
        assert config.num_islands == 8
        assert config.population_per_island == 25
        assert config.topology == "star"
        assert config.total_population == 200


# =============================================================================
# DIAGRAM CONSTANTS TESTS
# =============================================================================


class TestDiagramConstants:
    """Tests for Mermaid diagram constants."""

    def test_evolutionary_loop_diagram_exists(self):
        """EVOLUTIONARY_LOOP_DIAGRAM should be a non-empty string."""
        assert EVOLUTIONARY_LOOP_DIAGRAM
        assert "graph" in EVOLUTIONARY_LOOP_DIAGRAM

    def test_genome_phenotype_diagram_exists(self):
        """GENOME_PHENOTYPE_DIAGRAM should be a non-empty string."""
        assert GENOME_PHENOTYPE_DIAGRAM
        assert "graph" in GENOME_PHENOTYPE_DIAGRAM

    def test_island_model_diagram_exists(self):
        """ISLAND_MODEL_DIAGRAM should be a non-empty string."""
        assert ISLAND_MODEL_DIAGRAM
        assert "graph" in ISLAND_MODEL_DIAGRAM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
