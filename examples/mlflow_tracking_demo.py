#!/usr/bin/env python3
"""
MLflow Metrics Tracking Live Demo

Demonstrates comprehensive MLflow metrics tracking with the evolve framework.
Uses UnifiedConfig with TrackingConfig.comprehensive() to log all available metrics.

Run this script, then view results in MLflow UI:
    mlflow ui --port 5000

Go to: http://localhost:5000
"""

from random import Random

import numpy as np

from evolve.config.tracking import TrackingConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.population import Population
from evolve.core.types import Individual
from evolve.experiment.tracking.callback import TrackingCallback
from evolve.factory import create_engine
from evolve.representation.vector import VectorGenome

# -----------------------------------------------------------------------------
# Fitness Functions (receive decoded phenotype - numpy array)
# -----------------------------------------------------------------------------


def rastrigin(genes: np.ndarray) -> float:
    """
    Rastrigin function - highly multimodal test function.
    Global minimum: f(0,...,0) = 0
    """
    A = 10.0
    n = len(genes)
    return A * n + np.sum(genes**2 - A * np.cos(2 * np.pi * genes))


def sphere(genes: np.ndarray) -> float:
    """
    Sphere function - simple unimodal test function.
    Global minimum: f(0,...,0) = 0
    """
    return float(np.sum(genes**2))


def rosenbrock(genes: np.ndarray) -> float:
    """
    Rosenbrock function - banana-shaped valley.
    Global minimum: f(1,...,1) = 0
    """
    total = 0.0
    for i in range(len(genes) - 1):
        total += 100.0 * (genes[i + 1] - genes[i] ** 2) ** 2 + (1 - genes[i]) ** 2
    return total


# -----------------------------------------------------------------------------
# Initial Population Creator
# -----------------------------------------------------------------------------


def create_initial_population(
    size: int,
    dimensions: int,
    bounds: tuple[float, float],
    seed: int,
) -> Population:
    """Create random initial population."""
    rng = Random(seed)
    individuals = []

    for _ in range(size):
        genes = np.array([rng.uniform(bounds[0], bounds[1]) for _ in range(dimensions)])
        genome = VectorGenome(genes=genes)
        individuals.append(Individual(genome=genome))

    return Population(individuals=individuals)


# -----------------------------------------------------------------------------
# Main Demo
# -----------------------------------------------------------------------------


def run_comprehensive_tracking_demo():
    """Run evolution with comprehensive MLflow tracking."""

    print("=" * 70)
    print("MLflow Metrics Tracking Demo")
    print("=" * 70)

    # Configuration parameters
    SEED = 42
    DIMENSIONS = 10
    BOUNDS = (-5.12, 5.12)
    POPULATION_SIZE = 100
    GENERATIONS = 50

    # Create comprehensive tracking config with all metrics enabled
    tracking = TrackingConfig.comprehensive("evolve_live_demo")

    # Show what categories are enabled
    print("\n📊 Enabled Metric Categories:")
    for cat in sorted(tracking.categories, key=lambda c: c.value):
        print(f"   - {cat.value}")

    print("\n📋 Tracking Configuration:")
    print(f"   Backend: {tracking.backend}")
    print(f"   Experiment: {tracking.experiment_name}")
    print(f"   Buffer size: {tracking.buffer_size}")
    print(f"   Log interval: {tracking.log_interval}")
    print(f"   Diversity sample: {tracking.diversity_sample_size}")

    # Create UnifiedConfig with tracking
    config = UnifiedConfig(
        name="comprehensive_tracking_demo",
        description="Live demo of MLflow metrics tracking with all categories",
        tags=("demo", "mlflow", "tracking", "comprehensive"),
        seed=SEED,
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        elitism=2,
        # Selection
        selection="tournament",
        selection_params={"tournament_size": 5},
        # Crossover
        crossover="sbx",
        crossover_rate=0.9,
        crossover_params={"eta": 20.0},
        # Mutation
        mutation="gaussian",
        mutation_rate=0.1,
        mutation_params={"sigma": 0.5},
        # Representation
        genome_type="vector",
        genome_params={
            "dimensions": DIMENSIONS,
            "bounds": BOUNDS,
        },
        minimize=True,  # Minimizing fitness functions
        # Enable comprehensive tracking
        tracking=tracking,
    )

    print("\n🧬 Evolution Configuration:")
    print(f"   Population: {config.population_size}")
    print(f"   Generations: {config.max_generations}")
    print(f"   Selection: {config.selection}")
    print(f"   Crossover: {config.crossover} (rate={config.crossover_rate})")
    print(f"   Mutation: {config.mutation} (rate={config.mutation_rate})")
    print(f"   Genome: {config.genome_type} ({DIMENSIONS}D)")

    # Create engine
    print("\n🔧 Creating evolution engine...")
    engine = create_engine(config, rastrigin)

    # Create tracking callback with unified config and description
    # Note: For pure math functions like Rastrigin/Sphere, there's no evaluation
    # dataset - fitness is computed directly from the candidate solution.
    #
    # If your fitness function evaluates solutions against data (e.g., evolving
    # a neural network, symbolic regression, or trading strategy), pass the data:
    #
    #   tracking_callback = TrackingCallback(
    #       config=tracking,
    #       evaluation_data=your_training_data,  # DataFrame, np.ndarray, etc.
    #       evaluation_data_name="training_data",
    #   )
    #
    tracking_callback = TrackingCallback(
        config=tracking,
        unified_config_dict=config.to_dict(),
        description=config.description,
    )

    # Create initial population
    print("🌱 Generating initial population...")
    initial_pop = create_initial_population(
        size=POPULATION_SIZE,
        dimensions=DIMENSIONS,
        bounds=BOUNDS,
        seed=SEED,
    )

    # Run evolution with tracking callback
    print(f"\n🚀 Running evolution for {GENERATIONS} generations...")
    print("-" * 70)

    result = engine.run(initial_pop, callbacks=[tracking_callback])

    print("-" * 70)
    print("\n✅ Evolution Complete!")
    print(f"   Best fitness: {result.best.fitness.values[0]:.6f}")
    print(f"   Generations: {result.generations}")
    print(f"   Stop reason: {result.stop_reason}")

    # Show best solution
    print("\n🏆 Best Solution:")
    best_genes = result.best.genome.genes
    print(f"   Genes (first 5): {best_genes[:5]}")
    print(f"   Gene norm: {np.linalg.norm(best_genes):.6f}")

    print("\n" + "=" * 70)
    print("📈 View metrics in MLflow UI:")
    print("   mlflow ui --port 5000")
    print("   Then open: http://localhost:5000")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_comprehensive_tracking_demo()
