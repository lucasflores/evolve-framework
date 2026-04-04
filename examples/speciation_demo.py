#!/usr/bin/env python3
"""
Speciation via Assortative Mating - Standalone Example

Demonstrates how ERP can model speciation through assortative mating.
Individuals prefer mates similar to themselves (cosine similarity).

Run this example:
    python examples/speciation_demo.py

Expected behavior:
- Initial random population
- Over time, clusters form (proto-species)
- Within-cluster mating preferred
- Between-cluster mating rare
- Protocol diversity maintained
"""

from random import Random

import numpy as np

from evolve.core.callbacks import HistoryCallback
from evolve.core.operators.crossover import UniformCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.population import Population
from evolve.core.types import Individual
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome
from evolve.reproduction.engine import ERPConfig, ERPEngine
from evolve.reproduction.mutation import MutationConfig, ProtocolMutator
from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    MatchabilityFunction,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)

# Configuration
SEED = 123
DIMENSIONS = 10
INIT_RANGE = (-5.0, 5.0)
POPULATION_SIZE = 80
GENERATIONS = 100
SIMILARITY_THRESHOLD = 0.7  # High similarity required for mating


def rastrigin_function(genes: np.ndarray) -> float:
    """
    Rastrigin function: f(x) = -[An + sum(x_i^2 - A*cos(2*pi*x_i))]
    Multi-modal landscape - good for speciation
    """
    A = 10.0
    n = len(genes)
    return -(A * n + np.sum(genes**2 - A * np.cos(2 * np.pi * genes)))


def create_assortative_population(size: int, seed: int) -> Population:
    """
    Create population with assortative mating protocols.

    All individuals have:
    - Cosine similarity matchability (prefer similar mates)
    - High threshold (0.7) for strong assortment
    """
    rng = Random(seed)
    individuals = []

    for _ in range(size):
        # Random genome
        genes = np.array([rng.uniform(*INIT_RANGE) for _ in range(DIMENSIONS)])
        genome = VectorGenome(genes=genes)

        # Assortative protocol: prefer similar genomes
        protocol = ReproductionProtocol(
            intent=ReproductionIntentPolicy(type="always", params={}),
            matchability=MatchabilityFunction(
                type="cosine_similarity", params={"threshold": SIMILARITY_THRESHOLD}
            ),
            crossover=CrossoverProtocolSpec(
                type=CrossoverType.UNIFORM, params={"swap_probability": 0.5}
            ),
        )

        individual = Individual(genome=genome, fitness=None, protocol=protocol)
        individuals.append(individual)

    return Population(individuals=individuals)


def compute_pairwise_distances(population: Population) -> np.ndarray:
    """Compute pairwise cosine distances between all genomes."""
    n = len(population.individuals)
    distances = np.zeros((n, n))

    for i in range(n):
        genes_i = population.individuals[i].genome.genes
        norm_i = np.linalg.norm(genes_i)

        for j in range(i + 1, n):
            genes_j = population.individuals[j].genome.genes
            norm_j = np.linalg.norm(genes_j)

            # Cosine similarity → distance
            if norm_i > 0 and norm_j > 0:
                similarity = np.dot(genes_i, genes_j) / (norm_i * norm_j)
                distance = 1.0 - similarity
            else:
                distance = 1.0

            distances[i, j] = distance
            distances[j, i] = distance

    return distances


def analyze_speciation_results(
    initial_pop: Population, final_pop: Population, history: HistoryCallback
):
    """Analyze speciation patterns."""
    print("\n" + "=" * 70)
    print("SPECIATION ANALYSIS")
    print("=" * 70)

    # Diversity evolution
    print("\nPopulation Diversity:")
    initial_distances = compute_pairwise_distances(initial_pop)
    final_distances = compute_pairwise_distances(final_pop)

    initial_mean_dist = np.mean(initial_distances)
    final_mean_dist = np.mean(final_distances)

    print(f"  Initial mean distance: {initial_mean_dist:.4f}")
    print(f"  Final mean distance:   {final_mean_dist:.4f}")
    print(f"  Change:                {final_mean_dist - initial_mean_dist:+.4f}")

    # Cluster analysis (simple: count individuals within threshold of each other)
    threshold = 1.0 - SIMILARITY_THRESHOLD  # Convert to distance

    def count_clusters_simple(distances, thresh):
        """Simple cluster count: nodes with neighbors"""
        len(distances)
        has_neighbor = np.any(distances < thresh, axis=1)
        return np.sum(has_neighbor)

    initial_clustered = count_clusters_simple(initial_distances, threshold)
    final_clustered = count_clusters_simple(final_distances, threshold)

    print("\nAssortative Mating Effect:")
    print(f"  Individuals with similar neighbors (threshold={threshold:.2f}):")
    print(f"    Initial: {initial_clustered}/{len(initial_pop.individuals)}")
    print(f"    Final:   {final_clustered}/{len(final_pop.individuals)}")

    # Fitness
    best_fitness = [gen["best_fitness"] for gen in history.history]
    print("\nFitness Evolution:")
    print(f"  Initial: {best_fitness[0]:.4f}")
    print(f"  Final:   {best_fitness[-1]:.4f}")
    print(f"  Improvement: {best_fitness[-1] - best_fitness[0]:.4f}")

    if final_clustered > initial_clustered:
        print("\n✅ Assortative mating increased clustering!")
        print("   Proto-species forming through preferential mating")
    else:
        print("\n⚠️  No strong clustering - try higher threshold or more generations")

    print("=" * 70)


def main():
    """Run speciation demonstration."""
    print("=" * 70)
    print("SPECIATION VIA ASSORTATIVE MATING")
    print("=" * 70)
    print("\nScenario:")
    print(f"  • {POPULATION_SIZE} individuals with cosine similarity matchability")
    print(f"  • Threshold: {SIMILARITY_THRESHOLD} (prefer similar genomes)")
    print(f"  • {GENERATIONS} generations")
    print("  • Multi-modal landscape (Rastrigin function)")
    print("\nHypothesis:")
    print("  Individuals preferentially mate with similar partners")
    print("  → Proto-species clusters emerge")
    print("  → Within-cluster diversity decreases")
    print("  → Between-cluster diversity maintained")

    # Create population
    print("\n[1/4] Creating assortative mating population...")
    population = create_assortative_population(POPULATION_SIZE, SEED)
    initial_pop_copy = Population(
        individuals=[
            Individual(
                genome=VectorGenome(genes=ind.genome.genes.copy()),
                fitness=ind.fitness,
                protocol=ind.protocol,
            )
            for ind in population.individuals
        ]
    )

    # Configure ERP engine
    print("[2/4] Configuring ERP engine...")
    erp_config = ERPConfig(
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        enable_recovery=False,
        protocol_mutation_rate=0.05,  # Low mutation to maintain clustering
    )

    mutation_config = MutationConfig(
        param_mutation_rate=0.05, param_mutation_strength=0.03, type_mutation_rate=0.01
    )

    protocol_mutator = ProtocolMutator(config=mutation_config)

    erp_engine = ERPEngine(
        config=erp_config,
        evaluator=FunctionEvaluator(rastrigin_function),
        selection=TournamentSelection(tournament_size=3),
        crossover=UniformCrossover(swap_prob=0.5),
        mutation=GaussianMutation(mutation_rate=0.1, sigma=0.3),
        protocol_mutator=protocol_mutator,
        seed=SEED,
    )

    # Run evolution
    print("[3/4] Running evolution...")
    history = HistoryCallback()
    result = erp_engine.run(initial_population=population, callbacks=[history])

    # Analyze results
    print("[4/4] Analyzing speciation patterns...")
    analyze_speciation_results(initial_pop_copy, result.final_population, history)

    print("\n✅ Example complete!")
    print("\nTry modifying:")
    print("  • SIMILARITY_THRESHOLD - adjust clustering strength")
    print("  • GENERATIONS - observe longer-term speciation")
    print("  • protocol_mutation_rate - balance exploration/exploitation")


if __name__ == "__main__":
    main()
