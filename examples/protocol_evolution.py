#!/usr/bin/env python3
"""
Protocol Evolution Tracking - Standalone Example

Demonstrates how to track and visualize protocol parameter evolution
over generations. Shows how intent, matchability, and crossover protocols
adapt during evolution.

Run this example:
    python examples/protocol_evolution.py

Expected behavior:
- Protocols start with diverse parameters
- Selection pressure shapes protocol distributions
- Successful protocols proliferate
- Track parameter changes (thresholds, swap probabilities, etc.)
"""

from collections import defaultdict
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
SEED = 456
DIMENSIONS = 8
INIT_RANGE = (-10.0, 10.0)
POPULATION_SIZE = 60
GENERATIONS = 80


def rosenbrock_function(genes: np.ndarray) -> float:
    """Rosenbrock function: challenging optimization landscape."""
    result = 0.0
    for i in range(len(genes) - 1):
        result += 100 * (genes[i + 1] - genes[i] ** 2) ** 2 + (1 - genes[i]) ** 2
    return -result


def create_diverse_protocol_population(size: int, seed: int) -> Population:
    """
    Create population with diverse protocol parameters.

    Varies:
    - Matchability thresholds (0.2 - 0.9)
    - Crossover types (uniform, single-point, two-point)
    - Swap probabilities (0.3 - 0.7)
    """
    rng = Random(seed)
    individuals = []

    crossover_types = [CrossoverType.UNIFORM, CrossoverType.SINGLE_POINT, CrossoverType.TWO_POINT]

    for _ in range(size):
        # Random genome
        genes = np.array([rng.uniform(*INIT_RANGE) for _ in range(DIMENSIONS)])
        genome = VectorGenome(genes=genes)

        # Random protocol parameters
        threshold = rng.uniform(0.2, 0.9)
        swap_prob = rng.uniform(0.3, 0.7)
        crossover_type = rng.choice(crossover_types)

        protocol = ReproductionProtocol(
            intent=ReproductionIntentPolicy(type="always", params={}),
            matchability=MatchabilityFunction(
                type="fitness_threshold", params={"min_fitness": threshold}
            ),
            crossover=CrossoverProtocolSpec(
                type=crossover_type,
                params={"swap_probability": swap_prob}
                if crossover_type == CrossoverType.UNIFORM
                else {},
            ),
        )

        individual = Individual(genome=genome, fitness=None, protocol=protocol)
        individuals.append(individual)

    return Population(individuals=individuals)


class ProtocolTracker:
    """Tracks protocol parameter distributions over generations."""

    def __init__(self):
        self.generations = []
        self.threshold_stats = []  # (mean, std) per generation
        self.crossover_dist = []  # distribution dict per generation
        self.swap_prob_stats = []  # (mean, std) per generation

    def track(self, population: Population, generation: int):
        """Record protocol statistics for this generation."""
        self.generations.append(generation)

        # Extract parameters
        thresholds = []
        crossover_types = defaultdict(int)
        swap_probs = []

        for ind in population.individuals:
            if ind.protocol is None:
                continue

            # Matchability threshold
            if ind.protocol.matchability.type == "fitness_threshold":
                threshold = ind.protocol.matchability.params.get("min_fitness", 0.5)
                thresholds.append(threshold)

            # Crossover type
            crossover_type = ind.protocol.crossover.type
            crossover_types[crossover_type.name] += 1

            # Swap probability (for uniform crossover)
            if crossover_type == CrossoverType.UNIFORM:
                swap_prob = ind.protocol.crossover.params.get("swap_probability", 0.5)
                swap_probs.append(swap_prob)

        # Store statistics
        if thresholds:
            self.threshold_stats.append((np.mean(thresholds), np.std(thresholds)))
        else:
            self.threshold_stats.append((0.0, 0.0))

        self.crossover_dist.append(dict(crossover_types))

        if swap_probs:
            self.swap_prob_stats.append((np.mean(swap_probs), np.std(swap_probs)))
        else:
            self.swap_prob_stats.append((0.5, 0.0))

    def print_summary(self):
        """Print protocol evolution summary."""
        print("\n" + "=" * 70)
        print("PROTOCOL EVOLUTION SUMMARY")
        print("=" * 70)

        # Matchability thresholds
        print("\nMatchability Thresholds:")
        initial_mean, initial_std = self.threshold_stats[0]
        final_mean, final_std = self.threshold_stats[-1]
        print(f"  Generation 0:  mean={initial_mean:.3f}, std={initial_std:.3f}")
        print(
            f"  Generation {len(self.generations) - 1}: mean={final_mean:.3f}, std={final_std:.3f}"
        )
        print(f"  Change:        {final_mean - initial_mean:+.3f}")

        if abs(final_mean - initial_mean) > 0.1:
            if final_mean < initial_mean:
                print("  → Thresholds DECREASED (more permissive mating)")
            else:
                print("  → Thresholds INCREASED (more selective mating)")
        else:
            print("  → Thresholds remained stable")

        # Crossover types
        print("\nCrossover Type Distribution:")
        initial_dist = self.crossover_dist[0]
        final_dist = self.crossover_dist[-1]

        print("  Generation 0:")
        for ctype, count in sorted(initial_dist.items()):
            pct = 100 * count / sum(initial_dist.values())
            print(f"    {ctype}: {count} ({pct:.1f}%)")

        print(f"  Generation {len(self.generations) - 1}:")
        for ctype, count in sorted(final_dist.items()):
            pct = 100 * count / sum(final_dist.values())
            print(f"    {ctype}: {count} ({pct:.1f}%)")

        # Swap probabilities
        print("\nSwap Probabilities (Uniform Crossover):")
        initial_swap_mean, initial_swap_std = self.swap_prob_stats[0]
        final_swap_mean, final_swap_std = self.swap_prob_stats[-1]
        print(f"  Generation 0:  mean={initial_swap_mean:.3f}, std={initial_swap_std:.3f}")
        print(
            f"  Generation {len(self.generations) - 1}: mean={final_swap_mean:.3f}, std={final_swap_std:.3f}"
        )
        print(f"  Change:        {final_swap_mean - initial_swap_mean:+.3f}")

        print("=" * 70)

    def print_generation_samples(self, interval: int = 10):
        """Print protocol snapshots at regular intervals."""
        print("\n" + "=" * 70)
        print("PROTOCOL EVOLUTION TIMELINE")
        print("=" * 70)

        for i, gen in enumerate(self.generations):
            if gen % interval == 0 or gen == self.generations[-1]:
                thresh_mean, thresh_std = self.threshold_stats[i]
                swap_mean, swap_std = self.swap_prob_stats[i]

                print(f"\nGeneration {gen}:")
                print(f"  Threshold:   {thresh_mean:.3f} ± {thresh_std:.3f}")
                print(f"  Swap prob:   {swap_mean:.3f} ± {swap_std:.3f}")
                print(f"  Crossovers:  {dict(self.crossover_dist[i])}")

        print("=" * 70)


def main():
    """Run protocol evolution tracking demonstration."""
    print("=" * 70)
    print("PROTOCOL EVOLUTION TRACKING")
    print("=" * 70)
    print("\nScenario:")
    print(f"  • {POPULATION_SIZE} individuals with diverse protocols")
    print("  • Thresholds: 0.2 - 0.9 (uniform random)")
    print("  • Crossover types: uniform, single-point, two-point")
    print(f"  • {GENERATIONS} generations")
    print("\nHypothesis:")
    print("  Selection pressure favors certain protocol configurations")
    print("  → Successful protocols proliferate")
    print("  → Parameter distributions shift over time")

    # Create population
    print("\n[1/5] Creating diverse protocol population...")
    population = create_diverse_protocol_population(POPULATION_SIZE, SEED)

    # Initialize tracker
    print("[2/5] Initializing protocol tracker...")
    tracker = ProtocolTracker()

    # Configure ERP engine
    print("[3/5] Configuring ERP engine...")
    erp_config = ERPConfig(
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        enable_recovery=False,
        protocol_mutation_rate=0.15,
    )

    mutation_config = MutationConfig(
        param_mutation_rate=0.15, param_mutation_strength=0.1, type_mutation_rate=0.05
    )

    protocol_mutator = ProtocolMutator(config=mutation_config)

    erp_engine = ERPEngine(
        config=erp_config,
        evaluator=FunctionEvaluator(rosenbrock_function),
        selection=TournamentSelection(tournament_size=3),
        crossover=UniformCrossover(swap_prob=0.5),
        mutation=GaussianMutation(mutation_rate=0.1, sigma=0.5),
        protocol_mutator=protocol_mutator,
        seed=SEED,
    )

    # Run evolution with tracking
    print("[4/5] Running evolution with protocol tracking...")
    history = HistoryCallback()

    # Track initial population
    tracker.track(population, 0)

    # Custom callback to track protocols each generation
    class ProtocolTrackingCallback:
        def __init__(self, tracker):
            self.tracker = tracker

        def on_generation_end(self, _engine, population, generation):
            self.tracker.track(population, generation + 1)

    tracking_callback = ProtocolTrackingCallback(tracker)

    erp_engine.run(initial_population=population, callbacks=[history, tracking_callback])

    # Analyze results
    print("[5/5] Analyzing protocol evolution...")
    tracker.print_summary()
    tracker.print_generation_samples(interval=20)

    # Fitness summary
    print("\n" + "=" * 70)
    print("FITNESS EVOLUTION")
    print("=" * 70)
    best_fitness = [gen["best_fitness"] for gen in history.history]
    print(f"  Initial: {best_fitness[0]:.4f}")
    print(f"  Final:   {best_fitness[-1]:.4f}")
    print(f"  Improvement: {best_fitness[-1] - best_fitness[0]:.4f}")
    print("=" * 70)

    print("\n✅ Example complete!")
    print("\nTry modifying:")
    print("  • protocol_mutation_rate - observe stability vs exploration")
    print("  • GENERATIONS - track longer-term evolution")
    print("  • Initial parameter ranges - test different starting conditions")


if __name__ == "__main__":
    main()
