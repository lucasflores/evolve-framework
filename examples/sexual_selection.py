#!/usr/bin/env python3
"""
Sexual Selection with ERP - Standalone Example

Demonstrates asymmetric sexual selection using Evolvable Reproduction Protocols.
Models choosy vs. eager mating strategies similar to Tutorial 06, Part 7.

Run this example:
    python examples/sexual_selection.py

Expected behavior:
- Two groups form: choosy (high thresholds) and eager (low thresholds)
- Choosy individuals impose selection pressure
- Eager individuals adapt to meet choosy standards
- Over generations, population fitness increases
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
SEED = 42
DIMENSIONS = 5
INIT_RANGE = (-5.0, 5.0)
POPULATION_SIZE = 60
GENERATIONS = 50
CHOOSY_THRESHOLD = 0.8  # High selectivity
EAGER_THRESHOLD = 0.2  # Low selectivity
CHOOSY_FRACTION = 0.4  # 40% choosy, 60% eager


def sphere_function(genes: np.ndarray) -> float:
    """Simple sphere function: f(x) = -sum(x_i^2)"""
    return -np.sum(genes**2)


def create_sexual_selection_population(size: int, seed: int) -> Population:
    """
    Create a population with asymmetric mating strategies.

    40% "Choosy" individuals:
        - High fitness threshold (0.8)
        - Impose selection pressure

    60% "Eager" individuals:
        - Low fitness threshold (0.2)
        - Accept most mates
    """
    rng = Random(seed)
    individuals = []
    choosy_count = int(size * CHOOSY_FRACTION)

    for i in range(size):
        # Random genome
        genes = np.array([rng.uniform(*INIT_RANGE) for _ in range(DIMENSIONS)])
        genome = VectorGenome(genes=genes)

        # Asymmetric protocol assignment
        if i < choosy_count:
            # Choosy: High threshold
            protocol = ReproductionProtocol(
                intent=ReproductionIntentPolicy(type="always", params={}),
                matchability=MatchabilityFunction(
                    type="fitness_threshold", params={"min_fitness": CHOOSY_THRESHOLD}
                ),
                crossover=CrossoverProtocolSpec(
                    type=CrossoverType.UNIFORM, params={"swap_probability": 0.5}
                ),
            )
        else:
            # Eager: Low threshold
            protocol = ReproductionProtocol(
                intent=ReproductionIntentPolicy(type="always", params={}),
                matchability=MatchabilityFunction(
                    type="fitness_threshold", params={"min_fitness": EAGER_THRESHOLD}
                ),
                crossover=CrossoverProtocolSpec(
                    type=CrossoverType.UNIFORM, params={"swap_probability": 0.5}
                ),
            )

        individual = Individual(genome=genome, fitness=None, protocol=protocol)
        individuals.append(individual)

    return Population(individuals=individuals)


def analyze_sexual_selection_results(history: HistoryCallback):
    """Analyze and print results from sexual selection experiment."""
    print("\n" + "=" * 70)
    print("SEXUAL SELECTION RESULTS")
    print("=" * 70)

    # Fitness evolution
    best_fitness = [gen["best_fitness"] for gen in history.history]
    mean_fitness = [gen["mean_fitness"] for gen in history.history]

    print("\nFitness Evolution:")
    print(f"  Initial best: {best_fitness[0]:.4f}")
    print(f"  Final best:   {best_fitness[-1]:.4f}")
    print(f"  Improvement:  {best_fitness[-1] - best_fitness[0]:.4f}")
    print(f"\n  Initial mean: {mean_fitness[0]:.4f}")
    print(f"  Final mean:   {mean_fitness[-1]:.4f}")
    print(f"  Improvement:  {mean_fitness[-1] - mean_fitness[0]:.4f}")

    # Selection pressure effect
    fitness_improvement = best_fitness[-1] - best_fitness[0]
    if fitness_improvement > 5.0:
        print("\n✅ Strong sexual selection effect observed!")
        print("   Choosy individuals successfully imposed selection pressure")
    else:
        print("\n⚠️  Weak selection effect - may need higher choosy fraction")

    print("=" * 70)


def main():
    """Run sexual selection demonstration."""
    print("=" * 70)
    print("SEXUAL SELECTION WITH ERP")
    print("=" * 70)
    print("\nScenario:")
    print(
        f"  • {int(POPULATION_SIZE * CHOOSY_FRACTION)} choosy individuals (threshold={CHOOSY_THRESHOLD})"
    )
    print(
        f"  • {POPULATION_SIZE - int(POPULATION_SIZE * CHOOSY_FRACTION)} eager individuals (threshold={EAGER_THRESHOLD})"
    )
    print(f"  • {GENERATIONS} generations")
    print("\nHypothesis:")
    print("  Choosy individuals impose selection pressure")
    print("  → Eager individuals adapt to meet standards")
    print("  → Population fitness increases over time")

    # Create population
    print("\n[1/4] Creating population with sexual selection...")
    population = create_sexual_selection_population(POPULATION_SIZE, SEED)

    # Configure ERP engine
    print("[2/4] Configuring ERP engine...")
    erp_config = ERPConfig(
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        enable_recovery=False,
        protocol_mutation_rate=0.1,
    )

    mutation_config = MutationConfig(
        param_mutation_rate=0.1, param_mutation_strength=0.05, type_mutation_rate=0.02
    )

    protocol_mutator = ProtocolMutator(config=mutation_config)

    erp_engine = ERPEngine(
        config=erp_config,
        evaluator=FunctionEvaluator(sphere_function),
        selection=TournamentSelection(tournament_size=3),
        crossover=UniformCrossover(swap_prob=0.5),
        mutation=GaussianMutation(mutation_rate=0.1, sigma=0.5),
        protocol_mutator=protocol_mutator,
        seed=SEED,
    )

    # Run evolution
    print("[3/4] Running evolution...")
    history = HistoryCallback()
    result = erp_engine.run(initial_population=population, callbacks=[history])

    # Analyze results
    print("[4/4] Analyzing results...")
    analyze_sexual_selection_results(history)

    print("\n✅ Example complete!")
    print("\nTry modifying:")
    print("  • CHOOSY_FRACTION - change ratio of choosy/eager")
    print("  • CHOOSY_THRESHOLD - adjust selection pressure strength")
    print("  • GENERATIONS - observe longer-term evolution")


if __name__ == "__main__":
    main()
