"""
Test data generation utilities for ERP Tutorial.

Provides factory functions to create populations with diverse
reproduction protocols for demonstration purposes.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.core.types import Individual
    from evolve.reproduction.protocol import ReproductionProtocol
    from evolve.representation.vector import VectorGenome


def create_test_population_with_protocols(
    n: int,
    config: dict | None = None,
    seed: int = 42,
    rng: Random | None = None
) -> "Population":
    """Create a test population with diverse reproduction protocols.
    
    Args:
        n: Number of individuals
        config: Configuration dict with keys:
            - "strategy": "diverse", "selective", "promiscuous", "mixed"
            - "dimensions": genome dimensions (default: 10)
            - "bounds": (min, max) bounds (default: (-5, 5))
            - "fitness_range": (min, max) fitness values (default: (0, 100))
            - "diversity": "low", "medium", "high" (default: "medium")
        seed: Random seed (ignored if rng provided)
        rng: Random number generator (if None, creates from seed)
        
    Returns:
        Population with individuals that have reproduction protocols
    """
    from evolve.core.population import Population
    from evolve.core.types import Individual, Fitness
    from evolve.representation.vector import VectorGenome
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    config = config or {}
    strategy = config.get("strategy", "diverse")
    dimensions = config.get("dimensions", 10)
    bounds_tuple = config.get("bounds", (-5.0, 5.0))
    bounds = (np.full(dimensions, bounds_tuple[0]), np.full(dimensions, bounds_tuple[1]))
    fitness_range = config.get("fitness_range", (0.0, 100.0))
    
    if rng is None:
        rng = Random(seed)
    
    individuals = []
    
    for i in range(n):
        # Create genome
        genome = VectorGenome.random(dimensions, bounds, rng)
        
        # Create protocol based on strategy
        if strategy == "diverse":
            protocol = _create_diverse_protocol(i, n, rng)
        elif strategy == "selective":
            protocol = _create_selective_protocol(rng)
        elif strategy == "promiscuous":
            protocol = _create_promiscuous_protocol(rng)
        elif strategy == "mixed":
            protocol = _create_mixed_protocol(i, n, rng)
        else:
            protocol = _create_default_protocol()
        
        # Create individual with protocol
        individual = Individual(
            genome=genome,
            fitness=Fitness(values=(rng.uniform(*fitness_range),)),
            protocol=protocol
        )
        individuals.append(individual)
    
    return Population(individuals=individuals)


def _create_diverse_protocol(index: int, total: int, rng: Random) -> "ReproductionProtocol":
    """Create diverse protocol types across population."""
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    # Cycle through different protocol types
    strategy_index = index % 4
    
    if strategy_index == 0:
        # Always willing, accept similar genotypes
        intent = ReproductionIntentPolicy(type="always_willing")
        matchability = MatchabilityFunction(
            type="distance_threshold",
            params={"max_distance": rng.uniform(2.0, 8.0)}
        )
    elif strategy_index == 1:
        # Fitness threshold, accept all
        intent = ReproductionIntentPolicy(
            type="fitness_threshold",
            params={"threshold": rng.uniform(30.0, 70.0)}
        )
        matchability = MatchabilityFunction(type="accept_all")
    elif strategy_index == 2:
        # Always willing, fitness-based preference
        intent = ReproductionIntentPolicy(type="always_willing")
        matchability = MatchabilityFunction(
            type="fitness_proportional",
            params={"selectivity": rng.uniform(0.3, 0.9)}
        )
    else:
        # Stochastic intent, diverse preference
        intent = ReproductionIntentPolicy(
            type="stochastic",
            params={"probability": rng.uniform(0.5, 1.0)}
        )
        matchability = MatchabilityFunction(
            type="diversity_preference",
            params={"min_distance": rng.uniform(1.0, 3.0)}
        )
    
    # Random crossover type
    crossover_types = [
        CrossoverType.UNIFORM,
        CrossoverType.SINGLE_POINT,
        CrossoverType.TWO_POINT,
        CrossoverType.BLEND
    ]
    crossover_type = rng.choice(crossover_types)
    
    crossover = CrossoverProtocolSpec(
        type=crossover_type,
        params={"swap_prob": 0.5} if crossover_type == CrossoverType.UNIFORM else {}
    )
    
    return ReproductionProtocol(
        intent=intent,
        matchability=matchability,
        crossover=crossover
    )


def _create_selective_protocol(rng: Random) -> "ReproductionProtocol":
    """Create highly selective protocol."""
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    return ReproductionProtocol(
        intent=ReproductionIntentPolicy(
            type="fitness_threshold",
            params={"threshold": 70.0}
        ),
        matchability=MatchabilityFunction(
            type="distance_threshold",
            params={"max_distance": 2.0}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
    )


def _create_promiscuous_protocol(rng: Random) -> "ReproductionProtocol":
    """Create promiscuous protocol (accepts all)."""
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    return ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(type="accept_all"),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
    )


def _create_mixed_protocol(index: int, total: int, rng: Random) -> "ReproductionProtocol":
    """Create mixed strategies (half selective, half promiscuous)."""
    if index < total // 2:
        return _create_selective_protocol(rng)
    else:
        return _create_promiscuous_protocol(rng)


def _create_default_protocol() -> "ReproductionProtocol":
    """Create baseline protocol."""
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    return ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(
            type="distance_threshold",
            params={"max_distance": 5.0}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
    )


def create_diverse_protocol_set() -> list["ReproductionProtocol"]:
    """Create one protocol of each major type for demonstrations.
    
    Returns:
        List of protocols showcasing different strategies
    """
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    protocols = []
    
    # 1. Baseline: Always willing, accept all
    protocols.append(ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(type="accept_all"),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM),
        junk_data={"name": "Promiscuous"}
    ))
    
    # 2. Selective: Fitness threshold + distance-based
    protocols.append(ReproductionProtocol(
        intent=ReproductionIntentPolicy(
            type="fitness_threshold",
            params={"threshold": 50.0}
        ),
        matchability=MatchabilityFunction(
            type="distance_threshold",
            params={"max_distance": 3.0}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.SINGLE_POINT),
        junk_data={"name": "Selective"}
    ))
    
    # 3. Fitness-seeking: Always willing, prefer fit partners
    protocols.append(ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(
            type="fitness_proportional",
            params={"selectivity": 0.8}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.BLEND, params={"alpha": 0.5}),
        junk_data={"name": "Fitness-Seeking"}
    ))
    
    # 4. Diversity-seeking: Prefer genetically different partners
    protocols.append(ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(
            type="diversity_preference",
            params={"min_distance": 2.0}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.TWO_POINT),
        junk_data={"name": "Diversity-Seeking"}
    ))
    
    return protocols


def create_sexual_selection_population(
    n_males: int,
    n_females: int,
    seed: int = 42
) -> "Population":
    """Create population with asymmetric male/female protocols.
    
    Models sexual selection scenario:
    - Males: Accept all females (low selectivity)
    - Females: Prefer high-fitness males (high selectivity)
    
    Args:
        n_males: Number of males
        n_females: Number of females
        seed: Random seed
        
    Returns:
        Population with gendered protocols
    """
    from evolve.core.population import Population
    from evolve.core.types import Individual, Fitness
    from evolve.representation.vector import VectorGenome
    from evolve.reproduction.protocol import (
        ReproductionProtocol,
        ReproductionIntentPolicy,
        MatchabilityFunction,
        CrossoverProtocolSpec,
        CrossoverType
    )
    
    rng = Random(seed)
    individuals = []
    dimensions = 10
    bounds = (np.full(dimensions, -5.0), np.full(dimensions, 5.0))
    
    # Male protocol: accept all
    male_protocol = ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(type="accept_all"),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
    )
    
    # Female protocol: prefer high-fitness males
    female_protocol = ReproductionProtocol(
        intent=ReproductionIntentPolicy(type="always_willing"),
        matchability=MatchabilityFunction(
            type="fitness_based",
            params={"min_fitness": 60.0}
        ),
        crossover=CrossoverProtocolSpec(type=CrossoverType.UNIFORM)
    )
    
    # Create males
    for _ in range(n_males):
        genome = VectorGenome.random(dimensions, bounds, rng)
        individual = Individual(
            genome=genome,
            fitness=Fitness(values=(rng.uniform(0, 100),)),
            protocol=male_protocol
        )
        individuals.append(individual)
    
    # Create females
    for _ in range(n_females):
        genome = VectorGenome.random(dimensions, bounds, rng)
        individual = Individual(
            genome=genome,
            fitness=Fitness(values=(rng.uniform(0, 100),)),
            protocol=female_protocol
        )
        individuals.append(individual)
    
    return Population(individuals=individuals)


def generate_mock_erp_history(generations: int = 100, seed: int = 42) -> "ERPHistory":
    """Generate mock ERP history for testing visualizations.
    
    Args:
        generations: Number of generations to simulate
        seed: Random seed
        
    Returns:
        ERPHistory with synthetic data
    """
    from docs.tutorials.utils.tutorial_utils import ERPHistory
    
    rng = Random(seed)
    np.random.seed(seed)
    
    history = ERPHistory()
    
    for gen in range(generations):
        history.generations.append(gen)
        
        # Fitness improves over time
        best = 100 - 90 * np.exp(-gen / 30)
        mean = best - rng.uniform(10, 30)
        worst = mean - rng.uniform(20, 40)
        
        history.best_fitness.append(best + rng.uniform(-2, 2))
        history.mean_fitness.append(mean + rng.uniform(-5, 5))
        history.worst_fitness.append(worst + rng.uniform(-5, 5))
        history.std_fitness.append(rng.uniform(5, 15))
        
        # Diversity decreases over time
        diversity = 10 * np.exp(-gen / 50) + 1
        history.diversity.append(diversity + rng.uniform(-0.5, 0.5))
        
        # Mating success stabilizes
        success_rate = 0.5 + 0.4 * (1 - np.exp(-gen / 20))
        history.mating_success_rate.append(success_rate + rng.uniform(-0.05, 0.05))
        
        # Protocol diversity decreases as strategies converge
        protocol_div = 0.8 * np.exp(-gen / 40) + 0.2
        history.protocol_diversity.append(protocol_div + rng.uniform(-0.05, 0.05))
        
        # Matchability threshold decreases (more selective)
        threshold = 8 - 5 * (1 - np.exp(-gen / 30))
        history.mean_matchability_threshold.append(threshold + rng.uniform(-0.5, 0.5))
        
        # Intent threshold increases (higher standards)
        intent = 20 + 40 * (1 - np.exp(-gen / 40))
        history.mean_intent_threshold.append(intent + rng.uniform(-3, 3))
        
        # Recovery events at low points
        if gen > 0 and history.mating_success_rate[-1] < 0.3:
            if rng.random() < 0.2:  # 20% chance
                history.recovery_events.append(gen)
    
    return history
