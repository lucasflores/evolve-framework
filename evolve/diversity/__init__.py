"""
Diversity preservation module.

Provides mechanisms for maintaining population diversity:
- Island model: Parallel populations with migration
- Speciation: Grouping similar individuals
- Novelty search: Selection based on behavioral novelty
- Quality-diversity: MAP-Elites and related algorithms

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from evolve.diversity.islands import (
    BestMigration,
    Island,
    IslandEvolutionEngine,
    MigrationController,
    MigrationPolicy,
    RandomMigration,
    fully_connected_topology,
    hypercube_topology,
    ring_topology,
)
from evolve.diversity.niching import (
    clearing,
    crowding_distance,
    deterministic_crowding_pairing,
    explicit_fitness_sharing,
)
from evolve.diversity.novelty import (
    BehaviorCharacterization,
    FitnessBehavior,
    GenomeBehavior,
    NoveltyArchive,
    QDArchive,
    novelty_fitness,
)
from evolve.diversity.speciation import (
    DistanceFunction,
    KMeansSpeciator,
    Speciator,
    Species,
    ThresholdSpeciator,
    cosine_distance,
    euclidean_distance,
    hamming_distance,
    manhattan_distance,
    neat_distance,
)

__all__ = [
    # Island model
    "Island",
    "MigrationPolicy",
    "BestMigration",
    "RandomMigration",
    "MigrationController",
    "IslandEvolutionEngine",
    # Topologies
    "ring_topology",
    "fully_connected_topology",
    "hypercube_topology",
    # Distance functions
    "DistanceFunction",
    "euclidean_distance",
    "hamming_distance",
    "manhattan_distance",
    "cosine_distance",
    "neat_distance",
    # Speciation
    "Species",
    "Speciator",
    "ThresholdSpeciator",
    "KMeansSpeciator",
    # Niching
    "explicit_fitness_sharing",
    "crowding_distance",
    "clearing",
    "deterministic_crowding_pairing",
    # Novelty search
    "BehaviorCharacterization",
    "FitnessBehavior",
    "GenomeBehavior",
    "NoveltyArchive",
    "QDArchive",
    "novelty_fitness",
]
