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
    Island,
    MigrationPolicy,
    BestMigration,
    RandomMigration,
    MigrationController,
    IslandEvolutionEngine,
    ring_topology,
    fully_connected_topology,
    hypercube_topology,
)
from evolve.diversity.speciation import (
    DistanceFunction,
    euclidean_distance,
    hamming_distance,
    manhattan_distance,
    cosine_distance,
    neat_distance,
    Species,
    Speciator,
    ThresholdSpeciator,
    KMeansSpeciator,
)
from evolve.diversity.niching import (
    explicit_fitness_sharing,
    crowding_distance,
    clearing,
    deterministic_crowding_pairing,
)
from evolve.diversity.novelty import (
    BehaviorCharacterization,
    FitnessBehavior,
    GenomeBehavior,
    NoveltyArchive,
    QDArchive,
    novelty_fitness,
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
