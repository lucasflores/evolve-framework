"""
Island model for parallel population evolution.

Provides:
- Island: Isolated subpopulation with local configuration
- Topologies: Ring, fully connected, hypercube
- Migration policies: Best, random
- IslandEvolutionEngine: Coordinates multi-population evolution

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from evolve.diversity.islands.engine import IslandEvolutionEngine
from evolve.diversity.islands.island import Island
from evolve.diversity.islands.migration import (
    BestMigration,
    MigrationController,
    MigrationPolicy,
    RandomMigration,
)
from evolve.diversity.islands.topology import (
    fully_connected_topology,
    hypercube_topology,
    ring_topology,
)

__all__ = [
    "Island",
    "ring_topology",
    "fully_connected_topology",
    "hypercube_topology",
    "MigrationPolicy",
    "BestMigration",
    "RandomMigration",
    "MigrationController",
    "IslandEvolutionEngine",
]
