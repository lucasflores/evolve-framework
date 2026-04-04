"""
Migration policies and controller for island model.

Migration controls how individuals move between islands:
- MigrationPolicy: Protocol for selecting emigrants/immigrants
- BestMigration: Migrate best individuals
- RandomMigration: Migrate random individuals
- MigrationController: Coordinates synchronous migration

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from random import Random
from typing import Generic, Protocol, TypeVar

from evolve.core.types import Individual
from evolve.diversity.islands.island import Island

G = TypeVar("G")


class MigrationPolicy(Protocol[G]):
    """
    Protocol for migration policies.

    Controls how individuals are selected for emigration
    and how they are integrated into destination islands.
    """

    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: Random,
    ) -> list[Individual[G]]:
        """
        Select individuals to leave the island.

        Args:
            island: Source island
            n_emigrants: Number of emigrants to select
            rng: Random number generator

        Returns:
            List of individuals to migrate (copies)
        """
        ...

    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        target_island: Island[G],
        rng: Random,
    ) -> list[Individual[G]]:
        """
        Select which emigrants actually enter the target island.

        Args:
            emigrants: Candidates for immigration
            target_island: Destination island
            rng: Random number generator

        Returns:
            List of individuals to add
        """
        ...


@dataclass
class BestMigration(Generic[G]):
    """
    Migrate best individuals from each island.

    Selects the highest-fitness individuals as emigrants.
    This is the most common migration policy, promoting
    spread of good solutions while maintaining exploration.

    Example:
        >>> policy = BestMigration()
        >>> emigrants = policy.select_emigrants(island, n=3, rng=rng)
    """

    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        _rng: Random,
    ) -> list[Individual[G]]:
        """Select best individuals as emigrants."""
        if not island.population or n_emigrants <= 0:
            return []

        # Sort by fitness (descending - best first)
        sorted_pop = sorted(
            island.population,
            key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf"),
            reverse=True,
        )

        # Return copies to avoid modifying originals
        n = min(n_emigrants, len(sorted_pop))
        return [copy.deepcopy(ind) for ind in sorted_pop[:n]]

    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        _target_island: Island[G],
        _rng: Random,
    ) -> list[Individual[G]]:
        """Accept all emigrants as immigrants."""
        return emigrants


@dataclass
class RandomMigration(Generic[G]):
    """
    Migrate random individuals from each island.

    Selects random individuals as emigrants, promoting
    diversity over quality in migration.

    Example:
        >>> policy = RandomMigration()
        >>> emigrants = policy.select_emigrants(island, n=3, rng=rng)
    """

    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: Random,
    ) -> list[Individual[G]]:
        """Select random individuals as emigrants."""
        if not island.population or n_emigrants <= 0:
            return []

        n = min(n_emigrants, len(island.population))
        selected = rng.sample(island.population, n)

        # Return copies to avoid modifying originals
        return [copy.deepcopy(ind) for ind in selected]

    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        _target_island: Island[G],
        _rng: Random,
    ) -> list[Individual[G]]:
        """Accept all emigrants as immigrants."""
        return emigrants


@dataclass
class TournamentMigration(Generic[G]):
    """
    Migrate individuals selected via tournament.

    Uses tournament selection for emigrants, balancing
    fitness pressure with stochasticity.

    Attributes:
        tournament_size: Number of individuals per tournament
    """

    tournament_size: int = 3

    def select_emigrants(
        self,
        island: Island[G],
        n_emigrants: int,
        rng: Random,
    ) -> list[Individual[G]]:
        """Select emigrants via tournament selection."""
        if not island.population or n_emigrants <= 0:
            return []

        emigrants = []
        available = list(island.population)

        for _ in range(min(n_emigrants, len(available))):
            # Run tournament
            k = min(self.tournament_size, len(available))
            contestants = rng.sample(available, k)

            winner = max(
                contestants,
                key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf"),
            )

            emigrants.append(copy.deepcopy(winner))
            # Don't remove - same individual can migrate multiple times

        return emigrants

    def select_immigrants(
        self,
        emigrants: list[Individual[G]],
        _target_island: Island[G],
        _rng: Random,
    ) -> list[Individual[G]]:
        """Accept all emigrants as immigrants."""
        return emigrants


@dataclass
class MigrationController(Generic[G]):
    """
    Coordinates synchronous migration between islands.

    Synchronous model: all islands migrate at the same time
    to maintain reproducibility. Migration is deterministic
    given the same seed.

    Attributes:
        policy: Migration policy for selecting emigrants/immigrants
        migration_interval: Generations between migrations

    Example:
        >>> controller = MigrationController(
        ...     policy=BestMigration(),
        ...     migration_interval=10
        ... )
        >>> if controller.should_migrate(generation):
        ...     controller.migrate(islands, rng)
    """

    policy: MigrationPolicy[G]
    migration_interval: int = 10

    def should_migrate(self, generation: int) -> bool:
        """
        Check if migration should occur this generation.

        Args:
            generation: Current generation number

        Returns:
            True if migration should occur
        """
        return generation > 0 and generation % self.migration_interval == 0

    def migrate(
        self,
        islands: list[Island[G]],
        rng: Random,
    ) -> dict[str, int]:
        """
        Perform migration between all islands.

        DETERMINISM: Processes islands in fixed order,
        uses derived seeds for each island pair.

        Args:
            islands: List of all islands
            rng: Master random number generator

        Returns:
            Migration statistics
        """
        # Build island lookup
        {island.id: island for island in islands}

        # Collect emigrants from all islands first (deterministic order)
        emigrants: dict[int, list[Individual[G]]] = {}
        total_emigrants = 0

        for island in sorted(islands, key=lambda i: i.id):
            n_emigrants = max(1, int(island.size * island.migration_rate))

            if n_emigrants > 0 and island.topology:
                # Derive seed for this island to ensure determinism
                island_seed = rng.randint(0, 2**31 - 1)
                island_rng = Random(island_seed)

                selected = self.policy.select_emigrants(island, n_emigrants, island_rng)
                emigrants[island.id] = selected
                total_emigrants += len(selected)

        # Distribute immigrants (fixed topology order)
        total_immigrants = 0

        for island in sorted(islands, key=lambda i: i.id):
            incoming: list[Individual[G]] = []

            # Collect from all connected neighbors
            for neighbor_id in sorted(island.topology):
                if neighbor_id in emigrants:
                    incoming.extend(emigrants[neighbor_id])

            if incoming:
                # Derive seed for replacement
                replace_seed = rng.randint(0, 2**31 - 1)
                replace_rng = Random(replace_seed)

                # Filter through policy
                immigrants = self.policy.select_immigrants(incoming, island, replace_rng)

                if immigrants:
                    # Replace worst individuals
                    self._replace_worst(island, immigrants)
                    total_immigrants += len(immigrants)

            # Reset isolation time
            island.reset_isolation()

        return {
            "total_emigrants": total_emigrants,
            "total_immigrants": total_immigrants,
            "islands_involved": len([i for i in islands if i.topology]),
        }

    def _replace_worst(
        self,
        island: Island[G],
        immigrants: list[Individual[G]],
    ) -> None:
        """
        Replace worst individuals with immigrants.

        Args:
            island: Target island
            immigrants: Individuals to add
        """
        if not immigrants or not island.population:
            return

        # Sort population by fitness (ascending - worst first)
        island.population.sort(
            key=lambda ind: ind.fitness.values[0] if ind.fitness else float("-inf")
        )

        # Replace worst with immigrants
        n_replace = min(len(immigrants), len(island.population))
        island.population[:n_replace] = immigrants[:n_replace]
