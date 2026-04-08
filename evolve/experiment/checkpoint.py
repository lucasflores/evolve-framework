"""
Checkpointing for experiment state.

Provides mechanisms for saving and restoring complete
experiment state for resume capability.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from typing_extensions import Self

from evolve.core.types import Individual

G = TypeVar("G")


@dataclass
class Checkpoint:
    """
    Complete state for resuming an experiment.

    Includes everything needed to continue evolution
    from exactly this point.

    Example:
        >>> checkpoint = Checkpoint.from_engine(engine, config)
        >>> checkpoint.save("checkpoint.pkl")
        >>> # Later...
        >>> checkpoint = Checkpoint.load("checkpoint.pkl")
        >>> engine.restore_from_checkpoint(checkpoint)
    """

    # Identification
    experiment_name: str
    config_hash: str

    # State
    generation: int
    population: list[Individual]
    best_individual: Individual | None

    # RNG state for reproducibility (tuple from Random.getstate() or dict)
    rng_state: Any

    # History
    fitness_history: list[dict[str, float]] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_evaluations: int = 0
    elapsed_time: float = 0.0

    # Optional state for advanced algorithms
    species: list[Any] | None = None
    islands: list[Any] | None = None
    novelty_archive: Any | None = None
    pareto_front: list[Individual] | None = None

    def save(self, path: Path | str) -> None:
        """
        Save checkpoint to disk.

        Uses pickle for complex objects.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """
        Load checkpoint from disk.

        Args:
            path: Input file path

        Returns:
            Loaded Checkpoint
        """
        with open(path, "rb") as f:
            return cast(Self, pickle.load(f))  # noqa: S301

    @property
    def size_bytes(self) -> int:
        """Estimate checkpoint size in bytes."""
        return len(pickle.dumps(self))

    def __str__(self) -> str:
        return (
            f"Checkpoint(gen={self.generation}, "
            f"pop_size={len(self.population)}, "
            f"evals={self.total_evaluations})"
        )


class CheckpointManager:
    """
    Manages checkpoint saving and loading.

    Handles:
    - Saving checkpoints at intervals
    - Loading latest checkpoint for resume
    - Pruning old checkpoints to save disk space

    Example:
        >>> manager = CheckpointManager("./checkpoints", keep_last_n=5)
        >>> if manager.should_checkpoint(generation):
        ...     checkpoint = Checkpoint.from_engine(engine, config)
        ...     manager.save(checkpoint)
        >>> # Resume
        >>> latest = manager.load_latest()
    """

    def __init__(
        self,
        output_dir: Path | str,
        keep_last_n: int = 5,
        checkpoint_interval: int = 10,
    ) -> None:
        """
        Create checkpoint manager.

        Args:
            output_dir: Directory for checkpoint files
            keep_last_n: Number of recent checkpoints to keep
            checkpoint_interval: Save every N generations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_interval = checkpoint_interval

    def should_checkpoint(self, generation: int) -> bool:
        """
        Check if checkpoint should be saved at this generation.

        Args:
            generation: Current generation number

        Returns:
            True if checkpoint should be saved
        """
        return generation % self.checkpoint_interval == 0

    def save(self, checkpoint: Checkpoint) -> Path:
        """
        Save checkpoint and manage history.

        Prunes old checkpoints beyond keep_last_n.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Path to saved checkpoint file
        """
        filename = f"checkpoint_gen{checkpoint.generation:06d}.pkl"
        path = self.output_dir / filename
        checkpoint.save(path)

        # Prune old checkpoints
        self._prune_old_checkpoints()

        return path

    def load_latest(self) -> Checkpoint | None:
        """
        Load most recent checkpoint.

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(self.output_dir.glob("checkpoint_gen*.pkl"))
        if not checkpoints:
            return None
        return Checkpoint.load(checkpoints[-1])

    def load_generation(self, generation: int) -> Checkpoint | None:
        """
        Load checkpoint for specific generation.

        Args:
            generation: Generation number to load

        Returns:
            Checkpoint for that generation or None
        """
        path = self.output_dir / f"checkpoint_gen{generation:06d}.pkl"
        if path.exists():
            return Checkpoint.load(path)
        return None

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """
        List all checkpoints with generation numbers.

        Returns:
            List of (generation, path) tuples sorted by generation
        """
        checkpoints = []
        for path in self.output_dir.glob("checkpoint_gen*.pkl"):
            gen = int(path.stem.split("gen")[1])
            checkpoints.append((gen, path))
        return sorted(checkpoints)

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond keep_last_n."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_gen*.pkl"))
        while len(checkpoints) > self.keep_last_n:
            checkpoints[0].unlink()
            checkpoints.pop(0)

    def clear(self) -> int:
        """
        Remove all checkpoints.

        Returns:
            Number of checkpoints removed
        """
        checkpoints = list(self.output_dir.glob("checkpoint_gen*.pkl"))
        for cp in checkpoints:
            cp.unlink()
        return len(checkpoints)

    @property
    def latest_generation(self) -> int | None:
        """Get the latest checkpoint generation number."""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1][0]
        return None

    def __len__(self) -> int:
        """Number of checkpoints."""
        return len(list(self.output_dir.glob("checkpoint_gen*.pkl")))

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(dir={self.output_dir!r}, "
            f"keep={self.keep_last_n}, interval={self.checkpoint_interval})"
        )


__all__ = [
    "Checkpoint",
    "CheckpointManager",
]
