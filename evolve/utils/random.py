"""
Seeded random number generation utilities.

Provides deterministic RNG for reproducible experiments.
All randomness in the framework flows through these utilities.
"""

from __future__ import annotations

import hashlib
import struct
from random import Random
from typing import Any


def create_rng(seed: int | None = None) -> Random:
    """
    Create a seeded random number generator.
    
    Args:
        seed: Integer seed for reproducibility. If None, uses system entropy.
        
    Returns:
        A Random instance seeded with the given value.
        
    Example:
        >>> rng = create_rng(42)
        >>> rng.random()  # Always produces same value
        0.6394267984578837
    """
    rng = Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def derive_seed(master_seed: int, worker_id: int) -> int:
    """
    Derive a deterministic worker seed from master seed and worker ID.
    
    Uses SplitMix64-style seed derivation to ensure:
    - Different workers get different seeds
    - Same (master_seed, worker_id) always produces same derived seed
    - Seeds are well-distributed across the integer space
    
    Args:
        master_seed: The experiment's master random seed
        worker_id: Unique identifier for the worker (0, 1, 2, ...)
        
    Returns:
        A derived seed unique to this worker
        
    Example:
        >>> derive_seed(42, 0)
        ... # Returns deterministic seed for worker 0
        >>> derive_seed(42, 1)
        ... # Returns different but deterministic seed for worker 1
    """
    # Use SHA-256 for good mixing properties
    data = struct.pack(">QQ", master_seed & 0xFFFFFFFFFFFFFFFF, worker_id)
    digest = hashlib.sha256(data).digest()
    # Take first 8 bytes as unsigned 64-bit integer
    return struct.unpack(">Q", digest[:8])[0]


def get_rng_state(rng: Random) -> tuple[Any, ...]:
    """
    Get the internal state of a Random instance.
    
    Used for checkpointing to enable exact resumption.
    
    Args:
        rng: The Random instance
        
    Returns:
        A tuple representing the RNG state (can be pickled)
    """
    return rng.getstate()


def set_rng_state(rng: Random, state: tuple[Any, ...]) -> None:
    """
    Restore a Random instance to a saved state.
    
    Used for resuming from checkpoints.
    
    Args:
        rng: The Random instance to restore
        state: State tuple from get_rng_state()
    """
    rng.setstate(state)


def split_rng(rng: Random, n: int) -> list[Random]:
    """
    Create n independent child RNGs from a parent RNG.
    
    Each child is seeded deterministically from the parent,
    ensuring reproducible parallel execution.
    
    Args:
        rng: Parent Random instance
        n: Number of child RNGs to create
        
    Returns:
        List of n independent Random instances
        
    Example:
        >>> parent = create_rng(42)
        >>> children = split_rng(parent, 4)
        >>> [c.random() for c in children]  # Reproducible sequence
    """
    # Generate child seeds from parent
    child_seeds = [rng.randint(0, 2**63 - 1) for _ in range(n)]
    return [create_rng(seed) for seed in child_seeds]
