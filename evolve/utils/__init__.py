"""
Utilities module - Seeded RNG, validation helpers, timing utilities, and dependency checks.
"""

from evolve.utils.random import create_rng, derive_seed, get_rng_state, set_rng_state
from evolve.utils.timing import GenerationTimer, TimingResult, timing_context
from evolve.utils.dependencies import (
    DependencyCheck,
    check_dependency,
    require_dependencies,
    require_tracking,
)

__all__ = [
    # RNG
    "create_rng",
    "derive_seed",
    "get_rng_state",
    "set_rng_state",
    # Timing
    "GenerationTimer",
    "TimingResult",
    "timing_context",
    # Dependencies
    "DependencyCheck",
    "check_dependency",
    "require_dependencies",
    "require_tracking",
]
