"""
Engine Factory Module.

Provides one-line engine creation from unified configuration,
automatically resolving operators and genome types from registries.

Public API:
    create_engine(): Create a ready-to-run engine from configuration
    create_initial_population(): Create initial population from config
    OperatorCompatibilityError: Raised when operator-genome mismatch
"""

from evolve.factory.engine import (
    OperatorCompatibilityError,
    create_engine,
    create_initial_population,
)

__all__ = [
    "create_engine",
    "create_initial_population",
    "OperatorCompatibilityError",
]
