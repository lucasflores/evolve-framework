"""
Dependency checking utilities.

Provides utilities for verifying required dependencies are installed
before running experiments. Fails fast with clear error messages.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


@dataclass
class DependencyCheck:
    """Result of a dependency check."""

    name: str
    available: bool
    version: str | None = None
    error: str | None = None


def check_dependency(name: str, min_version: str | None = None) -> DependencyCheck:
    """
    Check if a package is installed.

    Args:
        name: Package name (e.g., "mlflow", "numpy")
        min_version: Optional minimum version requirement

    Returns:
        DependencyCheck with availability status
    """
    spec = importlib.util.find_spec(name)

    if spec is None:
        return DependencyCheck(
            name=name, available=False, error=f"Package '{name}' is not installed"
        )

    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)

        if min_version and version:
            from packaging.version import Version

            if Version(version) < Version(min_version):
                return DependencyCheck(
                    name=name,
                    available=False,
                    version=version,
                    error=f"Package '{name}' version {version} < required {min_version}",
                )

        return DependencyCheck(name=name, available=True, version=version)

    except ImportError as e:
        return DependencyCheck(name=name, available=False, error=str(e))


def require_dependencies(
    *dependencies: str | tuple[str, str], exit_on_failure: bool = True
) -> list[DependencyCheck]:
    """
    Verify required dependencies are installed, failing fast if not.

    Args:
        *dependencies: Package names or (name, min_version) tuples
        exit_on_failure: If True, exit with error code 1 on missing deps

    Returns:
        List of DependencyCheck results

    Raises:
        SystemExit: If exit_on_failure=True and dependencies are missing

    Example:
        >>> # At top of experiment script
        >>> from evolve.utils import require_dependencies
        >>> require_dependencies("mlflow", ("numpy", "1.20"))
    """
    results = []
    missing = []

    for dep in dependencies:
        if isinstance(dep, tuple):
            name, min_version = dep
        else:
            name, min_version = dep, None

        check = check_dependency(name, min_version)
        results.append(check)

        if not check.available:
            missing.append(check)

    if missing and exit_on_failure:
        print("=" * 60, file=sys.stderr)
        print("MISSING DEPENDENCIES", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for check in missing:
            print(f"  ✗ {check.name}: {check.error}", file=sys.stderr)
        print(file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  uv sync", file=sys.stderr)
        print("Or:", file=sys.stderr)
        names = " ".join(c.name for c in missing)
        print(f"  pip install {names}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    return results


def require_tracking() -> None:
    """
    Convenience function to verify MLflow tracking is available.

    Call at the top of experiment scripts that use TrackingCallback.

    Example:
        >>> from evolve.utils import require_tracking
        >>> require_tracking()  # Exits if mlflow not installed
    """
    require_dependencies(("mlflow", "2.0"))


__all__ = [
    "DependencyCheck",
    "check_dependency",
    "require_dependencies",
    "require_tracking",
]
