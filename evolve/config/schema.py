"""
Schema Versioning and Validation.

Provides schema version parsing, comparison, and migration utilities
for forward/backward compatibility of configuration files.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal

# Current schema version
CURRENT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """
    Semantic version representation for schema compatibility.
    
    Follows semantic versioning (major.minor.patch):
    - Major: Breaking changes that require migration
    - Minor: New features, backward compatible
    - Patch: Bug fixes, backward compatible
    
    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
    
    Example:
        >>> v = SchemaVersion.parse("1.2.3")
        >>> v.major, v.minor, v.patch
        (1, 2, 3)
        >>> v > SchemaVersion(1, 0, 0)
        True
    """
    
    major: int
    minor: int
    patch: int
    
    def __post_init__(self) -> None:
        """Validate version components."""
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version components must be non-negative")
    
    def __str__(self) -> str:
        """Format as version string."""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def parse(cls, version_string: str) -> "SchemaVersion":
        """
        Parse a version string.
        
        Args:
            version_string: Version in "major.minor.patch" format.
            
        Returns:
            SchemaVersion instance.
            
        Raises:
            ValueError: If version string is invalid.
        """
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_string.strip())
        if not match:
            raise ValueError(
                f"Invalid version format: {version_string!r}. "
                f"Expected 'major.minor.patch' (e.g., '1.0.0')"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )
    
    @classmethod
    def current(cls) -> "SchemaVersion":
        """Get the current framework schema version."""
        return cls.parse(CURRENT_SCHEMA_VERSION)
    
    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """
        Check if this version is compatible with another.
        
        Compatibility rules:
        - Same major version is compatible
        - Older minor/patch versions can be loaded (with warnings for deprecation)
        - Newer major versions are incompatible
        
        Args:
            other: Version to check compatibility with.
            
        Returns:
            True if versions are compatible.
        """
        return self.major == other.major


class SchemaVersionError(Exception):
    """Raised when schema version is incompatible."""
    
    def __init__(
        self,
        config_version: SchemaVersion,
        framework_version: SchemaVersion,
        message: str | None = None,
    ) -> None:
        self.config_version = config_version
        self.framework_version = framework_version
        if message is None:
            message = (
                f"Schema version mismatch: configuration uses v{config_version}, "
                f"but framework supports v{framework_version}. "
                f"Please upgrade the configuration or use a compatible framework version."
            )
        super().__init__(message)


def validate_schema_version(
    config_version: str,
    *,
    strict: bool = False,
) -> Literal["current", "older", "newer"]:
    """
    Validate a configuration schema version against the framework.
    
    Args:
        config_version: Schema version string from configuration.
        strict: If True, raise errors for any version mismatch.
        
    Returns:
        "current" if versions match exactly.
        "older" if config version is older (with deprecation warning).
        "newer" if config version is newer (raises error).
        
    Raises:
        SchemaVersionError: If config version is newer than framework (FR-008),
                           or if strict=True and versions don't match.
        ValueError: If version string is invalid.
    
    Example:
        >>> validate_schema_version("1.0.0")
        'current'
        >>> validate_schema_version("0.9.0")  # Issues deprecation warning
        'older'
        >>> validate_schema_version("2.0.0")  # Raises SchemaVersionError
        Traceback (most recent call last):
            ...
        SchemaVersionError: Schema version mismatch...
    """
    parsed = SchemaVersion.parse(config_version)
    current = SchemaVersion.current()
    
    if parsed == current:
        return "current"
    
    if parsed > current:
        # FR-008: Reject configurations with schema versions newer than framework
        raise SchemaVersionError(
            parsed,
            current,
            f"Configuration schema version v{parsed} is newer than "
            f"framework version v{current}. Please upgrade the framework "
            f"to load this configuration.",
        )
    
    # FR-007: Load older schemas with deprecation warnings
    if parsed.major < current.major:
        msg = (
            f"Configuration uses schema v{parsed} which is a major version "
            f"behind v{current}. Some features may not work correctly. "
            f"Consider migrating the configuration to the latest schema."
        )
        if strict:
            raise SchemaVersionError(parsed, current, msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    elif parsed.minor < current.minor:
        msg = (
            f"Configuration uses schema v{parsed} which is behind "
            f"the current v{current}. Loading with defaults for new fields."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Patch version differences are silent
    
    return "older"


def migrate_config_dict(
    data: dict,
    from_version: str,
    to_version: str | None = None,
) -> dict:
    """
    Migrate a configuration dictionary between schema versions.
    
    Args:
        data: Configuration dictionary to migrate.
        from_version: Source schema version.
        to_version: Target schema version (default: current).
        
    Returns:
        Migrated configuration dictionary.
        
    Note:
        Currently a placeholder. Migration logic will be added
        as schema versions evolve.
    """
    if to_version is None:
        to_version = CURRENT_SCHEMA_VERSION
    
    from_v = SchemaVersion.parse(from_version)
    to_v = SchemaVersion.parse(to_version)
    
    if from_v == to_v:
        return data.copy()
    
    # Copy data for migration
    result = data.copy()
    
    # Future migration steps would go here, e.g.:
    # if from_v < SchemaVersion(1, 1, 0):
    #     result = _migrate_1_0_to_1_1(result)
    
    # Update schema version
    result["schema_version"] = to_version
    
    return result
