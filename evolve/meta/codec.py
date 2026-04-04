"""
Configuration Codec.

Provides encoding/decoding between UnifiedConfig parameters and
vector genome representations for meta-evolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evolve.config.meta import ParameterSpec
    from evolve.config.unified import UnifiedConfig


@dataclass
class ConfigCodec:
    """
    Encode/decode UnifiedConfig parameters to/from vector genomes.

    Maps evolvable parameters to continuous vector dimensions for
    optimization by the outer evolutionary loop.

    Attributes:
        base_config: Base configuration template.
        param_specs: Parameter specifications for encoding.

    Example:
        >>> codec = ConfigCodec(base_config, param_specs)
        >>> vector = codec.encode(config)  # config -> [0.1, 0.5, 0.8]
        >>> config = codec.decode(vector)  # [0.1, 0.5, 0.8] -> config
    """

    base_config: UnifiedConfig
    """Base configuration template."""

    param_specs: tuple[ParameterSpec, ...]
    """Parameter specifications for encoding."""

    def __post_init__(self) -> None:
        """Precompute bounds for fast encoding/decoding."""
        self._bounds = self._compute_bounds()

    @property
    def dimensions(self) -> int:
        """Get total number of genome dimensions."""
        return sum(spec.num_dimensions for spec in self.param_specs)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Get bounds for each dimension."""
        return self._bounds

    def _compute_bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Compute genome bounds from parameter specifications (T058).

        All parameters are mapped to [0, 1] for uniform handling,
        with actual values decoded during reconstruction.

        Returns:
            Tuple of (min, max) for each dimension.
        """
        bounds_list: list[tuple[float, float]] = []

        for spec in self.param_specs:
            if spec.param_type == "continuous":
                # Map to [0, 1] for normalization
                bounds_list.append((0.0, 1.0))
            elif spec.param_type == "integer":
                # Map to [0, 1], decode to integer range
                bounds_list.append((0.0, 1.0))
            elif spec.param_type == "categorical":
                # Map to [0, 1), decode to index
                bounds_list.append((0.0, 1.0))

        return tuple(bounds_list)

    def encode(self, config: UnifiedConfig) -> list[float]:
        """
        Encode configuration parameters to vector genome (T059).

        Args:
            config: Configuration to encode.

        Returns:
            List of float values representing the genome.
        """
        vector: list[float] = []

        for spec in self.param_specs:
            value = _get_param(config, spec.path)

            if spec.param_type == "continuous":
                assert spec.bounds is not None
                lo, hi = spec.bounds

                if spec.log_scale:
                    # Log-scale encoding
                    log_val = math.log(value)
                    log_lo = math.log(lo)
                    log_hi = math.log(hi)
                    normalized = (log_val - log_lo) / (log_hi - log_lo)
                else:
                    normalized = (value - lo) / (hi - lo)

                vector.append(max(0.0, min(1.0, normalized)))

            elif spec.param_type == "integer":
                assert spec.bounds is not None
                lo, hi = spec.bounds
                normalized = (value - lo) / (hi - lo)
                vector.append(max(0.0, min(1.0, normalized)))

            elif spec.param_type == "categorical":
                assert spec.choices is not None
                if value in spec.choices:
                    idx = spec.choices.index(value)
                    # Map index to [0, 1)
                    normalized = idx / len(spec.choices)
                else:
                    normalized = 0.0
                vector.append(normalized)

        return vector

    def decode(self, vector: list[float]) -> UnifiedConfig:
        """
        Decode vector genome to configuration (T060).

        Args:
            vector: List of float values representing the genome.

        Returns:
            New configuration with decoded parameter values.
        """
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected vector of length {self.dimensions}, got {len(vector)}")

        # Start with base config as dictionary
        updates: dict[str, Any] = {}

        idx = 0
        for spec in self.param_specs:
            value = vector[idx]
            idx += 1

            if spec.param_type == "continuous":
                assert spec.bounds is not None
                lo, hi = spec.bounds

                if spec.log_scale:
                    # Log-scale decoding
                    log_lo = math.log(lo)
                    log_hi = math.log(hi)
                    decoded = math.exp(log_lo + value * (log_hi - log_lo))
                else:
                    decoded = lo + value * (hi - lo)

                updates[spec.path] = decoded

            elif spec.param_type == "integer":
                assert spec.bounds is not None
                lo, hi = int(spec.bounds[0]), int(spec.bounds[1])
                decoded = int(round(lo + value * (hi - lo)))
                decoded = max(lo, min(hi, decoded))
                updates[spec.path] = decoded

            elif spec.param_type == "categorical":
                assert spec.choices is not None
                # Map [0, 1) to index
                idx_choice = int(value * len(spec.choices))
                idx_choice = min(idx_choice, len(spec.choices) - 1)
                updates[spec.path] = spec.choices[idx_choice]

        # Apply updates to base config
        return _apply_updates(self.base_config, updates)


def _get_param(config: UnifiedConfig, path: str) -> Any:
    """
    Get parameter value using dot-notation path (T061).

    Args:
        config: Configuration to read from.
        path: Dot-notation path (e.g., 'mutation_params.sigma').

    Returns:
        Parameter value.

    Raises:
        KeyError: If path not found.
    """
    # Convert config to dict for navigation
    data = config.to_dict()

    parts = path.split(".")
    current: Any = data

    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Path '{path}' not found (missing '{part}')")
            current = current[part]
        else:
            raise KeyError(f"Path '{path}' not traversable at '{part}'")

    return current


def _set_param_update(data: dict[str, Any], path: str, value: Any) -> None:
    """
    Set parameter value in dict using dot-notation path (T062).

    Modifies dict in-place.

    Args:
        data: Dictionary to modify.
        path: Dot-notation path (e.g., 'mutation_params.sigma').
        value: New value.
    """
    parts = path.split(".")
    current = data

    # Navigate to parent
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Set final value
    current[parts[-1]] = value


def _apply_updates(config: UnifiedConfig, updates: dict[str, Any]) -> UnifiedConfig:
    """
    Apply parameter updates to configuration.

    Args:
        config: Base configuration.
        updates: Dictionary mapping paths to new values.

    Returns:
        New configuration with updates applied.
    """
    from evolve.config.unified import UnifiedConfig

    data = config.to_dict()

    for path, value in updates.items():
        _set_param_update(data, path, value)

    return UnifiedConfig.from_dict(data)
