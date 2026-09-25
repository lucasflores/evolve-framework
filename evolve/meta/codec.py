"""
Configuration Codec.

Provides encoding/decoding between UnifiedConfig parameters and
vector genome representations for meta-evolution, and decoding of a
vector genome against parameter specs into a plain nested dict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evolve.config.meta import ParameterSpec
    from evolve.config.unified import UnifiedConfig
    from evolve.representation.vector import VectorGenome


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
        """Refuse specs the codec cannot encode; precompute bounds."""
        for spec in self.param_specs:
            if spec.param_type == "subset" or spec.parent is not None:
                raise ValueError(
                    f"ConfigCodec does not support subset or dependent parameters "
                    f"('{spec.path}'); decode them with decode_parameters()"
                )
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

        for idx, spec in enumerate(self.param_specs):
            updates[spec.path] = decode_value(spec, vector[idx : idx + 1])

        # Apply updates to base config
        return _apply_updates(self.base_config, updates)


@dataclass
class ParameterDecoder:
    """
    Decode a VectorGenome on [0, 1] into a nested dict of parameter values.

    Registry name: ``"parameters"``; ``decoder_params={"params": [...]}``
    holds the specs as ``ParameterSpec.to_dict()`` dicts. The genome needs
    ``dimensions`` positions (``genome_params={"dimensions": ...,
    "bounds": (0.0, 1.0)}``).

    Attributes:
        specs: Parameter specifications, validated on construction.

    Example:
        >>> decoder = ParameterDecoder((ParameterSpec(path="a.b", bounds=(0.0, 2.0)),))
        >>> decoder.decode(VectorGenome(genes=np.array([0.5])))
        {'a': {'b': 1.0}}
    """

    specs: tuple[ParameterSpec, ...]
    """Parameter specifications, validated on construction."""

    def __post_init__(self) -> None:
        """Validate the specs' parents now rather than at the first decode."""
        _dependency_order(self.specs)

    @property
    def dimensions(self) -> int:
        """Get total number of genome dimensions."""
        return sum(spec.num_dimensions for spec in self.specs)

    def decode(self, genome: VectorGenome) -> dict[str, Any]:
        """Decode the genome's genes with decode_parameters()."""
        return decode_parameters(genome.genes.tolist(), self.specs)


def decode_value(
    spec: ParameterSpec,
    positions: Sequence[float],
    choices: Sequence[Any] | None = None,
) -> Any:
    """
    Map a parameter's genome positions on [0, 1] to its value.

    The one place the per-type math lives, shared by ConfigCodec and
    decode_parameters().

    Positions outside [0, 1] are clamped to it first, so an unbounded
    genome decodes to the nearest end of each range.

    Args:
        spec: Parameter specification.
        positions: The spec's ``num_dimensions`` genome positions.
        choices: For a categorical, options to use instead of ``spec.choices``
            (a dependent categorical's options for its parent's value); for a
            subset, the part of ``spec.choices`` it is limited to.

    Returns:
        Decoded value; a list in ``spec.choices`` order for a subset.
    """
    positions = [min(1.0, max(0.0, p)) for p in positions]

    if spec.param_type == "subset":
        # A choice is in the subset when its position is at least 0.5
        assert spec.choices is not None
        return [
            c
            for c, p in zip(spec.choices, positions)
            if p >= 0.5 and (choices is None or c in choices)
        ]

    value = positions[0]

    if spec.param_type == "continuous":
        assert spec.bounds is not None
        lo, hi = spec.bounds

        if spec.log_scale:
            # Log-scale decoding
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            return math.exp(log_lo + value * (log_hi - log_lo))
        return lo + value * (hi - lo)

    if spec.param_type == "integer":
        assert spec.bounds is not None
        lo, hi = int(spec.bounds[0]), int(spec.bounds[1])
        decoded = int(round(lo + value * (hi - lo)))
        return max(lo, min(hi, decoded))

    # Categorical: map [0, 1) to index
    options = spec.choices if choices is None else choices
    assert options is not None
    idx_choice = int(value * len(options))
    idx_choice = min(idx_choice, len(options) - 1)
    return options[idx_choice]


def decode_parameters(
    vector: Sequence[float],
    specs: Sequence[ParameterSpec],
) -> dict[str, Any]:
    """
    Decode a genome vector against parameter specs into a nested dict.

    Positions are laid out in ``specs`` order, ``num_dimensions`` each, and
    decoded parents first. A spec is inactive, and left out of the result,
    when its parent is inactive, its parent's value is not in its
    ``active_values``, or ``choices_by_parent`` has no entry for that value;
    an inactive spec's positions have no effect. A subset keeps only the
    choices allowed for its categorical parent's value (``choices_by_parent``)
    or, relative to a subset parent, the choices the parent's value also
    holds; either way in ``choices`` order, and positions of choices it
    cannot keep have no effect.
    Dot paths become nested keys.

    Args:
        vector: Genome positions on [0, 1].
        specs: Parameter specifications.

    Returns:
        Nested dict of the active parameters' values.

    Raises:
        ValueError: If the specs are inconsistent or the vector length is wrong.
    """
    order = _dependency_order(specs)
    dimensions = sum(spec.num_dimensions for spec in specs)
    if len(vector) != dimensions:
        raise ValueError(f"Expected vector of length {dimensions}, got {len(vector)}")

    positions: dict[str, Sequence[float]] = {}
    start = 0
    for spec in specs:
        positions[spec.path] = vector[start : start + spec.num_dimensions]
        start += spec.num_dimensions

    values: dict[str, Any] = {}
    for spec in order:
        choices = None
        parent_value: Any = None
        if spec.parent is not None:
            if spec.parent not in values:
                continue  # parent inactive
            parent_value = values[spec.parent]
            if spec.active_values is not None and parent_value not in spec.active_values:
                continue
            if spec.choices_by_parent is not None:
                choices = next((o for v, o in spec.choices_by_parent if v == parent_value), None)
                if choices is None:
                    continue
            elif spec.is_relative:
                choices = parent_value
        values[spec.path] = decode_value(spec, positions[spec.path], choices)

    decoded: dict[str, Any] = {}
    for spec in specs:
        if spec.path in values:
            _set_param_update(decoded, spec.path, values[spec.path])
    return decoded


def _dependency_order(specs: Sequence[ParameterSpec]) -> tuple[ParameterSpec, ...]:
    """
    Validate the specs' paths and parents, and order them parents first.

    Raises:
        ValueError: On a duplicate or clashing path, an unknown or
            type-incompatible parent, a value the parent cannot take,
            or a cycle of parents.
    """
    by_path: dict[str, ParameterSpec] = {}
    for spec in specs:
        if spec.path in by_path:
            raise ValueError(f"Duplicate parameter path '{spec.path}'")
        by_path[spec.path] = spec
    for path in by_path:
        # 'a' and 'a.b' cannot both be keys of the nested dict
        for other in by_path:
            if path.startswith(other + "."):
                raise ValueError(f"Parameter path '{path}' is nested under parameter '{other}'")

    for spec in specs:
        if spec.parent is None:
            continue
        parent = by_path.get(spec.parent)
        if parent is None:
            raise ValueError(f"Unknown parent '{spec.parent}' of '{spec.path}'")
        needed = "subset" if spec.is_relative else "categorical"
        if parent.param_type != needed:
            raise ValueError(
                f"Parent '{spec.parent}' of '{spec.path}' is {parent.param_type}; "
                f"it must be {needed}"
            )
        if needed == "categorical":
            can_take = (
                list(parent.choices or ())
                if parent.choices_by_parent is None
                else [v for _, options in parent.choices_by_parent for v in options]
            )
            listed = list(spec.active_values or ()) + [v for v, _ in spec.choices_by_parent or ()]
            unknown = [v for v in listed if v not in can_take]
            if unknown:
                raise ValueError(
                    f"'{spec.path}' lists values its parent '{spec.parent}' cannot take: {unknown}"
                )

    graph = {spec.path: [spec.parent] if spec.parent else [] for spec in specs}
    try:
        return tuple(by_path[path] for path in TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError(f"Parameter parents form a cycle: {' -> '.join(exc.args[1])}") from exc


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
