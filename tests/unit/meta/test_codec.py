"""Unit tests for evolve/meta/codec.py (T078)."""

from __future__ import annotations

import math
import pytest

from evolve.config.unified import UnifiedConfig
from evolve.config.meta import ParameterSpec
from evolve.meta.codec import ConfigCodec, _get_param, _set_param_update


@pytest.fixture
def base_config() -> UnifiedConfig:
    """Create base configuration for codec tests."""
    return UnifiedConfig(
        population_size=100,
        max_generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection="tournament",
        crossover="sbx",
        mutation="gaussian",
        genome_type="vector",
        mutation_params={"sigma": 0.5},
        selection_params={"tournament_size": 3},
    )


class TestParameterSpec:
    """Tests for ParameterSpec validation."""
    
    def test_continuous_param_requires_bounds(self) -> None:
        """Test that continuous parameters require bounds."""
        with pytest.raises(ValueError, match="bounds required"):
            ParameterSpec(path="mutation_rate", param_type="continuous")
    
    def test_integer_param_requires_bounds(self) -> None:
        """Test that integer parameters require bounds."""
        with pytest.raises(ValueError, match="bounds required"):
            ParameterSpec(path="population_size", param_type="integer")
    
    def test_categorical_param_requires_choices(self) -> None:
        """Test that categorical parameters require choices."""
        with pytest.raises(ValueError, match="choices required"):
            ParameterSpec(path="selection", param_type="categorical")
    
    def test_log_scale_requires_positive_bounds(self) -> None:
        """Test that log scale requires positive lower bound."""
        with pytest.raises(ValueError, match="positive lower bound"):
            ParameterSpec(
                path="learning_rate",
                param_type="continuous",
                bounds=(0.0, 1.0),  # 0 is not positive
                log_scale=True,
            )
    
    def test_bounds_order(self) -> None:
        """Test that bounds must be ordered min <= max."""
        with pytest.raises(ValueError, match="must be <="):
            ParameterSpec(
                path="rate",
                param_type="continuous",
                bounds=(1.0, 0.0),  # Reversed
            )
    
    def test_empty_path_rejected(self) -> None:
        """Test that empty path is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ParameterSpec(
                path="",
                param_type="continuous",
                bounds=(0.0, 1.0),
            )


class TestConfigCodecDimensions:
    """Tests for ConfigCodec dimension computation."""
    
    def test_dimensions_single_param(self, base_config: UnifiedConfig) -> None:
        """Test dimensions with single parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
            ),
        )
        
        assert codec.dimensions == 1
    
    def test_dimensions_multiple_params(self, base_config: UnifiedConfig) -> None:
        """Test dimensions with multiple parameters."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
                ParameterSpec(
                    path="population_size",
                    param_type="integer",
                    bounds=(50, 500),
                ),
                ParameterSpec(
                    path="selection",
                    param_type="categorical",
                    choices=("tournament", "roulette", "rank"),
                ),
            ),
        )
        
        assert codec.dimensions == 3
    
    def test_bounds_all_normalized(self, base_config: UnifiedConfig) -> None:
        """Test that all bounds are normalized to [0, 1]."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
                ParameterSpec(
                    path="population_size",
                    param_type="integer",
                    bounds=(50, 500),
                ),
            ),
        )
        
        for lo, hi in codec.bounds:
            assert lo == 0.0
            assert hi == 1.0


class TestConfigCodecEncode:
    """Tests for ConfigCodec.encode()."""
    
    def test_encode_continuous_param(self, base_config: UnifiedConfig) -> None:
        """Test encoding continuous parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.0, 0.5),  # 0.1 should map to 0.2
                ),
            ),
        )
        
        vector = codec.encode(base_config)
        
        assert len(vector) == 1
        # 0.1 in [0, 0.5] should encode to 0.2 in [0, 1]
        assert abs(vector[0] - 0.2) < 0.001
    
    def test_encode_integer_param(self, base_config: UnifiedConfig) -> None:
        """Test encoding integer parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="population_size",
                    param_type="integer",
                    bounds=(0, 200),  # 100 should map to 0.5
                ),
            ),
        )
        
        vector = codec.encode(base_config)
        
        assert len(vector) == 1
        assert abs(vector[0] - 0.5) < 0.001
    
    def test_encode_categorical_param(self, base_config: UnifiedConfig) -> None:
        """Test encoding categorical parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="selection",
                    param_type="categorical",
                    choices=("roulette", "tournament", "rank"),  # tournament is idx 1
                ),
            ),
        )
        
        vector = codec.encode(base_config)
        
        assert len(vector) == 1
        # tournament is at index 1, should encode to 1/3 ≈ 0.333
        assert abs(vector[0] - 1/3) < 0.001
    
    def test_encode_log_scale(self, base_config: UnifiedConfig) -> None:
        """Test encoding with log scale."""
        # Create config with specific mutation_rate
        config = base_config.with_params(mutation_rate=0.1)
        
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 1.0),
                    log_scale=True,
                ),
            ),
        )
        
        vector = codec.encode(config)
        
        # Log-scale: log(0.1) - log(0.01) / (log(1.0) - log(0.01))
        # = -2.3 - (-4.6) / (0 - (-4.6)) = 2.3 / 4.6 = 0.5
        assert abs(vector[0] - 0.5) < 0.001


class TestConfigCodecDecode:
    """Tests for ConfigCodec.decode()."""
    
    def test_decode_continuous_param(self, base_config: UnifiedConfig) -> None:
        """Test decoding continuous parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.0, 0.5),
                ),
            ),
        )
        
        decoded = codec.decode([0.6])  # Should give 0.3
        
        assert abs(decoded.mutation_rate - 0.3) < 0.001
    
    def test_decode_integer_param(self, base_config: UnifiedConfig) -> None:
        """Test decoding integer parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="population_size",
                    param_type="integer",
                    bounds=(50, 150),
                ),
            ),
        )
        
        decoded = codec.decode([0.5])  # Should give 100
        
        assert decoded.population_size == 100
    
    def test_decode_categorical_param(self, base_config: UnifiedConfig) -> None:
        """Test decoding categorical parameter."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="selection",
                    param_type="categorical",
                    choices=("tournament", "roulette", "rank"),
                ),
            ),
        )
        
        # Value 0.0 should give first choice
        decoded0 = codec.decode([0.0])
        assert decoded0.selection == "tournament"
        
        # Value 0.5 should give second choice
        decoded1 = codec.decode([0.5])
        assert decoded1.selection == "roulette"
        
        # Value 0.9 should give third choice
        decoded2 = codec.decode([0.9])
        assert decoded2.selection == "rank"
    
    def test_decode_invalid_length(self, base_config: UnifiedConfig) -> None:
        """Test that decode raises for wrong vector length."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    path="crossover_rate",
                    param_type="continuous",
                    bounds=(0.0, 1.0),
                ),
            ),
        )
        
        with pytest.raises(ValueError, match="Expected vector of length 2"):
            codec.decode([0.5])  # Only one value


class TestCodecRoundTrip:
    """Tests for encode/decode round-trip."""
    
    def test_roundtrip_continuous(self, base_config: UnifiedConfig) -> None:
        """Test round-trip encoding for continuous parameters."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
            ),
        )
        
        # Encode then decode
        vector = codec.encode(base_config)
        decoded = codec.decode(vector)
        
        # Should be close to original
        assert abs(decoded.mutation_rate - base_config.mutation_rate) < 0.001
    
    def test_roundtrip_multiple_params(self, base_config: UnifiedConfig) -> None:
        """Test round-trip with multiple parameters."""
        codec = ConfigCodec(
            base_config=base_config,
            param_specs=(
                ParameterSpec(
                    path="mutation_rate",
                    param_type="continuous",
                    bounds=(0.01, 0.5),
                ),
                ParameterSpec(
                    path="population_size",
                    param_type="integer",
                    bounds=(50, 200),
                ),
            ),
        )
        
        vector = codec.encode(base_config)
        decoded = codec.decode(vector)
        
        assert abs(decoded.mutation_rate - base_config.mutation_rate) < 0.001
        assert decoded.population_size == base_config.population_size


class TestPathHelpers:
    """Tests for _get_param and _set_param_update."""
    
    def test_get_param_top_level(self, base_config: UnifiedConfig) -> None:
        """Test getting top-level parameter."""
        value = _get_param(base_config, "population_size")
        assert value == 100
    
    def test_get_param_nested(self, base_config: UnifiedConfig) -> None:
        """Test getting nested parameter."""
        value = _get_param(base_config, "mutation_params.sigma")
        assert value == 0.5
    
    def test_get_param_not_found(self, base_config: UnifiedConfig) -> None:
        """Test getting non-existent parameter raises error."""
        with pytest.raises(KeyError, match="not found"):
            _get_param(base_config, "nonexistent.param")
    
    def test_set_param_top_level(self) -> None:
        """Test setting top-level parameter in dict."""
        data: dict = {"a": 1, "b": 2}
        _set_param_update(data, "a", 10)
        assert data["a"] == 10
    
    def test_set_param_nested(self) -> None:
        """Test setting nested parameter creates intermediate dicts."""
        data: dict = {}
        _set_param_update(data, "x.y.z", "value")
        assert data["x"]["y"]["z"] == "value"
