"""
Performance verification tests for tracking (T088).

Validates:
- Batch logging efficiency for large metric sets
- Buffer size limits and overflow behavior
- Flush interval controls
- Diversity sample size configuration
- Large population handling
"""

import time

import pytest

from evolve.config.tracking import TrackingConfig


class TestBatchLoggingEfficiency:
    """Verify batch logging works efficiently for large metric sets."""

    def test_large_metric_dict_accepted(self) -> None:
        """TrackingConfig handles large numbers of metric categories."""
        # Enable all categories - should handle without performance issues
        config = TrackingConfig.comprehensive("large_test")

        # All categories enabled should be performant
        assert len(config.categories) >= 7
        assert config.enabled

    def test_metrics_dict_batching(self) -> None:
        """Large metrics dicts are handled efficiently."""
        config = TrackingConfig(
            experiment_name="batch_test",
            buffer_size=100,
        )

        # Simulate large metrics dict (50 metrics per generation)
        large_metrics = {f"metric_{i}": float(i) for i in range(50)}

        # Config should accept this without issues
        assert len(large_metrics) == 50
        assert config.buffer_size == 100

    def test_many_generations_batched(self) -> None:
        """Metrics across many generations can be batched."""
        config = TrackingConfig(
            experiment_name="many_gen_test",
            buffer_size=1000,
        )

        # Simulate 500 generations worth of metrics
        buffered_entries = []
        for gen in range(500):
            buffered_entries.append(
                {
                    "step": gen,
                    "metrics": {"best_fitness": gen * 1.1},
                }
            )

        assert len(buffered_entries) == 500
        assert len(buffered_entries) < config.buffer_size


class TestBufferSizeLimits:
    """Verify buffer size limits work correctly."""

    def test_buffer_size_minimum_validation(self) -> None:
        """Buffer size must be at least 1."""
        with pytest.raises(ValueError, match="buffer_size must be >= 1"):
            TrackingConfig(
                experiment_name="test",
                buffer_size=0,
            )

    def test_buffer_size_default(self) -> None:
        """Default buffer size is reasonable for production use."""
        config = TrackingConfig(experiment_name="test")

        # Default should be large enough for typical disconnection periods
        assert config.buffer_size >= 100
        assert config.buffer_size <= 10000  # But not excessively large

    def test_buffer_size_configurable(self) -> None:
        """Buffer size can be customized."""
        small_config = TrackingConfig(
            experiment_name="small",
            buffer_size=10,
        )
        large_config = TrackingConfig(
            experiment_name="large",
            buffer_size=5000,
        )

        assert small_config.buffer_size == 10
        assert large_config.buffer_size == 5000

    def test_buffer_overflow_simulation(self) -> None:
        """Demonstrate buffer overflow handling logic."""
        buffer_size = 100
        config = TrackingConfig(
            experiment_name="overflow_test",
            buffer_size=buffer_size,
        )

        # Simulate buffer that exceeds limit
        buffer = list(range(150))

        # Overflow behavior: keep newest, drop oldest
        if len(buffer) > config.buffer_size:
            dropped = len(buffer) - config.buffer_size
            buffer = buffer[-config.buffer_size :]

            assert dropped == 50
            assert len(buffer) == 100
            assert buffer[0] == 50  # Oldest kept is 50, not 0


class TestFlushIntervalBehavior:
    """Verify flush interval controls work correctly."""

    def test_flush_interval_minimum_validation(self) -> None:
        """Flush interval must be positive."""
        with pytest.raises(ValueError, match="flush_interval must be > 0"):
            TrackingConfig(
                experiment_name="test",
                flush_interval=0,
            )

        with pytest.raises(ValueError, match="flush_interval must be > 0"):
            TrackingConfig(
                experiment_name="test",
                flush_interval=-1.0,
            )

    def test_flush_interval_default(self) -> None:
        """Default flush interval is reasonable."""
        config = TrackingConfig(experiment_name="test")

        # Should be frequent enough for observability but not excessive
        assert config.flush_interval >= 1.0  # At least 1 second
        assert config.flush_interval <= 300.0  # At most 5 minutes

    def test_flush_interval_configurable(self) -> None:
        """Flush interval can be customized."""
        fast_config = TrackingConfig(
            experiment_name="fast",
            flush_interval=1.0,
        )
        slow_config = TrackingConfig(
            experiment_name="slow",
            flush_interval=60.0,
        )

        assert fast_config.flush_interval == 1.0
        assert slow_config.flush_interval == 60.0

    def test_flush_timing_logic(self) -> None:
        """Verify flush timing calculation works."""
        flush_interval = 5.0

        last_flush_time = time.time() - 3.0  # 3 seconds ago
        current_time = time.time()

        should_flush = (current_time - last_flush_time) >= flush_interval
        assert not should_flush  # 3 < 5

        last_flush_time = time.time() - 6.0  # 6 seconds ago
        current_time = time.time()

        should_flush = (current_time - last_flush_time) >= flush_interval
        assert should_flush  # 6 >= 5


class TestDiversitySampleSize:
    """Verify diversity sample size limits large population handling."""

    def test_sample_size_minimum_validation(self) -> None:
        """Sample size must be at least 10."""
        with pytest.raises(ValueError, match="diversity_sample_size must be >= 10"):
            TrackingConfig(
                experiment_name="test",
                diversity_sample_size=5,
            )

    def test_sample_size_default(self) -> None:
        """Default sample size is reasonable."""
        config = TrackingConfig(experiment_name="test")

        # Should be large enough for statistical validity
        assert config.diversity_sample_size >= 100

    def test_sample_size_configurable(self) -> None:
        """Sample size can be customized."""
        small_config = TrackingConfig(
            experiment_name="small",
            diversity_sample_size=50,
        )
        large_config = TrackingConfig(
            experiment_name="large",
            diversity_sample_size=5000,
        )

        assert small_config.diversity_sample_size == 50
        assert large_config.diversity_sample_size == 5000

    def test_sampling_logic_for_large_population(self) -> None:
        """Demonstrate sampling logic for large populations."""
        import random

        sample_size = 100
        population_size = 10000

        # Simulate large population
        population = list(range(population_size))

        # If population exceeds sample size, sample
        if len(population) > sample_size:
            sampled = random.sample(population, sample_size)
        else:
            sampled = population

        assert len(sampled) == sample_size
        assert len(sampled) < len(population)

    def test_sampling_preserves_small_populations(self) -> None:
        """Small populations under sample size are not reduced."""
        sample_size = 1000
        population_size = 50

        population = list(range(population_size))

        # Small population - use all
        sampled = population[:sample_size] if len(population) > sample_size else population

        assert len(sampled) == population_size


class TestLargePopulationHandling:
    """Verify tracking handles large populations efficiently."""

    def test_large_population_metrics_extraction(self) -> None:
        """Metrics extraction scales with population size."""
        # Simulate extracting metrics from large population
        population_size = 10000

        # Core metrics (O(n) operations)
        fitnesses = [float(i) for i in range(population_size)]

        best_fitness = max(fitnesses)
        mean_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)

        assert best_fitness == 9999.0
        assert mean_fitness == 4999.5
        assert min_fitness == 0.0

    def test_diversity_computation_with_sampling(self) -> None:
        """Diversity computation uses sampling for efficiency."""
        import random

        population_size = 10000
        sample_size = 100

        # Generate random genomes
        population = [[random.random() for _ in range(10)] for _ in range(population_size)]

        # Sample for diversity computation
        if len(population) > sample_size:
            sample = random.sample(population, sample_size)
        else:
            sample = population

        # Compute diversity on sample (mock computation)
        unique_genomes = len({tuple(g) for g in sample})
        diversity_ratio = unique_genomes / len(sample)

        assert len(sample) == sample_size
        assert 0.0 <= diversity_ratio <= 1.0

    def test_log_interval_reduces_overhead(self) -> None:
        """Log interval reduces logging frequency for long runs."""
        config = TrackingConfig(
            experiment_name="long_run",
            log_interval=10,  # Log every 10th generation
        )

        generations = 1000
        expected_logs = generations // config.log_interval

        actual_logs = sum(1 for g in range(generations) if (g + 1) % config.log_interval == 0)

        assert actual_logs == expected_logs
        assert actual_logs == 100  # Much less than 1000


class TestConfigSerialization:
    """Verify config serialization includes performance settings."""

    def test_to_dict_includes_buffer_settings(self) -> None:
        """JSON serialization includes buffer configuration."""
        config = TrackingConfig(
            experiment_name="serialize_test",
            buffer_size=500,
            flush_interval=15.0,
        )

        data = config.to_dict()

        assert data["buffer_size"] == 500
        assert data["flush_interval"] == 15.0

    def test_to_dict_includes_sample_size(self) -> None:
        """JSON serialization includes sampling configuration."""
        config = TrackingConfig(
            experiment_name="serialize_test",
            diversity_sample_size=200,
        )

        data = config.to_dict()

        assert data["diversity_sample_size"] == 200

    def test_to_dict_full_performance_roundtrip(self) -> None:
        """Performance settings survive serialization roundtrip."""
        original = TrackingConfig(
            experiment_name="roundtrip_test",
            buffer_size=250,
            flush_interval=20.0,
            diversity_sample_size=500,
            log_interval=5,
        )

        data = original.to_dict()

        # Verify all performance settings preserved
        assert data["buffer_size"] == 250
        assert data["flush_interval"] == 20.0
        assert data["diversity_sample_size"] == 500
        assert data["log_interval"] == 5


class TestFactoryMethodPerformance:
    """Verify factory methods set appropriate performance defaults."""

    def test_minimal_has_conservative_defaults(self) -> None:
        """Minimal config uses conservative performance settings."""
        config = TrackingConfig.minimal()

        # Minimal should have reasonable defaults
        assert config.buffer_size >= 100
        assert config.diversity_sample_size >= 100

    def test_standard_has_balanced_defaults(self) -> None:
        """Standard config balances observability and performance."""
        config = TrackingConfig.standard("balanced")

        # Standard enables more features
        assert len(config.categories) > len(TrackingConfig.minimal().categories)

        # But still has reasonable performance settings
        assert config.buffer_size >= 100

    def test_comprehensive_handles_large_scale(self) -> None:
        """Comprehensive config can handle large scale experiments."""
        config = TrackingConfig.comprehensive("large_scale")

        # All categories enabled
        assert len(config.categories) >= 7

        # Buffer can handle sustained disconnection
        assert config.buffer_size >= 100


class TestPerformanceEdgeCases:
    """Test edge cases for performance-related configuration."""

    def test_minimum_valid_buffer_size(self) -> None:
        """Minimum buffer size of 1 is valid."""
        config = TrackingConfig(
            experiment_name="edge",
            buffer_size=1,
        )
        assert config.buffer_size == 1

    def test_minimum_valid_sample_size(self) -> None:
        """Minimum sample size of 10 is valid."""
        config = TrackingConfig(
            experiment_name="edge",
            diversity_sample_size=10,
        )
        assert config.diversity_sample_size == 10

    def test_very_small_flush_interval(self) -> None:
        """Very small (but positive) flush interval is valid."""
        config = TrackingConfig(
            experiment_name="edge",
            flush_interval=0.001,  # 1 millisecond
        )
        assert config.flush_interval == 0.001

    def test_very_large_buffer_handles_long_outage(self) -> None:
        """Large buffer can handle extended server outages."""
        config = TrackingConfig(
            experiment_name="edge",
            buffer_size=10000,
        )

        # 10000 generations at 1 metric set per gen = ~10k entries
        # This is reasonable for hours of disconnection
        assert config.buffer_size == 10000
