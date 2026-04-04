"""
Integration tests for tracking with create_engine().

Tests that TrackingCallback is correctly wired into the engine
when tracking configuration is provided.
"""

from unittest.mock import Mock

from evolve.config.tracking import TrackingConfig
from evolve.config.unified import UnifiedConfig
from evolve.experiment.tracking.callback import TrackingCallback
from evolve.factory.engine import create_engine


class TestTrackingWithCreateEngine:
    """Integration tests for tracking with create_engine()."""

    def test_no_tracking_callback_when_tracking_none(self) -> None:
        """No TrackingCallback created when tracking is None."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
        )

        # Create a simple evaluator
        def dummy_fitness(genome):
            return sum(genome)

        engine = create_engine(config, dummy_fitness)

        # Not using tracking callback
        callbacks = engine._callbacks
        tracking_callbacks = [cb for cb in callbacks if isinstance(cb, TrackingCallback)]

        assert len(tracking_callbacks) == 0

    def test_tracking_callback_added_when_tracking_enabled(self) -> None:
        """TrackingCallback added when tracking is configured."""
        tracking = TrackingConfig(
            experiment_name="test_integration",
            backend="null",  # Use null backend for testing
        )

        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=tracking,
        )

        def dummy_fitness(genome):
            return sum(genome)

        engine = create_engine(config, dummy_fitness)

        # Check that TrackingCallback exists
        callbacks = engine._callbacks
        tracking_callbacks = [cb for cb in callbacks if isinstance(cb, TrackingCallback)]

        assert len(tracking_callbacks) == 1
        assert tracking_callbacks[0].config.experiment_name == "test_integration"

    def test_tracking_callback_receives_unified_config_dict(self) -> None:
        """TrackingCallback receives serialized UnifiedConfig."""
        tracking = TrackingConfig(
            experiment_name="test_config_dict",
            backend="null",
        )

        config = UnifiedConfig(
            population_size=20,
            max_generations=10,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-2.0, 2.0)},
            mutation_rate=0.05,
            tracking=tracking,
        )

        def dummy_fitness(genome):
            return sum(genome)

        engine = create_engine(config, dummy_fitness)

        # Find TrackingCallback and check its unified_config_dict
        tracking_cb = next(cb for cb in engine._callbacks if isinstance(cb, TrackingCallback))

        assert tracking_cb.unified_config_dict is not None
        assert tracking_cb.unified_config_dict["population_size"] == 20
        assert tracking_cb.unified_config_dict["max_generations"] == 10
        assert tracking_cb.unified_config_dict["mutation_rate"] == 0.05

    def test_tracking_callback_not_added_when_disabled(self) -> None:
        """TrackingCallback not added when tracking.enabled=False."""
        tracking = TrackingConfig(
            experiment_name="disabled_tracking",
            enabled=False,
            backend="null",
        )

        config = UnifiedConfig(
            population_size=10,
            max_generations=5,
            genome_type="vector",
            genome_params={"dimensions": 3, "bounds": (-1.0, 1.0)},
            tracking=tracking,
        )

        def dummy_fitness(genome):
            return sum(genome)

        engine = create_engine(config, dummy_fitness)

        # is_tracking_enabled returns False when enabled=False
        assert config.is_tracking_enabled is False

        # No TrackingCallback
        callbacks = engine._callbacks
        tracking_callbacks = [cb for cb in callbacks if isinstance(cb, TrackingCallback)]

        assert len(tracking_callbacks) == 0


class TestTrackingCallbackLifecycle:
    """Test TrackingCallback lifecycle with null backend."""

    def test_callback_calls_tracker_on_evolution_start(self) -> None:
        """Callback calls tracker.start_run on evolution start."""
        from evolve.experiment.tracking.callback import TrackingCallback

        tracking = TrackingConfig(experiment_name="lifecycle_test", backend="null")
        callback = TrackingCallback(
            config=tracking,
            unified_config_dict={"population_size": 10},
        )

        # Mock population
        population = Mock()

        # Should not raise
        callback.on_evolution_start(population)

        assert callback._started is True

    def test_callback_logs_generation(self) -> None:
        """Callback logs generation metrics via tracker."""
        tracking = TrackingConfig(experiment_name="gen_test", backend="null")
        callback = TrackingCallback(
            config=tracking,
            unified_config_dict={"population_size": 10},
        )

        # Mock population and best individual
        population = Mock()
        best = Mock()

        # Start evolution
        callback.on_evolution_start(population)

        # Log generation
        callback.on_generation_end(
            generation=0,
            population=population,
            best=best,
            metrics={"best_fitness": 10.5},
        )

        # Should not raise
        assert callback._started is True

    def test_callback_ends_run_on_evolution_end(self) -> None:
        """Callback ends tracker run on evolution end."""
        tracking = TrackingConfig(experiment_name="end_test", backend="null")
        callback = TrackingCallback(
            config=tracking,
            unified_config_dict={"population_size": 10},
        )

        # Mock population and best
        population = Mock()
        best = Mock()

        # Full lifecycle
        callback.on_evolution_start(population)
        callback.on_generation_end(
            generation=0,
            population=population,
            best=best,
            metrics={"best_fitness": 10.5},
        )
        callback.on_evolution_end(population, best)

        assert callback._started is False


class TestTrackingLogInterval:
    """Test tracking respects log_interval configuration."""

    def test_log_interval_skips_generations(self) -> None:
        """Generations between log_intervals are skipped."""
        tracking = TrackingConfig(
            experiment_name="interval_test",
            backend="null",
            log_interval=3,  # Log every 3rd generation
        )
        callback = TrackingCallback(
            config=tracking,
            unified_config_dict={},
        )

        population = Mock()
        best = Mock()

        callback.on_evolution_start(population)

        # Generation 1, 2 - should be skipped (internal counter)
        # Generation 3 - should log
        callback.on_generation_end(0, population, best, {"f": 1.0})
        callback.on_generation_end(1, population, best, {"f": 2.0})
        callback.on_generation_end(2, population, best, {"f": 3.0})  # Logs here
        callback.on_generation_end(3, population, best, {"f": 4.0})
        callback.on_generation_end(4, population, best, {"f": 5.0})
        callback.on_generation_end(5, population, best, {"f": 6.0})  # Logs here

        # Should not raise
        callback.on_evolution_end(population, best)
