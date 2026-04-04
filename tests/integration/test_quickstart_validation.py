"""
Validation tests for quickstart.md examples.

Tests that all code examples in the quickstart documentation
compile and execute correctly (imports, config creation, serialization).

T087: Run quickstart.md examples as integration validation.
"""

from __future__ import annotations

import json

import numpy as np

from evolve.config.multiobjective import MultiObjectiveConfig, ObjectiveSpec
from evolve.config.tracking import MetricCategory, TrackingConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.types import Fitness


class TestQuickstartSection1BasicSetup:
    """Section 1: Basic Setup with UnifiedConfig."""

    def test_basic_config_with_tracking(self):
        """Test the basic setup example compiles and runs."""
        config = UnifiedConfig(
            name="sphere_optimization",
            population_size=100,
            max_generations=50,
            selection="tournament",
            crossover="sbx",
            mutation="gaussian",
            genome_type="vector",
            genome_params={"dimensions": 10, "bounds": (-5.12, 5.12)},
            # Enable MLflow tracking with standard metrics
            tracking=TrackingConfig(
                backend="null",  # Use null backend for testing
                experiment_name="evolve_experiments",
                run_name="sphere_run_001",
            ),
        )

        assert config.tracking is not None
        assert config.tracking.experiment_name == "evolve_experiments"
        assert config.tracking.run_name == "sphere_run_001"


class TestQuickstartSection2MetricCategories:
    """Section 2: Metric Categories."""

    def test_minimal_factory(self):
        """Test TrackingConfig.minimal() factory method."""
        tracking = TrackingConfig.minimal()

        assert MetricCategory.CORE in tracking.categories
        assert len(tracking.categories) == 1

    def test_standard_categories(self):
        """Test standard category configuration."""
        tracking = TrackingConfig(
            experiment_name="my_experiment",
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.EXTENDED_POPULATION,
                    MetricCategory.TIMING,
                }
            ),
        )

        assert tracking.has_category(MetricCategory.CORE)
        assert tracking.has_category(MetricCategory.EXTENDED_POPULATION)
        assert tracking.has_category(MetricCategory.TIMING)

    def test_comprehensive_factory(self):
        """Test TrackingConfig.comprehensive() factory method."""
        tracking = TrackingConfig.comprehensive("detailed_analysis")

        assert tracking.experiment_name == "detailed_analysis"
        # Comprehensive should have many categories
        assert len(tracking.categories) >= 5


class TestQuickstartSection3DomainSpecific:
    """Section 3: Domain-Specific Metrics."""

    def test_multiobjective_config(self):
        """Test multi-objective configuration example."""
        config = UnifiedConfig(
            name="moo_experiment",
            population_size=50,
            max_generations=20,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(
                backend="null",
                categories=frozenset(
                    {
                        MetricCategory.CORE,
                        MetricCategory.MULTIOBJECTIVE,
                    }
                ),
                hypervolume_reference=(10.0, 0.0),
            ),
        )

        # Add multiobjective config
        config = config.with_multiobjective(
            objectives=(
                ObjectiveSpec(name="obj1", direction="minimize"),
                ObjectiveSpec(name="obj2", direction="maximize"),
            ),
        )

        assert config.multiobjective is not None
        assert len(config.multiobjective.objectives) == 2

    def test_erp_config(self):
        """Test ERP configuration example."""
        base_config = UnifiedConfig(
            name="erp_experiment",
            population_size=50,
            max_generations=20,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(
                backend="null",
                categories=frozenset(
                    {
                        MetricCategory.CORE,
                        MetricCategory.ERP,
                    }
                ),
            ),
        )

        # Add ERP config
        config = base_config.with_erp()

        assert config.erp is not None
        # ERP category should be auto-enabled
        assert config.tracking.has_category(MetricCategory.ERP)

    def test_speciation_categories(self):
        """Test speciation tracking example."""
        tracking = TrackingConfig(
            backend="null",
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.SPECIATION,
                }
            ),
        )

        assert tracking.has_category(MetricCategory.SPECIATION)


class TestQuickstartSection4FitnessMetadata:
    """Section 4: Fitness Metadata Extraction."""

    def test_metadata_tracking_config(self):
        """Test metadata extraction configuration."""
        config = UnifiedConfig(
            name="rl_experiment",
            population_size=50,
            max_generations=20,
            genome_type="vector",
            genome_params={"dimensions": 5, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(
                backend="null",
                categories=frozenset(
                    {
                        MetricCategory.CORE,
                        MetricCategory.METADATA,
                    }
                ),
                metadata_threshold=0.5,
                metadata_prefix="meta_",
            ),
        )

        assert config.tracking.metadata_threshold == 0.5
        assert config.tracking.metadata_prefix == "meta_"

    def test_fitness_with_metadata(self):
        """Test Fitness creation with metadata."""
        fitness = Fitness(
            values=np.array([100.0]),
            metadata={
                "episode_reward": 100.0,
                "steps": 500,
                "collisions": 3,
                "goal_reached": 1,
            },
        )

        assert fitness.metadata["episode_reward"] == 100.0
        assert fitness.metadata["steps"] == 500


class TestQuickstartSection5Timing:
    """Section 5: Timing Instrumentation."""

    def test_timing_config(self):
        """Test timing configuration."""
        tracking = TrackingConfig(
            backend="null",
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.TIMING,
                }
            ),
            timing_breakdown=True,
        )

        assert tracking.timing_breakdown is True
        assert tracking.has_category(MetricCategory.TIMING)


class TestQuickstartSection6DerivedAnalytics:
    """Section 6: Derived Analytics."""

    def test_derived_config(self):
        """Test derived analytics configuration."""
        tracking = TrackingConfig(
            backend="null",
            categories=frozenset(
                {
                    MetricCategory.CORE,
                    MetricCategory.DERIVED,
                }
            ),
        )

        assert tracking.has_category(MetricCategory.DERIVED)


class TestQuickstartSection7RemoteServer:
    """Section 7: Remote MLflow Server."""

    def test_remote_server_config(self):
        """Test remote server configuration with resilience settings."""
        tracking = TrackingConfig(
            backend="mlflow",
            tracking_uri="http://mlflow.mycompany.com:5000",
            experiment_name="production_runs",
            buffer_size=500,
            flush_interval=60.0,
        )

        assert tracking.tracking_uri == "http://mlflow.mycompany.com:5000"
        assert tracking.buffer_size == 500
        assert tracking.flush_interval == 60.0


class TestQuickstartSection8JsonSerialization:
    """Section 8: JSON Configuration."""

    def test_config_to_dict_and_back(self):
        """Test JSON serialization round-trip."""
        config = UnifiedConfig(
            name="my_experiment",
            population_size=100,
            max_generations=50,
            genome_type="vector",
            genome_params={"dimensions": 10, "bounds": (-1.0, 1.0)},
            tracking=TrackingConfig(
                enabled=True,
                backend="mlflow",
                experiment_name="evolve_experiments",
                categories=frozenset(
                    {
                        MetricCategory.CORE,
                        MetricCategory.TIMING,
                        MetricCategory.EXTENDED_POPULATION,
                    }
                ),
                log_interval=1,
                timing_breakdown=False,
            ),
        )

        # Convert to dict
        config_dict = config.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(config_dict, indent=2)
        assert len(json_str) > 0

        # Restore from dict
        restored = UnifiedConfig.from_dict(config_dict)

        assert restored.name == "my_experiment"
        assert restored.tracking is not None
        assert restored.tracking.experiment_name == "evolve_experiments"
        assert restored.tracking.has_category(MetricCategory.CORE)


class TestQuickstartImportsWork:
    """Test that all imports mentioned in quickstart work."""

    def test_all_imports(self):
        """Verify all quickstart imports are available."""
        # Section 1
        from evolve.config.erp import ERPSettings

        # Section 3
        from evolve.config.multiobjective import ObjectiveSpec

        # Section 2
        from evolve.config.tracking import MetricCategory, TrackingConfig
        from evolve.config.unified import UnifiedConfig

        # Section 4
        from evolve.core.types import Fitness
        from evolve.factory import create_engine

        # All imports should work
        assert UnifiedConfig is not None
        assert TrackingConfig is not None
        assert create_engine is not None
        assert MetricCategory is not None
        assert MultiObjectiveConfig is not None
        assert ObjectiveSpec is not None
        assert ERPSettings is not None
        assert Fitness is not None
