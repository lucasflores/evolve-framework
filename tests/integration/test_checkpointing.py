"""
Integration tests for experiment checkpointing and resume.

Tests the complete checkpoint -> kill -> resume workflow
to ensure identical continuation.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from random import Random
from uuid import uuid4

import numpy as np
import pytest

from evolve.core.engine import EvolutionConfig, EvolutionEngine, create_initial_population
from evolve.core.operators import GaussianMutation, TournamentSelection, UniformCrossover
from evolve.core.population import Population
from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.evaluation.reference.functions import sphere
from evolve.experiment.checkpoint import Checkpoint, CheckpointManager
from evolve.experiment.config import ExperimentConfig
from evolve.experiment.metrics import LocalTracker, compute_generation_metrics
from evolve.representation.vector import VectorGenome


class TestCheckpointSaveRestore:
    """Test checkpoint serialization."""

    def test_checkpoint_roundtrip(self) -> None:
        """Checkpoint saves and loads correctly."""
        # Create sample population
        rng = Random(42)
        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(np.random.randn(10)),
                fitness=Fitness.scalar(float(i)),
                metadata=IndividualMetadata(),
                created_at=0,
            )
            for i in range(10)
        ]
        population = Population(individuals=individuals, generation=5)

        # Create checkpoint
        checkpoint = Checkpoint(
            experiment_name="test",
            config_hash="abc123",
            generation=5,
            population=population.individuals,
            best_individual=individuals[0],
            rng_state=rng.getstate(),
            fitness_history=[{"gen": i, "best": float(i)} for i in range(5)],
        )

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pkl"
            checkpoint.save(path)

            loaded = Checkpoint.load(path)

            assert loaded.experiment_name == checkpoint.experiment_name
            assert loaded.config_hash == checkpoint.config_hash
            assert loaded.generation == checkpoint.generation
            assert len(loaded.population) == len(checkpoint.population)
            assert loaded.best_individual.id == checkpoint.best_individual.id
            assert loaded.rng_state == checkpoint.rng_state
            assert len(loaded.fitness_history) == len(checkpoint.fitness_history)


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_interval(self) -> None:
        """Manager respects checkpoint interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                output_dir=Path(tmpdir),
                checkpoint_interval=10,
            )

            # 0 is divisible by 10, so it should checkpoint
            assert manager.should_checkpoint(0)  # 0 % 10 == 0
            assert not manager.should_checkpoint(5)
            assert manager.should_checkpoint(10)
            assert not manager.should_checkpoint(15)
            assert manager.should_checkpoint(20)

    def test_save_and_load_latest(self) -> None:
        """Can save and load latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                output_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            # Create and save checkpoints
            for gen in range(3):
                checkpoint = Checkpoint(
                    experiment_name="test",
                    config_hash="abc",
                    generation=gen,
                    population=[],
                    best_individual=None,  # type: ignore
                    rng_state=None,
                )
                manager.save(checkpoint)

            # Load latest
            latest = manager.load_latest()
            assert latest is not None
            assert latest.generation == 2

    def test_load_specific_generation(self) -> None:
        """Can load checkpoint for specific generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                output_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            # Create and save checkpoints
            for gen in range(3):
                checkpoint = Checkpoint(
                    experiment_name="test",
                    config_hash="abc",
                    generation=gen,
                    population=[],
                    best_individual=None,  # type: ignore
                    rng_state=None,
                )
                manager.save(checkpoint)

            # Load specific
            gen1 = manager.load_generation(1)
            assert gen1 is not None
            assert gen1.generation == 1


@pytest.mark.integration
class TestResumeEquivalence:
    """Test that resumed experiments produce identical results."""

    @pytest.fixture
    def simple_engine_setup(self) -> dict:
        """Create simple evolution engine setup."""
        config = EvolutionConfig(
            population_size=20,
            max_generations=10,
            elitism=1,
            crossover_rate=0.9,
            mutation_rate=0.1,
            minimize=True,
        )

        evaluator = FunctionEvaluator(sphere)
        selection = TournamentSelection(tournament_size=3)
        crossover = UniformCrossover()
        mutation = GaussianMutation(sigma=0.1)

        return {
            "config": config,
            "evaluator": evaluator,
            "selection": selection,
            "crossover": crossover,
            "mutation": mutation,
        }

    def test_checkpoint_resume_identical(self, simple_engine_setup: dict) -> None:
        """
        Checkpoint, kill, resume produces identical results.

        This is the key reproducibility test:
        1. Run evolution for N generations
        2. Save checkpoint at generation M
        3. Continue to generation N and record results
        4. Load checkpoint at M
        5. Continue to generation N
        6. Results should be identical
        """
        setup = simple_engine_setup
        seed = 42

        # Run full evolution
        engine_full = EvolutionEngine(
            config=setup["config"],
            evaluator=setup["evaluator"],
            selection=setup["selection"],
            crossover=setup["crossover"],
            mutation=setup["mutation"],
            seed=seed,
        )

        # Create initial population
        from evolve.representation.vector import VectorGenome

        rng = Random(seed)
        init_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome(np.array([r.gauss(0, 1) for _ in range(5)])),
            population_size=setup["config"].population_size,
            rng=rng,
        )

        # Store initial population for later
        Population(
            individuals=[
                Individual(
                    id=ind.id,
                    genome=ind.genome.copy(),
                    metadata=ind.metadata,
                    created_at=ind.created_at,
                )
                for ind in init_pop.individuals
            ],
            generation=0,
        )

        # Run full evolution
        result_full = engine_full.run(init_pop)

        # Now run with checkpoint at generation 5
        checkpoint_gen = 5
        engine_partial = EvolutionEngine(
            config=EvolutionConfig(
                **{**setup["config"].__dict__, "max_generations": checkpoint_gen}
            ),
            evaluator=setup["evaluator"],
            selection=setup["selection"],
            crossover=setup["crossover"],
            mutation=setup["mutation"],
            seed=seed,
        )

        # Reset RNG for same initial conditions
        rng2 = Random(seed)
        init_pop2 = create_initial_population(
            genome_factory=lambda r: VectorGenome(np.array([r.gauss(0, 1) for _ in range(5)])),
            population_size=setup["config"].population_size,
            rng=rng2,
        )

        # Run to checkpoint generation
        result_partial = engine_partial.run(init_pop2)

        # Save checkpoint
        checkpoint = Checkpoint(
            experiment_name="test",
            config_hash="test123",
            generation=checkpoint_gen,
            population=result_partial.population.individuals,
            best_individual=result_partial.best,
            rng_state=engine_partial.get_rng_state(),
            fitness_history=engine_partial.history,
        )

        # Resume from checkpoint
        remaining_gens = setup["config"].max_generations - checkpoint_gen
        engine_resumed = EvolutionEngine(
            config=EvolutionConfig(
                **{**setup["config"].__dict__, "max_generations": remaining_gens}
            ),
            evaluator=setup["evaluator"],
            selection=setup["selection"],
            crossover=setup["crossover"],
            mutation=setup["mutation"],
            seed=seed,  # Seed doesn't matter - we restore RNG state
        )

        # Restore RNG state
        engine_resumed.set_rng_state(checkpoint.rng_state)

        # Recreate population from checkpoint
        resumed_pop = Population(
            individuals=checkpoint.population,
            generation=checkpoint.generation,
        )

        # Run remaining generations
        result_resumed = engine_resumed.run(resumed_pop)

        # Compare final results
        # Best fitness should be identical
        assert (
            abs(result_full.best.fitness.values[0] - result_resumed.best.fitness.values[0]) < 1e-10
        )

        # Final population should have same fitness values
        full_fitnesses = sorted(
            [ind.fitness.values[0] for ind in result_full.population.individuals]
        )
        resumed_fitnesses = sorted(
            [ind.fitness.values[0] for ind in result_resumed.population.individuals]
        )

        for f1, f2 in zip(full_fitnesses, resumed_fitnesses):
            assert abs(f1 - f2) < 1e-10, f"Fitness mismatch: {f1} vs {f2}"


@pytest.mark.integration
class TestLocalTracker:
    """Test local file tracking."""

    def test_tracker_creates_files(self) -> None:
        """Tracker creates expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test_exp",
                seed=42,
                output_dir=tmpdir,
            )

            tracker = LocalTracker()
            tracker.start_run(config)

            # Log some metrics
            for gen in range(5):
                tracker.log_generation(
                    gen,
                    {
                        "best_fitness": float(gen),
                        "mean_fitness": float(gen) * 0.5,
                        "std_fitness": 0.1,
                    },
                )

            tracker.log_params({"custom_param": "value"})
            tracker.end_run()

            # Check files exist
            output_dir = Path(tmpdir) / "test_exp"
            assert (output_dir / "config.json").exists()
            assert (output_dir / "metrics.csv").exists()
            assert (output_dir / "params.json").exists()
            assert (output_dir / "summary.json").exists()

    def test_metrics_csv_content(self) -> None:
        """Metrics CSV contains correct data."""
        import csv

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test_exp",
                seed=42,
                output_dir=tmpdir,
            )

            tracker = LocalTracker()
            tracker.start_run(config)

            tracker.log_generation(0, {"best_fitness": 10.0, "mean_fitness": 5.0})
            tracker.log_generation(1, {"best_fitness": 8.0, "mean_fitness": 4.0})
            tracker.end_run()

            # Read CSV
            metrics_file = Path(tmpdir) / "test_exp" / "metrics.csv"
            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert float(rows[0]["best_fitness"]) == 10.0
            assert float(rows[1]["best_fitness"]) == 8.0


class TestComputeGenerationMetrics:
    """Test metric computation helper."""

    def test_computes_standard_metrics(self) -> None:
        """Computes all standard metrics."""
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        metrics = compute_generation_metrics(fitness_values)

        assert metrics["best_fitness"] == 5.0
        assert metrics["min_fitness"] == 1.0
        assert metrics["mean_fitness"] == 3.0
        assert "std_fitness" in metrics

    def test_includes_diversity(self) -> None:
        """Includes diversity if provided."""
        metrics = compute_generation_metrics([1.0, 2.0, 3.0], diversity=0.5)

        assert metrics["diversity"] == 0.5


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_config_hash_deterministic(self) -> None:
        """Same config produces same hash."""
        config1 = ExperimentConfig(name="test", seed=42)
        config2 = ExperimentConfig(name="test", seed=42)

        assert config1.hash() == config2.hash()

    def test_config_hash_different_for_different_params(self) -> None:
        """Different configs produce different hashes."""
        config1 = ExperimentConfig(name="test", seed=42)
        config2 = ExperimentConfig(name="test", seed=43)

        assert config1.hash() != config2.hash()

    def test_config_json_roundtrip(self) -> None:
        """Config survives JSON serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test",
                seed=42,
                population_size=100,
                n_generations=50,
                mutation_rate=0.01,
            )

            path = Path(tmpdir) / "config.json"
            config.to_json(path)
            loaded = ExperimentConfig.from_json(path)

            assert loaded.name == config.name
            assert loaded.seed == config.seed
            assert loaded.population_size == config.population_size
            assert loaded.n_generations == config.n_generations
            assert loaded.mutation_rate == config.mutation_rate


class TestSCMGenomeCheckpointing:
    """T091: Verify SCMGenome checkpoint compatibility."""

    def test_scm_genome_checkpoint_roundtrip(self) -> None:
        """SCMGenome checkpoints save and load correctly."""
        from uuid import uuid4

        from evolve.representation.scm import SCMConfig, SCMGenome

        rng = Random(42)

        # Create SCM config (only observed_variables is required)
        scm_config = SCMConfig(observed_variables=("X", "Y", "Z"))

        # Create individuals with SCM genomes
        individuals = [
            Individual(
                id=uuid4(),
                genome=SCMGenome.random(scm_config, length=50, rng=Random(i)),
                fitness=Fitness.scalar(float(i)),
                metadata=IndividualMetadata(),
                created_at=0,
            )
            for i in range(10)
        ]

        # Create checkpoint
        checkpoint = Checkpoint(
            experiment_name="scm_test",
            config_hash="scm123",
            generation=5,
            population=individuals,
            best_individual=individuals[0],
            rng_state=rng.getstate(),
            fitness_history=[{"gen": i, "best": float(i)} for i in range(5)],
        )

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scm_checkpoint.pkl"
            checkpoint.save(path)

            loaded = Checkpoint.load(path)

            # Verify checkpoint metadata
            assert loaded.experiment_name == "scm_test"
            assert loaded.generation == 5
            assert len(loaded.population) == 10

            # Verify SCMGenome data preserved
            for orig, restored in zip(individuals, loaded.population):
                assert isinstance(restored.genome, SCMGenome)
                assert list(restored.genome.genes) == list(orig.genome.genes)
                assert restored.genome.config.observed_variables == scm_config.observed_variables

    def test_scm_genome_pickle_directly(self) -> None:
        """SCMGenome can be pickled directly."""
        from evolve.representation.scm import SCMConfig, SCMGenome

        scm_config = SCMConfig(observed_variables=("A", "B", "C", "D"))

        genome = SCMGenome.random(scm_config, length=80, rng=Random(42))

        # Pickle round-trip
        pickled = pickle.dumps(genome)
        restored = pickle.loads(pickled)

        assert isinstance(restored, SCMGenome)
        assert list(restored.genes) == list(genome.genes)
        assert restored.config == genome.config

    def test_checkpoint_manager_with_scm_population(self) -> None:
        """CheckpointManager handles SCM populations correctly."""
        from uuid import uuid4

        from evolve.representation.scm import SCMConfig, SCMGenome

        scm_config = SCMConfig(observed_variables=("X", "Y"))

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                output_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            # Create population with SCM genomes
            individuals = [
                Individual(
                    id=uuid4(),
                    genome=SCMGenome.random(scm_config, length=30, rng=Random(i)),
                    fitness=Fitness.scalar(float(i)),
                    metadata=IndividualMetadata(),
                    created_at=0,
                )
                for i in range(5)
            ]

            # Save checkpoint
            checkpoint = Checkpoint(
                experiment_name="scm_manager_test",
                config_hash="mgr123",
                generation=10,
                population=individuals,
                best_individual=individuals[-1],
                rng_state=Random(99).getstate(),
            )
            manager.save(checkpoint)

            # Load latest
            loaded = manager.load_latest()
            assert loaded is not None
            assert loaded.generation == 10
            assert len(loaded.population) == 5

            # Verify genome types preserved
            for ind in loaded.population:
                assert isinstance(ind.genome, SCMGenome)
