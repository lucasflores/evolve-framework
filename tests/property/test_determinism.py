"""
Property-based tests for determinism.

Verifies that evolution is reproducible with the same seed.
Uses Hypothesis for generating random test cases.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evolve.core.engine import EvolutionConfig, EvolutionEngine, create_initial_population
from evolve.core.operators.crossover import BlendCrossover, UniformCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.evaluation.reference.functions import sphere
from evolve.representation.vector import VectorGenome
from evolve.utils.random import create_rng, derive_seed


@pytest.mark.property
class TestDeterminism:
    """Property tests for deterministic reproducibility."""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=10, deadline=None)
    def test_same_seed_produces_identical_results(self, seed: int):
        """
        Running evolution twice with the same seed should produce
        identical results (fitness values, best individual, etc.)
        """
        n_dims = 5
        bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))

        config = EvolutionConfig(
            population_size=20,
            max_generations=10,
            elitism=1,
        )

        # Run 1
        engine1 = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(mutation_rate=0.1, sigma=0.1),
            seed=seed,
        )

        rng1 = create_rng(seed)
        pop1 = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng1,
        )
        result1 = engine1.run(pop1)

        # Run 2 (same seed)
        engine2 = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(mutation_rate=0.1, sigma=0.1),
            seed=seed,
        )

        rng2 = create_rng(seed)
        pop2 = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng2,
        )
        result2 = engine2.run(pop2)

        # Results should be identical
        assert result1.generations == result2.generations
        assert len(result1.history) == len(result2.history)

        # Best fitness should be identical
        assert result1.best.fitness is not None
        assert result2.best.fitness is not None
        np.testing.assert_array_almost_equal(
            result1.best.fitness.values,
            result2.best.fitness.values,
            decimal=10,
        )

        # Best genome should be identical
        np.testing.assert_array_almost_equal(
            result1.best.genome.genes,
            result2.best.genome.genes,
            decimal=10,
        )

        # History should be identical
        for h1, h2 in zip(result1.history, result2.history):
            assert h1["generation"] == h2["generation"]
            np.testing.assert_almost_equal(
                h1["best_fitness"],
                h2["best_fitness"],
                decimal=10,
            )

    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=10, deadline=None)
    def test_different_seeds_produce_different_results(self, seed1: int, seed2: int):
        """
        Different seeds should produce different evolution trajectories.
        (Statistical test - may occasionally fail for very similar seeds)
        """
        if seed1 == seed2:
            # Skip if seeds are equal
            return

        n_dims = 5
        bounds = (np.full(n_dims, -5.0), np.full(n_dims, 5.0))

        config = EvolutionConfig(
            population_size=20,
            max_generations=10,
            elitism=1,
        )

        # Run with seed1
        engine1 = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.2),
            seed=seed1,
        )

        rng1 = create_rng(seed1)
        pop1 = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng1,
        )
        result1 = engine1.run(pop1)

        # Run with seed2
        engine2 = EvolutionEngine(
            config=config,
            evaluator=FunctionEvaluator(sphere),
            selection=TournamentSelection(),
            crossover=BlendCrossover(),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.2),
            seed=seed2,
        )

        rng2 = create_rng(seed2)
        pop2 = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_dims, bounds, r),
            population_size=config.population_size,
            rng=rng2,
        )
        result2 = engine2.run(pop2)

        # At least the final best genomes should differ
        # (very unlikely to be identical with different seeds)
        assert not np.allclose(
            result1.best.genome.genes,
            result2.best.genome.genes,
            rtol=1e-5,
        ), "Different seeds should produce different genomes"


@pytest.mark.property
class TestSeedDerivation:
    """Tests for deterministic seed derivation."""

    @given(
        master_seed=st.integers(min_value=0, max_value=2**63 - 1),
        worker_id=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50)
    def test_derive_seed_is_deterministic(self, master_seed: int, worker_id: int):
        """Same inputs should always produce same derived seed."""
        seed1 = derive_seed(master_seed, worker_id)
        seed2 = derive_seed(master_seed, worker_id)
        assert seed1 == seed2

    @given(
        master_seed=st.integers(min_value=0, max_value=2**63 - 1),
    )
    @settings(max_examples=20)
    def test_different_workers_get_different_seeds(self, master_seed: int):
        """Different worker IDs should produce different seeds."""
        seeds = [derive_seed(master_seed, i) for i in range(10)]
        # All seeds should be unique
        assert len(set(seeds)) == 10, "Each worker should get a unique seed"

    @given(
        master_seed1=st.integers(min_value=0, max_value=2**63 - 1),
        master_seed2=st.integers(min_value=0, max_value=2**63 - 1),
    )
    @settings(max_examples=20)
    def test_different_masters_produce_different_derived_seeds(
        self, master_seed1: int, master_seed2: int
    ):
        """Different master seeds should produce different worker seeds."""
        if master_seed1 == master_seed2:
            return

        derived1 = derive_seed(master_seed1, 0)
        derived2 = derive_seed(master_seed2, 0)
        assert derived1 != derived2


@pytest.mark.property
class TestOperatorDeterminism:
    """Tests for operator determinism."""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20)
    def test_mutation_is_deterministic(self, seed: int):
        """Same seed should produce same mutation."""
        genome = VectorGenome(
            genes=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            bounds=(np.zeros(5), np.ones(5) * 10),
        )
        mutation = GaussianMutation(mutation_rate=0.5, sigma=0.5)

        rng1 = create_rng(seed)
        result1 = mutation.mutate(genome, rng1)

        rng2 = create_rng(seed)
        result2 = mutation.mutate(genome, rng2)

        np.testing.assert_array_equal(result1.genes, result2.genes)

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20)
    def test_crossover_is_deterministic(self, seed: int):
        """Same seed should produce same crossover result."""
        parent1 = VectorGenome(genes=np.zeros(5))
        parent2 = VectorGenome(genes=np.ones(5))

        crossover = UniformCrossover(swap_prob=0.5)

        rng1 = create_rng(seed)
        child1a, child1b = crossover.crossover(parent1, parent2, rng1)

        rng2 = create_rng(seed)
        child2a, child2b = crossover.crossover(parent1, parent2, rng2)

        np.testing.assert_array_equal(child1a.genes, child2a.genes)
        np.testing.assert_array_equal(child1b.genes, child2b.genes)
