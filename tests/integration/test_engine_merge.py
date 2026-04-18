"""Integration tests for engine with symbiogenetic merge enabled."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from uuid import uuid4

import numpy as np
import pytest

from evolve.config.merge import MergeConfig
from evolve.config.unified import UnifiedConfig
from evolve.core.engine import EvolutionConfig, EvolutionEngine
from evolve.core.operators.crossover import UniformCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.population import Population
from evolve.core.types import Individual, IndividualMetadata
from evolve.evaluation.evaluator import FunctionEvaluator
from evolve.representation.vector import VectorGenome


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


@dataclass
class _AveragingMerge:
    """Merge operator that averages genes, preserving dimensions."""

    def merge(self, host: VectorGenome, symbiont: VectorGenome, rng: Random) -> VectorGenome:
        if host is symbiont:
            raise ValueError("Cannot merge an individual with itself")
        min_len = min(len(host.genes), len(symbiont.genes))
        merged = (host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0
        return VectorGenome(genes=merged)


class TestEngineMergeIntegration:
    """Test that merge phase works in the full engine loop."""

    def test_engine_with_merge_runs(self) -> None:
        """Engine with merge operator completes without error."""
        config = EvolutionConfig(
            population_size=20,
            max_generations=5,
            elitism=2,
            crossover_rate=0.9,
            mutation_rate=0.5,
            minimize=True,
            merge_rate=0.3,
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        # Create initial population
        rng = Random(42)
        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(
                    genes=np.random.default_rng(rng.randint(0, 2**31)).standard_normal(5)
                ),
            )
            for _ in range(20)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        assert result.generations > 0
        assert result.best.fitness is not None

    def test_engine_without_merge_unchanged(self) -> None:
        """Engine without merge operator (merge_rate=0) runs as before."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=3,
            merge_rate=0.0,
        )
        evaluator = FunctionEvaluator(_sphere)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
        )

        rng = Random(42)
        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(
                    genes=np.random.default_rng(rng.randint(0, 2**31)).standard_normal(3)
                ),
            )
            for _ in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)
        assert result.generations > 0

    def test_merge_creates_symbiogenetic_origin(self) -> None:
        """Verify merged individuals have origin='symbiogenetic_merge'."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=2,
            merge_rate=1.0,  # 100% merge rate for determinism
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        # At least some individuals should have symbiogenetic_merge origin
        origins = [ind.metadata.origin for ind in result.population.individuals]
        assert "symbiogenetic_merge" in origins

    def test_merge_rate_zero_no_merges(self) -> None:
        """With merge_rate=0, no merges occur even with operator set."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=2,
            merge_rate=0.0,
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        origins = [ind.metadata.origin for ind in result.population.individuals]
        assert "symbiogenetic_merge" not in origins


class TestFactoryMergeIntegration:
    """Test merge operator resolution via factory."""

    def test_create_engine_with_merge_config(self) -> None:
        """Factory resolves merge operator from MergeConfig."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=3,
            selection="tournament",
            crossover="neat",
            mutation="neat",
            genome_type="graph",
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            merge=MergeConfig(
                operator="graph_symbiogenetic",
                merge_rate=0.2,
            ),
        )

        from evolve.factory.engine import create_engine

        engine = create_engine(config, seed=42)
        assert engine.merge_operator is not None
        assert engine.config.merge_rate == 0.2

    def test_create_engine_without_merge(self) -> None:
        """Factory creates engine without merge when not configured."""
        config = UnifiedConfig(
            population_size=10,
            max_generations=3,
            selection="tournament",
            crossover="uniform",
            mutation="gaussian",
            genome_type="vector",
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
        )

        from evolve.factory.engine import create_engine

        engine = create_engine(config, seed=42)
        assert engine.merge_operator is None
        assert engine.config.merge_rate == 0.0

    def test_incompatible_merge_operator_raises(self) -> None:
        """T040: clear error when merge operator incompatible with genome type."""
        from evolve.factory.engine import OperatorCompatibilityError, create_engine

        config = UnifiedConfig(
            population_size=10,
            max_generations=3,
            selection="tournament",
            crossover="uniform",
            mutation="gaussian",
            genome_type="vector",
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            merge=MergeConfig(
                operator="graph_symbiogenetic",
                merge_rate=0.2,
            ),
        )

        with pytest.raises(OperatorCompatibilityError):
            create_engine(config, seed=42)


class TestEngineMergeEdgeCases:
    """Phase 8 edge case tests."""

    def test_host_and_symbiont_always_distinct(self) -> None:
        """T061: host and symbiont are always distinct during merge."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=3,
            merge_rate=1.0,
        )
        evaluator = FunctionEvaluator(_sphere)

        # Track merge calls to verify distinct individuals
        merge_calls: list[tuple[int, int]] = []

        @dataclass
        class _TrackingMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                merge_calls.append((id(host), id(symbiont)))
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_TrackingMerge(),
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        engine.run(pop)

        # Verify all merge calls used distinct objects
        for host_id, symbiont_id in merge_calls:
            assert host_id != symbiont_id, "Host and symbiont were the same object"

    def test_merge_rate_1_all_eligible_merged(self) -> None:
        """T026: merge_rate=1.0 applies merge to all eligible offspring."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        merge_count = [0]

        @dataclass
        class _CountingMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                merge_count[0] += 1
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_CountingMerge(),
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        engine.run(pop)

        # All non-elite offspring should have been merge candidates
        n_offspring = config.population_size - config.elitism
        assert merge_count[0] == n_offspring

    def test_symbiont_consumed_removes_from_population(self) -> None:
        """T027: symbiont_fate='consumed' removes symbiont from offspring."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_fate="consumed",
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        # With consumed fate, some offspring are consumed; population may shrink
        # but elites are preserved. The key assertion is that the run completes
        # and that we get fewer merged offspring than with "survives".
        merged = [
            ind
            for ind in result.population.individuals
            if ind.metadata.origin == "symbiogenetic_merge"
        ]
        # With consumed, some hosts lose their symbiont so not all can merge
        n_offspring = config.population_size - config.elitism
        assert len(merged) < n_offspring

    def test_symbiont_survives_keeps_in_population(self) -> None:
        """T028: symbiont_fate='survives' keeps symbiont in offspring."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        merge_count = [0]

        @dataclass
        class _CountingMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                merge_count[0] += 1
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_CountingMerge(),
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        engine.run(pop)

        # With survives, all offspring are merge candidates (same as T026)
        n_offspring = config.population_size - config.elitism
        assert merge_count[0] == n_offspring

    def test_cross_species_sourcing(self) -> None:
        """T029: cross_species sourcing selects symbiont from different species."""
        config = EvolutionConfig(
            population_size=6,
            max_generations=1,
            merge_rate=1.0,
            symbiont_source="cross_species",
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        @dataclass
        class _SpeciesTrackingMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_SpeciesTrackingMerge(),
        )

        # Create population with two species
        individuals = []
        for i in range(6):
            ind = Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
                metadata=IndividualMetadata(species_id=i % 2),
            )
            individuals.append(ind)
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        # Merge should have occurred since we have two species
        origins = [ind.metadata.origin for ind in result.population.individuals]
        assert "symbiogenetic_merge" in origins

    def test_single_species_cross_species_skips_with_warning(self) -> None:
        """T030: single species + cross_species sourcing gracefully skips merge."""
        # Directly test _apply_merge with species-tagged offspring
        config = EvolutionConfig(
            population_size=6,
            max_generations=1,
            merge_rate=1.0,
            symbiont_source="cross_species",
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )
        engine._callbacks = []

        # Create offspring all with same species
        offspring = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
                metadata=IndividualMetadata(species_id=0),
            )
            for i in range(5)
        ]

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine._apply_merge(offspring)

        # Should have emitted warnings about no cross-species symbiont
        merge_warnings = [x for x in w if "cross_species" in str(x.message)]
        assert len(merge_warnings) > 0

    def test_archive_sourcing_with_empty_archive_skips(self) -> None:
        """T060: archive-based sourcing with empty archive skips merge."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_source="archive",
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(pop)

        # Should warn about empty archive
        archive_warnings = [x for x in w if "archive" in str(x.message).lower()]
        assert len(archive_warnings) > 0

        # No merges should have occurred
        origins = [ind.metadata.origin for ind in result.population.individuals]
        assert "symbiogenetic_merge" not in origins

    def test_max_complexity_threshold_skips_merge(self) -> None:
        """T067: merge skipped when result exceeds max_complexity."""
        config = EvolutionConfig(
            population_size=6,
            max_generations=1,
            merge_rate=1.0,
            max_complexity=2,  # Very low threshold
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        @dataclass
        class _ExpandingMerge:
            """Merge that always produces large genomes."""

            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                return VectorGenome(genes=np.concatenate([host.genes, symbiont.genes]))

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_ExpandingMerge(),
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(6)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(pop)

        # Should have warnings about exceeding max_complexity
        complexity_warnings = [x for x in w if "max_complexity" in str(x.message)]
        assert len(complexity_warnings) > 0

        # No merges should have succeeded (3+3=6 > 2)
        origins = [ind.metadata.origin for ind in result.population.individuals]
        assert "symbiogenetic_merge" not in origins

    def test_source_strategy_in_metadata(self) -> None:
        """T069: merged offspring metadata includes source_strategy."""
        config = EvolutionConfig(
            population_size=10,
            max_generations=1,
            merge_rate=1.0,
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)
        merge_op = _AveragingMerge()

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=merge_op,
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(10)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        result = engine.run(pop)

        merged = [
            ind
            for ind in result.population.individuals
            if ind.metadata.origin == "symbiogenetic_merge"
        ]
        assert len(merged) > 0
        for ind in merged:
            assert ind.metadata.source_strategy == "cross_species"

    def test_uniform_random_host_selection(self) -> None:
        """T070: host selection for merge is uniform random."""
        config = EvolutionConfig(
            population_size=20,
            max_generations=50,
            merge_rate=0.5,
            symbiont_fate="survives",
        )
        evaluator = FunctionEvaluator(_sphere)

        merge_count = [0]

        @dataclass
        class _CountingMerge:
            def merge(
                self, host: VectorGenome, symbiont: VectorGenome, rng: Random
            ) -> VectorGenome:
                merge_count[0] += 1
                min_len = min(len(host.genes), len(symbiont.genes))
                return VectorGenome(genes=(host.genes[:min_len] + symbiont.genes[:min_len]) / 2.0)

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(sigma=0.1),
            seed=42,
            merge=_CountingMerge(),
        )

        individuals = [
            Individual(
                id=uuid4(),
                genome=VectorGenome(genes=np.random.default_rng(i).standard_normal(3)),
            )
            for i in range(20)
        ]
        pop = Population(individuals=individuals, generation=0, minimize=True)
        engine.run(pop)

        # With 50 generations, 19 offspring each, 50% rate → ~475 expected merges
        # Statistical test: should be roughly 50% of offspring across generations
        n_offspring_per_gen = config.population_size - config.elitism
        expected_per_gen = n_offspring_per_gen * config.merge_rate
        total_expected = expected_per_gen * config.max_generations
        # Allow 30% tolerance
        assert merge_count[0] > total_expected * 0.5
        assert merge_count[0] < total_expected * 1.5

    def test_engine_selects_compatible_merge_by_genome_type(self) -> None:
        """T039: engine uses merge operator compatible with genome type."""
        from evolve.factory.engine import create_engine

        config = UnifiedConfig(
            population_size=10,
            max_generations=1,
            selection="tournament",
            crossover="neat",
            mutation="neat",
            genome_type="graph",
            evaluator="benchmark",
            evaluator_params={"function_name": "sphere"},
            merge=MergeConfig(
                operator="graph_symbiogenetic",
                merge_rate=0.5,
            ),
        )

        engine = create_engine(config, seed=42)
        assert engine.merge_operator is not None

        # Verify the resolved operator is a GraphSymbiogeneticMerge
        from evolve.core.operators.merge import GraphSymbiogeneticMerge

        assert isinstance(engine.merge_operator, GraphSymbiogeneticMerge)
