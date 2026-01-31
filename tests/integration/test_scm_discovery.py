"""
Integration tests for SCM (Structural Causal Model) discovery.

Tests cover:
- T094 [US7]: ERP-enabled evolution with SCM matchability
- T103: End-to-end SCM discovery on synthetic data (Phase 10)
- T104: Population evolution at scale (Phase 10)

These tests verify that SCM genomes integrate correctly with the
evolutionary framework including ERP matchability.
"""

from __future__ import annotations

from random import Random
from uuid import uuid4

import numpy as np
import pytest

from evolve.core.types import Fitness, Individual, IndividualMetadata
from evolve.reproduction.matchability import evaluate_matchability
from evolve.reproduction.protocol import MateContext, ReproductionProtocol
from evolve.reproduction.sandbox import StepCounter


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Simple 3-variable SCM configuration."""
    from evolve.representation.scm import SCMConfig
    return SCMConfig(observed_variables=("A", "B", "C"))


@pytest.fixture
def rng() -> Random:
    """Seeded random number generator."""
    return Random(42)


@pytest.fixture
def counter() -> StepCounter:
    """Step counter for resource tracking."""
    return StepCounter(limit=1000)


def create_scm_individual(
    genome,
    fitness_value: float = 1.0,
) -> Individual:
    """Create a test individual with SCMGenome."""
    from evolve.reproduction.protocol import ReproductionProtocol
    
    return Individual(
        id=uuid4(),
        genome=genome,
        protocol=ReproductionProtocol.default(),
        fitness=Fitness.scalar(fitness_value),
        metadata=IndividualMetadata(),
        created_at=0,
    )


def create_scm_mate_context(
    genome_a,
    genome_b,
    decoder,
    structural_weight: float = 0.5,
) -> MateContext:
    """Create MateContext for SCM genomes using SCM distance."""
    from evolve.representation.scm import scm_distance
    
    distance = scm_distance(genome_a, genome_b, decoder, structural_weight)
    
    return MateContext(
        partner_distance=distance,
        partner_fitness_rank=0,
        partner_fitness_ratio=1.0,
        partner_niche_id=None,
        population_diversity=0.5,
        crowding_distance=None,
        self_fitness=np.array([1.0]),
        partner_fitness=np.array([1.0]),
    )


# =============================================================================
# T094: Integration Test with ERP-Enabled Evolution
# =============================================================================

class TestSCMERPIntegration:
    """Integration tests for SCM genomes with ERP matchability (T094)."""
    
    def test_scm_distance_integrates_with_mate_context(self, basic_config, rng):
        """Test that SCM distance can be used to create MateContext."""
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        
        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))
        decoder = SCMDecoder(basic_config)
        
        context = create_scm_mate_context(genome_a, genome_b, decoder, structural_weight=0.5)
        
        assert isinstance(context.partner_distance, float)
        assert context.partner_distance >= 0.0
    
    def test_scm_matchability_with_similarity_threshold(
        self, basic_config, rng, counter
    ):
        """Test SCM genomes with SimilarityThresholdMatchability."""
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.reproduction.matchability import SimilarityThresholdMatchability
        
        # Create two different genomes
        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = SCMGenome.random(basic_config, length=20, rng=Random(2))
        decoder = SCMDecoder(basic_config)
        
        # Get their distance
        distance = scm_distance(genome_a, genome_b, decoder, structural_weight=0.5)
        
        context = MateContext(
            partner_distance=distance,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.0]),
        )
        
        evaluator = SimilarityThresholdMatchability()
        
        # With high threshold, should accept
        result = evaluator.evaluate(context, {"max_distance": 2.0}, rng, counter)
        assert result is True
        
        # With very low threshold, should reject (assuming different genomes)
        if distance > 0.01:
            result = evaluator.evaluate(context, {"max_distance": 0.01}, rng, counter)
            assert result is False
    
    def test_scm_matchability_with_distance_threshold(
        self, basic_config, rng, counter
    ):
        """Test SCM genomes with DistanceThresholdMatchability (diversity-seeking)."""
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.reproduction.matchability import DistanceThresholdMatchability
        
        # Create identical genomes
        genome_a = SCMGenome.random(basic_config, length=20, rng=Random(1))
        genome_b = genome_a.copy()  # Same genome
        decoder = SCMDecoder(basic_config)
        
        distance = scm_distance(genome_a, genome_b, decoder, structural_weight=0.5)
        assert distance == 0.0  # Identical genomes have zero distance
        
        context = MateContext(
            partner_distance=distance,
            partner_fitness_rank=0,
            partner_fitness_ratio=1.0,
            partner_niche_id=None,
            population_diversity=0.5,
            crowding_distance=None,
            self_fitness=np.array([1.0]),
            partner_fitness=np.array([1.0]),
        )
        
        evaluator = DistanceThresholdMatchability()
        
        # Require some minimum distance - should reject identical genomes
        result = evaluator.evaluate(context, {"min_distance": 0.1}, rng, counter)
        assert result is False
        
        # With zero threshold, should accept
        result = evaluator.evaluate(context, {"min_distance": 0.0}, rng, counter)
        assert result is True
    
    def test_structural_weight_affects_matchability_outcome(
        self, basic_config, rng, counter
    ):
        """
        Test acceptance scenario: structural_weight > 0 makes structural 
        differences affect matchability decisions.
        """
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.representation.sequence import SequenceGenome
        from evolve.reproduction.matchability import SimilarityThresholdMatchability
        
        # Create two genomes with similar sequences but different structures
        # genome_a: A = B (edge B -> A)
        genes_a = ("B", "STORE_A", "1", "2", "+", "3")
        genome_a = SCMGenome(
            inner=SequenceGenome(genes=genes_a),
            config=basic_config,
            erc_values=(),
        )
        
        # genome_b: C = B (edge B -> C) - similar genes, different target
        genes_b = ("B", "STORE_C", "1", "2", "+", "3")
        genome_b = SCMGenome(
            inner=SequenceGenome(genes=genes_b),
            config=basic_config,
            erc_values=(),
        )
        
        decoder = SCMDecoder(basic_config)
        
        # Compute distances with different structural weights
        dist_seq_only = scm_distance(genome_a, genome_b, decoder, structural_weight=0.0)
        dist_with_struct = scm_distance(genome_a, genome_b, decoder, structural_weight=1.0)
        
        # They should have similar sequences (just STORE target differs)
        # so sequence distance should be small
        # But structural difference (different edges) should be larger
        
        # Using a threshold between the two distances, the matchability 
        # decision should change based on structural_weight
        
        evaluator = SimilarityThresholdMatchability()
        
        # Find a threshold that demonstrates the difference
        # If structural distance is higher than sequence distance:
        if dist_with_struct > dist_seq_only:
            threshold = (dist_seq_only + dist_with_struct) / 2
            
            # With sequence-only (weight=0), should be accepted
            context_seq = MateContext(
                partner_distance=dist_seq_only,
                partner_fitness_rank=0,
                partner_fitness_ratio=1.0,
                partner_niche_id=None,
                population_diversity=0.5,
                crowding_distance=None,
                self_fitness=np.array([1.0]),
                partner_fitness=np.array([1.0]),
            )
            result_seq = evaluator.evaluate(context_seq, {"max_distance": threshold}, rng, counter)
            
            # With structural (weight=1), should be rejected
            context_struct = MateContext(
                partner_distance=dist_with_struct,
                partner_fitness_rank=0,
                partner_fitness_ratio=1.0,
                partner_niche_id=None,
                population_diversity=0.5,
                crowding_distance=None,
                self_fitness=np.array([1.0]),
                partner_fitness=np.array([1.0]),
            )
            result_struct = evaluator.evaluate(context_struct, {"max_distance": threshold}, rng, counter)
            
            # The key assertion: structural weight changes the outcome
            assert result_seq != result_struct or dist_with_struct == dist_seq_only
    
    def test_matchability_evaluation_completes_without_error(
        self, basic_config, rng, counter
    ):
        """Test that matchability evaluation with SCM genomes completes successfully."""
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.reproduction.matchability import (
            AcceptAllMatchability,
            DistanceThresholdMatchability,
            SimilarityThresholdMatchability,
            ProbabilisticMatchability,
        )
        
        # Create a population of genomes
        genomes = [
            SCMGenome.random(basic_config, length=20, rng=Random(i))
            for i in range(10)
        ]
        decoder = SCMDecoder(basic_config)
        
        evaluators = [
            (AcceptAllMatchability(), {}),
            (SimilarityThresholdMatchability(), {"max_distance": 1.0}),
            (DistanceThresholdMatchability(), {"min_distance": 0.0}),
            (ProbabilisticMatchability(), {"base_probability": 0.5}),
        ]
        
        # Test all evaluators with all genome pairs
        for genome_a in genomes[:3]:  # Sample subset
            for genome_b in genomes[:3]:
                distance = scm_distance(genome_a, genome_b, decoder, structural_weight=0.5)
                
                context = MateContext(
                    partner_distance=distance,
                    partner_fitness_rank=0,
                    partner_fitness_ratio=1.0,
                    partner_niche_id=None,
                    population_diversity=0.5,
                    crowding_distance=None,
                    self_fitness=np.array([1.0]),
                    partner_fitness=np.array([1.0]),
                )
                
                for evaluator, params in evaluators:
                    # Should complete without error
                    result = evaluator.evaluate(context, params, rng, counter)
                    assert isinstance(result, (bool, float))


# =============================================================================
# T103: End-to-end SCM Discovery on Synthetic Data
# =============================================================================

class TestEndToEndSCMDiscovery:
    """T103: Integration tests for end-to-end SCM discovery."""
    
    @pytest.fixture
    def synthetic_3var_data(self):
        """Generate synthetic data from a known causal model: A -> B -> C."""
        rng = np.random.default_rng(42)
        n_samples = 100
        
        # True model: A is exogenous, B = 2*A + noise, C = 0.5*B + noise
        A = rng.normal(0, 1, n_samples)
        B = 2 * A + rng.normal(0, 0.1, n_samples)
        C = 0.5 * B + rng.normal(0, 0.1, n_samples)
        
        data = np.column_stack([A, B, C])
        return data, ("A", "B", "C")
    
    def test_scm_genome_evaluator_workflow(self, synthetic_3var_data, rng):
        """Test complete workflow: genome -> decode -> evaluate."""
        from evolve.representation.scm import SCMConfig, SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
        
        data, var_names = synthetic_3var_data
        config = SCMConfig(observed_variables=var_names)
        decoder = SCMDecoder(config)
        
        # Create evaluator
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=data, 
            variable_names=var_names,
            config=fitness_config, 
            decoder=decoder
        )
        
        # Create a population of genomes
        population_size = 20
        genomes = [
            SCMGenome.random(config, length=50, rng=Random(i))
            for i in range(population_size)
        ]
        
        # Wrap in Individuals
        from evolve.core.types import Individual, IndividualMetadata
        individuals = [
            Individual(
                id=uuid4(),
                genome=g,
                protocol=ReproductionProtocol.default(),
                fitness=None,
                metadata=IndividualMetadata(),
                created_at=0,
            )
            for g in genomes
        ]
        
        # Evaluate all individuals
        fitnesses = evaluator.evaluate(individuals)
        
        assert len(fitnesses) == population_size
        # Some genomes may produce cyclic graphs and be rejected (returning None)
        # At minimum, we should have some valid fitnesses
        valid_fitnesses = [f for f in fitnesses if f is not None]
        assert len(valid_fitnesses) > 0, "No genomes produced valid fitness"
        for f in valid_fitnesses:
            assert len(f.values) > 0
    
    def test_population_fitness_variance(self, synthetic_3var_data, rng):
        """Test that population has meaningful fitness variance."""
        from evolve.representation.scm import SCMConfig, SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
        
        data, var_names = synthetic_3var_data
        config = SCMConfig(observed_variables=var_names)
        decoder = SCMDecoder(config)
        
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=data, 
            variable_names=var_names,
            config=fitness_config, 
            decoder=decoder
        )
        
        # Create diverse population
        genomes = [
            SCMGenome.random(config, length=50, rng=Random(i * 100))
            for i in range(50)
        ]
        
        individuals = [
            Individual(
                id=uuid4(),
                genome=g,
                protocol=ReproductionProtocol.default(),
                fitness=None,
                metadata=IndividualMetadata(),
                created_at=0,
            )
            for g in genomes
        ]
        
        fitnesses = evaluator.evaluate(individuals)
        # Filter out rejected genomes (cyclic, etc.)
        valid_fitnesses = [f for f in fitnesses if f is not None]
        
        # Should have at least some valid fitnesses
        assert len(valid_fitnesses) > 5, "Too few valid genomes in population"
        
        fitness_values = [f.values[0] for f in valid_fitnesses]
        
        # Should have variance (not all same fitness)
        assert np.std(fitness_values) > 0, "Population has no fitness variance"


# =============================================================================
# T104: Population Evolution at Scale
# =============================================================================

class TestPopulationEvolution:
    """T104: Integration tests for population-scale evolution."""
    
    def test_large_population_creation(self, basic_config, rng):
        """Test creating and evaluating 1000 individuals."""
        from evolve.representation.scm import SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
        
        # Generate synthetic data
        np_rng = np.random.default_rng(42)
        n_samples = 50
        A = np_rng.normal(0, 1, n_samples)
        B = A + np_rng.normal(0, 0.1, n_samples)
        C = B + np_rng.normal(0, 0.1, n_samples)
        data = np.column_stack([A, B, C])
        var_names = ("A", "B", "C")
        
        decoder = SCMDecoder(basic_config)
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=data, 
            variable_names=var_names,
            config=fitness_config, 
            decoder=decoder
        )
        
        # Create 1000 individuals
        population_size = 1000
        genomes = [
            SCMGenome.random(basic_config, length=50, rng=Random(i))
            for i in range(population_size)
        ]
        
        individuals = [
            Individual(
                id=uuid4(),
                genome=g,
                protocol=ReproductionProtocol.default(),
                fitness=None,
                metadata=IndividualMetadata(),
                created_at=0,
            )
            for g in genomes
        ]
        
        # Evaluate all - should complete without error
        fitnesses = evaluator.evaluate(individuals)
        
        assert len(fitnesses) == population_size
        # All should return something (None for rejected, Fitness for valid)
        # At least some should be valid
        valid_fitnesses = [f for f in fitnesses if f is not None]
        assert len(valid_fitnesses) > population_size // 10, "Too few valid genomes"
    
    def test_simulated_generations(self, basic_config, rng):
        """Test simulating multiple generations of selection/reproduction."""
        from evolve.representation.scm import SCMGenome, scm_distance
        from evolve.representation.scm_decoder import SCMDecoder
        from evolve.evaluation.scm_evaluator import SCMEvaluator, SCMFitnessConfig
        
        # Generate synthetic data
        np_rng = np.random.default_rng(42)
        n_samples = 30
        A = np_rng.normal(0, 1, n_samples)
        B = 2 * A + np_rng.normal(0, 0.1, n_samples)
        C = B + np_rng.normal(0, 0.1, n_samples)
        data = np.column_stack([A, B, C])
        var_names = ("A", "B", "C")
        
        decoder = SCMDecoder(basic_config)
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=data, 
            variable_names=var_names,
            config=fitness_config, 
            decoder=decoder
        )
        
        # Small population for speed
        population_size = 50
        n_generations = 10
        
        # Initial population
        genomes = [
            SCMGenome.random(basic_config, length=50, rng=Random(i))
            for i in range(population_size)
        ]
        
        # Track best fitness over generations
        best_fitnesses = []
        
        for gen in range(n_generations):
            # Create individuals
            individuals = [
                Individual(
                    id=uuid4(),
                    genome=g,
                    protocol=ReproductionProtocol.default(),
                    fitness=None,
                    metadata=IndividualMetadata(),
                    created_at=gen,
                )
                for g in genomes
            ]
            
            # Evaluate
            fitnesses = evaluator.evaluate(individuals)
            
            # Track best (handle None for rejected genomes)
            # Pair genomes with fitnesses, filter out None
            genome_fitness_pairs = [
                (g, f.values[0] if f is not None else float("-inf")) 
                for g, f in zip(genomes, fitnesses)
            ]
            
            valid_fitness_values = [
                fv for _, fv in genome_fitness_pairs if fv > float("-inf")
            ]
            if valid_fitness_values:
                best_fitnesses.append(max(valid_fitness_values))
            
            # Simple tournament selection + random mutation (simulated)
            # Sort by fitness (higher is better for negated MSE)
            sorted_pairs = sorted(
                genome_fitness_pairs, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Keep top half, regenerate rest
            n_keep = population_size // 2
            survivors = [g for g, _ in sorted_pairs[:n_keep]]
            
            # Create new genomes (simplified - just random in this test)
            new_genomes = survivors + [
                SCMGenome.random(basic_config, length=50, rng=Random(gen * 1000 + i))
                for i in range(population_size - n_keep)
            ]
            genomes = new_genomes
        
        # Should complete all generations
        assert len(best_fitnesses) == n_generations


# =============================================================================
# T104a: Memory Profiling Test
# =============================================================================

class TestMemoryProfiling:
    """T104a: Memory profiling tests for large-scale evolution."""
    
    def test_population_memory_bounded(self, basic_config, rng):
        """Test that 1000+ population evolution doesn't cause memory issues."""
        import gc
        import sys
        
        from evolve.representation.scm import SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        
        decoder = SCMDecoder(basic_config)
        
        # Create and discard populations to test memory cleanup
        for iteration in range(5):
            # Create large population
            genomes = [
                SCMGenome.random(basic_config, length=100, rng=Random(iteration * 1000 + i))
                for i in range(1000)
            ]
            
            # Decode all (creates DecodedSCM objects with graphs)
            decoded = [decoder.decode(g) for g in genomes]
            
            # Clear references
            del decoded
            del genomes
            gc.collect()
        
        # If we reach here without MemoryError, test passes
        assert True


# =============================================================================
# T105: Performance Test for Decoding
# =============================================================================

class TestDecodingPerformance:
    """T105: Performance tests for decoding large genomes."""
    
    def test_decode_500_genes_under_threshold(self, basic_config, rng):
        """Test that decoding 500-gene genome completes in reasonable time."""
        import time
        
        from evolve.representation.scm import SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        
        decoder = SCMDecoder(basic_config)
        
        # Create a 500-gene genome
        genome = SCMGenome.random(basic_config, length=500, rng=Random(42))
        
        # Warm up
        _ = decoder.decode(genome)
        
        # Measure decode time (average over multiple runs)
        n_runs = 10
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = decoder.decode(genome)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / n_runs) * 1000
        
        # Should complete in under 10ms on average
        # Note: Being lenient with 50ms to account for CI variability
        assert avg_time_ms < 50, f"Decoding took {avg_time_ms:.2f}ms on average"
    
    def test_batch_decode_performance(self, basic_config, rng):
        """Test decoding batch of genomes has acceptable throughput."""
        import time
        
        from evolve.representation.scm import SCMGenome
        from evolve.representation.scm_decoder import SCMDecoder
        
        decoder = SCMDecoder(basic_config)
        
        # Create 100 genomes of 100 genes each
        genomes = [
            SCMGenome.random(basic_config, length=100, rng=Random(i))
            for i in range(100)
        ]
        
        # Measure total decode time
        start = time.perf_counter()
        for g in genomes:
            _ = decoder.decode(g)
        elapsed = time.perf_counter() - start
        
        # Should complete 100 decodes in under 1 second
        assert elapsed < 1.0, f"Batch decoding took {elapsed:.2f}s"
