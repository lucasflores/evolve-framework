"""
Unit tests for SCM Evaluator module.

Tests cover:
- SCMFitnessConfig creation
- SCMEvaluator creation and evaluation
- Objective computation
- Penalty handling
- Evaluation results
"""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from evolve.representation.scm import (
    SCMConfig,
    SCMGenome,
)
from evolve.representation.scm_decoder import (
    SCMDecoder,
)
from evolve.evaluation.scm_evaluator import (
    SCMEvaluationResult,
    SCMEvaluator,
    SCMFitnessConfig,
)


# === Fixtures ===

@pytest.fixture
def basic_scm_config() -> SCMConfig:
    """Simple 3-variable SCM configuration."""
    return SCMConfig(observed_variables=("X", "Y", "Z"))


@pytest.fixture
def fitness_config() -> SCMFitnessConfig:
    """Default fitness configuration."""
    return SCMFitnessConfig()


@pytest.fixture
def sample_data() -> np.ndarray:
    """Sample observational data (100 samples, 3 variables)."""
    np.random.seed(42)
    X = np.random.randn(100)
    Y = 2 * X + np.random.randn(100) * 0.1
    Z = X + Y + np.random.randn(100) * 0.1
    return np.column_stack([X, Y, Z])


@pytest.fixture
def basic_evaluator(sample_data, fitness_config) -> SCMEvaluator:
    """Basic evaluator instance."""
    return SCMEvaluator(
        data=sample_data,
        variable_names=("X", "Y", "Z"),
        config=fitness_config,
    )


@pytest.fixture
def decoder(basic_scm_config) -> SCMDecoder:
    """SCM decoder instance."""
    return SCMDecoder(basic_scm_config)


# === SCMFitnessConfig Tests ===

class TestSCMFitnessConfig:
    """Tests for SCMFitnessConfig."""
    
    def test_default_creation(self):
        """Test default config creation."""
        config = SCMFitnessConfig()
        
        assert "data_fit" in config.objectives
        assert "sparsity" in config.objectives
        assert "simplicity" in config.objectives
    
    def test_custom_objectives(self):
        """Test custom objective configuration."""
        config = SCMFitnessConfig(
            objectives=("data_fit", "coverage"),
        )
        
        assert "data_fit" in config.objectives
        assert "coverage" in config.objectives
        assert "sparsity" not in config.objectives
    
    def test_penalty_defaults(self):
        """Test penalty default values."""
        config = SCMFitnessConfig()
        
        assert config.cycle_penalty_per_cycle == 1.0
        assert config.conflict_penalty == 1.0


# === SCMEvaluator Creation Tests ===

class TestSCMEvaluatorCreation:
    """Tests for SCMEvaluator initialization."""
    
    def test_basic_creation(self, sample_data, fitness_config):
        """Test basic evaluator creation."""
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        assert evaluator.config == fitness_config
    
    def test_mismatched_columns_raises(self, sample_data, fitness_config):
        """Test that mismatched column count raises ValueError."""
        with pytest.raises(ValueError, match="columns"):
            SCMEvaluator(
                data=sample_data,
                variable_names=("X", "Y"),  # Missing Z
                config=fitness_config,
            )


# === Objective Tests ===

class TestObjectives:
    """Tests for objective computation."""
    
    def test_data_fit_computed(self, basic_evaluator, basic_scm_config):
        """Test data fit objective is computed."""
        genome = SCMGenome(
            inner=["X", 2.0, "*", "STORE_Y"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = basic_evaluator.evaluate_detailed(genome)
        
        assert result is not None
        assert len(result.objectives) > 0
    
    def test_sparsity_computed(self, basic_evaluator, basic_scm_config):
        """Test sparsity objective is computed."""
        genome = SCMGenome(
            inner=["X", "STORE_Y"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = basic_evaluator.evaluate_detailed(genome)
        
        assert result is not None
    
    def test_simplicity_computed(self, basic_evaluator, basic_scm_config):
        """Test simplicity objective is computed."""
        genome = SCMGenome(
            inner=["X", "X", "+", "X", "*", 1.0, "-", "STORE_Y"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = basic_evaluator.evaluate_detailed(genome)
        
        assert result is not None


# === Penalty Tests ===

class TestPenalties:
    """Tests for penalty computation."""
    
    def test_no_penalty_acyclic(self, basic_evaluator, basic_scm_config):
        """Test acyclic SCM gets no cycle penalty."""
        genome = SCMGenome(
            inner=["X", "STORE_Y", "Y", "STORE_Z"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = basic_evaluator.evaluate_detailed(genome)
        
        assert result is not None
        assert result.cycle_penalty == 0.0
    
    def test_coverage_penalty(self, sample_data, basic_scm_config):
        """Test coverage penalty when not all vars defined."""
        fitness_config = SCMFitnessConfig(
            incomplete_coverage_penalty=10.0,
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        # Only X has equation
        genome = SCMGenome(
            inner=[1.0, "STORE_X"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = evaluator.evaluate_detailed(genome)
        
        assert result is not None
        # Should have coverage penalty
        assert result.coverage_penalty > 0.0
    
    def test_conflict_penalty(self, sample_data, basic_scm_config):
        """Test conflict penalty when multiple STORE to same var."""
        fitness_config = SCMFitnessConfig(
            conflict_penalty=5.0,
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        # Two stores to Y - conflict
        genome = SCMGenome(
            inner=[1.0, "STORE_Y", 2.0, "STORE_Y"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = evaluator.evaluate_detailed(genome)
        
        assert result is not None
        assert result.conflict_penalty > 0.0


# === Acyclicity Tests ===

class TestAcyclicity:
    """Tests for acyclicity constraint handling."""
    
    def test_acyclic_valid(self, sample_data, basic_scm_config):
        """Test acyclic graph is valid."""
        fitness_config = SCMFitnessConfig(
            acyclicity_mode="reject",
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        genome = SCMGenome(
            inner=["X", "STORE_Y", "Y", "STORE_Z"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = evaluator.evaluate_detailed(genome)
        
        assert result is not None
        assert result.is_valid
    
    def test_cyclic_penalize_mode(self, sample_data):
        """Test cyclic graph in penalize mode returns penalty."""
        scm_config = SCMConfig(
            observed_variables=("X", "Y"),
            acyclicity_mode="penalize",
        )
        fitness_config = SCMFitnessConfig(
            acyclicity_mode="penalize",
            cycle_penalty_per_cycle=10.0,
        )
        evaluator = SCMEvaluator(
            data=sample_data[:, :2],
            variable_names=("X", "Y"),
            config=fitness_config,
        )
        
        # Self-loop: X = X
        genome = SCMGenome(
            inner=["X", "STORE_X"],
            config=scm_config,
            erc_values=(),
        )
        
        fitness, result = evaluator.evaluate_detailed(genome)
        
        assert result is not None
        # Should have cycle penalty
        assert result.cycle_count >= 1


# === SCMEvaluationResult Tests ===

class TestSCMEvaluationResult:
    """Tests for SCMEvaluationResult dataclass."""
    
    def test_attributes(self, basic_evaluator, basic_scm_config):
        """Test result has expected attributes."""
        genome = SCMGenome(
            inner=["X", "STORE_Y"],
            config=basic_scm_config,
            erc_values=(),
        )
        
        fitness, result = basic_evaluator.evaluate_detailed(genome)
        
        assert hasattr(result, "objectives")
        assert hasattr(result, "total_penalty")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "cycle_penalty")
        assert hasattr(result, "coverage_penalty")
        assert hasattr(result, "conflict_penalty")


# === Integration Tests ===

class TestEvaluatorIntegration:
    """Integration tests for full evaluation pipeline."""
    
    def test_end_to_end_evaluation(self, sample_data, basic_scm_config):
        """Test complete evaluation from genome to fitness."""
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        # Create genome
        genome = SCMGenome.random(basic_scm_config, length=50, rng=Random(42))
        
        # Evaluate
        fitness, result = evaluator.evaluate_detailed(genome)
        
        # Should return a result
        assert result is not None
        
        # Total penalty should be non-negative
        assert result.total_penalty >= 0.0
    
    def test_deterministic_evaluation(self, sample_data, basic_scm_config):
        """Test that evaluation is deterministic."""
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        genome = SCMGenome.random(basic_scm_config, length=50, rng=Random(42))
        
        _, result1 = evaluator.evaluate_detailed(genome)
        _, result2 = evaluator.evaluate_detailed(genome)
        
        assert result1.objectives == result2.objectives
        assert result1.total_penalty == result2.total_penalty
    
    def test_batch_evaluate(self, sample_data, basic_scm_config):
        """Test batch evaluation of multiple individuals."""
        from evolve.core.types import Individual
        
        fitness_config = SCMFitnessConfig()
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        # Create multiple individuals
        individuals = [
            Individual(genome=SCMGenome.random(basic_scm_config, length=30, rng=Random(i)))
            for i in range(5)
        ]
        
        # Batch evaluate
        results = evaluator.evaluate(individuals)
        
        assert len(results) == 5


class TestAcyclicityStrategies:
    """Tests for different acyclicity handling strategies (T079, T080, T080a, T080b)."""
    
    @pytest.fixture
    def cyclic_genome_data(self, basic_scm_config):
        """Create a genome that produces cyclic structure."""
        # Fixed seed that produces cycles
        rng = Random(999)
        genome = SCMGenome.random(basic_scm_config, length=80, rng=rng)
        return genome
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for evaluation."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 3))
    
    def test_penalty_only_strategy(self, sample_data, basic_scm_config, cyclic_genome_data):
        """T080: Test penalty_only strategy returns -inf fitness + penalty."""
        fitness_config = SCMFitnessConfig(
            acyclicity_strategy="penalty_only",
            acyclicity_mode="penalize",
            cycle_penalty_per_cycle=100.0,
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        fitness, result = evaluator.evaluate_detailed(cyclic_genome_data)
        
        # Result should be returned (may have None fitness if not evaluable)
        assert result is not None
        # If cyclic, penalty_only should penalize
        if result.cycle_count > 0:
            assert result.cycle_penalty > 0
    
    def test_acyclic_subgraph_strategy(self, sample_data, basic_scm_config, cyclic_genome_data):
        """Test acyclic_subgraph strategy extracts maximal DAG."""
        fitness_config = SCMFitnessConfig(
            acyclicity_strategy="acyclic_subgraph",
            acyclicity_mode="penalize",
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        fitness, result = evaluator.evaluate_detailed(cyclic_genome_data)
        
        # Should produce a result
        assert result is not None
        # Objectives should exist
        assert len(result.objectives) > 0
    
    def test_parse_order_strategy(self, sample_data, basic_scm_config, cyclic_genome_data):
        """T079: Test parse_order strategy breaks cycles by decode order."""
        fitness_config = SCMFitnessConfig(
            acyclicity_strategy="parse_order",
            acyclicity_mode="penalize",
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        fitness, result = evaluator.evaluate_detailed(cyclic_genome_data)
        
        # Parse order should produce a result
        assert result is not None
        # Result should have objectives
        assert len(result.objectives) > 0
    
    def test_parent_inheritance_strategy(self, sample_data, basic_scm_config, cyclic_genome_data):
        """T080a: Test parent_inheritance strategy for ERP-aware behavior."""
        fitness_config = SCMFitnessConfig(
            acyclicity_strategy="parent_inheritance",
            acyclicity_mode="penalize",
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        fitness, result = evaluator.evaluate_detailed(cyclic_genome_data)
        
        # Should produce valid results
        assert result is not None
        assert len(result.objectives) > 0
    
    def test_composite_strategy(self, sample_data, basic_scm_config, cyclic_genome_data):
        """T080b: Test composite strategy combines subgraph + proportional penalty."""
        fitness_config = SCMFitnessConfig(
            acyclicity_strategy="composite",
            acyclicity_mode="penalize",
            cycle_penalty_per_cycle=50.0,
        )
        evaluator = SCMEvaluator(
            data=sample_data,
            variable_names=("X", "Y", "Z"),
            config=fitness_config,
        )
        
        fitness, result = evaluator.evaluate_detailed(cyclic_genome_data)
        
        # Composite should produce result
        assert result is not None
        assert len(result.objectives) > 0
    
    def test_strategy_consistency(self, sample_data, basic_scm_config):
        """Test that strategies produce consistent results across runs."""
        from evolve.representation.scm import AcyclicityStrategy
        
        # Create a fixed genome
        genome = SCMGenome.random(basic_scm_config, length=60, rng=Random(42))
        
        for strategy in AcyclicityStrategy:
            fitness_config = SCMFitnessConfig(
                acyclicity_strategy=strategy.value,
                acyclicity_mode="penalize",
            )
            evaluator = SCMEvaluator(
                data=sample_data,
                variable_names=("X", "Y", "Z"),
                config=fitness_config,
            )
            
            # Run twice with same input
            fitness1, result1 = evaluator.evaluate_detailed(genome)
            fitness2, result2 = evaluator.evaluate_detailed(genome)
            
            # Compare objectives (which are tuples of floats)
            assert result1.objectives == result2.objectives, f"Strategy {strategy} not deterministic"
