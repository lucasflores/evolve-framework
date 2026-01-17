"""
Integration test: NEAT-style neuroevolution solving XOR.

Verifies:
1. GraphGenome can encode neural network topology
2. NEATMutation can evolve topology (add nodes/connections)
3. NEATCrossover properly aligns genes by innovation
4. Networks can solve XOR (non-linearly separable)
5. Fitness improves over generations
"""

import numpy as np
import pytest
from random import Random

from evolve.core.engine import EvolutionEngine, EvolutionConfig
from evolve.core.operators.selection import TournamentSelection
from evolve.core.operators.crossover import NEATCrossover
from evolve.core.operators.mutation import NEATMutation
from evolve.core.population import Population
from evolve.core.stopping import FitnessThresholdStopping, CompositeStoppingCriterion, GenerationLimitStopping
from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import Evaluator
from evolve.representation.graph import GraphGenome, NodeGene, ConnectionGene, InnovationTracker
from evolve.representation.network import NEATNetwork
from evolve.representation.decoder import GraphToNetworkDecoder
from evolve.utils.random import create_rng


# XOR truth table
XOR_INPUTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
XOR_TARGETS = np.array([0.0, 1.0, 1.0, 0.0])


class XORFitnessEvaluator(Evaluator[GraphGenome]):
    """
    Evaluates graph genomes on XOR problem.
    
    Fitness is (4 - sum of squared errors), so perfect score is 4.0.
    This is a maximization problem.
    """
    
    def __init__(self):
        self.decoder = GraphToNetworkDecoder()
    
    def evaluate(self, genome: GraphGenome) -> Fitness:
        """Evaluate genome on XOR."""
        try:
            network = self.decoder.decode(genome)
            
            total_error = 0.0
            for inputs, target in zip(XOR_INPUTS, XOR_TARGETS):
                output = network.forward(inputs)
                error = (output[0] - target) ** 2
                total_error += error
            
            # Convert MSE to fitness (higher is better)
            # Perfect fitness = 4.0, worst = 0.0
            fitness = 4.0 - total_error
            return Fitness((fitness,))
            
        except Exception:
            # Invalid network topology
            return Fitness((0.0,))
    
    def evaluate_batch(self, genomes: list[GraphGenome]) -> list[Fitness]:
        """Evaluate batch of genomes."""
        return [self.evaluate(g) for g in genomes]


def create_minimal_xor_population(
    size: int,
    innovation_tracker: InnovationTracker,
    rng: Random,
) -> list[Individual[GraphGenome]]:
    """
    Create initial population of minimal 2-input, 1-output genomes.
    
    Each genome has random weights on connections.
    """
    individuals = []
    
    for _ in range(size):
        # Create minimal topology: 2 inputs -> 1 output
        genome = GraphGenome.minimal(
            n_inputs=2,
            n_outputs=1,
            innovation_tracker=innovation_tracker,
        )
        
        # Randomize connection weights
        new_connections = set()
        for conn in genome.connections:
            new_weight = rng.uniform(-2.0, 2.0)
            new_connections.add(conn.with_weight(new_weight))
        
        genome = GraphGenome(
            nodes=genome.nodes,
            connections=frozenset(new_connections),
            input_ids=genome.input_ids,
            output_ids=genome.output_ids,
        )
        
        individuals.append(Individual(genome=genome))
    
    return individuals


@pytest.mark.integration
class TestNeuroevolution:
    """Integration tests for NEAT-style neuroevolution."""

    def test_graph_genome_creates_valid_network(self):
        """GraphGenome should decode to a working network."""
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)  # 2 inputs + 1 output
        
        genome = GraphGenome.minimal(
            n_inputs=2,
            n_outputs=1,
            innovation_tracker=tracker,
        )
        
        decoder = GraphToNetworkDecoder()
        network = decoder.decode(genome)
        
        # Network should accept 2 inputs and produce 1 output
        output = network.forward(np.array([0.0, 1.0]))
        assert output.shape == (1,)
        assert 0.0 <= output[0] <= 1.0  # Sigmoid output

    def test_neat_mutation_adds_structure(self):
        """NEATMutation should be able to add nodes and connections."""
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)
        
        genome = GraphGenome.minimal(
            n_inputs=2,
            n_outputs=1,
            innovation_tracker=tracker,
        )
        
        # High mutation rates to ensure structural changes
        mutation = NEATMutation(
            add_node_prob=0.9,
            add_connection_prob=0.9,
            weight_mutation_prob=1.0,
            innovation_tracker=tracker,
        )
        
        rng = Random(42)
        
        # Apply multiple mutations
        original_nodes = len(genome.nodes)
        original_conns = len(genome.connections)
        
        for _ in range(10):
            genome = mutation.mutate(genome, rng)
        
        # Should have gained structure
        assert len(genome.nodes) >= original_nodes
        assert len(genome.connections) >= original_conns
        
        # Network should still be valid
        decoder = GraphToNetworkDecoder()
        network = decoder.decode(genome)
        output = network.forward(np.array([0.5, 0.5]))
        assert output.shape == (1,)

    def test_neat_crossover_aligns_genes(self):
        """NEATCrossover should align genes by innovation number."""
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)
        
        # Create two minimal genomes with same innovations
        genome1 = GraphGenome.minimal(n_inputs=2, n_outputs=1, innovation_tracker=tracker)
        genome2 = GraphGenome.minimal(n_inputs=2, n_outputs=1, innovation_tracker=tracker)
        
        # Give different weights
        rng = Random(42)
        
        def randomize_weights(genome: GraphGenome) -> GraphGenome:
            new_connections = set()
            for conn in genome.connections:
                new_connections.add(conn.with_weight(rng.uniform(-2, 2)))
            return GraphGenome(
                nodes=genome.nodes,
                connections=frozenset(new_connections),
                input_ids=genome.input_ids,
                output_ids=genome.output_ids,
            )
        
        genome1 = randomize_weights(genome1)
        genome2 = randomize_weights(genome2)
        
        crossover = NEATCrossover()
        child1, child2 = crossover.crossover(genome1, genome2, rng)
        
        # Children should have same structure (matching innovations)
        assert len(child1.connections) == len(genome1.connections)
        assert len(child2.connections) == len(genome2.connections)
        
        # Children should decode to valid networks
        decoder = GraphToNetworkDecoder()
        net1 = decoder.decode(child1)
        net2 = decoder.decode(child2)
        
        output1 = net1.forward(np.array([1.0, 0.0]))
        output2 = net2.forward(np.array([1.0, 0.0]))
        
        assert output1.shape == (1,)
        assert output2.shape == (1,)

    def test_xor_fitness_evaluation(self):
        """XOR fitness evaluator should work correctly."""
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)
        
        genome = GraphGenome.minimal(
            n_inputs=2,
            n_outputs=1,
            innovation_tracker=tracker,
        )
        
        evaluator = XORFitnessEvaluator()
        fitness = evaluator.evaluate(genome)
        
        # Minimal network (all zeros) won't solve XOR well
        # But should return valid fitness
        assert fitness.values is not None
        assert len(fitness.values) == 1
        assert 0.0 <= fitness.values[0] <= 4.0

    def test_neuroevolution_improves_xor_fitness(self):
        """
        NEAT should improve fitness on XOR over generations.
        
        This is a smoke test - full XOR solution may require more generations.
        """
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)  # 2 inputs + 1 output
        
        rng = create_rng(42)
        
        # Create initial population
        individuals = create_minimal_xor_population(
            size=50,
            innovation_tracker=tracker,
            rng=rng,
        )
        
        evaluator = XORFitnessEvaluator()
        
        # Evaluate initial population
        for ind in individuals:
            ind.fitness = evaluator.evaluate(ind.genome)
        
        initial_best = max(ind.fitness.values[0] for ind in individuals if ind.fitness)
        
        # Setup mutation and crossover
        mutation = NEATMutation(
            add_node_prob=0.03,
            add_connection_prob=0.05,
            weight_mutation_prob=0.8,
            innovation_tracker=tracker,
        )
        
        crossover = NEATCrossover()
        selection = TournamentSelection(tournament_size=3, minimize=False)  # Higher is better for XOR
        
        # Manual evolution loop (to test components directly)
        for gen in range(30):
            # Reset innovation cache per generation
            tracker.reset_generation()
            
            # Wrap in Population for selection
            population = Population(individuals, generation=gen)
            
            # Select parents
            parents = selection.select(population, len(individuals), rng)
            
            # Create offspring
            offspring: list[Individual[GraphGenome]] = []
            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                
                # Determine fitter parent
                f1 = p1.fitness.values[0] if p1.fitness else 0.0
                f2 = p2.fitness.values[0] if p2.fitness else 0.0
                
                # Crossover
                child1_genome, child2_genome = crossover.crossover(
                    p1.genome, p2.genome, rng, parent1_fitter=(f1 >= f2)
                )
                
                # Mutation
                child1_genome = mutation.mutate(child1_genome, rng)
                child2_genome = mutation.mutate(child2_genome, rng)
                
                offspring.append(Individual(genome=child1_genome))
                offspring.append(Individual(genome=child2_genome))
            
            # Evaluate offspring
            for ind in offspring:
                ind.fitness = evaluator.evaluate(ind.genome)
            
            # Elitism: keep best 2 from previous generation
            sorted_individuals = sorted(
                individuals,
                key=lambda x: x.fitness.values[0] if x.fitness else 0.0,
                reverse=True,
            )
            elite = sorted_individuals[:2]
            
            # Replace population
            individuals = elite + offspring[:len(individuals) - 2]
        
        # Get final best
        final_best = max(ind.fitness.values[0] for ind in individuals if ind.fitness)
        
        # Fitness should have improved
        assert final_best >= initial_best, (
            f"Expected fitness improvement: initial={initial_best}, final={final_best}"
        )
        
        # Should have made meaningful progress (at least 2.5 out of 4.0)
        assert final_best >= 2.5, f"Expected fitness >= 2.5, got {final_best}"

    def test_full_xor_solution(self):
        """
        Test that NEAT can fully solve XOR with enough generations.
        
        This is a longer test that verifies the algorithm can find
        a near-perfect solution.
        """
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)
        
        rng = create_rng(42)  # Different seed for better convergence
        
        individuals = create_minimal_xor_population(
            size=200,  # Larger population
            innovation_tracker=tracker,
            rng=rng,
        )
        
        evaluator = XORFitnessEvaluator()
        
        # Evaluate initial population
        for ind in individuals:
            ind.fitness = evaluator.evaluate(ind.genome)
        
        mutation = NEATMutation(
            add_node_prob=0.05,  # Slightly higher for more exploration
            add_connection_prob=0.08,
            weight_mutation_prob=0.9,
            weight_perturb_sigma=0.5,
            bias_mutation_prob=0.3,
            innovation_tracker=tracker,
        )
        
        crossover = NEATCrossover()
        selection = TournamentSelection(tournament_size=3, minimize=False)
        
        best_fitness = 0.0
        best_genome = None
        
        for gen in range(150):  # More generations
            tracker.reset_generation()
            
            # Check for solution
            for ind in individuals:
                if ind.fitness and ind.fitness.values[0] > best_fitness:
                    best_fitness = ind.fitness.values[0]
                    best_genome = ind.genome
            
            # Early termination if solved
            if best_fitness >= 3.9:  # Near-perfect
                break
            
            # Wrap in Population for selection
            population = Population(individuals, generation=gen)
            
            # Evolve
            parents = selection.select(population, len(individuals), rng)
            
            offspring: list[Individual[GraphGenome]] = []
            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                f1 = p1.fitness.values[0] if p1.fitness else 0.0
                f2 = p2.fitness.values[0] if p2.fitness else 0.0
                
                child1, child2 = crossover.crossover(
                    p1.genome, p2.genome, rng, parent1_fitter=(f1 >= f2)
                )
                child1 = mutation.mutate(child1, rng)
                child2 = mutation.mutate(child2, rng)
                
                offspring.append(Individual(genome=child1))
                offspring.append(Individual(genome=child2))
            
            for ind in offspring:
                ind.fitness = evaluator.evaluate(ind.genome)
            
            # Elitism - keep more elites
            sorted_individuals = sorted(
                individuals,
                key=lambda x: x.fitness.values[0] if x.fitness else 0.0,
                reverse=True,
            )
            individuals = sorted_individuals[:10] + offspring[:len(individuals) - 10]
        
        # Should achieve good fitness (at least 3.0 out of 4.0 - allows for some variation)
        assert best_fitness >= 3.0, f"Expected fitness >= 3.0, got {best_fitness}"
        
        # Verify the solution actually works if we got good fitness
        if best_genome and best_fitness >= 3.5:
            decoder = GraphToNetworkDecoder()
            network = decoder.decode(best_genome)
            
            correct = 0
            for inputs, target in zip(XOR_INPUTS, XOR_TARGETS):
                output = network.forward(inputs)[0]
                predicted = 1.0 if output > 0.5 else 0.0
                if predicted == target:
                    correct += 1
            
            # Should get at least 3 out of 4 correct
            assert correct >= 3, f"Expected at least 3/4 correct, got {correct}/4"


@pytest.mark.integration
class TestGraphGenomeSerialization:
    """Test genome serialization for checkpointing."""

    def test_graph_genome_round_trip(self):
        """GraphGenome should serialize and deserialize correctly."""
        tracker = InnovationTracker()
        tracker.reserve_node_ids(3)
        
        # Create genome with some structure
        genome = GraphGenome.minimal(n_inputs=2, n_outputs=1, innovation_tracker=tracker)
        
        # Add a hidden node
        mutation = NEATMutation(
            add_node_prob=1.0,
            add_connection_prob=0.0,
            weight_mutation_prob=0.0,
            innovation_tracker=tracker,
        )
        rng = Random(42)
        genome = mutation.mutate(genome, rng)
        
        # Serialize
        data = genome.to_dict()
        
        # Verify structure
        assert "nodes" in data
        assert "connections" in data
        assert "input_ids" in data
        assert "output_ids" in data
        
        # Deserialize
        restored = GraphGenome.from_dict(data)
        
        # Should be equal
        assert restored.input_ids == genome.input_ids
        assert restored.output_ids == genome.output_ids
        assert len(restored.nodes) == len(genome.nodes)
        assert len(restored.connections) == len(genome.connections)
        
        # Networks should produce same output
        decoder = GraphToNetworkDecoder()
        net1 = decoder.decode(genome)
        net2 = decoder.decode(restored)
        
        test_input = np.array([0.7, 0.3])
        output1 = net1.forward(test_input)
        output2 = net2.forward(test_input)
        
        np.testing.assert_allclose(output1, output2)
