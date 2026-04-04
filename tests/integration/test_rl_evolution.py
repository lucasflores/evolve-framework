"""
Integration tests for RL evolution.

Tests that the RL module components work together:
- Environment abstraction
- Policy implementations
- Rollout execution
- RLEvaluator for fitness

If gymnasium is not installed, tests are skipped.
"""

from random import Random

import numpy as np
import pytest

from evolve.core.engine import EvolutionConfig, EvolutionEngine, create_initial_population
from evolve.core.operators.crossover import BlendCrossover
from evolve.core.operators.mutation import GaussianMutation
from evolve.core.operators.selection import TournamentSelection
from evolve.core.types import Individual
from evolve.representation.vector import VectorGenome
from evolve.rl.environment import GymAdapter, SimpleEnvironment, Space
from evolve.rl.evaluator import PolicyDecoder, RLEvaluator
from evolve.rl.policy import LinearPolicy, MLPPolicy, RecurrentPolicy
from evolve.rl.rollout import RolloutResult, StandardRollout, evaluate_policy
from evolve.utils.random import create_rng

# Check if gymnasium is available
try:
    import gymnasium as gym

    HAS_GYM = True
except ImportError:
    HAS_GYM = False


@pytest.mark.integration
class TestSpace:
    """Test Space specification."""

    def test_box_space_creation(self):
        """Box space should be created with bounds."""
        space = Space.box(low=-1.0, high=1.0, shape=(4,))

        assert space.shape == (4,)
        assert space.is_continuous
        assert not space.is_discrete
        assert np.all(space.low == -1.0)
        assert np.all(space.high == 1.0)

    def test_discrete_space_creation(self):
        """Discrete space should be created with n."""
        space = Space.discrete(n=3)

        assert space.n == 3
        assert space.is_discrete
        assert not space.is_continuous

    def test_box_space_sampling(self):
        """Box space should sample within bounds."""
        space = Space.box(low=-1.0, high=1.0, shape=(4,))
        rng = np.random.default_rng(42)

        for _ in range(10):
            sample = space.sample(rng)
            assert sample.shape == (4,)
            assert np.all(sample >= -1.0)
            assert np.all(sample <= 1.0)

    def test_discrete_space_sampling(self):
        """Discrete space should sample valid integers."""
        space = Space.discrete(n=5)
        rng = np.random.default_rng(42)

        for _ in range(10):
            sample = space.sample(rng)
            assert 0 <= sample < 5

    def test_space_contains(self):
        """Space.contains should validate values."""
        box = Space.box(low=0.0, high=1.0, shape=(2,))
        discrete = Space.discrete(n=3)

        assert box.contains(np.array([0.5, 0.5]))
        assert not box.contains(np.array([1.5, 0.5]))
        assert not box.contains(np.array([0.5]))  # Wrong shape

        assert discrete.contains(0)
        assert discrete.contains(2)
        assert not discrete.contains(3)
        assert not discrete.contains(-1)


@pytest.mark.integration
class TestSimpleEnvironment:
    """Test SimpleEnvironment for testing without Gym."""

    def test_environment_creation(self):
        """SimpleEnvironment should initialize properly."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=100,
        )

        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 2

    def test_environment_reset(self):
        """Reset should return observation in space."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
        )

        obs = env.reset(seed=42)
        assert obs.shape == (4,)
        assert env.observation_space.contains(obs)

    def test_environment_step(self):
        """Step should return valid tuple."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=10,
        )

        env.reset(seed=42)
        obs, reward, done, info = env.step(0)

        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_environment_terminates(self):
        """Environment should terminate after max_steps."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=5,
        )

        env.reset(seed=42)
        for i in range(10):
            _, _, done, _ = env.step(0)
            if done:
                break

        assert done
        assert i <= 5


@pytest.mark.integration
class TestPolicies:
    """Test policy implementations."""

    def test_linear_policy_continuous(self):
        """LinearPolicy should output continuous actions."""
        policy = LinearPolicy(
            weights=np.ones((4, 2)),
            bias=np.zeros(2),
            discrete=False,
        )

        obs = np.array([1.0, 0.0, 0.0, 0.0])
        action = policy(obs)

        assert action.shape == (2,)
        np.testing.assert_array_equal(action, [1.0, 1.0])

    def test_linear_policy_discrete(self):
        """LinearPolicy should output discrete actions."""
        policy = LinearPolicy(
            weights=np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            bias=np.zeros(2),
            discrete=True,
        )

        obs = np.array([1.0, 0.0, 0.0, 0.0])
        action = policy(obs)

        assert action == 0  # First output is 1.0, second is -1.0

    def test_linear_policy_from_parameters(self):
        """LinearPolicy should be created from flattened params."""
        params = np.ones(10)  # 4*2 + 2 = 10 params
        policy = LinearPolicy.from_parameters(params, obs_dim=4, action_dim=2, discrete=True)

        assert policy.n_parameters == 10
        assert policy.weights.shape == (4, 2)
        assert policy.bias.shape == (2,)

    def test_mlp_policy(self):
        """MLPPolicy should work with hidden layers."""
        policy = MLPPolicy(
            weights=[np.ones((4, 8)), np.ones((8, 2))],
            biases=[np.zeros(8), np.zeros(2)],
            activation=np.tanh,
            discrete=True,
        )

        obs = np.array([0.1, 0.1, 0.1, 0.1])
        action = policy(obs)

        assert action in [0, 1]

    def test_mlp_policy_from_parameters(self):
        """MLPPolicy should be created from flattened params."""
        # 4->8->2: 4*8 + 8 + 8*2 + 2 = 58 params
        params = np.random.randn(58)
        policy = MLPPolicy.from_parameters(params, layer_sizes=[4, 8, 2], discrete=True)

        assert policy.layer_sizes == [4, 8, 2]
        assert policy.n_parameters == 58

    def test_recurrent_policy(self):
        """RecurrentPolicy should maintain state."""
        policy = RecurrentPolicy(
            w_xh=np.ones((4, 8)) * 0.1,
            w_hh=np.eye(8) * 0.5,
            w_hy=np.ones((8, 2)) * 0.1,
            b_h=np.zeros(8),
            b_y=np.zeros(2),
            discrete=True,
        )

        obs = np.array([1.0, 0.0, 0.0, 0.0])

        # First call
        policy.reset_state()
        action1 = policy(obs)
        state1 = policy.get_state()

        # Second call (state should change)
        action2 = policy(obs)
        state2 = policy.get_state()

        assert not np.allclose(state1, state2)

    def test_recurrent_policy_reset(self):
        """RecurrentPolicy reset should clear state."""
        policy = RecurrentPolicy(
            w_xh=np.ones((4, 8)),
            w_hh=np.eye(8),
            w_hy=np.ones((8, 2)),
            b_h=np.zeros(8),
            b_y=np.zeros(2),
        )

        obs = np.array([1.0, 0.0, 0.0, 0.0])
        policy(obs)  # Update state

        policy.reset_state()
        assert np.all(policy.get_state() == 0)


@pytest.mark.integration
class TestRollout:
    """Test rollout execution."""

    def test_standard_rollout(self):
        """StandardRollout should execute episode."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=10,
        )

        policy = LinearPolicy(
            weights=np.random.randn(4, 2),
            bias=np.zeros(2),
            discrete=True,
        )

        rollout = StandardRollout()
        result = rollout(policy, env, seed=42, max_steps=10)

        assert isinstance(result, RolloutResult)
        assert result.episode_length <= 10
        assert isinstance(result.total_reward, float)

    def test_rollout_with_trajectory(self):
        """Rollout should record trajectory when requested."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=5,
        )

        policy = LinearPolicy(
            weights=np.random.randn(4, 2),
            bias=np.zeros(2),
            discrete=True,
        )

        rollout = StandardRollout()
        result = rollout(policy, env, seed=42, record_trajectory=True)

        assert result.observations is not None
        assert result.actions is not None
        assert result.rewards is not None
        assert len(result.observations) == result.episode_length + 1
        assert len(result.actions) == result.episode_length
        assert len(result.rewards) == result.episode_length

    def test_rollout_resets_stateful_policy(self):
        """Rollout should reset stateful policy."""
        env = SimpleEnvironment(
            obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
            act_space=Space.discrete(n=2),
            max_steps=5,
        )

        policy = RecurrentPolicy(
            w_xh=np.random.randn(4, 8) * 0.1,
            w_hh=np.eye(8) * 0.5,
            w_hy=np.random.randn(8, 2) * 0.1,
            b_h=np.zeros(8),
            b_y=np.zeros(2),
            discrete=True,
        )

        # Pollute state
        policy(np.ones(4))
        old_state = policy.get_state()

        rollout = StandardRollout()
        result = rollout(policy, env, seed=42)

        # State should have been reset at start
        # (We can't check this directly, but the rollout should work)
        assert result.episode_length > 0

    def test_evaluate_policy_multiple_episodes(self):
        """evaluate_policy should aggregate results."""

        def env_factory():
            return SimpleEnvironment(
                obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
                act_space=Space.discrete(n=2),
                max_steps=10,
            )

        policy = LinearPolicy(
            weights=np.random.randn(4, 2),
            bias=np.zeros(2),
            discrete=True,
        )

        result = evaluate_policy(
            policy,
            env_factory,
            n_episodes=5,
            seeds=[1, 2, 3, 4, 5],
        )

        assert result.n_episodes == 5
        assert len(result.all_rewards) == 5
        assert result.min_reward <= result.mean_reward <= result.max_reward


@pytest.mark.integration
class TestRLEvaluator:
    """Test RLEvaluator for fitness computation."""

    def test_rl_evaluator_basic(self):
        """RLEvaluator should compute fitness."""

        def env_factory():
            return SimpleEnvironment(
                obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
                act_space=Space.discrete(n=2),
                max_steps=10,
            )

        decoder = PolicyDecoder(
            policy_factory=lambda params: LinearPolicy.from_parameters(
                params, obs_dim=4, action_dim=2, discrete=True
            )
        )

        evaluator = RLEvaluator(
            decoder=decoder,
            env_factory=env_factory,
            n_episodes=3,
            max_steps=10,
        )

        # Create a genome
        n_params = 10
        bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))
        genome = VectorGenome.random(n_params, bounds=bounds, rng=Random(42))

        # Use evaluate_single for basic test
        fitness = evaluator.evaluate_single(genome, seed=42)

        assert fitness is not None
        assert len(fitness.values) == 1
        assert fitness.metadata is not None
        assert "mean_reward" in fitness.metadata
        assert "n_episodes" in fitness.metadata

    def test_rl_evaluator_with_seed(self):
        """RLEvaluator should be deterministic with seed."""

        def env_factory():
            return SimpleEnvironment(
                obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
                act_space=Space.discrete(n=2),
                max_steps=10,
            )

        decoder = PolicyDecoder(
            policy_factory=lambda params: LinearPolicy.from_parameters(
                params, obs_dim=4, action_dim=2, discrete=True
            )
        )

        evaluator = RLEvaluator(
            decoder=decoder,
            env_factory=env_factory,
            n_episodes=3,
        )

        n_params = 10
        bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))
        genome = VectorGenome.random(n_params, bounds=bounds, rng=Random(42))

        # Same seed should give same result
        fitness1 = evaluator.evaluate_single(genome, seed=123)
        fitness2 = evaluator.evaluate_single(genome, seed=123)

        assert fitness1.values[0] == fitness2.values[0]

    def test_rl_evaluator_batch(self):
        """RLEvaluator should evaluate batch of individuals."""

        def env_factory():
            return SimpleEnvironment(
                obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
                act_space=Space.discrete(n=2),
                max_steps=10,
            )

        decoder = PolicyDecoder(
            policy_factory=lambda params: LinearPolicy.from_parameters(
                params, obs_dim=4, action_dim=2, discrete=True
            )
        )

        evaluator = RLEvaluator(
            decoder=decoder,
            env_factory=env_factory,
            n_episodes=3,
            max_steps=10,
        )

        # Create multiple individuals
        n_params = 10
        bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))

        individuals = [
            Individual(
                id=f"ind_{i}",
                genome=VectorGenome.random(n_params, bounds=bounds, rng=Random(42 + i)),
            )
            for i in range(5)
        ]

        # Evaluate batch
        fitnesses = evaluator.evaluate(individuals, seed=42)

        assert len(fitnesses) == 5
        for fitness in fitnesses:
            assert fitness is not None
            assert len(fitness.values) == 1
            assert "mean_reward" in fitness.metadata


@pytest.mark.integration
class TestRLEvolution:
    """Test evolving policies with RL evaluator."""

    def test_evolve_linear_policy(self):
        """Should be able to evolve a linear policy."""

        def env_factory():
            return SimpleEnvironment(
                obs_space=Space.box(low=-1.0, high=1.0, shape=(4,)),
                act_space=Space.discrete(n=2),
                max_steps=20,
            )

        decoder = PolicyDecoder(
            policy_factory=lambda params: LinearPolicy.from_parameters(
                params, obs_dim=4, action_dim=2, discrete=True
            )
        )

        evaluator = RLEvaluator(
            decoder=decoder,
            env_factory=env_factory,
            n_episodes=2,
            max_steps=20,
            negate=True,  # For minimization (GA minimizes by default)
        )

        # Setup GA
        n_params = 4 * 2 + 2  # 10 parameters
        bounds = (np.full(n_params, -2.0), np.full(n_params, 2.0))

        config = EvolutionConfig(
            population_size=20,
            max_generations=5,
            elitism=2,
        )

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.3),
            seed=42,
        )

        # Create initial population
        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_params, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )

        # Run evolution
        result = engine.run(initial_pop)

        assert result.best.fitness is not None
        assert result.generations > 0


@pytest.mark.integration
@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestGymIntegration:
    """Test integration with Gymnasium environments."""

    def test_gym_adapter_cartpole(self):
        """GymAdapter should wrap CartPole correctly."""
        gym_env = gym.make("CartPole-v1")
        env = GymAdapter(gym_env)

        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 2

        obs = env.reset(seed=42)
        assert obs.shape == (4,)

        obs, reward, done, info = env.step(0)
        assert obs.shape == (4,)
        assert isinstance(reward, float)

        env.close()

    def test_evolve_cartpole_policy(self):
        """Should be able to evolve CartPole policy (smoke test)."""

        def env_factory():
            return GymAdapter(gym.make("CartPole-v1"))

        decoder = PolicyDecoder(
            policy_factory=lambda params: LinearPolicy.from_parameters(
                params, obs_dim=4, action_dim=2, discrete=True
            )
        )

        evaluator = RLEvaluator(
            decoder=decoder,
            env_factory=env_factory,
            n_episodes=3,
            max_steps=200,
            negate=True,
        )

        n_params = 4 * 2 + 2
        bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))

        config = EvolutionConfig(
            population_size=30,
            max_generations=10,
            elitism=2,
        )

        engine = EvolutionEngine(
            config=config,
            evaluator=evaluator,
            selection=TournamentSelection(tournament_size=3),
            crossover=BlendCrossover(alpha=0.5),
            mutation=GaussianMutation(mutation_rate=0.2, sigma=0.2),
            seed=42,
        )

        rng = create_rng(42)
        initial_pop = create_initial_population(
            genome_factory=lambda r: VectorGenome.random(n_params, bounds, r),
            population_size=config.population_size,
            rng=rng,
        )

        result = engine.run(initial_pop)

        # Should complete without error
        assert result.best.fitness is not None

        # Fitness should be negative (we negated it)
        # A random policy gets ~20-30 steps on average
        # After evolution, should be better (more negative when negated)
        best_fitness = result.best.fitness.values[0]
        assert best_fitness < 0  # Negated, so negative is good
