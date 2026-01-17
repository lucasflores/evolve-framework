# Reinforcement Learning Integration Interfaces Contract

**Module**: `evolve.rl`  
**Purpose**: Define environment, policy, and rollout abstractions for neuroevolution in RL

---

## Environment Protocol

```python
from typing import Protocol, TypeVar, Any, Callable
from dataclasses import dataclass
import numpy as np

Observation = TypeVar('Observation')
Action = TypeVar('Action')


class Environment(Protocol[Observation, Action]):
    """
    Environment interface for policy evaluation.
    
    Framework-neutral: works with NumPy, Gym, or custom envs.
    """
    
    @property
    def observation_space(self) -> 'Space':
        """Specification of observation shape/type."""
        ...
    
    @property
    def action_space(self) -> 'Space':
        """Specification of action shape/type."""
        ...
    
    def reset(self, seed: int | None = None) -> Observation:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for deterministic reset
            
        Returns:
            Initial observation
        """
        ...
    
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, info)
        """
        ...


class VectorizedEnvironment(Protocol[Observation, Action]):
    """
    Batched environment for parallel evaluation.
    
    Runs multiple episodes simultaneously.
    """
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        ...
    
    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        """
        Reset all environments.
        
        Args:
            seeds: Per-environment seeds
            
        Returns:
            Batch of observations, shape: (num_envs, *obs_shape)
        """
        ...
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Step all environments.
        
        Args:
            actions: Batch of actions, shape: (num_envs, *action_shape)
            
        Returns:
            (observations, rewards, dones, infos)
        """
        ...
```

---

## Space Specification

```python
@dataclass
class Space:
    """
    Specification of observation/action space.
    
    Framework-neutral representation of Gym-style spaces.
    """
    shape: tuple[int, ...]
    dtype: np.dtype
    low: np.ndarray | None = None  # Bounds for Box spaces
    high: np.ndarray | None = None
    n: int | None = None  # Cardinality for Discrete spaces
    
    @classmethod
    def box(
        cls,
        low: np.ndarray | float,
        high: np.ndarray | float,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32
    ) -> 'Space':
        """Create Box (continuous) space."""
        if isinstance(low, (int, float)):
            low = np.full(shape, low, dtype=dtype)
        if isinstance(high, (int, float)):
            high = np.full(shape, high, dtype=dtype)
        return cls(shape=shape, dtype=dtype, low=low, high=high)
    
    @classmethod
    def discrete(cls, n: int) -> 'Space':
        """Create Discrete space."""
        return cls(shape=(), dtype=np.int64, n=n)
    
    def sample(self, rng: 'Random') -> np.ndarray:
        """Sample random element from space."""
        if self.n is not None:
            return rng.integers(0, self.n)
        return rng.uniform(self.low, self.high)


@dataclass
class GymAdapter:
    """
    Wraps a Gym environment to conform to our protocol.
    """
    env: Any  # gym.Env
    
    @property
    def observation_space(self) -> Space:
        s = self.env.observation_space
        return Space(shape=s.shape, dtype=s.dtype, low=s.low, high=s.high)
    
    @property
    def action_space(self) -> Space:
        s = self.env.action_space
        if hasattr(s, 'n'):
            return Space.discrete(s.n)
        return Space(shape=s.shape, dtype=s.dtype, low=s.low, high=s.high)
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _ = self.env.reset(seed=seed)
        return obs
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info
```

---

## Policy Protocol

```python
from evolve.representation import Phenotype


class Policy(Phenotype, Protocol[Observation, Action]):
    """
    Policy maps observations to actions.
    
    A Policy IS A Phenotype (evaluable decoded genome).
    
    Policies MAY be:
    - Deterministic (always same action for same obs)
    - Stochastic (sample from action distribution)
    - Stateful (e.g., RNN with hidden state)
    """
    
    def __call__(self, observation: Observation) -> Action:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Action to take
        """
        ...


class StochasticPolicy(Policy[Observation, Action], Protocol):
    """Policy that can sample or return distribution."""
    
    def distribution(self, observation: Observation) -> 'Distribution':
        """Get action distribution."""
        ...
    
    def sample(self, observation: Observation, rng: 'Random') -> Action:
        """Sample action from distribution."""
        ...


class StatefulPolicy(Policy[Observation, Action], Protocol):
    """Policy with internal state (RNN, LSTM)."""
    
    def reset_state(self) -> None:
        """Reset hidden state."""
        ...
    
    def get_state(self) -> Any:
        """Get current state for checkpointing."""
        ...
    
    def set_state(self, state: Any) -> None:
        """Restore state."""
        ...
```

---

## Reference Policy Implementations

```python
class LinearPolicy:
    """
    Simple linear policy: action = W @ obs + b
    
    CPU reference implementation.
    """
    
    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        discrete: bool = False
    ):
        self.weights = weights
        self.bias = bias
        self.discrete = discrete
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        output = observation @ self.weights + self.bias
        if self.discrete:
            return np.argmax(output)
        return output


class MLPPolicy:
    """
    Multi-layer perceptron policy.
    
    CPU reference implementation using NumPy.
    """
    
    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activation: Callable = np.tanh,
        output_activation: Callable | None = None
    ):
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.output_activation = output_activation
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        x = observation
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(x @ w + b)
        
        # Output layer
        x = x @ self.weights[-1] + self.biases[-1]
        if self.output_activation:
            x = self.output_activation(x)
        return x


class RecurrentPolicy:
    """
    Simple recurrent policy with hidden state.
    
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    y_t = W_hy @ h_t + b_y
    """
    
    def __init__(
        self,
        w_xh: np.ndarray,
        w_hh: np.ndarray,
        w_hy: np.ndarray,
        b_h: np.ndarray,
        b_y: np.ndarray
    ):
        self.w_xh = w_xh
        self.w_hh = w_hh
        self.w_hy = w_hy
        self.b_h = b_h
        self.b_y = b_y
        self.hidden_size = w_hh.shape[0]
        self.reset_state()
    
    def reset_state(self) -> None:
        self.hidden = np.zeros(self.hidden_size)
    
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.hidden = np.tanh(
            self.w_xh @ observation +
            self.w_hh @ self.hidden +
            self.b_h
        )
        return self.w_hy @ self.hidden + self.b_y
    
    def get_state(self) -> np.ndarray:
        return self.hidden.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        self.hidden = state.copy()
```

---

## Rollout Protocol

```python
@dataclass
class RolloutResult:
    """
    Result of evaluating a policy in an environment.
    """
    total_reward: float
    episode_length: int
    observations: list[np.ndarray] | None = None  # Optional trajectory
    actions: list[np.ndarray] | None = None
    rewards: list[float] | None = None
    info: dict | None = None  # Custom metrics


class Rollout(Protocol[G]):
    """
    Evaluates a policy by running it in an environment.
    """
    
    def __call__(
        self,
        policy: Policy,
        env: Environment,
        seed: int | None = None,
        *,
        max_steps: int | None = None,
        render: bool = False,
        record_trajectory: bool = False
    ) -> RolloutResult:
        """
        Run policy in environment for one episode.
        
        Args:
            policy: Policy to evaluate
            env: Environment to run in
            seed: Seed for environment reset
            max_steps: Maximum episode length
            render: Whether to render
            record_trajectory: Whether to store obs/actions
            
        Returns:
            Evaluation result
        """
        ...


class StandardRollout:
    """
    Standard episode rollout.
    
    Deterministic given seed (for reproducibility).
    """
    
    def __call__(
        self,
        policy: Policy,
        env: Environment,
        seed: int | None = None,
        *,
        max_steps: int | None = None,
        render: bool = False,
        record_trajectory: bool = False
    ) -> RolloutResult:
        observation = env.reset(seed=seed)
        
        total_reward = 0.0
        steps = 0
        
        observations = [observation] if record_trajectory else None
        actions = [] if record_trajectory else None
        rewards = [] if record_trajectory else None
        
        # Reset stateful policy
        if hasattr(policy, 'reset_state'):
            policy.reset_state()
        
        while True:
            action = policy(observation)
            
            if record_trajectory:
                actions.append(action)
            
            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if record_trajectory:
                observations.append(observation)
                rewards.append(reward)
            
            if done or (max_steps and steps >= max_steps):
                break
        
        return RolloutResult(
            total_reward=total_reward,
            episode_length=steps,
            observations=observations,
            actions=actions,
            rewards=rewards,
            info=info
        )
```

---

## Multi-Episode Evaluation

```python
@dataclass
class AggregatedResult:
    """
    Aggregated results from multiple episodes.
    """
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    n_episodes: int
    all_rewards: list[float]


def evaluate_policy(
    policy: Policy,
    env_factory: Callable[[], Environment],
    n_episodes: int,
    seeds: list[int],
    *,
    max_steps: int | None = None
) -> AggregatedResult:
    """
    Evaluate policy over multiple episodes.
    
    Creates fresh environment for each episode to avoid
    state leakage.
    
    Args:
        policy: Policy to evaluate
        env_factory: Creates new environment instance
        n_episodes: Number of episodes to run
        seeds: Seeds for each episode (len = n_episodes)
        max_steps: Max steps per episode
        
    Returns:
        Aggregated statistics
    """
    rollout = StandardRollout()
    rewards = []
    lengths = []
    
    for i in range(n_episodes):
        env = env_factory()
        result = rollout(policy, env, seed=seeds[i], max_steps=max_steps)
        rewards.append(result.total_reward)
        lengths.append(result.episode_length)
    
    return AggregatedResult(
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        mean_length=float(np.mean(lengths)),
        n_episodes=n_episodes,
        all_rewards=rewards
    )
```

---

## RL-Specific Evaluator

```python
from evolve.evaluation import Evaluator, EvaluatorCapabilities
from evolve.core import Individual, Fitness


@dataclass
class RLEvaluator(Generic[G]):
    """
    Evaluates individuals as RL policies.
    
    Combines decoder + environment + rollout.
    """
    decoder: 'Decoder[G, Policy]'
    env_factory: Callable[[], Environment]
    n_episodes: int = 1
    max_steps: int | None = None
    aggregate: str = "mean"  # "mean", "min", "median"
    
    @property
    def capabilities(self) -> EvaluatorCapabilities:
        return EvaluatorCapabilities(
            batched=False,
            stochastic=True,  # Episodes may vary
            stateful=False
        )
    
    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None
    ) -> Sequence[Fitness]:
        rng = np.random.default_rng(seed)
        results = []
        
        for ind in individuals:
            policy = self.decoder.decode(ind.genome)
            
            # Generate episode seeds
            episode_seeds = [rng.integers(0, 2**31) for _ in range(self.n_episodes)]
            
            agg_result = evaluate_policy(
                policy,
                self.env_factory,
                self.n_episodes,
                episode_seeds,
                max_steps=self.max_steps
            )
            
            if self.aggregate == "mean":
                value = agg_result.mean_reward
            elif self.aggregate == "min":
                value = agg_result.min_reward
            elif self.aggregate == "median":
                value = float(np.median(agg_result.all_rewards))
            else:
                value = agg_result.mean_reward
            
            results.append(Fitness(
                value=value,
                metadata={
                    'mean_reward': agg_result.mean_reward,
                    'std_reward': agg_result.std_reward,
                    'min_reward': agg_result.min_reward,
                    'max_reward': agg_result.max_reward,
                    'mean_length': agg_result.mean_length,
                    'n_episodes': self.n_episodes
                }
            ))
        
        return results
```

---

## Behavior Extraction for Novelty Search

```python
from evolve.diversity import BehaviorCharacterization


class FinalStateBehavior:
    """
    Behavior = final observation of episode.
    
    Common for maze/navigation tasks.
    """
    
    def __init__(self, decoder: 'Decoder', env_factory: Callable):
        self.decoder = decoder
        self.env_factory = env_factory
    
    def characterize(self, individual: Individual) -> np.ndarray:
        policy = self.decoder.decode(individual.genome)
        env = self.env_factory()
        
        rollout = StandardRollout()
        result = rollout(policy, env, record_trajectory=True)
        
        return result.observations[-1]


class TrajectoryBehavior:
    """
    Behavior = summary statistics of trajectory.
    
    More informative for complex behaviors.
    """
    
    def __init__(
        self,
        decoder: 'Decoder',
        env_factory: Callable,
        n_samples: int = 10
    ):
        self.decoder = decoder
        self.env_factory = env_factory
        self.n_samples = n_samples
    
    def characterize(self, individual: Individual) -> np.ndarray:
        policy = self.decoder.decode(individual.genome)
        env = self.env_factory()
        
        rollout = StandardRollout()
        result = rollout(policy, env, record_trajectory=True)
        
        # Sample positions from trajectory
        trajectory = np.array(result.observations)
        if len(trajectory) <= self.n_samples:
            indices = list(range(len(trajectory)))
        else:
            indices = np.linspace(0, len(trajectory) - 1, self.n_samples, dtype=int)
        
        return trajectory[indices].flatten()
```
