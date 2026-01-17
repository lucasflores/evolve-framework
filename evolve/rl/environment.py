"""
Environment abstractions for reinforcement learning.

Provides framework-neutral environment interfaces that work
with NumPy, Gymnasium, or custom environments.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol, TypeVar, Generic, runtime_checkable

import numpy as np


Observation = TypeVar("Observation", bound=np.ndarray)
Action = TypeVar("Action", bound=np.ndarray | int)


@dataclass
class Space:
    """
    Specification of observation/action space.
    
    Framework-neutral representation of Gym-style spaces.
    Supports Box (continuous) and Discrete spaces.
    
    Attributes:
        shape: Shape of the space
        dtype: NumPy dtype for elements
        low: Lower bounds for Box spaces
        high: Upper bounds for Box spaces
        n: Cardinality for Discrete spaces
    
    Example:
        >>> # Box space: 4-dimensional continuous
        >>> obs_space = Space.box(low=-10, high=10, shape=(4,))
        >>> obs_space.sample(rng)  # array of 4 floats
        
        >>> # Discrete space: 2 actions
        >>> act_space = Space.discrete(n=2)
        >>> act_space.sample(rng)  # 0 or 1
    """
    
    shape: tuple[int, ...]
    dtype: np.dtype
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    n: int | None = None
    
    @classmethod
    def box(
        cls,
        low: np.ndarray | float,
        high: np.ndarray | float,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> "Space":
        """
        Create Box (continuous) space.
        
        Args:
            low: Lower bound (scalar or array)
            high: Upper bound (scalar or array)
            shape: Shape of the space
            dtype: Data type (default: float32)
            
        Returns:
            Box space
        """
        if isinstance(low, (int, float)):
            low = np.full(shape, low, dtype=dtype)
        if isinstance(high, (int, float)):
            high = np.full(shape, high, dtype=dtype)
        return cls(shape=shape, dtype=np.dtype(dtype), low=low, high=high)
    
    @classmethod
    def discrete(cls, n: int) -> "Space":
        """
        Create Discrete space.
        
        Args:
            n: Number of discrete values (0 to n-1)
            
        Returns:
            Discrete space
        """
        return cls(shape=(), dtype=np.dtype(np.int64), n=n)
    
    def sample(self, rng: Random | np.random.Generator) -> np.ndarray | int:
        """
        Sample random element from space.
        
        Args:
            rng: Random number generator (Python Random or NumPy Generator)
            
        Returns:
            Random element from space
        """
        if self.n is not None:
            # Discrete space
            if isinstance(rng, np.random.Generator):
                return int(rng.integers(0, self.n))
            return rng.randint(0, self.n - 1)
        
        # Box space
        if self.low is None or self.high is None:
            raise ValueError("Box space requires low and high bounds")
        
        if isinstance(rng, np.random.Generator):
            return rng.uniform(self.low, self.high).astype(self.dtype)
        
        # Python Random - sample element by element
        result = np.empty(self.shape, dtype=self.dtype)
        for idx in np.ndindex(self.shape):
            result[idx] = rng.uniform(float(self.low[idx]), float(self.high[idx]))
        return result
    
    def contains(self, x: np.ndarray | int) -> bool:
        """
        Check if value is in space.
        
        Args:
            x: Value to check
            
        Returns:
            True if value is in space
        """
        if self.n is not None:
            # Discrete space
            if isinstance(x, (int, np.integer)):
                return 0 <= int(x) < self.n
            return False
        
        # Box space
        if self.low is None or self.high is None:
            return True  # Unbounded
        
        x = np.asarray(x)
        if x.shape != self.shape:
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))
    
    @property
    def is_discrete(self) -> bool:
        """Whether this is a discrete space."""
        return self.n is not None
    
    @property
    def is_continuous(self) -> bool:
        """Whether this is a continuous (box) space."""
        return self.n is None
    
    def __repr__(self) -> str:
        if self.n is not None:
            return f"Discrete({self.n})"
        return f"Box({self.shape}, {self.dtype})"


@runtime_checkable
class Environment(Protocol[Observation, Action]):
    """
    Environment interface for policy evaluation.
    
    Framework-neutral: works with NumPy, Gym, or custom envs.
    All implementations must be deterministic given the same seed.
    """
    
    @property
    def observation_space(self) -> Space:
        """Specification of observation shape/type."""
        ...
    
    @property
    def action_space(self) -> Space:
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
    
    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, info)
        """
        ...


@runtime_checkable
class VectorizedEnvironment(Protocol[Observation, Action]):
    """
    Batched environment for parallel evaluation.
    
    Runs multiple episodes simultaneously for efficiency.
    """
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        ...
    
    @property
    def observation_space(self) -> Space:
        """Specification of observation shape/type."""
        ...
    
    @property
    def action_space(self) -> Space:
        """Specification of action shape/type."""
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
    
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            actions: Batch of actions, shape: (num_envs, *action_shape)
            
        Returns:
            (observations, rewards, dones, infos)
        """
        ...


@dataclass
class GymAdapter:
    """
    Wraps a Gymnasium environment to conform to our Environment protocol.
    
    Handles the gymnasium API (which returns (obs, info) from reset and
    has separate terminated/truncated flags) and converts to our simpler
    interface.
    
    Attributes:
        env: The gymnasium.Env instance
    
    Example:
        >>> import gymnasium as gym
        >>> gym_env = gym.make("CartPole-v1")
        >>> env = GymAdapter(gym_env)
        >>> obs = env.reset(seed=42)
        >>> obs, reward, done, info = env.step(0)
    """
    
    env: Any  # gymnasium.Env - not typed to avoid import
    
    @property
    def observation_space(self) -> Space:
        """Convert Gym observation space to our Space."""
        s = self.env.observation_space
        if hasattr(s, "n"):
            # Discrete observation space (rare but possible)
            return Space.discrete(s.n)
        return Space(
            shape=s.shape,
            dtype=np.dtype(s.dtype),
            low=np.array(s.low, dtype=s.dtype),
            high=np.array(s.high, dtype=s.dtype),
        )
    
    @property
    def action_space(self) -> Space:
        """Convert Gym action space to our Space."""
        s = self.env.action_space
        if hasattr(s, "n"):
            return Space.discrete(s.n)
        return Space(
            shape=s.shape,
            dtype=np.dtype(s.dtype),
            low=np.array(s.low, dtype=s.dtype),
            high=np.array(s.high, dtype=s.dtype),
        )
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset environment.
        
        Args:
            seed: Random seed for deterministic reset
            
        Returns:
            Initial observation
        """
        obs, _ = self.env.reset(seed=seed)
        return np.asarray(obs)
    
    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, info)
        """
        # Convert numpy scalar to python int for discrete actions
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = int(action)
        elif isinstance(action, np.integer):
            action = int(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.asarray(obs), float(reward), done, info
    
    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()
    
    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()


@dataclass
class SimpleEnvironment:
    """
    Simple environment for testing without Gymnasium dependency.
    
    A minimal environment that can be used for unit tests.
    
    Attributes:
        obs_space: Observation space specification
        act_space: Action space specification
        max_steps: Maximum steps per episode
    """
    
    obs_space: Space
    act_space: Space
    max_steps: int = 100
    _step: int = field(default=0, init=False)
    _rng: np.random.Generator | None = field(default=None, init=False)
    _state: np.ndarray | None = field(default=None, init=False)
    
    @property
    def observation_space(self) -> Space:
        return self.obs_space
    
    @property
    def action_space(self) -> Space:
        return self.act_space
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to random initial state."""
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._state = self._rng.uniform(-1, 1, size=self.obs_space.shape).astype(
            self.obs_space.dtype
        )
        return self._state.copy()
    
    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Take a step - simple dynamics for testing."""
        if self._rng is None:
            raise RuntimeError("Environment not reset")
        
        self._step += 1
        
        # Simple dynamics: state moves toward action
        if self.act_space.is_discrete:
            # Discrete: random state change
            delta = self._rng.uniform(-0.1, 0.1, size=self.obs_space.shape)
        else:
            # Continuous: state moves toward action
            action = np.asarray(action)
            delta = 0.1 * (action[:len(self._state)] - self._state)  # type: ignore
        
        self._state = np.clip(
            self._state + delta.astype(self.obs_space.dtype),  # type: ignore
            self.obs_space.low,
            self.obs_space.high,
        )
        
        # Simple reward: negative distance from origin
        reward = -float(np.sum(self._state**2))
        
        done = self._step >= self.max_steps
        
        return self._state.copy(), reward, done, {"step": self._step}
