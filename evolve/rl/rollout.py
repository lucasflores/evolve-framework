"""
Rollout execution for reinforcement learning.

Provides episode execution and multi-episode evaluation
for policy fitness computation.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from evolve.rl.environment import Environment
    from evolve.rl.policy import Policy


@dataclass
class RolloutResult:
    """
    Result of evaluating a policy in an environment for one episode.

    Attributes:
        total_reward: Sum of rewards over the episode
        episode_length: Number of steps taken
        observations: List of observations (if recorded)
        actions: List of actions taken (if recorded)
        rewards: List of rewards received (if recorded)
        info: Final info dict from environment
    """

    total_reward: float
    episode_length: int
    observations: list[np.ndarray] | None = None
    actions: list[np.ndarray | int] | None = None
    rewards: list[float] | None = None
    info: dict[str, Any] | None = None


@dataclass
class AggregatedResult:
    """
    Aggregated results from multiple episodes.

    Attributes:
        mean_reward: Average total reward
        std_reward: Standard deviation of rewards
        min_reward: Minimum reward achieved
        max_reward: Maximum reward achieved
        mean_length: Average episode length
        n_episodes: Number of episodes run
        all_rewards: List of all episode rewards
    """

    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    n_episodes: int
    all_rewards: list[float]


class StandardRollout:
    """
    Standard episode rollout.

    Executes a policy in an environment for one episode,
    accumulating rewards and optionally recording trajectory.

    Deterministic given seed (for reproducibility).

    Example:
        >>> rollout = StandardRollout()
        >>> result = rollout(policy, env, seed=42, max_steps=200)
        >>> print(f"Reward: {result.total_reward}")
    """

    def __call__(
        self,
        policy: Policy,
        env: Environment,
        seed: int | None = None,
        *,
        max_steps: int | None = None,
        render: bool = False,
        record_trajectory: bool = False,
    ) -> RolloutResult:
        """
        Run policy in environment for one episode.

        Args:
            policy: Policy to evaluate
            env: Environment to run in
            seed: Seed for environment reset
            max_steps: Maximum episode length (None = no limit)
            render: Whether to render (calls env.render())
            record_trajectory: Whether to store observations/actions

        Returns:
            Evaluation result with total reward and episode length
        """
        observation = env.reset(seed=seed)

        total_reward = 0.0
        steps = 0

        observations = [observation.copy()] if record_trajectory else None
        actions: list[np.ndarray | int] | None = [] if record_trajectory else None
        rewards: list[float] | None = [] if record_trajectory else None

        # Reset stateful policy if applicable
        if hasattr(policy, "reset_state"):
            policy.reset_state()  # type: ignore

        info: dict[str, Any] = {}

        while True:
            if render and hasattr(env, "render"):
                env.render()  # type: ignore

            action = policy(observation)

            if record_trajectory and actions is not None:
                actions.append(action)

            observation, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if record_trajectory:
                if observations is not None:
                    observations.append(observation.copy())
                if rewards is not None:
                    rewards.append(reward)

            if done or (max_steps is not None and steps >= max_steps):
                break

        return RolloutResult(
            total_reward=total_reward,
            episode_length=steps,
            observations=observations,
            actions=actions,
            rewards=rewards,
            info=info,
        )


def evaluate_policy(
    policy: Policy,
    env_factory: Callable[[], Environment],
    n_episodes: int,
    seeds: list[int],
    *,
    max_steps: int | None = None,
) -> AggregatedResult:
    """
    Evaluate policy over multiple episodes.

    Creates fresh environment for each episode to avoid
    state leakage between episodes.

    Args:
        policy: Policy to evaluate
        env_factory: Callable that creates new environment instance
        n_episodes: Number of episodes to run
        seeds: Seeds for each episode (len must equal n_episodes)
        max_steps: Max steps per episode

    Returns:
        Aggregated statistics over all episodes

    Example:
        >>> def make_env():
        ...     return GymAdapter(gym.make("CartPole-v1"))
        >>> result = evaluate_policy(
        ...     policy, make_env, n_episodes=10,
        ...     seeds=list(range(10)), max_steps=500
        ... )
        >>> print(f"Mean reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
    """
    if len(seeds) != n_episodes:
        raise ValueError(f"Expected {n_episodes} seeds, got {len(seeds)}")

    rollout = StandardRollout()
    all_rewards: list[float] = []
    all_lengths: list[int] = []

    for i in range(n_episodes):
        env = env_factory()
        result = rollout(policy, env, seed=seeds[i], max_steps=max_steps)
        all_rewards.append(result.total_reward)
        all_lengths.append(result.episode_length)

        # Close environment if it has a close method
        if hasattr(env, "close"):
            env.close()  # type: ignore

    return AggregatedResult(
        mean_reward=float(np.mean(all_rewards)),
        std_reward=float(np.std(all_rewards)),
        min_reward=float(np.min(all_rewards)),
        max_reward=float(np.max(all_rewards)),
        mean_length=float(np.mean(all_lengths)),
        n_episodes=n_episodes,
        all_rewards=all_rewards,
    )
