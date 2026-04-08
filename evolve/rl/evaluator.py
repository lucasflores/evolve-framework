"""
RL Evaluator for neuroevolution.

Combines decoder, environment, and rollout to evaluate
individuals as RL policies and compute fitness.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from evolve.core.types import Fitness, Individual
from evolve.evaluation.evaluator import EvaluatorCapabilities
from evolve.rl.rollout import AggregatedResult, evaluate_policy

G = TypeVar("G")  # Genome type


@dataclass
class RLEvaluator(Generic[G]):
    """
    Evaluates individuals as RL policies.

    Combines:
    - Decoder: genome -> policy
    - Environment factory: creates fresh env for each evaluation
    - Rollout: executes policy in environment

    Fitness is computed from episode rewards, optionally aggregated
    over multiple episodes.

    Attributes:
        decoder: Converts genome to policy
        env_factory: Creates new environment instance
        n_episodes: Number of episodes per evaluation (default: 1)
        max_steps: Maximum steps per episode (None = no limit)
        aggregate: How to aggregate rewards ("mean", "min", "median")
        negate: If True, negate fitness (for minimization)

    Example:
        >>> evaluator = RLEvaluator(
        ...     decoder=policy_decoder,
        ...     env_factory=lambda: GymAdapter(gym.make("CartPole-v1")),
        ...     n_episodes=5,
        ...     max_steps=500,
        ...     aggregate="mean"
        ... )
        >>> fitness = evaluator.evaluate(individuals, seed=42)
    """

    decoder: Any  # Decoder[G, Policy] - using Any to avoid circular import
    env_factory: Callable[[], Any]  # Callable[[], Environment]
    n_episodes: int = 1
    max_steps: int | None = None
    aggregate: str = "mean"  # "mean", "min", "median"
    negate: bool = False  # Set True if minimizing fitness

    @property
    def capabilities(self) -> EvaluatorCapabilities:
        """Evaluator capabilities."""
        return EvaluatorCapabilities(
            batchable=False,  # Environments typically not vectorized
            stochastic=True,  # Episodes may vary with seed
            stateful=False,
        )

    def evaluate(
        self,
        individuals: Sequence[Individual[G]],
        seed: int | None = None,
    ) -> Sequence[Fitness]:
        """
        Evaluate batch of individuals as RL policies.

        Args:
            individuals: Sequence of individuals to evaluate
            seed: Random seed for episode generation

        Returns:
            Sequence of fitness values
        """
        rng = np.random.default_rng(seed)
        results = []

        for ind in individuals:
            # Generate episode seeds
            episode_seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_episodes)]

            # Decode genome to policy
            policy = self.decoder.decode(ind.genome)

            # Run evaluation
            agg_result = evaluate_policy(
                policy,
                self.env_factory,
                self.n_episodes,
                episode_seeds,
                max_steps=self.max_steps,
            )

            # Compute fitness value
            fitness_value = self._compute_fitness(agg_result)

            if self.negate:
                fitness_value = -fitness_value

            results.append(
                Fitness(
                    values=np.array([fitness_value]),
                    metadata={
                        "mean_reward": agg_result.mean_reward,
                        "std_reward": agg_result.std_reward,
                        "min_reward": agg_result.min_reward,
                        "max_reward": agg_result.max_reward,
                        "mean_length": agg_result.mean_length,
                        "n_episodes": self.n_episodes,
                    },
                )
            )

        return results

    def evaluate_single(self, genome: G, seed: int | None = None) -> Fitness:
        """
        Evaluate single genome as RL policy.

        Args:
            genome: Genome to evaluate
            seed: Seed for episode generation

        Returns:
            Fitness based on episode rewards
        """
        # Decode genome to policy
        policy = self.decoder.decode(genome)

        # Generate episode seeds
        rng = np.random.default_rng(seed)
        episode_seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_episodes)]

        # Run evaluation
        agg_result = evaluate_policy(
            policy,
            self.env_factory,
            self.n_episodes,
            episode_seeds,
            max_steps=self.max_steps,
        )

        # Compute fitness value
        fitness_value = self._compute_fitness(agg_result)

        if self.negate:
            fitness_value = -fitness_value

        return Fitness(
            values=np.array([fitness_value]),
            metadata={
                "mean_reward": agg_result.mean_reward,
                "std_reward": agg_result.std_reward,
                "min_reward": agg_result.min_reward,
                "max_reward": agg_result.max_reward,
                "mean_length": agg_result.mean_length,
                "n_episodes": self.n_episodes,
            },
        )

    def _compute_fitness(self, result: AggregatedResult) -> float:
        """Compute fitness from aggregated result."""
        if self.aggregate == "mean":
            return result.mean_reward
        elif self.aggregate == "min":
            return result.min_reward
        elif self.aggregate == "median":
            return float(np.median(result.all_rewards))
        elif self.aggregate == "max":
            return result.max_reward
        else:
            return result.mean_reward


@dataclass
class PolicyDecoder(Generic[G]):
    """
    Simple decoder that converts vector genomes to policies.

    For use with VectorGenome where genes directly encode
    policy parameters.

    Attributes:
        policy_factory: Creates policy from parameter array

    Example:
        >>> decoder = PolicyDecoder(
        ...     policy_factory=lambda params: LinearPolicy.from_parameters(
        ...         params, obs_dim=4, action_dim=2, discrete=True
        ...     )
        ... )
    """

    policy_factory: Callable[[np.ndarray], Any]  # params -> Policy

    def decode(self, genome: G) -> Any:
        """
        Decode genome to policy.

        Args:
            genome: Genome with .genes attribute (VectorGenome)

        Returns:
            Policy instance
        """
        # Assume genome has genes attribute (VectorGenome)
        return self.policy_factory(genome.genes)  # type: ignore
