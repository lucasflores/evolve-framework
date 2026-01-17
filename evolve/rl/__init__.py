"""
Reinforcement Learning integration module.

Provides environment, policy, and rollout abstractions for
neuroevolution in reinforcement learning settings.

Key components:
- Space: Observation/action space specification
- Environment: Protocol for RL environments
- GymAdapter: Wrapper for Gymnasium environments
- Policy: Protocol for policies (LinearPolicy, MLPPolicy, RecurrentPolicy)
- Rollout: Episode execution and evaluation
- RLEvaluator: Combines decoder + environment + rollout for fitness
"""

from evolve.rl.environment import (
    Space,
    Environment,
    VectorizedEnvironment,
    GymAdapter,
)
from evolve.rl.policy import (
    Policy,
    StatefulPolicy,
    LinearPolicy,
    MLPPolicy,
    RecurrentPolicy,
)
from evolve.rl.rollout import (
    RolloutResult,
    AggregatedResult,
    StandardRollout,
    evaluate_policy,
)
from evolve.rl.evaluator import (
    RLEvaluator,
)

__all__ = [
    # Environment
    "Space",
    "Environment",
    "VectorizedEnvironment",
    "GymAdapter",
    # Policies
    "Policy",
    "StatefulPolicy",
    "LinearPolicy",
    "MLPPolicy",
    "RecurrentPolicy",
    # Rollout
    "RolloutResult",
    "AggregatedResult",
    "StandardRollout",
    "evaluate_policy",
    # Evaluator
    "RLEvaluator",
]
