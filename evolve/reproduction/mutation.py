"""
Protocol mutation operators for Evolvable Reproduction Protocols.

This module implements mutation operators that modify reproduction protocols,
enabling the evolution of reproductive strategies. Mutations can:
- Modify parameter values (continuous)
- Switch between strategy types (discrete)
- Activate/deactivate components
- Add/remove/modify junk_data (neutral drift)

Key Components:
- ProtocolMutator: Main mutation interface
- Parameter mutation operators
- Junk data mutation operators
- Activation toggle mutations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    MatchabilityFunction,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MutationConfig:
    """
    Configuration for protocol mutation.

    Attributes:
        param_mutation_rate: Probability of mutating each parameter
        param_mutation_strength: Standard deviation for Gaussian noise
        type_mutation_rate: Probability of switching to a different type
        activation_mutation_rate: Probability of toggling active flag
        junk_add_rate: Probability of adding new junk data entry
        junk_remove_rate: Probability of removing a junk data entry
        junk_modify_rate: Probability of modifying existing junk data
        junk_activate_rate: Probability of promoting junk to active param
    """

    param_mutation_rate: float = 0.1
    param_mutation_strength: float = 0.1
    type_mutation_rate: float = 0.05
    activation_mutation_rate: float = 0.02
    junk_add_rate: float = 0.05
    junk_remove_rate: float = 0.03
    junk_modify_rate: float = 0.1
    junk_activate_rate: float = 0.02


# =============================================================================
# Type Choices
# =============================================================================

# Available matchability types
MATCHABILITY_TYPES = [
    "accept_all",
    "reject_all",
    "distance_threshold",
    "similarity_threshold",
    "fitness_ratio",
    "different_niche",
    "probabilistic",
    "diversity_seeking",
]

# Available intent types
INTENT_TYPES = [
    "always_willing",
    "never_willing",
    "fitness_threshold",
    "fitness_rank_threshold",
    "resource_budget",
    "age_dependent",
    "probabilistic",
]

# Crossover types for mutation
CROSSOVER_TYPES = list(CrossoverType)


# =============================================================================
# Parameter Mutation
# =============================================================================


def mutate_params(
    params: dict[str, float],
    rng: Random,
    mutation_rate: float,
    mutation_strength: float,
) -> dict[str, float]:
    """
    Mutate parameter values with Gaussian noise.

    Args:
        params: Original parameters
        rng: Random number generator
        mutation_rate: Probability of mutating each parameter
        mutation_strength: Standard deviation of Gaussian noise

    Returns:
        New parameter dictionary with mutated values
    """
    new_params = dict(params)
    for key, value in new_params.items():
        if rng.random() < mutation_rate:
            # Add Gaussian noise
            noise = rng.gauss(0, mutation_strength)
            new_value = value + noise
            # Clamp to reasonable range for probability-like params
            if key.endswith("_prob") or key == "probability" or key.endswith("_ratio"):
                new_value = max(0.0, min(1.0, new_value))
            new_params[key] = new_value
    return new_params


def add_random_param(
    params: dict[str, float],
    rng: Random,
) -> dict[str, float]:
    """
    Add a random parameter (useful for promoting junk).

    Args:
        params: Original parameters
        rng: Random number generator

    Returns:
        New parameter dictionary with added parameter
    """
    new_params = dict(params)
    # Generate a random parameter name
    param_names = ["threshold", "ratio", "alpha", "beta", "weight", "factor"]
    name = rng.choice(param_names)
    # Ensure unique key
    if name in new_params:
        name = f"{name}_{rng.randint(1, 100)}"
    new_params[name] = rng.random()
    return new_params


# =============================================================================
# Junk Data Mutation
# =============================================================================


def mutate_junk_data(
    junk_data: dict[str, Any],
    rng: Random,
    config: MutationConfig,
) -> dict[str, Any]:
    """
    Mutate junk data for neutral drift.

    Args:
        junk_data: Original junk data dictionary
        rng: Random number generator
        config: Mutation configuration

    Returns:
        New junk data dictionary with mutations
    """
    new_junk = dict(junk_data)

    # Possibly add new junk entry
    if rng.random() < config.junk_add_rate:
        key = f"junk_{rng.randint(0, 10000)}"
        # Random value type
        choice = rng.random()
        if choice < 0.5:
            new_junk[key] = rng.random()  # float
        elif choice < 0.8:
            new_junk[key] = rng.randint(0, 100)  # int
        else:
            new_junk[key] = {"nested": rng.random()}  # dict

    # Possibly remove existing junk
    if new_junk and rng.random() < config.junk_remove_rate:
        key_to_remove = rng.choice(list(new_junk.keys()))
        del new_junk[key_to_remove]

    # Modify existing junk values
    for key in list(new_junk.keys()):
        if rng.random() < config.junk_modify_rate:
            value = new_junk[key]
            if isinstance(value, float):
                new_junk[key] = value + rng.gauss(0, 0.1)
            elif isinstance(value, int):
                new_junk[key] = value + rng.randint(-5, 5)
            elif isinstance(value, dict):
                # Recursively modify nested dict
                for nested_key in value:
                    if isinstance(value[nested_key], int | float):
                        value[nested_key] = float(value[nested_key]) + rng.gauss(0, 0.1)

    return new_junk


def promote_junk_to_param(
    junk_data: dict[str, Any],
    params: dict[str, float],
    rng: Random,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Promote a junk data entry to an active parameter.

    This enables "dormant" strategies to become active through mutation.

    Args:
        junk_data: Original junk data
        params: Current active parameters
        rng: Random number generator

    Returns:
        Tuple of (new_junk_data, new_params)
    """
    if not junk_data:
        return dict(junk_data), dict(params)

    new_junk = dict(junk_data)
    new_params = dict(params)

    # Find promotable entries (numeric values only)
    promotable = [(k, v) for k, v in junk_data.items() if isinstance(v, int | float)]
    if not promotable:
        return new_junk, new_params

    # Choose one to promote
    key, value = rng.choice(promotable)

    # Remove from junk
    del new_junk[key]

    # Add to params with a suitable name
    param_name = key.replace("junk_", "")
    if param_name in new_params:
        param_name = f"{param_name}_{rng.randint(1, 100)}"
    new_params[param_name] = float(value)

    return new_junk, new_params


def demote_param_to_junk(
    params: dict[str, float],
    junk_data: dict[str, Any],
    rng: Random,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Demote an active parameter to junk data.

    This preserves the value for potential future reactivation.

    Args:
        params: Current active parameters
        junk_data: Original junk data
        rng: Random number generator

    Returns:
        Tuple of (new_params, new_junk_data)
    """
    if not params:
        return dict(params), dict(junk_data)

    new_params = dict(params)
    new_junk = dict(junk_data)

    # Choose a parameter to demote
    key = rng.choice(list(params.keys()))
    value = new_params.pop(key)

    # Add to junk
    junk_key = f"dormant_{key}"
    new_junk[junk_key] = value

    return new_params, new_junk


# =============================================================================
# Component Mutators
# =============================================================================


def mutate_matchability(
    matchability: MatchabilityFunction,
    rng: Random,
    config: MutationConfig,
) -> MatchabilityFunction:
    """
    Mutate a matchability function.

    Args:
        matchability: Original matchability function
        rng: Random number generator
        config: Mutation configuration

    Returns:
        New mutated matchability function
    """
    new_type = matchability.type
    new_params = mutate_params(
        matchability.params,
        rng,
        config.param_mutation_rate,
        config.param_mutation_strength,
    )
    new_active = matchability.active

    # Possibly switch type
    if rng.random() < config.type_mutation_rate:
        new_type = rng.choice(MATCHABILITY_TYPES)
        # Reset params for new type
        new_params = {}

    # Possibly toggle activation
    if rng.random() < config.activation_mutation_rate:
        new_active = not new_active

    return MatchabilityFunction(
        type=new_type,
        params=new_params,
        active=new_active,
    )


def mutate_intent(
    intent: ReproductionIntentPolicy,
    rng: Random,
    config: MutationConfig,
) -> ReproductionIntentPolicy:
    """
    Mutate an intent policy.

    Args:
        intent: Original intent policy
        rng: Random number generator
        config: Mutation configuration

    Returns:
        New mutated intent policy
    """
    new_type = intent.type
    new_params = mutate_params(
        intent.params,
        rng,
        config.param_mutation_rate,
        config.param_mutation_strength,
    )
    new_active = intent.active

    # Possibly switch type
    if rng.random() < config.type_mutation_rate:
        new_type = rng.choice(INTENT_TYPES)
        new_params = {}

    # Possibly toggle activation
    if rng.random() < config.activation_mutation_rate:
        new_active = not new_active

    return ReproductionIntentPolicy(
        type=new_type,
        params=new_params,
        active=new_active,
    )


def mutate_crossover(
    crossover: CrossoverProtocolSpec,
    rng: Random,
    config: MutationConfig,
) -> CrossoverProtocolSpec:
    """
    Mutate a crossover specification.

    Args:
        crossover: Original crossover spec
        rng: Random number generator
        config: Mutation configuration

    Returns:
        New mutated crossover spec
    """
    new_type = crossover.type
    new_params = mutate_params(
        crossover.params,
        rng,
        config.param_mutation_rate,
        config.param_mutation_strength,
    )
    new_active = crossover.active

    # Possibly switch type
    if rng.random() < config.type_mutation_rate:
        new_type = rng.choice(CROSSOVER_TYPES)
        new_params = {}

    # Possibly toggle activation
    if rng.random() < config.activation_mutation_rate:
        new_active = not new_active

    return CrossoverProtocolSpec(
        type=new_type,
        params=new_params,
        active=new_active,
    )


# =============================================================================
# Main Protocol Mutator
# =============================================================================


@dataclass
class ProtocolMutator:
    """
    Mutates reproduction protocols.

    This operator applies mutations to all components of a reproduction
    protocol, enabling the evolution of reproductive strategies.

    Attributes:
        config: Mutation configuration
    """

    config: MutationConfig = field(default_factory=MutationConfig)

    def mutate(
        self,
        protocol: ReproductionProtocol,
        rng: Random,
    ) -> ReproductionProtocol:
        """
        Apply mutations to a protocol.

        Args:
            protocol: Original protocol to mutate
            rng: Random number generator

        Returns:
            New mutated protocol
        """
        # Mutate each component
        new_matchability = mutate_matchability(protocol.matchability, rng, self.config)
        new_intent = mutate_intent(protocol.intent, rng, self.config)
        new_crossover = mutate_crossover(protocol.crossover, rng, self.config)
        new_junk = mutate_junk_data(protocol.junk_data, rng, self.config)

        # Possibly promote junk to active params
        if rng.random() < self.config.junk_activate_rate:
            choice = rng.random()
            if choice < 0.33:
                new_junk, new_params = promote_junk_to_param(
                    new_junk, dict(new_matchability.params), rng
                )
                new_matchability = MatchabilityFunction(
                    type=new_matchability.type,
                    params=new_params,
                    active=new_matchability.active,
                )
            elif choice < 0.67:
                new_junk, new_params = promote_junk_to_param(new_junk, dict(new_intent.params), rng)
                new_intent = ReproductionIntentPolicy(
                    type=new_intent.type,
                    params=new_params,
                    active=new_intent.active,
                )
            else:
                new_junk, new_params = promote_junk_to_param(
                    new_junk, dict(new_crossover.params), rng
                )
                new_crossover = CrossoverProtocolSpec(
                    type=new_crossover.type,
                    params=new_params,
                    active=new_crossover.active,
                )

        return ReproductionProtocol(
            matchability=new_matchability,
            intent=new_intent,
            crossover=new_crossover,
            junk_data=new_junk,
        )

    def mutate_single_component(
        self,
        protocol: ReproductionProtocol,
        rng: Random,
        component: str = "matchability",
    ) -> ReproductionProtocol:
        """
        Mutate only a single component of the protocol.

        Args:
            protocol: Original protocol
            rng: Random number generator
            component: One of "matchability", "intent", "crossover", "junk"

        Returns:
            Protocol with only the specified component mutated
        """
        if component == "matchability":
            new_matchability = mutate_matchability(protocol.matchability, rng, self.config)
            return ReproductionProtocol(
                matchability=new_matchability,
                intent=protocol.intent,
                crossover=protocol.crossover,
                junk_data=protocol.junk_data,
            )
        elif component == "intent":
            new_intent = mutate_intent(protocol.intent, rng, self.config)
            return ReproductionProtocol(
                matchability=protocol.matchability,
                intent=new_intent,
                crossover=protocol.crossover,
                junk_data=protocol.junk_data,
            )
        elif component == "crossover":
            new_crossover = mutate_crossover(protocol.crossover, rng, self.config)
            return ReproductionProtocol(
                matchability=protocol.matchability,
                intent=protocol.intent,
                crossover=new_crossover,
                junk_data=protocol.junk_data,
            )
        elif component == "junk":
            new_junk = mutate_junk_data(protocol.junk_data, rng, self.config)
            return ReproductionProtocol(
                matchability=protocol.matchability,
                intent=protocol.intent,
                crossover=protocol.crossover,
                junk_data=new_junk,
            )
        else:
            raise ValueError(f"Unknown component: {component}")
