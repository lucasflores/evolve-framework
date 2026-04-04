"""
Evolvable Reproduction Protocols (ERP) module.

This module enables individuals to encode, evolve, and execute their own
reproductive compatibility logic and offspring construction strategies.

Key Components:
- ReproductionProtocol: The evolvable reproduction protocol genome (RPG)
- MatchabilityFunction: Determines mate acceptability
- ReproductionIntentPolicy: Governs when reproduction is attempted
- CrossoverProtocolSpec: Specifies offspring genome construction
- ERPEngine: Evolution engine with ERP support
"""

from evolve.reproduction.crossover_protocol import (
    BlendCrossoverExecutor,
    CloneCrossoverExecutor,
    CrossoverExecutor,
    CrossoverRegistry,
    # Built-in executors
    SinglePointCrossoverExecutor,
    TwoPointCrossoverExecutor,
    UniformCrossoverExecutor,
    execute_crossover,
    inherit_protocol,
    safe_execute_crossover,
    validate_offspring,
)
from evolve.reproduction.intent import (
    AgeDependentIntent,
    # Built-in evaluators
    AlwaysWillingIntent,
    FitnessRankThresholdIntent,
    FitnessThresholdIntent,
    IntentEvaluator,
    IntentRegistry,
    NeverWillingIntent,
    ProbabilisticIntent,
    ResourceBudgetIntent,
    evaluate_intent,
    safe_evaluate_intent,
)
from evolve.reproduction.matchability import (
    # Built-in evaluators
    AcceptAllMatchability,
    DifferentNicheMatchability,
    DistanceThresholdMatchability,
    DiversitySeekingMatchability,
    FitnessRatioMatchability,
    MatchabilityEvaluator,
    MatchabilityRegistry,
    ProbabilisticMatchability,
    RejectAllMatchability,
    SimilarityThresholdMatchability,
    evaluate_matchability,
    safe_evaluate_matchability,
)
from evolve.reproduction.mutation import (
    MutationConfig,
    ProtocolMutator,
    demote_param_to_junk,
    mutate_crossover,
    mutate_intent,
    mutate_junk_data,
    mutate_matchability,
    promote_junk_to_param,
)
from evolve.reproduction.protocol import (
    CrossoverProtocolSpec,
    CrossoverType,
    IntentContext,
    MatchabilityFunction,
    MateContext,
    ReproductionEvent,
    ReproductionIntentPolicy,
    ReproductionProtocol,
)
from evolve.reproduction.recovery import (
    CompositeRecovery,
    ImmigrationRecovery,
    MutationBoostRecovery,
    RecoveryStrategy,
    RelaxedMatchingRecovery,
)
from evolve.reproduction.sandbox import (
    StepCounter,
    StepLimitExceeded,
    safe_execute,
    sandboxed_execute,
)

# Note: Callbacks and ERPEngine are imported separately to avoid circular imports:
#   from evolve.reproduction.callbacks import ERPMetricsCallback, ERPLoggerCallback
#   from evolve.reproduction.engine import ERPConfig, ERPEngine

__all__ = [
    # Core protocol types
    "ReproductionProtocol",
    "MatchabilityFunction",
    "ReproductionIntentPolicy",
    "CrossoverProtocolSpec",
    "CrossoverType",
    # Context types
    "MateContext",
    "IntentContext",
    # Sandbox
    "StepCounter",
    "StepLimitExceeded",
    "sandboxed_execute",
    "safe_execute",
    # Events
    "ReproductionEvent",
    # Matchability
    "MatchabilityEvaluator",
    "MatchabilityRegistry",
    "evaluate_matchability",
    "safe_evaluate_matchability",
    "AcceptAllMatchability",
    "RejectAllMatchability",
    "DistanceThresholdMatchability",
    "SimilarityThresholdMatchability",
    "FitnessRatioMatchability",
    "DifferentNicheMatchability",
    "ProbabilisticMatchability",
    "DiversitySeekingMatchability",
    # Crossover
    "CrossoverExecutor",
    "CrossoverRegistry",
    "execute_crossover",
    "safe_execute_crossover",
    "inherit_protocol",
    "validate_offspring",
    "SinglePointCrossoverExecutor",
    "TwoPointCrossoverExecutor",
    "UniformCrossoverExecutor",
    "BlendCrossoverExecutor",
    "CloneCrossoverExecutor",
    # Intent
    "IntentEvaluator",
    "IntentRegistry",
    "evaluate_intent",
    "safe_evaluate_intent",
    "AlwaysWillingIntent",
    "NeverWillingIntent",
    "FitnessThresholdIntent",
    "FitnessRankThresholdIntent",
    "ResourceBudgetIntent",
    "AgeDependentIntent",
    "ProbabilisticIntent",
    # Mutation
    "MutationConfig",
    "ProtocolMutator",
    "mutate_matchability",
    "mutate_intent",
    "mutate_crossover",
    "mutate_junk_data",
    "promote_junk_to_param",
    "demote_param_to_junk",
    # Recovery
    "RecoveryStrategy",
    "ImmigrationRecovery",
    "MutationBoostRecovery",
    "RelaxedMatchingRecovery",
    "CompositeRecovery",
    # Callbacks - import separately: from evolve.reproduction.callbacks import ERPMetricsCallback
    # "ERPMetrics",
    # "ERPMetricsCallback",
    # "ERPLoggerCallback",
    # Engine - import separately: from evolve.reproduction.engine import ERPConfig, ERPEngine
    # "ERPConfig",
    # "ERPEngine",
]
